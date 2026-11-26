//! Layer 2 — Blackwell (`sm_120`) Discrete Cliff Pruner & Whole-Model Roofline Ranker.
//!
//! Filters candidate kernel knob tuples `(BM, BN, BK, stages, warps, cluster_ctas)` against
//! the Lean 4 `sm_120` hardware laws BEFORE invoking `nvcc`/`ptxas`, then ranks surviving
//! candidates by whole-model algorithmic bandwidth & TFLOP/s ceilings.

use leansonic_core::generated_spec::{
    SM120_COMBINED_L1_SMEM_KIB, SM120_MAX_SMEM_PER_SM_BYTES, SM120_MAX_WARPS_PER_SM,
    SM120_REGS_PER_SM, SM120_SYS_RESERVED_SMEM_BYTES,
};

#[derive(Debug, Clone)]
pub struct BlackwellSkuProfile {
    pub name: &'static str,
    pub sm_count: usize,
    pub vram_gib: f64,
    pub l2_mib: f64,
    pub sustained_dram_gbps: f64,
    pub dtod_copy_gbps: f64,
    pub pcie_htod_gbps: f64,
    pub fp8_tflops: f64,
}

impl BlackwellSkuProfile {
    pub fn rtx_pro_5000_300w() -> Self {
        Self {
            name: "NVIDIA RTX PRO 5000 72GB GDDR7 (300W, sm_120)",
            sm_count: 110,
            vram_gib: 72.0,
            l2_mib: 96.0,
            sustained_dram_gbps: 1344.0,
            dtod_copy_gbps: 890.0,
            pcie_htod_gbps: 51.4,
            fp8_tflops: 380.0,
        }
    }

    pub fn rtx_5070_ti_300w() -> Self {
        Self {
            name: "NVIDIA GeForce RTX 5070 Ti 16GB GDDR7 (300W, sm_120, Spilled Host)",
            sm_count: 70,
            vram_gib: 16.0,
            l2_mib: 48.0,
            sustained_dram_gbps: 759.8,
            dtod_copy_gbps: 447.0,
            pcie_htod_gbps: 25.7, // Coalesced PCIe Gen4 x16 pinned DDR4/DDR5
            fp8_tflops: 215.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CandidateKnob {
    pub label: String,
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
    pub stages: usize,
    pub threads_per_cta: usize,
    pub est_regs_per_thread: usize,
    pub user_smem_bytes: usize,
    pub ctas_per_sm: usize,
    pub cluster_ctas: usize,
    pub min_required_l1_kib: usize,
}

#[derive(Debug, Clone)]
pub struct EvaluatedCandidate {
    pub knob: CandidateKnob,
    pub carveout_kib: usize,
    pub remaining_l1_kib: usize,
    pub active_warps_per_sm: usize,
    pub wave_efficiency: f64,
    pub composite_score: f64,
}

pub fn sm120_carveout_kib(total_smem_per_sm_bytes: usize) -> Option<usize> {
    if total_smem_per_sm_bytes == 0 {
        Some(0)
    } else if total_smem_per_sm_bytes <= 8 * 1024 {
        Some(8)
    } else if total_smem_per_sm_bytes <= 16 * 1024 {
        Some(16)
    } else if total_smem_per_sm_bytes <= 32 * 1024 {
        Some(32)
    } else if total_smem_per_sm_bytes <= 64 * 1024 {
        Some(64)
    } else if total_smem_per_sm_bytes <= SM120_MAX_SMEM_PER_SM_BYTES {
        Some(100)
    } else {
        None
    }
}

/// Evaluate & prune a candidate knob configuration against `sm_120` discrete cliffs.
pub fn evaluate_candidate(
    sku: &BlackwellSkuProfile,
    knob: CandidateKnob,
) -> Result<EvaluatedCandidate, &'static str> {
    if knob.threads_per_cta == 0 || !knob.threads_per_cta.is_multiple_of(32) {
        return Err("threads_per_cta must be a positive multiple of 32");
    }
    if knob.cluster_ctas == 0 || knob.cluster_ctas > 2 {
        return Err("cluster_ctas > 2 rejected due to multi-wave cluster tail idling cliff");
    }

    let aligned_regs = knob.est_regs_per_thread.div_ceil(8) * 8;
    let total_regs = knob.threads_per_cta * aligned_regs * knob.ctas_per_sm;
    if total_regs > SM120_REGS_PER_SM {
        return Err("exceeds 65,536 register file budget (risks local-memory stack spill)");
    }

    let warps_per_sm = (knob.threads_per_cta / 32) * knob.ctas_per_sm;
    if warps_per_sm > SM120_MAX_WARPS_PER_SM {
        return Err("exceeds 48 warps/SM hardware limit");
    }

    // Include 1 KiB swizzle alignment pad + 1 KiB system SMEM reservation per CTA
    let smem_per_sm =
        knob.ctas_per_sm * (knob.user_smem_bytes + 1024 + SM120_SYS_RESERVED_SMEM_BYTES);
    let carveout_kib =
        sm120_carveout_kib(smem_per_sm).ok_or("exceeds 100 KiB SMEM/SM hardware ceiling")?;
    let remaining_l1_kib = SM120_COMBINED_L1_SMEM_KIB - carveout_kib;
    if remaining_l1_kib < knob.min_required_l1_kib {
        return Err("crossed 64->100 KiB SMEM carveout cliff, starving required L1 cache");
    }

    let sms_per_cluster = knob.cluster_ctas.div_ceil(knob.ctas_per_sm).max(1);
    let usable_sms = (sku.sm_count / sms_per_cluster) * sms_per_cluster;
    let wave_efficiency = (usable_sms as f64) / (sku.sm_count as f64);

    // Score favors balanced square/wide Tensor Core reuse, warp residency, and L1 headroom
    let arithmetic_reuse =
        ((knob.tile_m * knob.tile_n) as f64) / ((knob.tile_m + knob.tile_n) as f64).max(1.0);
    let warp_occ = (warps_per_sm as f64) / (SM120_MAX_WARPS_PER_SM as f64);
    let stage_bonus = 1.0 + 0.06 * (knob.stages.min(4) as f64);
    let composite_score =
        arithmetic_reuse * (0.55 + 0.45 * warp_occ) * wave_efficiency * stage_bonus;

    Ok(EvaluatedCandidate {
        knob,
        carveout_kib,
        remaining_l1_kib,
        active_warps_per_sm: warps_per_sm,
        wave_efficiency,
        composite_score,
    })
}

/// Rank a grid of candidate tile knobs, pruning all cliff-violating configurations.
pub fn rank_valid_candidates(
    sku: &BlackwellSkuProfile,
    candidates: Vec<CandidateKnob>,
) -> Vec<EvaluatedCandidate> {
    let mut valid: Vec<EvaluatedCandidate> = candidates
        .into_iter()
        .filter_map(|c| evaluate_candidate(sku, c).ok())
        .collect();
    valid.sort_by(|a, b| {
        b.composite_score
            .partial_cmp(&a.composite_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    valid
}

/// Whole-Model Speculative & Multi-Slot Algorithmic Bandwidth Roofline (`tok/s` ceiling).
#[derive(Debug, Clone)]
pub struct ModelRooflineInput {
    pub dense_and_head_bytes: f64,
    pub total_expert_bytes: f64,
    pub num_experts: usize,
    pub top_k: usize,
    pub kv_bytes_per_slot: f64,
    pub recurrent_state_bytes_per_slot: f64,
    pub draft_bytes_per_step: f64,
    pub draft_steps_k: usize,
    pub mean_accepted_plus_one: f64,
    pub concurrency_slots: usize,
}

pub fn compute_decode_ceiling_tok_s(sku: &BlackwellSkuProfile, m: &ModelRooflineInput) -> f64 {
    let total_tokens_in_verify = ((m.draft_steps_k + 1) * m.concurrency_slots) as f64;
    let miss_prob_single = 1.0 - (m.top_k as f64) / (m.num_experts as f64);
    let activated_expert_frac = 1.0 - miss_prob_single.powf(total_tokens_in_verify);

    let b_verify = m.dense_and_head_bytes
        + m.total_expert_bytes * activated_expert_frac
        + (m.concurrency_slots as f64)
            * (m.kv_bytes_per_slot + 2.0 * m.recurrent_state_bytes_per_slot);
    // Batched drafter reads draft weights ONCE across all active slots!
    let b_draft = (m.draft_steps_k as f64) * m.draft_bytes_per_step;

    let emitted_tokens = m.mean_accepted_plus_one * (m.concurrency_slots as f64);
    let bw_bytes_per_sec = sku.sustained_dram_gbps * 1e9;
    (emitted_tokens * bw_bytes_per_sec) / (b_verify + b_draft)
}

/// Overlapped Spilled-Expert + Host `Engram` PCIe Roofline (`tok/s` ceiling).
/// Models `h2d_stream` Ping-Pong Late-Wait overlap against VRAM compute time:
/// - `Engram` PCIe bytes are hidden behind Visits `0..ENGRAM_RESIDUAL_INJECT_VISIT`.
/// - Spilled Expert PCIe bytes + `DtoD` promoted copies overlap with Attention + Router/Quantize.
#[derive(Debug, Clone)]
pub struct SpilledEngramRooflineInput {
    pub vram_compute_bytes_per_step: f64,
    pub overlap_hideable_vram_fraction: f64,
    pub spilled_pcie_bytes_per_step: f64,
    pub dtod_promoted_bytes_per_step: f64,
    pub engram_pcie_bytes_per_step: f64,
    pub emitted_tokens_per_step: f64,
}

pub fn compute_spilled_engram_ceiling_tok_s(
    sku: &BlackwellSkuProfile,
    inp: &SpilledEngramRooflineInput,
) -> (f64, f64) {
    let t_vram_total = inp.vram_compute_bytes_per_step / (sku.sustained_dram_gbps * 1e9);
    let t_hideable = t_vram_total * inp.overlap_hideable_vram_fraction.clamp(0.0, 1.0);
    let t_non_hideable = t_vram_total - t_hideable;

    let t_pcie = (inp.spilled_pcie_bytes_per_step + inp.engram_pcie_bytes_per_step)
        / (sku.pcie_htod_gbps * 1e9);
    let t_dtod = inp.dtod_promoted_bytes_per_step / (sku.dtod_copy_gbps * 1e9);
    let t_transfer = t_pcie + t_dtod;

    let exposed_stall_sec = (t_transfer - t_hideable).max(0.0);
    let step_sec = t_non_hideable + t_hideable.max(t_transfer);
    let tok_s = inp.emitted_tokens_per_step / step_sec.max(1e-12);
    let exposed_pct = (exposed_stall_sec / step_sec.max(1e-12)) * 100.0;
    (tok_s, exposed_pct)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prunes_smem_carveout_cliff_and_wide_cluster_traps() {
        let sku = BlackwellSkuProfile::rtx_pro_5000_300w();

        // Anti-Pattern #1: 74.2 KiB user SMEM jumps to 100 KiB carveout -> 28 KiB L1 < 64 KiB required!
        let cliff_knob = CandidateKnob {
            label: "mmq_74kib_cliff".into(),
            tile_m: 128,
            tile_n: 128,
            tile_k: 64,
            stages: 2,
            threads_per_cta: 256,
            est_regs_per_thread: 96,
            user_smem_bytes: 75_980,
            ctas_per_sm: 1,
            cluster_ctas: 1,
            min_required_l1_kib: 64,
        };
        assert!(evaluate_candidate(&sku, cliff_knob).is_err());

        // Anti-Pattern #13: cluster_ctas = 8 rejected!
        let wide_cluster = CandidateKnob {
            label: "cluster_8_tail_trap".into(),
            tile_m: 128,
            tile_n: 128,
            tile_k: 32,
            stages: 3,
            threads_per_cta: 128,
            est_regs_per_thread: 128,
            user_smem_bytes: 24_576,
            ctas_per_sm: 2,
            cluster_ctas: 8,
            min_required_l1_kib: 28,
        };
        assert!(evaluate_candidate(&sku, wide_cluster).is_err());

        // Valid 60 KiB Q8 Planar TMA GEMM stays inside 64 KiB carveout (`64 KiB` L1)!
        let good_q8 = CandidateKnob {
            label: "q8_planar_128x128_tma".into(),
            tile_m: 128,
            tile_n: 128,
            tile_k: 64,
            stages: 3,
            threads_per_cta: 512,
            est_regs_per_thread: 128,
            user_smem_bytes: 61_440,
            ctas_per_sm: 1,
            cluster_ctas: 1,
            min_required_l1_kib: 64,
        };
        let eval = evaluate_candidate(&sku, good_q8).expect("valid SM120 Q8 kernel");
        assert_eq!(eval.carveout_kib, 64);
        assert_eq!(eval.remaining_l1_kib, 64);
    }
}
