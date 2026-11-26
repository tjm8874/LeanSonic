//! Phase 1 Minimal End-to-End Inference Slice (`MinimalEngine`).
//!
//! Executes `VisitBytecode` models (`Inkling-2-mini` 24 physical layers -> 48 looped
//! visits, and `Qwen3.8-Flash-Next` 3:1 GDN + QSA + 4-branch Gated Residual + 51B
//! Host `Engram`), supporting both All-Resident VRAM and Spilled Ping-Pong + `DtoD`
//! offload modes with `0-ULP` parity verification for Gate 3.

use crate::bytecode::{ResidualTopology, SequenceMixOp, VisitProgram};
use crate::generated_spec::ENGRAM_RESIDUAL_INJECT_VISIT;
use crate::kernels::{
    moe_dual_regime_forward, quantize_planar_q8, split_query_22bit_fp16, MoeExecutionRegime,
};
use crate::offload::{HostEngramTable, SpilledPingPongRuntime, TieredExpertLayer};
use crate::state_pool::{CudaGraphTopologyCache, HybridSlotPool};
use crate::weights::{plan_async_htod_staging, AsyncHtoDPlan, PlanarNvfp4Experts};

pub struct MinimalEngine {
    pub program: VisitProgram,
    pub hidden_dim: usize,
    pub layer_experts: Vec<TieredExpertLayer>,
    pub spill_runtime: SpilledPingPongRuntime,
    pub engram_table: Option<HostEngramTable>,
    pub pool: HybridSlotPool,
    pub graph_cache: CudaGraphTopologyCache,
}

#[derive(Debug, Clone)]
pub struct StepTelemetry {
    pub visits_executed: usize,
    pub full_attn_visits: usize,
    pub swa_visits: usize,
    pub qsa_visits: usize,
    pub gdn_visits: usize,
    pub moe_regime: MoeExecutionRegime,
    pub used_cuda_graph: bool,
    pub htod_plan: AsyncHtoDPlan,
    pub prefetch_launches: usize,
    /// Per-slot final hidden vectors (`[num_slots][hidden_dim]`).
    pub final_hidden: Vec<Vec<f32>>,
}

impl MinimalEngine {
    /// Construct an All-Resident miniature engine (`spilled_experts == 0`).
    pub fn new_synthetic(program: VisitProgram, hidden_dim: usize, seed: u64) -> Self {
        let max_experts = program.visits[0].num_experts;
        Self::new_synthetic_tiered(program, hidden_dim, seed, max_experts)
    }

    /// Construct a Tiered/Spilled miniature engine where `resident_experts_per_layer`
    /// reside in VRAM and remaining experts reside in pinned Host DDR staged via
    /// 2-buffer Ping-Pong Late-Wait + `DtoD` promotion.
    pub fn new_synthetic_tiered(
        program: VisitProgram,
        hidden_dim: usize,
        seed: u64,
        resident_experts_per_layer: usize,
    ) -> Self {
        assert_eq!(hidden_dim % 32, 0);
        program.validate().expect("invalid VisitProgram schedule");

        let k_blocks = hidden_dim / 32;
        let num_experts_layer0 = program.visits[0].num_experts;
        let layer_experts: Vec<TieredExpertLayer> = (0..program.physical_layers)
            .map(|l| {
                let planar = PlanarNvfp4Experts::deterministic_synthetic(
                    seed.wrapping_add(l as u64 * 131),
                    num_experts_layer0,
                    hidden_dim,
                    k_blocks,
                );
                TieredExpertLayer::from_planar(planar, resident_experts_per_layer)
            })
            .collect();

        let total_spilled: usize = layer_experts.iter().map(|l| l.spilled_expert_count()).sum();
        let engram_table = if program.has_engram_host_table {
            Some(HostEngramTable::new(hidden_dim, seed ^ 0x51B0_E49A))
        } else {
            None
        };

        Self {
            program,
            hidden_dim,
            layer_experts,
            spill_runtime: SpilledPingPongRuntime::new(total_spilled),
            engram_table,
            pool: HybridSlotPool::new(hidden_dim),
            graph_cache: CudaGraphTopologyCache::new(24),
        }
    }

    /// Execute one forward verification step across `slot_inputs` (`[slot_id, hidden_vec]`).
    pub fn forward_step(&mut self, slot_inputs: &[(usize, Vec<f32>)]) -> StepTelemetry {
        self.forward_step_with_tokens(slot_inputs, None)
    }

    /// Execute one forward step with optional input token IDs for pre-Visit-0 `Engram` DMA staging.
    pub fn forward_step_with_tokens(
        &mut self,
        slot_inputs: &[(usize, Vec<f32>)],
        slot_tokens: Option<&[u32]>,
    ) -> StepTelemetry {
        assert!(!slot_inputs.is_empty());
        let num_rows = slot_inputs.len();
        let used_cuda_graph = self.graph_cache.observe_and_should_replay_graph(num_rows);

        // Pre-Visit-0 (`ENGRAM_PREFETCH_ISSUE_VISIT == 0`): Stage Host `Engram` trigram embeddings!
        let mut staged_engram_per_row: Option<Vec<Vec<f32>>> = None;
        if let Some(ref engram) = self.engram_table {
            let mut rows_emb = Vec::with_capacity(num_rows);
            for (r, &(slot_id, _)) in slot_inputs.iter().enumerate() {
                let tok = slot_tokens.and_then(|ts| ts.get(r).copied()).unwrap_or(1);
                let tail = self.pool.committed_engram_tail(slot_id);
                let (mut embs, new_tail) = engram.stage_sequence_embeddings(tail, &[tok]);
                self.pool.set_committed_engram_tail(slot_id, new_tail);
                rows_emb.push(embs.swap_remove(0));
            }
            staged_engram_per_row = Some(rows_emb);
        }

        let spilled_bytes = self.spill_runtime.total_spilled_experts * self.hidden_dim / 2;
        let engram_bytes = if staged_engram_per_row.is_some() {
            num_rows * self.hidden_dim * 4
        } else {
            0
        };
        let htod_plan = plan_async_htod_staging(spilled_bytes, engram_bytes);
        let launches_before = self.spill_runtime.prefetch_kernel_launches;

        let num_branches = self.program.visits[0].residual.num_branches();
        let mut streams: Vec<Vec<Vec<f32>>> = slot_inputs
            .iter()
            .map(|(_, x)| {
                assert_eq!(x.len(), self.hidden_dim);
                vec![x.clone(); num_branches]
            })
            .collect();

        let mut full_attn_visits = 0;
        let mut swa_visits = 0;
        let mut qsa_visits = 0;
        let mut gdn_visits = 0;
        let mut last_regime = MoeExecutionRegime::SimtReg64;

        for visit in &self.program.visits {
            // Inject pre-staged Host `Engram` embedding at `ENGRAM_RESIDUAL_INJECT_VISIT` (`2`)!
            if visit.visit_index == ENGRAM_RESIDUAL_INJECT_VISIT {
                if let Some(ref engram_rows) = staged_engram_per_row {
                    for (r, emb) in engram_rows.iter().enumerate() {
                        for branch in streams[r].iter_mut().take(num_branches) {
                            for d in 0..self.hidden_dim {
                                branch[d] = std::hint::black_box(branch[d] + emb[d]);
                            }
                        }
                    }
                }
            }

            // 1. Sequence Mix (Full / SWA / QSA / GDN)
            match visit.mix {
                SequenceMixOp::FullAttention => {
                    full_attn_visits += 1;
                    for (r, &(slot_id, _)) in slot_inputs.iter().enumerate() {
                        let _blks = self.pool.active_kv_blocks(slot_id, 1, visit.mix, None);
                        for v in &mut streams[r][0] {
                            let (hi, lo) = split_query_22bit_fp16(*v);
                            *v = std::hint::black_box(hi + lo * (1.0 / 2048.0));
                        }
                    }
                }
                SequenceMixOp::SlidingWindow { .. } => {
                    swa_visits += 1;
                    for (r, &(slot_id, _)) in slot_inputs.iter().enumerate() {
                        let _blks = self.pool.active_kv_blocks(slot_id, 1, visit.mix, None);
                        for v in &mut streams[r][0] {
                            let (hi, lo) = split_query_22bit_fp16(*v);
                            *v = std::hint::black_box(hi + lo * (1.0 / 2048.0));
                        }
                    }
                }
                SequenceMixOp::QwenSparse { .. } => {
                    qsa_visits += 1;
                    for (r, &(slot_id, _)) in slot_inputs.iter().enumerate() {
                        let _blks = self.pool.active_kv_blocks(slot_id, 1, visit.mix, None);
                        for v in &mut streams[r][0] {
                            let (hi, lo) = split_query_22bit_fp16(*v);
                            *v = std::hint::black_box(hi + lo * (1.0 / 2048.0));
                        }
                    }
                }
                SequenceMixOp::GatedDeltaNet { .. } => {
                    gdn_visits += 1;
                    for (r, &(slot_id, _)) in slot_inputs.iter().enumerate() {
                        let sample = streams[r][0][0];
                        let delta: Vec<f32> = streams[r][0].iter().map(|&v| v * 0.05).collect();
                        self.pool
                            .stage_speculative_recurrent_step(slot_id, 1, 0.95, &delta, sample);
                    }
                }
            }

            // 2. Deterministic Planar `Q8` Quantization + Top-K Routing
            let mut q8_codes = Vec::with_capacity(num_rows);
            let mut q8_scales = Vec::with_capacity(num_rows);
            let mut routed_pairs = Vec::with_capacity(num_rows * visit.top_k_experts);

            for (r, branch_set) in streams.iter().enumerate() {
                let (c, s) = quantize_planar_q8(&branch_set[0]);
                let sig = c
                    .iter()
                    .enumerate()
                    .fold(visit.visit_index, |acc, (i, &qv)| {
                        acc.wrapping_add(((qv as i32 + 128) as usize).wrapping_mul(i + 3))
                    });
                for k in 0..visit.top_k_experts {
                    let expert_id = (sig + k * 17) % visit.num_experts;
                    routed_pairs.push((expert_id, r));
                }
                q8_codes.push(c);
                q8_scales.push(s);
            }

            // 3. Tiered Staging (`0` launches if all-resident; Ping-Pong Late-Wait if spilled) + Dual-Regime MoE
            let tiered_layer = &self.layer_experts[visit.physical_layer_id];
            let staged_planar = self.spill_runtime.stage_visit_experts(
                visit.visit_index,
                tiered_layer,
                &routed_pairs,
            );
            let (regime, pair_outputs) =
                moe_dual_regime_forward(&q8_codes, &q8_scales, staged_planar, &routed_pairs);
            last_regime = regime;

            // 4. Ordered Deterministic Combine + Residual Stream Update
            let inv_topk = 1.0f32 / (visit.top_k_experts as f32);
            for r in 0..num_rows {
                let mut combined = vec![0.0f32; self.hidden_dim];
                for k in 0..visit.top_k_experts {
                    let pout = &pair_outputs[r * visit.top_k_experts + k];
                    for d in 0..self.hidden_dim {
                        combined[d] = std::hint::black_box(
                            combined[d] + std::hint::black_box(pout[d] * inv_topk * 0.02),
                        );
                    }
                }

                match visit.residual {
                    ResidualTopology::SingleStream { gated_loop_blend } => {
                        let gate = if gated_loop_blend { 0.875f32 } else { 1.0f32 };
                        for d in 0..self.hidden_dim {
                            streams[r][0][d] = std::hint::black_box(
                                std::hint::black_box(streams[r][0][d] * gate) + combined[d],
                            );
                        }
                    }
                    ResidualTopology::QuadGatedStream => {
                        let gates = [0.95f32, 0.85f32, 0.75f32, 0.65f32];
                        for b in 0..4 {
                            for d in 0..self.hidden_dim {
                                streams[r][b][d] = std::hint::black_box(
                                    std::hint::black_box(streams[r][b][d] * gates[b]) + combined[d],
                                );
                            }
                        }
                    }
                }
            }
        }

        for &(slot_id, _) in slot_inputs {
            self.pool.commit_speculative_tokens(slot_id, 0);
        }

        StepTelemetry {
            visits_executed: self.program.visits.len(),
            full_attn_visits,
            swa_visits,
            qsa_visits,
            gdn_visits,
            moe_regime: last_regime,
            used_cuda_graph,
            htod_plan,
            prefetch_launches: self.spill_runtime.prefetch_kernel_launches - launches_before,
            final_hidden: streams.into_iter().map(|mut b| b.swap_remove(0)).collect(),
        }
    }
}

/// Gate 3 Check A: `C = 1` (`SimtReg64`) vs `C = 64` (`GroupedMmaInt8`) `0-ULP` parity.
pub fn verify_inkling2_batch_invariance(hidden_dim: usize) -> Result<(), String> {
    let probe: Vec<f32> = (0..hidden_dim)
        .map(|i| ((i as f32) * 0.173 - 1.2).cos() * 0.8)
        .collect();

    let mut solo_engine =
        MinimalEngine::new_synthetic(VisitProgram::inkling2_mini(), hidden_dim, 20261125);
    let solo_step = solo_engine.forward_step(&[(0, probe.clone())]);
    if solo_step.prefetch_launches != 0 {
        return Err("All-resident engine launched unexpected prefetch kernels!".into());
    }

    let mut batch_engine =
        MinimalEngine::new_synthetic(VisitProgram::inkling2_mini(), hidden_dim, 20261125);
    let mut batch_inputs = Vec::with_capacity(64);
    for slot in 0..64 {
        if slot == 41 {
            batch_inputs.push((slot, probe.clone()));
        } else {
            let v: Vec<f32> = (0..hidden_dim)
                .map(|i| (((i + slot * 5) as f32) * 0.09).sin())
                .collect();
            batch_inputs.push((slot, v));
        }
    }
    let batch_step = batch_engine.forward_step(&batch_inputs);

    for (idx, (&a, &b)) in solo_step.final_hidden[0]
        .iter()
        .zip(batch_step.final_hidden[41].iter())
        .enumerate()
    {
        if a.to_bits() != b.to_bits() {
            return Err(format!(
                "48-visit C=1 vs C=64 0-ULP parity failed at dim {idx}: {:#010x} != {:#010x}",
                a.to_bits(),
                b.to_bits()
            ));
        }
    }
    Ok(())
}

/// Gate 3 Check B: 72GB All-Resident (`128/128` VRAM) vs 16GB Spilled (`48/128` VRAM + `80` Host DDR
/// Ping-Pong Late-Wait + `DtoD` promoted) `0-ULP` bitwise parity across all 48 visits!
pub fn verify_inkling2_resident_vs_spilled_0ulp(hidden_dim: usize) -> Result<(), String> {
    let probe: Vec<f32> = (0..hidden_dim)
        .map(|i| ((i as f32) * 0.211 + 0.4).sin() * 0.9)
        .collect();

    let mut resident_eng = MinimalEngine::new_synthetic_tiered(
        VisitProgram::inkling2_mini(),
        hidden_dim,
        20261125,
        128,
    );
    let mut spilled_eng = MinimalEngine::new_synthetic_tiered(
        VisitProgram::inkling2_mini(),
        hidden_dim,
        20261125,
        48,
    );

    let res_step = resident_eng.forward_step(&[(0, probe.clone())]);
    let spl_step = spilled_eng.forward_step(&[(0, probe)]);

    if res_step.prefetch_launches != 0 {
        return Err("Expected 0 prefetch launches in All-Resident mode".into());
    }
    if spl_step.prefetch_launches != 48 {
        return Err(format!(
            "Expected 48 Ping-Pong staging visits in Spilled mode, got {}",
            spl_step.prefetch_launches
        ));
    }

    for (idx, (&a, &b)) in res_step.final_hidden[0]
        .iter()
        .zip(spl_step.final_hidden[0].iter())
        .enumerate()
    {
        if a.to_bits() != b.to_bits() {
            return Err(format!(
                "Resident vs Spilled 0-ULP parity failed at dim {idx}: {:#010x} != {:#010x}",
                a.to_bits(),
                b.to_bits()
            ));
        }
    }
    Ok(())
}

/// Gate 3 Check C: Qwen3.8-Flash-Next `Engram` (51B N-gram Host Offload) boundary & speculative
/// rollback `0-ULP` invariance (`scanTrigrams_append_exact` runtime verification).
pub fn verify_qwen38_engram_boundary_0ulp(hidden_dim: usize) -> Result<(), String> {
    let tokens: [u32; 6] = [101, 205, 309, 412, 518, 627];
    let inputs: Vec<Vec<f32>> = tokens
        .iter()
        .enumerate()
        .map(|(pos, &t)| {
            (0..hidden_dim)
                .map(|d| (((t as usize + pos * 11 + d) as f32) * 0.07).cos() * 0.5)
                .collect()
        })
        .collect();

    // Path 1: Continuous unsplit token stream on Slot 0
    let mut mono_eng =
        MinimalEngine::new_synthetic(VisitProgram::qwen38_flash_next(), hidden_dim, 20261125);
    let mut mono_last = Vec::new();
    for i in 0..tokens.len() {
        let step = mono_eng.forward_step_with_tokens(&[(0, inputs[i].clone())], Some(&[tokens[i]]));
        mono_last = step.final_hidden[0].clone();
    }

    // Path 2: Split after token 2, stage 4 bogus speculative draft trigram tails (`steps 1..=4`),
    // reject all 4 (`commit_speculative_tokens(0, 0)` -> O(1) rollback!), then resume tokens `2..6`!
    let mut split_eng =
        MinimalEngine::new_synthetic(VisitProgram::qwen38_flash_next(), hidden_dim, 20261125);
    for i in 0..2 {
        let _ = split_eng.forward_step_with_tokens(&[(0, inputs[i].clone())], Some(&[tokens[i]]));
    }
    for bogus_step in 1..=4 {
        split_eng.pool.stage_speculative_engram_tail(
            0,
            bogus_step,
            (99999, 88888 + bogus_step as u32),
        );
    }
    split_eng.pool.commit_speculative_tokens(0, 0);

    let mut split_last = Vec::new();
    for i in 2..tokens.len() {
        let step =
            split_eng.forward_step_with_tokens(&[(0, inputs[i].clone())], Some(&[tokens[i]]));
        split_last = step.final_hidden[0].clone();
    }

    if mono_eng.pool.committed_engram_tail(0) != split_eng.pool.committed_engram_tail(0) {
        return Err("Engram (prev2, prev1) tail diverged after speculative rollback!".into());
    }
    for (idx, (&a, &b)) in mono_last.iter().zip(split_last.iter()).enumerate() {
        if a.to_bits() != b.to_bits() {
            return Err(format!(
                "Qwen3.8 Engram chunk/rollback 0-ULP failed at dim {idx}: {:#010x} != {:#010x}",
                a.to_bits(),
                b.to_bits()
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase1_inkling2_mini_48_visits_0ulp_batch_identical() {
        verify_inkling2_batch_invariance(64).expect("48-visit C=1 == C=64 0-ULP parity");
    }

    #[test]
    fn phase1_inkling2_mini_resident_vs_spilled_0ulp_identical() {
        verify_inkling2_resident_vs_spilled_0ulp(64)
            .expect("48-visit Resident == Spilled 0-ULP parity");
    }

    #[test]
    fn qwen38_flash_next_engram_chunk_and_rollback_0ulp_identical() {
        verify_qwen38_engram_boundary_0ulp(64)
            .expect("Qwen3.8 Engram boundary & rollback 0-ULP parity");
    }
}
