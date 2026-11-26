//! Phase 1 Minimal End-to-End Inference Slice (`MinimalEngine`).
//!
//! Executes a miniature `VisitBytecode` model (`Inkling-2-mini` 24 physical layers ->
//! 48 looped visits, or `Qwen3.8-Flash-Next` 3:1 GDN + QSA + 4-branch Gated Residual)
//! and provides deterministic `C = 1` vs `C = 64` `0-ULP` parity verification for Gate 3.

use crate::bytecode::{ResidualTopology, SequenceMixOp, VisitProgram};
use crate::kernels::{
    moe_dual_regime_forward, quantize_planar_q8, split_query_22bit_fp16, MoeExecutionRegime,
};
use crate::state_pool::{CudaGraphTopologyCache, HybridSlotPool};
use crate::weights::{plan_async_htod_staging, AsyncHtoDPlan, PlanarNvfp4Experts};

pub struct MinimalEngine {
    pub program: VisitProgram,
    pub hidden_dim: usize,
    pub layer_experts: Vec<PlanarNvfp4Experts>,
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
    /// Per-slot final hidden vectors (`[num_slots][hidden_dim]`).
    pub final_hidden: Vec<Vec<f32>>,
}

impl MinimalEngine {
    /// Construct a deterministic miniature engine (`hidden_dim` must be a multiple of 32)
    /// allocating ONLY `program.physical_layers` weight sets (e.g. `24` for Inkling-2-mini's `48` visits).
    pub fn new_synthetic(program: VisitProgram, hidden_dim: usize, seed: u64) -> Self {
        assert_eq!(hidden_dim % 32, 0);
        program.validate().expect("invalid VisitProgram schedule");

        let k_blocks = hidden_dim / 32;
        let num_experts_layer0 = program.visits[0].num_experts;
        let layer_experts = (0..program.physical_layers)
            .map(|l| {
                PlanarNvfp4Experts::deterministic_synthetic(
                    seed.wrapping_add(l as u64 * 131),
                    num_experts_layer0,
                    hidden_dim,
                    k_blocks,
                )
            })
            .collect();

        Self {
            program,
            hidden_dim,
            layer_experts,
            pool: HybridSlotPool::new(hidden_dim),
            graph_cache: CudaGraphTopologyCache::new(24),
        }
    }

    /// Execute one forward verification step across `slot_inputs` (`[slot_id, hidden_vec]`).
    pub fn forward_step(&mut self, slot_inputs: &[(usize, Vec<f32>)]) -> StepTelemetry {
        assert!(!slot_inputs.is_empty());
        let num_rows = slot_inputs.len();
        let used_cuda_graph = self.graph_cache.observe_and_should_replay_graph(num_rows);
        let engram_bytes = if self.program.has_engram_host_table {
            num_rows * self.hidden_dim * 2
        } else {
            0
        };
        let htod_plan = plan_async_htod_staging(0, engram_bytes);

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
            // 1. Sequence Mix (Full / SWA / QSA / GDN) with 22-bit FP16 query split or GDN state step
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

            // 2. Deterministic Planar `Q8` Quantization + Deterministic Top-K Routing
            let mut q8_codes = Vec::with_capacity(num_rows);
            let mut q8_scales = Vec::with_capacity(num_rows);
            let mut routed_pairs = Vec::with_capacity(num_rows * visit.top_k_experts);

            for (r, branch_set) in streams.iter().enumerate() {
                let (c, s) = quantize_planar_q8(&branch_set[0]);
                // Deterministic router tie-break based solely on row's own quantized codes
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

            // 3. Dual-Regime `0-ULP` Planar-SoA MoE Execution (reusing physical weight layer!)
            let w = &self.layer_experts[visit.physical_layer_id];
            let (regime, pair_outputs) =
                moe_dual_regime_forward(&q8_codes, &q8_scales, w, &routed_pairs);
            last_regime = regime;

            // 4. Ordered Deterministic Combine (NEVER unordered global `atomicAdd`!)
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

                // 5. Residual Stream Update (`SingleStream` Loop Blend vs `QuadGatedStream`)
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
            final_hidden: streams.into_iter().map(|mut b| b.swap_remove(0)).collect(),
        }
    }
}

/// Gate 3 Self-Check: Run `Inkling-2-mini` 48-visit looped forward pass at `C = 1` (`SimtReg64`)
/// and `C = 64` (`GroupedMmaInt8`) and assert `0-ULP` bitwise identity across all 48 visits!
pub fn verify_inkling2_batch_invariance(hidden_dim: usize) -> Result<(), String> {
    let probe: Vec<f32> = (0..hidden_dim)
        .map(|i| ((i as f32) * 0.173 - 1.2).cos() * 0.8)
        .collect();

    let mut solo_engine =
        MinimalEngine::new_synthetic(VisitProgram::inkling2_mini(), hidden_dim, 20261125);
    let solo_step = solo_engine.forward_step(&[(0, probe.clone())]);
    if solo_step.moe_regime != MoeExecutionRegime::SimtReg64 {
        return Err(format!(
            "expected SimtReg64 at C=1, got {:?}",
            solo_step.moe_regime
        ));
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
    if batch_step.moe_regime != MoeExecutionRegime::GroupedMmaInt8 {
        return Err(format!(
            "expected GroupedMmaInt8 at C=64, got {:?}",
            batch_step.moe_regime
        ));
    }

    for (idx, (&a, &b)) in solo_step.final_hidden[0]
        .iter()
        .zip(batch_step.final_hidden[41].iter())
        .enumerate()
    {
        if a.to_bits() != b.to_bits() {
            return Err(format!(
                "48-visit 0-ULP parity failed at dim {idx}: C=1 {:#010x} != C=64 {:#010x}",
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
    fn qwen38_flash_next_40_visits_executes_gdn_qsa_and_engram_plan() {
        let mut eng = MinimalEngine::new_synthetic(VisitProgram::qwen38_flash_next(), 64, 20261125);
        let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.02).collect();
        let step = eng.forward_step(&[(0, x)]);
        assert_eq!(step.visits_executed, 40);
        assert_eq!(step.gdn_visits, 30);
        assert_eq!(step.qsa_visits, 10);
        assert!(matches!(step.htod_plan, AsyncHtoDPlan::Scheduled { .. }));
    }
}
