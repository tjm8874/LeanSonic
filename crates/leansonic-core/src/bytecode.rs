//! Unified `VisitBytecode` schedule compiler & validator.
//!
//! Replaces rigid `for layer in 0..N` loops with an explicit instruction stream
//! so Inkling-2's 24-layer -> 48-visit Looped Transformer and Qwen3.8/4's 3:1
//! Gated DeltaNet + Qwen Sparse Attention + 4-branch Gated Residual run on the
//! exact same minimal dispatcher.

use crate::generated_spec::{
    INKLING2_MINI_NUM_EXPERTS, INKLING2_MINI_PHYSICAL_LAYERS, INKLING2_MINI_TOP_K_EXPERTS,
    INKLING2_MINI_TOTAL_VISITS, QWEN38_FLASH_NEXT_NUM_EXPERTS, QWEN38_FLASH_NEXT_PHYSICAL_LAYERS,
    QWEN38_FLASH_NEXT_TOP_K_EXPERTS, SWA_WINDOW_TOKENS,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceMixOp {
    FullAttention,
    SlidingWindow {
        window_tokens: usize,
    },
    QwenSparse {
        top_blocks: usize,
        block_tokens: usize,
    },
    GatedDeltaNet {
        conv_kernel: usize,
        state_heads: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualTopology {
    SingleStream { gated_loop_blend: bool },
    QuadGatedStream,
}

impl ResidualTopology {
    pub fn num_branches(self) -> usize {
        match self {
            Self::SingleStream { .. } => 1,
            Self::QuadGatedStream => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisitInstruction {
    pub visit_index: usize,
    pub physical_layer_id: usize,
    pub state_layer_slot: usize,
    pub mix: SequenceMixOp,
    pub residual: ResidualTopology,
    pub num_experts: usize,
    pub top_k_experts: usize,
    pub has_shared_expert: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisitProgram {
    pub model_name: String,
    pub physical_layers: usize,
    pub has_engram_host_table: bool,
    pub visits: Vec<VisitInstruction>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduleError {
    EmptyVisits,
    NonContiguousVisit {
        expected: usize,
        got: usize,
    },
    OutOfBoundsPhysicalLayer {
        visit: usize,
        layer_id: usize,
        max_layers: usize,
    },
    InvalidExpertTopK {
        visit: usize,
        top_k: usize,
        num_experts: usize,
    },
}

impl VisitProgram {
    /// Validate invariants mirroring `LeanSonic.checkScheduleValid`.
    pub fn validate(&self) -> Result<(), ScheduleError> {
        if self.visits.is_empty() {
            return Err(ScheduleError::EmptyVisits);
        }
        for (idx, op) in self.visits.iter().enumerate() {
            if op.visit_index != idx {
                return Err(ScheduleError::NonContiguousVisit {
                    expected: idx,
                    got: op.visit_index,
                });
            }
            if op.physical_layer_id >= self.physical_layers {
                return Err(ScheduleError::OutOfBoundsPhysicalLayer {
                    visit: idx,
                    layer_id: op.physical_layer_id,
                    max_layers: self.physical_layers,
                });
            }
            if op.top_k_experts == 0 || op.top_k_experts > op.num_experts {
                return Err(ScheduleError::InvalidExpertTopK {
                    visit: idx,
                    top_k: op.top_k_experts,
                    num_experts: op.num_experts,
                });
            }
        }
        Ok(())
    }

    /// Compile the verified Phase 1 `Inkling-2-mini` schedule:
    /// - `24` physical weight layers (`0..5` Prelude, `6..17` 3x Loop Core, `18..23` Coda)
    /// - `48` scheduled visits (`12` Full Attention + `36` SWA `window=1024`)
    /// - `128` routed experts (`top-8`)
    pub fn inkling2_mini() -> Self {
        let mut visits = Vec::with_capacity(INKLING2_MINI_TOTAL_VISITS);
        for v in 0..INKLING2_MINI_TOTAL_VISITS {
            let physical_layer_id = if v < 6 {
                v
            } else if v < 42 {
                6 + ((v - 6) % 12)
            } else {
                18 + (v - 42)
            };
            let gated_loop_blend = (18..42).contains(&v);
            let mix = if v % 4 == 3 {
                SequenceMixOp::FullAttention
            } else {
                SequenceMixOp::SlidingWindow {
                    window_tokens: SWA_WINDOW_TOKENS,
                }
            };
            visits.push(VisitInstruction {
                visit_index: v,
                physical_layer_id,
                state_layer_slot: v,
                mix,
                residual: ResidualTopology::SingleStream { gated_loop_blend },
                num_experts: INKLING2_MINI_NUM_EXPERTS,
                top_k_experts: INKLING2_MINI_TOP_K_EXPERTS,
                has_shared_expert: false,
            });
        }
        Self {
            model_name: "Inkling-2-Mini".to_string(),
            physical_layers: INKLING2_MINI_PHYSICAL_LAYERS,
            has_engram_host_table: false,
            visits,
        }
    }

    /// Compile the verified `Qwen3.8-Flash-Next` (`Qwen4` preview) schedule:
    /// - `3:1` repeating `[GDN, GDN, GDN, QSA]` macro-layers
    /// - `4`-branch Gated Residual (`QuadGatedStream`)
    /// - `512` routed experts (`top-10`) + dedicated shared expert + 51B host `Engram` prefetch
    pub fn qwen38_flash_next() -> Self {
        let mut visits = Vec::with_capacity(QWEN38_FLASH_NEXT_PHYSICAL_LAYERS);
        for v in 0..QWEN38_FLASH_NEXT_PHYSICAL_LAYERS {
            let mix = if v % 4 == 3 {
                SequenceMixOp::QwenSparse {
                    top_blocks: 64,
                    block_tokens: 64,
                }
            } else {
                SequenceMixOp::GatedDeltaNet {
                    conv_kernel: 4,
                    state_heads: 32,
                }
            };
            visits.push(VisitInstruction {
                visit_index: v,
                physical_layer_id: v,
                state_layer_slot: v,
                mix,
                residual: ResidualTopology::QuadGatedStream,
                num_experts: QWEN38_FLASH_NEXT_NUM_EXPERTS,
                top_k_experts: QWEN38_FLASH_NEXT_TOP_K_EXPERTS,
                has_shared_expert: true,
            });
        }
        Self {
            model_name: "Qwen3.8-Flash-Next".to_string(),
            physical_layers: QWEN38_FLASH_NEXT_PHYSICAL_LAYERS,
            has_engram_host_table: true,
            visits,
        }
    }

    pub fn count_full_attention_visits(&self) -> usize {
        self.visits
            .iter()
            .filter(|v| matches!(v.mix, SequenceMixOp::FullAttention))
            .count()
    }

    pub fn count_swa_visits(&self) -> usize {
        self.visits
            .iter()
            .filter(|v| matches!(v.mix, SequenceMixOp::SlidingWindow { .. }))
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inkling2_mini_schedule_matches_lean_spec() {
        let prog = VisitProgram::inkling2_mini();
        assert_eq!(prog.validate(), Ok(()));
        assert_eq!(prog.visits.len(), 48);
        assert_eq!(prog.count_full_attention_visits(), 12);
        assert_eq!(prog.count_swa_visits(), 36);
        assert!(prog.visits.iter().all(|v| v.physical_layer_id < 24));
    }

    #[test]
    fn qwen38_flash_next_schedule_matches_lean_spec() {
        let prog = VisitProgram::qwen38_flash_next();
        assert_eq!(prog.validate(), Ok(()));
        assert_eq!(prog.visits.len(), 40);
        assert!(prog.has_engram_host_table);
        assert!(prog
            .visits
            .iter()
            .all(|v| v.residual.num_branches() == 4 && v.has_shared_expert));
    }
}
