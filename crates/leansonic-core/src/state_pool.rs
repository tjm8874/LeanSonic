//! Unified Hybrid State Pool (`HybridSlotPool`) & 2-Hit CUDA Graph Topology Filter.
//!
//! Enforces the Lean 4 `SafeHybridStatePool` invariants:
//! - 64-token aligned block lists unifying Full Attention, SWA (`1088`-token ring),
//!   and Qwen Sparse Attention (QSA micro-blocks).
//! - `C * (1 + W_spec)` = `64 * 17 = 1088` recurrent state slots supporting `O(1)`
//!   speculative commit & rollback without overwriting active GDN/conv history.
//! - Device-driven CUDA Graph topology filter requiring 2 hits before graph capture.

use std::collections::HashMap;

use crate::bytecode::SequenceMixOp;
use crate::generated_spec::{
    KV_BLOCK_ALIGN_TOKENS, MAX_BATCH_ROWS, MAX_CONCURRENT_SLOTS, RECURRENT_STATE_SLOTS,
    SPEC_VERIFY_WINDOW, SWA_RING_CAPACITY_TOKENS,
};

#[derive(Debug, Clone)]
pub struct RecurrentSnapshot {
    pub gdn_matrix: Vec<f32>,
    pub conv_ring: [f32; 4],
}

impl RecurrentSnapshot {
    pub fn zeros(state_dim: usize) -> Self {
        Self {
            gdn_matrix: vec![0.0; state_dim],
            conv_ring: [0.0; 4],
        }
    }
}

#[derive(Debug, Clone)]
pub struct HybridSlotPool {
    pub max_slots: usize,
    pub spec_window: usize,
    pub state_dim: usize,
    /// Current committed sequence token length per conversation slot.
    pub seq_lens: Vec<usize>,
    /// Active physical recurrent state index (`0 .. C * (1 + W)`) for each slot's committed tip.
    pub committed_recurrent_idx: Vec<usize>,
    /// Pre-allocated pool of `RECURRENT_STATE_SLOTS` (`1088`) states.
    pub recurrent_pool: Vec<RecurrentSnapshot>,
}

impl HybridSlotPool {
    pub fn new(state_dim: usize) -> Self {
        let total_rec_slots = RECURRENT_STATE_SLOTS;
        assert_eq!(
            total_rec_slots,
            MAX_CONCURRENT_SLOTS * (1 + SPEC_VERIFY_WINDOW)
        );
        assert_eq!(SWA_RING_CAPACITY_TOKENS % KV_BLOCK_ALIGN_TOKENS, 0);

        let stride = 1 + SPEC_VERIFY_WINDOW;
        let committed_recurrent_idx = (0..MAX_CONCURRENT_SLOTS).map(|s| s * stride).collect();

        Self {
            max_slots: MAX_CONCURRENT_SLOTS,
            spec_window: SPEC_VERIFY_WINDOW,
            state_dim,
            seq_lens: vec![0; MAX_CONCURRENT_SLOTS],
            committed_recurrent_idx,
            recurrent_pool: (0..total_rec_slots)
                .map(|_| RecurrentSnapshot::zeros(state_dim))
                .collect(),
        }
    }

    /// Compute the 64-key aligned block list scanned by `attn_blocklist_tc_split_kv`
    /// for Full Attention, SWA ring, or QSA selected sparse blocks.
    pub fn active_kv_blocks(
        &self,
        slot_id: usize,
        draft_rows: usize,
        mix: SequenceMixOp,
        qsa_selected_blocks: Option<&[usize]>,
    ) -> Vec<usize> {
        assert!(slot_id < self.max_slots);
        assert!(draft_rows <= MAX_BATCH_ROWS);
        let total_tokens = self.seq_lens[slot_id] + draft_rows;
        if total_tokens == 0 {
            return Vec::new();
        }
        let end_block = total_tokens.div_ceil(KV_BLOCK_ALIGN_TOKENS);

        match mix {
            SequenceMixOp::FullAttention => (0..end_block).collect(),
            SequenceMixOp::SlidingWindow { window_tokens } => {
                let lo_token = total_tokens.saturating_sub(window_tokens + draft_rows);
                // Global 64-key boundary alignment (`c_min = lo_min >> 6`)
                let start_block = lo_token / KV_BLOCK_ALIGN_TOKENS;
                let ring_blocks = SWA_RING_CAPACITY_TOKENS / KV_BLOCK_ALIGN_TOKENS;
                (start_block..end_block).map(|b| b % ring_blocks).collect()
            }
            SequenceMixOp::QwenSparse { top_blocks, .. } => {
                if let Some(sel) = qsa_selected_blocks {
                    sel.iter().copied().take(top_blocks).collect()
                } else {
                    let start = end_block.saturating_sub(top_blocks);
                    (start..end_block).collect()
                }
            }
            SequenceMixOp::GatedDeltaNet { .. } => Vec::new(),
        }
    }

    /// Stage `draft_count <= SPEC_VERIFY_WINDOW` candidate recurrent states in the slot's
    /// dedicated `1 + W` sub-band without touching `committed_recurrent_idx[slot_id]`.
    pub fn stage_speculative_recurrent_step(
        &mut self,
        slot_id: usize,
        draft_step: usize,
        decay_gate: f32,
        delta_vec: &[f32],
        conv_sample: f32,
    ) -> usize {
        assert!(slot_id < self.max_slots);
        assert!((1..=self.spec_window).contains(&draft_step));
        assert_eq!(delta_vec.len(), self.state_dim);

        let band_base = slot_id * (1 + self.spec_window);
        let prev_phys = if draft_step == 1 {
            self.committed_recurrent_idx[slot_id]
        } else {
            band_base + ((draft_step - 1) % (1 + self.spec_window))
        };
        let dst_phys = (self.committed_recurrent_idx[slot_id] + draft_step)
            .rem_euclid(1 + self.spec_window)
            + band_base;
        assert_ne!(dst_phys, self.committed_recurrent_idx[slot_id]);

        let prev_clone = self.recurrent_pool[prev_phys].clone();
        let dst = &mut self.recurrent_pool[dst_phys];
        for (i, out) in dst.gdn_matrix.iter_mut().enumerate() {
            *out = prev_clone.gdn_matrix[i].mul_add(decay_gate, delta_vec[i]);
        }
        // Write ONLY trailing `k_conv = 4` positions (prevents short-conv clobber)
        dst.conv_ring = [
            prev_clone.conv_ring[1],
            prev_clone.conv_ring[2],
            prev_clone.conv_ring[3],
            conv_sample,
        ];
        dst_phys
    }

    /// `O(1)` Speculative Commit / Rollback:
    /// Advances `committed_recurrent_idx` to the accepted draft state (`0..=W`);
    /// rejected trailing draft states are simply abandoned in-place with `0` memcpy!
    pub fn commit_speculative_tokens(&mut self, slot_id: usize, accepted_drafts: usize) {
        assert!(slot_id < self.max_slots);
        assert!(accepted_drafts <= self.spec_window);
        let band_base = slot_id * (1 + self.spec_window);
        if accepted_drafts > 0 {
            let new_phys = (self.committed_recurrent_idx[slot_id] + accepted_drafts)
                .rem_euclid(1 + self.spec_window)
                + band_base;
            self.committed_recurrent_idx[slot_id] = new_phys;
        }
        // +1 bonus token from target verification
        self.seq_lens[slot_id] += accepted_drafts + 1;
    }
}

/// 2-Hit CUDA Graph Topology Filter (`<= 24` coarse batch bucket topologies).
/// Prevents the `2x` `eager + capture` thrash penalty on transient batch shapes.
#[derive(Debug, Default)]
pub struct CudaGraphTopologyCache {
    max_topologies: usize,
    hit_counts: HashMap<usize, usize>,
    captured_buckets: Vec<usize>,
}

impl CudaGraphTopologyCache {
    pub fn new(max_topologies: usize) -> Self {
        Self {
            max_topologies: max_topologies.min(24),
            hit_counts: HashMap::new(),
            captured_buckets: Vec::new(),
        }
    }

    pub fn bucket_for_rows(total_rows: usize) -> usize {
        match total_rows {
            0..=1 => 1,
            2 => 2,
            3..=4 => 4,
            5..=8 => 8,
            9..=16 => 16,
            17..=32 => 32,
            _ => 64,
        }
    }

    /// Returns `true` iff this batch row count should execute via a captured CUDA Graph.
    /// First encounter runs eager (`hit = 1`); second encounter captures & replays (`hit >= 2`).
    pub fn observe_and_should_replay_graph(&mut self, total_rows: usize) -> bool {
        let bucket = Self::bucket_for_rows(total_rows);
        let hits = self.hit_counts.entry(bucket).or_insert(0);
        *hits += 1;
        if *hits >= 2 {
            if !self.captured_buckets.contains(&bucket)
                && self.captured_buckets.len() < self.max_topologies
            {
                self.captured_buckets.push(bucket);
            }
            self.captured_buckets.contains(&bucket)
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swa_ring_never_clobbers_active_window_across_16_draft_rows() {
        let mut pool = HybridSlotPool::new(16);
        pool.seq_lens[0] = 8192;
        let blks = pool.active_kv_blocks(
            0,
            16,
            SequenceMixOp::SlidingWindow {
                window_tokens: 1024,
            },
            None,
        );
        // 1088 / 64 = 17 physical ring blocks; 1024 + 16 = 1040 tokens <= 17 blocks -> zero duplicate ring alias!
        let mut sorted = blks.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(blks.len(), sorted.len(), "SWA ring blocks must not alias!");
    }

    #[test]
    fn o1_speculative_rollback_preserves_committed_gdn_state() {
        let mut pool = HybridSlotPool::new(4);
        let orig = pool.recurrent_pool[pool.committed_recurrent_idx[0]].clone();
        for step in 1..=8 {
            pool.stage_speculative_recurrent_step(0, step, 0.9, &[1.0, 2.0, 3.0, 4.0], step as f32);
        }
        // Reject all 8 drafts (`accepted_drafts = 0`) -> committed state must be 100% untouched!
        pool.commit_speculative_tokens(0, 0);
        assert_eq!(
            pool.recurrent_pool[pool.committed_recurrent_idx[0]].gdn_matrix,
            orig.gdn_matrix
        );
    }

    #[test]
    fn two_hit_graph_filter_prevents_first_hit_capture_thrash() {
        let mut cache = CudaGraphTopologyCache::new(24);
        assert!(!cache.observe_and_should_replay_graph(8));
        assert!(cache.observe_and_should_replay_graph(8));
    }
}
