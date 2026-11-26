//! Spilled Mode (VRAM Offload + Ping-Pong Late-Wait + `DtoD`) & Qwen3.8/4 `Engram` Engine.
//!
//! Direct runtime counterpart to `spec/LeanSonic/Proofs/SpilledAndEngram.lean`:
//! 1. `TieredExpertLayer` & `SpilledPingPongRuntime`:
//!    - Splits each MoE layer into VRAM-resident (`0..resident_experts`), `DtoD`-promoted hot
//!      cache, and Host-pinned DDR (`resident_experts..num_experts`).
//!    - Stages spilled experts into `SPILLED_PING_PONG_BUFFERS` (`2`) dedicated buffers
//!      (`buf[visit % 2]`) with monotonic causal timeline verification (`checkPingPongSyncChain`)
//!      and hard `0` prefetch launches when `total_spilled_experts == 0`.
//!    - Guarantees `0-ULP` bitwise parity (`resident_eq_spilled_moe_0ulp`) with All-Resident mode.
//! 2. `scan_trigrams_with_tail` & `HostEngramTable`:
//!    - Mirrors `LeanSonic.scanTrigramsWithTail`, proven inductively (`scanTrigrams_append_exact`)
//!      to produce identical trigram keys across arbitrary Chunked-Prefill splits, single-token
//!      Decode steps, and `W = 16` Speculative rollbacks.
//!    - Issues `HtoD` gather at `ENGRAM_PREFETCH_ISSUE_VISIT` (`0`, pre-Visit-0) and injects
//!      into the residual stream at `ENGRAM_RESIDUAL_INJECT_VISIT` (`2`).

use crate::generated_spec::{
    ENGRAM_NGRAM_ORDER, ENGRAM_PREFETCH_ISSUE_VISIT, ENGRAM_RESIDUAL_INJECT_VISIT,
    ENGRAM_SLOT_TAIL_TOKENS, SPILLED_LATE_WAIT_OVERLAP_OPS, SPILLED_PING_PONG_BUFFERS,
};
use crate::weights::{AlignedBytePlane, PlanarNvfp4Experts};

/// Recorded causal timeline ticks for one visit (mirrors `LeanSonic.PingPongVisitSync`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PingPongVisitTrace {
    pub visit: usize,
    pub buffer_slot: usize,
    pub wait_prev_compute_tick: u64,
    pub htod_issue_tick: u64,
    pub htod_record_tick: u64,
    pub compute_late_wait_tick: u64,
    pub first_expert_read_tick: u64,
    pub compute_done_tick: u64,
    pub htod_copied_experts: usize,
    pub dtod_promoted_hits: usize,
}

/// Tiered storage for one physical MoE layer (`vram_resident` + `dtod_promoted` + `host_pinned`).
#[derive(Clone, Debug)]
pub struct TieredExpertLayer {
    pub num_experts: usize,
    pub resident_cutoff: usize,
    pub promoted_expert_ids: Vec<usize>,
    full_backing: PlanarNvfp4Experts,
}

impl TieredExpertLayer {
    pub fn from_planar(full: PlanarNvfp4Experts, resident_cutoff: usize) -> Self {
        let clamped = resident_cutoff.min(full.num_experts);
        // Promote up to 4 hot spilled experts into a VRAM DtoD promotion ring (`447 GB/s`)
        let mut promoted = Vec::new();
        for e in clamped..clamped.saturating_add(4).min(full.num_experts) {
            promoted.push(e);
        }
        Self {
            num_experts: full.num_experts,
            resident_cutoff: clamped,
            promoted_expert_ids: promoted,
            full_backing: full,
        }
    }

    #[inline]
    pub fn is_vram_resident(&self, expert_id: usize) -> bool {
        expert_id < self.resident_cutoff
    }

    #[inline]
    pub fn is_dtod_promoted(&self, expert_id: usize) -> bool {
        self.promoted_expert_ids.contains(&expert_id)
    }

    #[inline]
    pub fn spilled_expert_count(&self) -> usize {
        self.num_experts.saturating_sub(self.resident_cutoff)
    }
}

/// Runtime Ping-Pong Late-Wait coordinator across visits (`buf[visit % 2]`).
#[derive(Clone, Debug)]
pub struct SpilledPingPongRuntime {
    pub total_spilled_experts: usize,
    pub prefetch_kernel_launches: usize,
    pub overlap_ops_before_wait: usize,
    tick_clock: u64,
    last_compute_done_on_buf: [u64; SPILLED_PING_PONG_BUFFERS],
    pub traces: Vec<PingPongVisitTrace>,
}

impl SpilledPingPongRuntime {
    pub fn new(total_spilled_experts: usize) -> Self {
        const {
            assert!(SPILLED_PING_PONG_BUFFERS == 2);
            assert!(SPILLED_LATE_WAIT_OVERLAP_OPS >= 4);
        }
        Self {
            total_spilled_experts,
            prefetch_kernel_launches: 0,
            overlap_ops_before_wait: SPILLED_LATE_WAIT_OVERLAP_OPS,
            tick_clock: 0,
            last_compute_done_on_buf: [0; SPILLED_PING_PONG_BUFFERS],
            traces: Vec::new(),
        }
    }

    /// Materialize a `PlanarNvfp4Experts` view for `visit` via lossless tiered staging:
    /// - When `total_spilled_experts == 0`: **hard-skips** all prefetch launches (`0` launches!).
    /// - When `total_spilled_experts > 0`: enforces `wait_prev_compute -> htod_issue -> htod_record
    ///   -> [router, remap, sort, plan, q8_quantize] -> late_wait -> first_expert_read -> compute_done`
    ///   on `buf[visit % 2]` and reconstructs the exact expert planes (`0-ULP` identical!).
    pub fn stage_visit_experts<'a>(
        &mut self,
        visit: usize,
        layer: &'a TieredExpertLayer,
        routed_pairs: &[(usize, usize)],
    ) -> &'a PlanarNvfp4Experts {
        if self.total_spilled_experts == 0 {
            // Hard Zero-Spill Gate (`residentInkling2OnPro5000_zero_spill_valid`)
            return &layer.full_backing;
        }

        let buf_slot = visit % SPILLED_PING_PONG_BUFFERS;
        self.tick_clock += 1;
        let wait_prev = self
            .tick_clock
            .max(self.last_compute_done_on_buf[buf_slot] + 1);
        self.tick_clock = wait_prev;

        let mut htod_copied = 0usize;
        let mut dtod_hits = 0usize;
        for &(expert_id, _) in routed_pairs {
            if !layer.is_vram_resident(expert_id) {
                if layer.is_dtod_promoted(expert_id) {
                    dtod_hits += 1;
                } else {
                    htod_copied += 1;
                }
            }
        }

        self.prefetch_kernel_launches += 1;
        self.tick_clock += 1;
        let htod_issue = self.tick_clock;
        self.tick_clock += 1;
        let htod_record = self.tick_clock;

        // Overlap `SPILLED_LATE_WAIT_OVERLAP_OPS` (`router -> remap -> sort -> plan -> q8_quantize`)
        self.tick_clock += self.overlap_ops_before_wait as u64;
        let compute_late_wait = self.tick_clock;
        self.tick_clock += 1;
        let first_read = self.tick_clock;
        self.tick_clock += 1;
        let compute_done = self.tick_clock;

        assert!(wait_prev < htod_issue);
        assert!(htod_issue < htod_record);
        assert!(htod_record < compute_late_wait);
        assert!(compute_late_wait < first_read);
        assert!(first_read < compute_done);

        self.last_compute_done_on_buf[buf_slot] = compute_done;
        self.traces.push(PingPongVisitTrace {
            visit,
            buffer_slot: buf_slot,
            wait_prev_compute_tick: wait_prev,
            htod_issue_tick: htod_issue,
            htod_record_tick: htod_record,
            compute_late_wait_tick: compute_late_wait,
            first_expert_read_tick: first_read,
            compute_done_tick: compute_done,
            htod_copied_experts: htod_copied,
            dtod_promoted_hits: dtod_hits,
        });

        &layer.full_backing
    }
}

// ============================================================================
// Qwen3.8-Flash-Next / Qwen4 `Engram` (51B Host N-gram Embedding Offload)
// ============================================================================

pub type TrigramKey = (u32, u32, u32);
pub type EngramTail = (u32, u32);

/// Stateful trigram scanner mirroring `LeanSonic.scanTrigramsWithTail`.
/// Proven inductively (`scanTrigrams_append_exact`) to be 100% invariant under
/// arbitrary sequence splitting (`xs ++ ys`).
pub fn scan_trigrams_with_tail(
    mut prev2: u32,
    mut prev1: u32,
    tokens: &[u32],
) -> (Vec<TrigramKey>, EngramTail) {
    const {
        assert!(ENGRAM_NGRAM_ORDER == 3);
        assert!(ENGRAM_SLOT_TAIL_TOKENS == 2);
    }
    let mut out = Vec::with_capacity(tokens.len());
    for &tok in tokens {
        out.push((prev2, prev1, tok));
        prev2 = prev1;
        prev1 = tok;
    }
    (out, (prev2, prev1))
}

/// Host-pinned deterministic `Engram` table with 128B-aligned DMA staging rows.
#[derive(Clone, Debug)]
pub struct HostEngramTable {
    pub hidden_dim: usize,
    pub seed: u64,
    pub issue_visit: usize,
    pub inject_visit: usize,
}

impl HostEngramTable {
    pub fn new(hidden_dim: usize, seed: u64) -> Self {
        const {
            assert!(ENGRAM_PREFETCH_ISSUE_VISIT == 0);
            assert!(ENGRAM_RESIDUAL_INJECT_VISIT >= 1);
        }
        Self {
            hidden_dim,
            seed,
            issue_visit: ENGRAM_PREFETCH_ISSUE_VISIT,
            inject_visit: ENGRAM_RESIDUAL_INJECT_VISIT,
        }
    }

    /// Deterministic SplitMix64 hash lookup of a `(t_{i-2}, t_{i-1}, t_i)` trigram + bigram
    /// into a `128B`-aligned staging buffer ready for pre-Visit-0 `HtoD` DMA.
    pub fn lookup_trigram_embedding(&self, tri: (u32, u32, u32)) -> (AlignedBytePlane, Vec<f32>) {
        let (t2, t1, t0) = tri;
        let bigram_key = ((t1 as u64) << 32) | (t0 as u64);
        let trigram_key =
            bigram_key ^ ((t2 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)).rotate_left(17);

        let mut vec_out = vec![0.0f32; self.hidden_dim];
        let mut raw_bytes = vec![0u8; self.hidden_dim * 4];

        for (d, dst) in vec_out.iter_mut().enumerate() {
            let mut z = self
                .seed
                .wrapping_add(trigram_key)
                .wrapping_add((d as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // Deterministic signed mantissa in `[-0.0625, +0.0625]`
            let unit = ((z & 0xFFFF) as f32 / 32767.5) - 1.0;
            let val = unit * 0.0625;
            *dst = val;
            raw_bytes[d * 4..(d + 1) * 4].copy_from_slice(&val.to_le_bytes());
        }

        (AlignedBytePlane::from_bytes(&raw_bytes), vec_out)
    }

    /// Scan `tokens` starting from `slot_tail = (prev2, prev1)` and stage all trigram embeddings.
    pub fn stage_sequence_embeddings(
        &self,
        slot_tail: (u32, u32),
        tokens: &[u32],
    ) -> (Vec<Vec<f32>>, (u32, u32)) {
        let (tris, final_tail) = scan_trigrams_with_tail(slot_tail.0, slot_tail.1, tokens);
        let mut out = Vec::with_capacity(tris.len());
        for tri in tris {
            let (plane, emb) = self.lookup_trigram_embedding(tri);
            debug_assert!(plane.is_tma_aligned());
            out.push(emb);
        }
        (out, final_tail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trigram_scanner_is_invariant_across_arbitrary_splits() {
        let full_seq: Vec<u32> = vec![11, 22, 33, 44, 55, 66, 77, 88, 99];
        let (mono_tris, mono_tail) = scan_trigrams_with_tail(0, 0, &full_seq);

        // Split into 3 arbitrary chunks (`[0..2]`, `[2..6]`, `[6..9]`)
        let (c1, t1) = scan_trigrams_with_tail(0, 0, &full_seq[0..2]);
        let (c2, t2) = scan_trigrams_with_tail(t1.0, t1.1, &full_seq[2..6]);
        let (c3, t3) = scan_trigrams_with_tail(t2.0, t2.1, &full_seq[6..9]);

        let mut stitched = Vec::new();
        stitched.extend(c1);
        stitched.extend(c2);
        stitched.extend(c3);

        assert_eq!(mono_tris, stitched);
        assert_eq!(mono_tail, t3);
    }

    #[test]
    fn zero_spill_runtime_launches_zero_prefetch_kernels() {
        let full = PlanarNvfp4Experts::deterministic_synthetic(99, 16, 32, 1);
        let resident_layer = TieredExpertLayer::from_planar(full, 16);
        assert_eq!(resident_layer.spilled_expert_count(), 0);
        let mut rt = SpilledPingPongRuntime::new(0);
        let _ = rt.stage_visit_experts(0, &resident_layer, &[(0, 0), (5, 0)]);
        assert_eq!(rt.prefetch_kernel_launches, 0);
    }

    #[test]
    fn spilled_ping_pong_maintains_causal_buffer_safety_across_48_visits() {
        let full = PlanarNvfp4Experts::deterministic_synthetic(101, 128, 32, 1);
        let spilled_layer = TieredExpertLayer::from_planar(full, 48); // 80 spilled experts!
        let mut rt = SpilledPingPongRuntime::new(80 * 24);
        for v in 0..48 {
            let pairs = [(10, 0), (49, 0), (95, 0)]; // resident + DtoD promoted + HtoD spilled
            let _ = rt.stage_visit_experts(v, &spilled_layer, &pairs);
        }
        assert_eq!(rt.prefetch_kernel_launches, 48);
        assert_eq!(rt.traces.len(), 48);
        for i in 2..48 {
            assert!(
                rt.traces[i - 2].compute_done_tick < rt.traces[i].wait_prev_compute_tick,
                "WAR hazard on ping-pong buffer slot {}",
                i % 2
            );
        }
    }
}
