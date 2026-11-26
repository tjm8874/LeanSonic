//! 128B-Aligned Planar SoA Weights (`0 B` extra VRAM) & Late-Wait Async `HtoD` Offloader.
//!
//! Unifies:
//! 1. In-place Planar SoA separation of quantized codes (`NVFP4 E2M1` / `Q8`) and scales
//!    on 128-byte boundaries (`SM120_TMA_ALIGNMENT_BYTES`) for TMA 2D Prefill + `LDG.128` Decode.
//! 2. Late-Wait Async `HtoD` scheduler serving both Inkling-2 Spilled Experts and
//!    Qwen3.8/4 51B host `Engram` N-gram embedding lookups, with hard suppression of
//!    prefetch launches when `spilled_bytes == 0`.

use crate::generated_spec::SM120_TMA_ALIGNMENT_BYTES;

#[repr(C, align(128))]
#[derive(Clone, Copy, Debug)]
pub struct Align128Chunk(pub [u8; SM120_TMA_ALIGNMENT_BYTES]);

impl Default for Align128Chunk {
    fn default() -> Self {
        Self([0u8; SM120_TMA_ALIGNMENT_BYTES])
    }
}

/// 128-byte aligned byte buffer guaranteed to satisfy `sm_120` TMA base address alignment.
#[derive(Clone, Debug)]
pub struct AlignedBytePlane {
    chunks: Vec<Align128Chunk>,
    byte_len: usize,
}

impl AlignedBytePlane {
    pub fn from_bytes(src: &[u8]) -> Self {
        let n_chunks = src.len().div_ceil(SM120_TMA_ALIGNMENT_BYTES).max(1);
        let mut chunks = vec![Align128Chunk::default(); n_chunks];
        let flat = unsafe {
            std::slice::from_raw_parts_mut(
                chunks.as_mut_ptr() as *mut u8,
                n_chunks * SM120_TMA_ALIGNMENT_BYTES,
            )
        };
        flat[..src.len()].copy_from_slice(src);
        Self {
            chunks,
            byte_len: src.len(),
        }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.chunks.as_ptr() as *const u8
    }

    pub fn is_tma_aligned(&self) -> bool {
        (self.as_ptr() as usize).is_multiple_of(SM120_TMA_ALIGNMENT_BYTES)
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.byte_len) }
    }
}

/// Planar SoA NVFP4 (`E2M1`) expert weight tensor (`[num_experts, n_rows, k_blocks, 16B]`).
/// Each 32-element block packs 32 4-bit `E2M1` codes into 16 bytes + 1 `f32` block scale.
#[derive(Clone, Debug)]
pub struct PlanarNvfp4Experts {
    pub num_experts: usize,
    pub n_rows: usize,
    pub k_blocks: usize,
    pub code_plane: AlignedBytePlane,
    pub scale_plane: Vec<f32>,
}

impl PlanarNvfp4Experts {
    pub fn deterministic_synthetic(
        seed: u64,
        num_experts: usize,
        n_rows: usize,
        k_blocks: usize,
    ) -> Self {
        let total_blocks = num_experts * n_rows * k_blocks;
        let mut codes = vec![0u8; total_blocks * 16];
        let mut scales = vec![0.0f32; total_blocks];

        let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
        for b in 0..total_blocks {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let scale_step = ((s >> 32) & 0xFF) as f32 / 255.0;
            scales[b] = 0.02 + 0.08 * scale_step;
            for byte_idx in 0..16 {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                let lo_nib = ((s >> 40) & 0x0F) as u8;
                let hi_nib = ((s >> 48) & 0x0F) as u8;
                codes[b * 16 + byte_idx] = (hi_nib << 4) | lo_nib;
            }
        }

        Self {
            num_experts,
            n_rows,
            k_blocks,
            code_plane: AlignedBytePlane::from_bytes(&codes),
            scale_plane: scales,
        }
    }

    pub fn block_codes(&self, expert: usize, row: usize, kb: usize) -> &[u8] {
        let blk = (expert * self.n_rows + row) * self.k_blocks + kb;
        let off = blk * 16;
        &self.code_plane.as_slice()[off..off + 16]
    }

    pub fn block_scale(&self, expert: usize, row: usize, kb: usize) -> f32 {
        let blk = (expert * self.n_rows + row) * self.k_blocks + kb;
        self.scale_plane[blk]
    }
}

/// Late-Wait Async `HtoD` staging descriptor for Spilled Experts & Host `Engram` tables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AsyncHtoDPlan {
    /// Zero spilled/host bytes -> hard-disable prefetch kernel launch (`-49.8 ms` idle scan trap avoided!).
    SuppressedZeroSpill,
    /// Stage `bytes` on non-blocking `h2d_stream` and defer `cudaStreamWaitEvent` until `wait_before_op`.
    Scheduled {
        bytes: usize,
        issue_at_stage: &'static str,
        wait_before_op: &'static str,
    },
}

pub fn plan_async_htod_staging(
    spilled_expert_bytes: usize,
    engram_lookup_bytes: usize,
) -> AsyncHtoDPlan {
    let total = spilled_expert_bytes + engram_lookup_bytes;
    if total == 0 {
        AsyncHtoDPlan::SuppressedZeroSpill
    } else if engram_lookup_bytes > 0 && spilled_expert_bytes == 0 {
        // Input token bigrams/trigrams are 100% known at token ingestion before Visit 0!
        AsyncHtoDPlan::Scheduled {
            bytes: engram_lookup_bytes,
            issue_at_stage: "token_ingest_pre_visit_0",
            wait_before_op: "engram_residual_inject",
        }
    } else {
        // Defer stream wait past `router -> remap -> moe_sort -> quantize` right up to first expert read!
        AsyncHtoDPlan::Scheduled {
            bytes: total,
            issue_at_stage: "post_prev_chunk_down_gemm",
            wait_before_op: "first_spilled_expert_global_read",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planar_soa_plane_is_128b_aligned() {
        let w = PlanarNvfp4Experts::deterministic_synthetic(42, 4, 8, 2);
        assert!(w.code_plane.is_tma_aligned());
    }

    #[test]
    fn zero_spill_hard_suppresses_idle_prefetch_launch() {
        assert_eq!(
            plan_async_htod_staging(0, 0),
            AsyncHtoDPlan::SuppressedZeroSpill
        );
        assert!(matches!(
            plan_async_htod_staging(0, 4096),
            AsyncHtoDPlan::Scheduled {
                issue_at_stage: "token_ingest_pre_visit_0",
                ..
            }
        ));
    }
}
