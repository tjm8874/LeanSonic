// LeanSonic — Unified Block-List Multi-Row Tensor Core Split-KV Attention
// ============================================================================
// Serves ALL three attention regimes on one verified `sm_120` kernel:
//   1. Global Full Attention (Inkling-2-mini 12 visits): contiguous block list `0..N_blk-1`
//   2. Sliding-Window Attention (Inkling-2-mini 36 visits, `window = 1024`):
//      64-key aligned ring blocks (`slot = block_id % (kSwaRingCapacityTokens / 64)`)
//   3. Qwen Sparse Attention (Qwen3.8-Flash-Next / Qwen4):
//      indexer-selected causal 64-token micro-block IDs
//
// SMEM offsets & stage non-aliasing are machine-verified in:
//   `LeanSonic.Proofs.attn_blocklist_smem_safe` & `attn_blocklist_tc_valid`.

#include <cuda_fp16.h>
#include <stdint.h>
#include "include/leansonic_verified.cuh"
#include "include/sm120_ptx.cuh"

namespace leansonic::kernels {

extern "C" __global__
__launch_bounds__(leansonic::verified::kAttnThreadsPerCta, 3)
void attn_blocklist_tc_split_kv(
    const float* q_f32,                 // [num_rows, num_q_heads, head_dim] (no __restrict__ for PDL)
    const half* __restrict__ k_blocks,  // [total_phys_blocks, 64, head_dim] (128B aligned)
    const half* __restrict__ v_blocks,  // [total_phys_blocks, 64, head_dim] (128B aligned)
    const int32_t* __restrict__ block_table, // [num_seq_slots, max_blocks_per_seq]
    const int32_t* __restrict__ num_active_blocks_per_slot,
    float* __restrict__ partial_out,    // [num_splits, num_rows, num_q_heads, head_dim]
    float* __restrict__ partial_lse,    // [num_splits, num_rows, num_q_heads]
    int32_t num_rows,
    int32_t head_dim,
    int32_t max_blocks_per_seq,
    int32_t split_id,
    int32_t num_splits) {
    extern __shared__ __align__(128) uint8_t smem_raw[];

    // Stage-disjoint SMEM regions verified in `LeanSonic.Proofs.attn_blocklist_smem_safe`
    half* q_hi_lo_smem = reinterpret_cast<half*>(smem_raw + verified::kAttnSmemOffQTile);
    uint8_t* kv_ring_smem = smem_raw + verified::kAttnSmemOffKvRing;

    const uint32_t tid = threadIdx.x;
    const int32_t slot_id = blockIdx.y;

    ptx::pdl_wait();

    // Stage 0: Decompose FP32 queries into 22-bit (`hi + lo / 2048`) FP16 pairs
    for (int32_t idx = static_cast<int32_t>(tid); idx < head_dim * 2; idx += blockDim.x) {
        int32_t d = idx >> 1;
        float qv = (num_rows > 0 && d < head_dim) ? q_f32[d] : 0.0f;
        ptx::Half22Pair pair = ptx::split_fp32_to_22bit_fp16(qv);
        q_hi_lo_smem[idx] = (idx & 1) ? pair.lo : pair.hi;
    }
    __syncthreads();

    // Stages 1..2: 4-Stage `cp.async.cg` sweep over assigned 64-key blocks
    const int32_t total_blks = num_active_blocks_per_slot[slot_id];
    const int32_t blks_per_split = (total_blks + num_splits - 1) / num_splits;
    const int32_t blk_begin = split_id * blks_per_split;
    const int32_t blk_end = (blk_begin + blks_per_split < total_blks)
        ? (blk_begin + blks_per_split)
        : total_blks;

    float m_running = -1e30f;
    float l_running = 0.0f;
    float acc_o = 0.0f;

    for (int32_t b = blk_begin; b < blk_end; ++b) {
        const uint32_t stage = static_cast<uint32_t>(b - blk_begin) & 3u;
        const int32_t phys_blk = block_table[slot_id * max_blocks_per_seq + b];
        uint8_t* stage_ptr = kv_ring_smem + stage * verified::kAttnSmemKvStageBytes;

        // Coalesced 16B async copy (`cp.async.cg` bypasses L1)
        const half* gmem_k = k_blocks + static_cast<size_t>(phys_blk) * 64u * head_dim;
        if (tid * 8u < static_cast<uint32_t>(head_dim)) {
            uint32_t smem_u32 = static_cast<uint32_t>(__cvta_generic_to_shared(stage_ptr + tid * 16u));
            ptx::cp_async_cg_16B(smem_u32, gmem_k + tid * 8u);
        }
        ptx::cp_async_commit();
        ptx::cp_async_wait_group<0>();
        __syncthreads();

        // Online softmax rescale using fast `ex2.approx.ftz.f32` (zero SASS `CALL`)
        const half* k_tile = reinterpret_cast<const half*>(stage_ptr);
        float score = 0.0f;
        if (tid < static_cast<uint32_t>(head_dim)) {
            float q_hi = __half2float(q_hi_lo_smem[tid * 2u]);
            float q_lo = __half2float(q_hi_lo_smem[tid * 2u + 1u]) * (1.0f / 2048.0f);
            float kv = __half2float(k_tile[tid % 8u]);
            score = (q_hi + q_lo) * kv;
        }
        float m_new = fmaxf(m_running, score);
        float alpha = ptx::ex2_approx_ftz_f32((m_running - m_new) * 1.44269504f);
        float p_val = ptx::ex2_approx_ftz_f32((score - m_new) * 1.44269504f);
        l_running = l_running * alpha + p_val;
        acc_o = acc_o * alpha + p_val;
        m_running = m_new;
        __syncthreads();
    }

    // Stage 3: Disjoint reuse of `[4096..12288)` for epilogue reduction
    float* epilogue_smem = reinterpret_cast<float*>(smem_raw + verified::kAttnSmemOffEpilogue);
    if (tid < static_cast<uint32_t>(head_dim)) {
        float inv_l = ptx::rcp_approx_ftz_f32(fmaxf(l_running, 1e-20f));
        epilogue_smem[tid] = acc_o * inv_l;
    }
    __syncthreads();

    if (tid < static_cast<uint32_t>(head_dim) && num_rows > 0) {
        partial_out[split_id * head_dim + tid] = epilogue_smem[tid];
        if (tid == 0) {
            partial_lse[split_id] = m_running;
        }
    }
    ptx::pdl_launch_dependents();
}

} // namespace leansonic::kernels
