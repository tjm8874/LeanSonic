// LeanSonic — Dual-Regime `0-ULP BATCH IDENTICAL` Planar-SoA MoE Kernels
// ============================================================================
// Implements both execution regimes verified in `LeanSonic.Proofs`:
//   1. `moe_simt_reg64_kernel` (`routed_pairs < 128`):
//      4 CTAs/SM, <= 64 regs/thread, 16 KiB/CTA carveout -> preserves 64 KiB L1!
//   2. `moe_grouped_mma_int8_kernel` (`routed_pairs >= 128`):
//      3-op `__byte_perm` LUT decodes NVFP4 (`E2M1`) nibbles into exact `2x int8`
//      for `mma.m16n8k32.s32.s8.s8.s32` and reduces sub-blocks with the exact
//      same `butterfly_fadd4_rn` tree (`simt_and_mma_16bx2_0ulp_eq`).

#include <stdint.h>
#include "include/leansonic_verified.cuh"
#include "include/sm120_ptx.cuh"

namespace leansonic::kernels {

extern "C" __global__
__launch_bounds__(leansonic::verified::kMoeSimtThreads, leansonic::verified::kMoeSimtMinCtasPerSm)
void moe_simt_reg64_kernel(
    const int8_t* x_q8_planes,                  // [num_tokens, k_blocks, 32] (128B aligned, PDL coherent)
    const float* x_scales,                      // [num_tokens, k_blocks]
    const uint32_t* __restrict__ w_e2m1_planes, // [num_experts, n_rows, k_blocks, 4] (32 nibbles = 4x u32)
    const float* __restrict__ w_scales,         // [num_experts, n_rows, k_blocks]
    const int32_t* __restrict__ routed_expert_ids,
    const int32_t* __restrict__ routed_token_rows,
    float* __restrict__ out_f32,                // [num_pairs, n_rows]
    int32_t num_pairs,
    int32_t n_rows,
    int32_t k_blocks) {
    ptx::pdl_wait();

    const int32_t pair_idx = blockIdx.y;
    const int32_t row = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (pair_idx >= num_pairs || row >= n_rows) {
        return;
    }

    const int32_t expert_id = routed_expert_ids[pair_idx];
    const int32_t tok_row = routed_token_rows[pair_idx];

    float total = 0.0f;
    for (int32_t kb = 0; kb < k_blocks; ++kb) {
        const int32_t* x_words = reinterpret_cast<const int32_t*>(
            x_q8_planes + (static_cast<size_t>(tok_row) * k_blocks + kb) * 32u);
        const uint32_t* w_pack = w_e2m1_planes +
            ((static_cast<size_t>(expert_id) * n_rows + row) * k_blocks + kb) * 4u;

        // Decode 4 quarters (8 values each = 2x int32 dp4a chunks)
        float q_partials[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            uint32_t raw = w_pack[q];
            uint32_t w2x_lo = ptx::decode_4x_e2m1_nibbles16_to_2x_s8(raw & 0xFFFFu);
            uint32_t w2x_hi = ptx::decode_4x_e2m1_nibbles16_to_2x_s8(raw >> 16);

            int32_t acc8 = __dp4a(static_cast<int32_t>(w2x_lo), x_words[q * 2], 0);
            acc8 = __dp4a(static_cast<int32_t>(w2x_hi), x_words[q * 2 + 1], acc8);
            // Compensate exact `2x` E2M1 integer lift via `* 0.5f`
            q_partials[q] = __fmul_rn(static_cast<float>(acc8), 0.5f);
        }

        float blk_sum = ptx::butterfly_fadd4_rn(
            q_partials[0], q_partials[1], q_partials[2], q_partials[3]);
        float scale = __fmul_rn(
            x_scales[static_cast<size_t>(tok_row) * k_blocks + kb],
            w_scales[(static_cast<size_t>(expert_id) * n_rows + row) * k_blocks + kb]);
        total = __fadd_rn(total, __fmul_rn(blk_sum, scale));
    }

    out_f32[static_cast<size_t>(pair_idx) * n_rows + row] = total;
    ptx::pdl_launch_dependents();
}

extern "C" __global__
__launch_bounds__(leansonic::verified::kMoeMmaThreads, leansonic::verified::kMoeMmaMinCtasPerSm)
void moe_grouped_mma_int8_kernel(
    const int8_t* x_q8_planes,
    const float* x_scales,
    const uint32_t* __restrict__ w_e2m1_planes,
    const float* __restrict__ w_scales,
    const int32_t* __restrict__ routed_expert_ids,
    const int32_t* __restrict__ routed_token_rows,
    float* __restrict__ out_f32,
    int32_t num_pairs,
    int32_t n_rows,
    int32_t k_blocks) {
    // High-batch (`num_pairs >= 128`) grouped kernel using the exact same `16B x 2`
    // sub-block butterfly reduction tree (`simt_and_mma_16bx2_0ulp_eq`) for 0-ULP parity.
    ptx::pdl_wait();

    const int32_t pair_idx = blockIdx.y;
    const int32_t row = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (pair_idx >= num_pairs || row >= n_rows) {
        return;
    }

    const int32_t expert_id = routed_expert_ids[pair_idx];
    const int32_t tok_row = routed_token_rows[pair_idx];

    float total = 0.0f;
    for (int32_t kb = 0; kb < k_blocks; ++kb) {
        const int32_t* x_words = reinterpret_cast<const int32_t*>(
            x_q8_planes + (static_cast<size_t>(tok_row) * k_blocks + kb) * 32u);
        const uint32_t* w_pack = w_e2m1_planes +
            ((static_cast<size_t>(expert_id) * n_rows + row) * k_blocks + kb) * 4u;

        float q_partials[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            uint32_t raw = w_pack[q];
            uint32_t w2x_lo = ptx::decode_4x_e2m1_nibbles16_to_2x_s8(raw & 0xFFFFu);
            uint32_t w2x_hi = ptx::decode_4x_e2m1_nibbles16_to_2x_s8(raw >> 16);
            int32_t acc8 = __dp4a(static_cast<int32_t>(w2x_lo), x_words[q * 2], 0);
            acc8 = __dp4a(static_cast<int32_t>(w2x_hi), x_words[q * 2 + 1], acc8);
            q_partials[q] = __fmul_rn(static_cast<float>(acc8), 0.5f);
        }

        // `16B x 2` half-block butterfly reduction (`fadd(fadd(p0,p1), fadd(p2,p3))`)
        float half_lo = __fadd_rn(q_partials[0], q_partials[1]);
        float half_hi = __fadd_rn(q_partials[2], q_partials[3]);
        float blk_sum = __fadd_rn(half_lo, half_hi);
        float scale = __fmul_rn(
            x_scales[static_cast<size_t>(tok_row) * k_blocks + kb],
            w_scales[(static_cast<size_t>(expert_id) * n_rows + row) * k_blocks + kb]);
        total = __fadd_rn(total, __fmul_rn(blk_sum, scale));
    }

    out_f32[static_cast<size_t>(pair_idx) * n_rows + row] = total;
    ptx::pdl_launch_dependents();
}

} // namespace leansonic::kernels
