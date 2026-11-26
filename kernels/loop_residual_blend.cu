// LeanSonic — Fused RMSNorm + Loop / 4-Branch Gated Residual Blend + Planar Q8 Quantize
// ============================================================================
// Unifies:
//   - Inkling-2-mini Gated Loop Blend (`R = 1`, passes 2 & 3 of Loop Core)
//   - Qwen3.8-Flash-Next / Qwen4 4-Branch Gated Residual (`R = 4`)
//   - Downstream 128B-aligned Planar `Q8` activation quantization (`0` intermediate DRAM trips)
// Fixed reduction tree width (`256` threads) prevents 1-ULP re-association drift across 48 visits.

#include <stdint.h>
#include "include/leansonic_verified.cuh"
#include "include/sm120_ptx.cuh"

namespace leansonic::kernels {

extern "C" __global__
__launch_bounds__(256, 2)
void loop_residual_blend_q8_kernel(
    float* residual_streams,            // [num_rows, num_branches, hidden_dim] (PDL coherent)
    const float* branch_delta,          // [num_rows, hidden_dim]
    const float* __restrict__ rms_weight,
    const float* __restrict__ branch_gates, // [num_rows, num_branches]
    int8_t* __restrict__ out_q8_planes, // [num_rows, hidden_dim / 32, 32] (128B aligned)
    float* __restrict__ out_q8_scales,  // [num_rows, hidden_dim / 32]
    int32_t hidden_dim,
    int32_t num_branches,
    float rms_eps) {
    __shared__ float smem_sq_sum[256];

    ptx::pdl_wait();

    const int32_t row = blockIdx.x;
    const int32_t tid = threadIdx.x;

    // 1. Blend 1-branch (Looped) or 4-branch (Qwen GR) residual streams & accumulate fixed-tree sumsq
    float local_sq = 0.0f;
    for (int32_t d = tid; d < hidden_dim; d += 256) {
        float acc = branch_delta[static_cast<size_t>(row) * hidden_dim + d];
        for (int32_t b = 0; b < num_branches; ++b) {
            size_t off = (static_cast<size_t>(row) * num_branches + b) * hidden_dim + d;
            float g = branch_gates[static_cast<size_t>(row) * num_branches + b];
            float updated = __fadd_rn(__fmul_rn(residual_streams[off], g), acc);
            residual_streams[off] = updated;
            if (b == 0) {
                acc = updated;
            }
        }
        local_sq = __fadd_rn(local_sq, __fmul_rn(acc, acc));
    }

    // Deterministic 256-thread power-of-two reduction tree (prevents 48-visit 1-ULP drift)
    smem_sq_sum[tid] = local_sq;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem_sq_sum[tid] = __fadd_rn(smem_sq_sum[tid], smem_sq_sum[tid + stride]);
        }
        __syncthreads();
    }

    const float mean_sq = smem_sq_sum[0] * ptx::rcp_approx_ftz_f32(static_cast<float>(hidden_dim));
    const float inv_rms = rsqrtf(mean_sq + rms_eps);

    // 2. Fused RMSNorm + 32-element Planar `Q8` Quantization (1 thread per 32-dim block)
    const int32_t num_q8_blocks = hidden_dim / 32;
    for (int32_t qb = tid; qb < num_q8_blocks; qb += 256) {
        float vals[32];
        float amax = 0.0f;
        const size_t base = (static_cast<size_t>(row) * num_branches * hidden_dim) + qb * 32u;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            float v = residual_streams[base + i] * inv_rms * rms_weight[qb * 32 + i];
            vals[i] = v;
            amax = fmaxf(amax, fabsf(v));
        }
        float scale = amax * (1.0f / 127.0f);
        float inv_scale = (amax > 0.0f) ? (127.0f * ptx::rcp_approx_ftz_f32(amax)) : 0.0f;
        out_q8_scales[static_cast<size_t>(row) * num_q8_blocks + qb] = scale;

        int8_t* dst = out_q8_planes + (static_cast<size_t>(row) * num_q8_blocks + qb) * 32u;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            int32_t q = __float2int_rn(vals[i] * inv_scale);
            q = (q < -127) ? -127 : ((q > 127) ? 127 : q);
            dst[i] = static_cast<int8_t>(q);
        }
    }

    ptx::pdl_launch_dependents();
}

} // namespace leansonic::kernels
