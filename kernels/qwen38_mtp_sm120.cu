// LeanSonic — Qwen3.8-Flash-Next built-in MTP drafter (`language_model.mtp.*`) input fusion.
// The MTP decoder layer itself reuses the `kernels/qwen38_sm120.cu` kernels (same C ABI).
#include <stdint.h>
#include "include/leansonic_verified.cuh"

namespace leansonic::qwen38_mtp {

constexpr int kD = 2560;
constexpr int kHcD = 10240;

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    return v;
}

} // namespace leansonic::qwen38_mtp

using namespace leansonic::qwen38_mtp;

// Gemma RMSNorm over `n` values (`pre_fc_norm_embedding`: 2560, `pre_fc_norm_hidden`: 10240 =
// all four hyper-connection streams together): `out = x * rsqrt(mean(x^2) + eps) * w_eff`,
// `w_eff` already `1 + w` for zero-centered checkpoints.
// Grid: `(1, 1)`, Block: `256`
extern "C" __global__ void q38_mtp_rmsnorm(
    const float* __restrict__ x,
    const float* __restrict__ w_eff,
    int32_t n,
    float* __restrict__ out) {
    __shared__ float smem_sq[8];
    __shared__ float smem_inv;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    float sq = 0.0f;
    for (int i = tid; i < n; i += 256) {
        const float v = x[i];
        sq += v * v;
    }
    sq = warp_sum(sq);
    if (lane == 0) smem_sq[warp_id] = sq;
    __syncthreads();
    if (warp_id == 0) {
        float tot = (lane < 8) ? smem_sq[lane] : 0.0f;
        tot = warp_sum(tot);
        // Constant reciprocals (no IEEE division call in SASS); other sizes use the fast path.
        const float inv_n = (n == kD) ? (1.0f / 2560.0f)
                          : (n == kHcD) ? (1.0f / 10240.0f)
                                        : __fdividef(1.0f, static_cast<float>(n));
        if (lane == 0) smem_inv = rsqrtf(tot * inv_n + 1e-6f);
    }
    __syncthreads();
    const float inv = smem_inv;
    for (int i = tid; i < n; i += 256) out[i] = x[i] * inv * w_eff[i];
}

// `hc[s, d] += e[d]` for the four streams (`fc_hidden(h_s) + fc_embedding(e)`).
// Grid: `(40, 1)`, Block: `256`
extern "C" __global__ void q38_mtp_add_bcast4(
    float* __restrict__ hc,          // [4, 2560]
    const float* __restrict__ e) {   // [2560]
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= kHcD) return;
    hc[i] += e[i % kD];
}
