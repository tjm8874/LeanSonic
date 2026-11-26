// LeanSonic — Full-GPU `Inkling-2-Mini` (`sm_120`) Kernel Suite (`inkling2_sm120.cu`)
// ============================================================================
// Executes ALL 24 physical layers (`48` scheduled visits: `0..19` NVFP4 MoE +
// `20..23` MXFP8 MoE, depthwise attention residuals, loop core controllers,
// 4-tap static/dynamic short-convs, L2 clips, log-scaled global attention, and
// 1024-extent relative-bias + sink GQA) 100% on NVIDIA Blackwell GPUs.
//
// Obeys all Gate 2 static SASS rules (`ci/audit_sass.py`):
//   - `STACK: 0` (zero local-memory spills)
//   - Zero `CALL.REL.NOINC` (inline PTX `tanh.approx.f32`, `ex2.approx.ftz.f32`)
//   - Zero 8-bit scalar `LDG.E.[U/S]8` (all byte/nibble tables read as 32/64/128-bit words)

#ifndef __CUDACC_RTC__
#include <stdint.h>
#include "include/leansonic_verified.cuh"
#else
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
#endif

namespace leansonic::inkling2 {

static constexpr int kD = 2304;
static constexpr int kHead = 24;
static constexpr int kKv = 8;
static constexpr int kHd = 128;
static constexpr int kRel = 16;
static constexpr int kExtent = 1024;
static constexpr int kExp = 128;
static constexpr int kTopK = 8;
static constexpr int kFf = 1152;
static constexpr int kRank = 64;
static constexpr int kVocab = 200058;

__device__ __forceinline__ float ptx_tanh(float x) {
    float y;
    asm("tanh.approx.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float ptx_exp(float x) {
    float y;
    // exp(x) = 2^(x * log2(e))
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x * 1.4426950408889634f));
    return y;
}

__device__ __forceinline__ float ptx_ln(float x) {
    float y;
    asm("lg2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y * 0.6931471805599453f;
}

__device__ __forceinline__ float ptx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float ptx_sqrt(float x) {
    float y;
    asm("sqrt.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float ptx_rsqrt(float x) {
    float y;
    asm("rsqrt.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float ptx_sigmoid(float x) {
    return ptx_rcp(1.0f + ptx_exp(-x));
}

__device__ __forceinline__ float bf16_to_f32(uint16_t b) {
    return __uint_as_float(static_cast<uint32_t>(b) << 16);
}

__device__ __forceinline__ uint16_t f32_to_fp16_bits(float v) {
    uint16_t h;
    asm("cvt.rn.f16.f32 %0, %1;\n" : "=h"(h) : "f"(v));
    return h;
}

__device__ __forceinline__ float fp16_bits_to_f32(uint16_t h) {
    float v;
    asm("cvt.f32.f16 %0, %1;\n" : "=f"(v) : "h"(h));
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, mask);
    }
    return v;
}

__device__ __forceinline__ float e4m3_byte_to_f32(uint32_t b) {
    uint32_t mag = (b & 0x7Fu) << 20;
    uint32_t sgn = (b & 0x80u) << 24;
    return __uint_as_float(mag | sgn) * __uint_as_float(247u << 23);
}

// Verified `__byte_perm` LUT (`LeanSonic.Proofs.e2m1_all_16_codes_exact`):
// Decodes 4 contiguous 4-bit `E2M1` nibbles in `nibbles_lo16` (`bits 15..0`) into
// 4 packed signed `int8` values (`2 * E2M1`) in bytes `0..3`!
__device__ __forceinline__ uint32_t decode_4x_e2m1_nibbles16_to_2x_s8(uint32_t nibbles_lo16) {
    uint32_t sel = nibbles_lo16 & 0x7777u;
    uint32_t pos = __byte_perm(0x03020100u, 0x0c080604u, sel);
    uint32_t neg = __byte_perm(0xfdfeff00u, 0xf4f8fafcu, sel);
    uint32_t s4 = ((nibbles_lo16 >> 3) & 0x1u)
                | (((nibbles_lo16 >> 7) & 0x1u) << 8)
                | (((nibbles_lo16 >> 11) & 0x1u) << 16)
                | (((nibbles_lo16 >> 15) & 0x1u) << 24);
    uint32_t sign_mask = s4 * 0xFFu;
    return (pos & ~sign_mask) | (neg & sign_mask);
}

} // namespace leansonic::inkling2

using namespace leansonic::inkling2;

// ----------------------------------------------------------------------------
// 1. Token Embedding Lookup (from BF16 row) + RMSNorm (`[D = 2304]`)
// ----------------------------------------------------------------------------
extern "C" __global__ void ink2_embed_rmsnorm(
    const uint32_t* __restrict__ embed_bf16_pairs, // [vocab_padded, D / 2]
    const uint32_t* __restrict__ norm_bf16_pairs,  // [D / 2]
    const int32_t* __restrict__ token_id_ptr,
    float* __restrict__ out_x) {
    __shared__ float smem_red[8];
    const int tid = threadIdx.x; // 256 threads
    const int32_t token_id = *token_id_ptr;
    const uint32_t* row = embed_bf16_pairs + static_cast<size_t>(token_id) * (kD / 2);

    float sq = 0.0f;
    for (int idx = tid; idx < kD / 2; idx += 256) {
        uint32_t pair = row[idx];
        float v0 = bf16_to_f32(static_cast<uint16_t>(pair & 0xFFFFu));
        float v1 = bf16_to_f32(static_cast<uint16_t>(pair >> 16));
        out_x[idx * 2] = v0;
        out_x[idx * 2 + 1] = v1;
        sq += v0 * v0 + v1 * v1;
    }
    sq = warp_sum(sq);
    if ((tid & 31) == 0) smem_red[tid >> 5] = sq;
    __syncthreads();
    if (tid < 32) {
        float bsum = (tid < 8) ? smem_red[tid] : 0.0f;
        bsum = warp_sum(bsum);
        if (tid == 0) smem_red[0] = ptx_rsqrt(bsum * (1.0f / static_cast<float>(kD)) + 1e-6f);
    }
    __syncthreads();
    const float scale = smem_red[0];
    for (int idx = tid; idx < kD / 2; idx += 256) {
        uint32_t nw = norm_bf16_pairs[idx];
        out_x[idx * 2] *= scale * bf16_to_f32(static_cast<uint16_t>(nw & 0xFFFFu));
        out_x[idx * 2 + 1] *= scale * bf16_to_f32(static_cast<uint16_t>(nw >> 16));
    }
}

// ----------------------------------------------------------------------------
// 2. RMSNorm (in-place or out-of-place, optional BF16 gamma, `len = 2304`)
// ----------------------------------------------------------------------------
extern "C" __global__ void ink2_rmsnorm_d(
    const float* __restrict__ inp,
    const uint32_t* __restrict__ gamma_bf16_pairs, // nullable if `has_gamma == 0`
    float* __restrict__ out,
    int32_t has_gamma) {
    __shared__ float smem_red[8];
    const int tid = threadIdx.x;
    float sq = 0.0f;
    for (int i = tid; i < kD; i += 256) {
        float v = inp[i];
        sq += v * v;
    }
    sq = warp_sum(sq);
    if ((tid & 31) == 0) smem_red[tid >> 5] = sq;
    __syncthreads();
    if (tid < 32) {
        float bsum = (tid < 8) ? smem_red[tid] : 0.0f;
        bsum = warp_sum(bsum);
        if (tid == 0) smem_red[0] = ptx_rsqrt(bsum * (1.0f / static_cast<float>(kD)) + 1e-6f);
    }
    __syncthreads();
    const float k = smem_red[0];
    if (has_gamma) {
        for (int idx = tid; idx < kD / 2; idx += 256) {
            uint32_t gw = gamma_bf16_pairs[idx];
            float x0 = inp[idx * 2];
            float x1 = inp[idx * 2 + 1];
            out[idx * 2] = x0 * k * bf16_to_f32(static_cast<uint16_t>(gw & 0xFFFFu));
            out[idx * 2 + 1] = x1 * k * bf16_to_f32(static_cast<uint16_t>(gw >> 16));
        }
    } else {
        for (int i = tid; i < kD; i += 256) {
            out[i] = inp[i] * k;
        }
    }
}

// ----------------------------------------------------------------------------
// 3. Depthwise Attention-Residual `mix(states, queries)` (`n_states <= 6`, `n_q <= 4`)
//    Grid: `n_q` CTAs (`256` threads/CTA). If `append_cur == 1`, `cur` is the last state!
// ----------------------------------------------------------------------------
extern "C" __global__ void ink2_depth_mix(
    const float* __restrict__ blocks_buf, // [max_blocks, D]
    int32_t n_blocks,
    const float* __restrict__ cur_opt,    // [D] if `append_cur == 1`
    int32_t append_cur,
    const float* __restrict__ queries,    // [n_q, D]
    float* __restrict__ out_mixed) {      // [n_q, D]
    __shared__ float smem_w[8];
    __shared__ float smem_inv_rms[6];
    __shared__ float smem_scores[6];
    __shared__ float smem_probs[6];

    const int q_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int total_states = n_blocks + append_cur;
    const float* q = queries + static_cast<size_t>(q_idx) * kD;
    const float inv_sqrt_d = 0.020833333333333332f; // 1.0 / sqrt(2304.0) = 1.0 / 48.0

    for (int s = 0; s < total_states; ++s) {
        const float* st = (s < n_blocks) ? (blocks_buf + static_cast<size_t>(s) * kD) : cur_opt;
        float sq = 0.0f;
        float dq = 0.0f;
        for (int i = tid; i < kD; i += 256) {
            float v = st[i];
            sq += v * v;
            dq += v * q[i];
        }
        sq = warp_sum(sq);
        dq = warp_sum(dq);
        if ((tid & 31) == 0) smem_w[tid >> 5] = sq;
        __syncthreads();
        float total_sq = 0.0f;
        if (tid < 8) total_sq = smem_w[tid];
        total_sq = warp_sum(total_sq);
        if (tid == 0) {
            smem_inv_rms[s] = ptx_rsqrt(total_sq * (1.0f / static_cast<float>(kD)) + 1e-6f);
        }
        __syncthreads();

        if ((tid & 31) == 0) smem_w[tid >> 5] = dq;
        __syncthreads();
        float total_dq = 0.0f;
        if (tid < 8) total_dq = smem_w[tid];
        total_dq = warp_sum(total_dq);
        if (tid == 0) {
            smem_scores[s] = total_dq * smem_inv_rms[s] * inv_sqrt_d;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float m = -1e30f;
        for (int s = 0; s < total_states; ++s) {
            if (smem_scores[s] > m) m = smem_scores[s];
        }
        float z = 0.0f;
        for (int s = 0; s < total_states; ++s) {
            float e = ptx_exp(smem_scores[s] - m);
            smem_probs[s] = e;
            z += e;
        }
        float inv_z = ptx_rcp(z);
        for (int s = 0; s < total_states; ++s) {
            smem_probs[s] = smem_probs[s] * inv_z * smem_inv_rms[s];
        }
    }
    __syncthreads();

    float* dst = out_mixed + static_cast<size_t>(q_idx) * kD;
    for (int i = tid; i < kD; i += 256) {
        float acc = 0.0f;
        for (int s = 0; s < total_states; ++s) {
            const float* st = (s < n_blocks) ? (blocks_buf + static_cast<size_t>(s) * kD) : cur_opt;
            acc += smem_probs[s] * st[i];
        }
        dst[i] = acc;
    }
}

// ----------------------------------------------------------------------------
// 4. Loop Controller Blend (`control(cur, msg, ctrl)`)
//     Stage A (`grid = 64`, `block = 256`): compute `g[64] = down[64, 4608] * [norm(cur); norm(msg)]`
//     Stage B (`grid = 288`, `block = 256` -> 8 warps/CTA = 8 rows/CTA):
//             `out[i] = cur[i] - tanh(forget[i]) * cur[i] + tanh(gain[i]) * msg[i]`
// ----------------------------------------------------------------------------
extern "C" __global__ void ink2_ctrl_down(
    const float* __restrict__ cur,
    const float* __restrict__ msg,
    const uint4* __restrict__ down_bf16_u128, // [64, 4608 / 8 = 576]
    float* __restrict__ g_out) {              // [64]
    __shared__ float smem_red[8];
    __shared__ float inv_a;
    __shared__ float inv_b;

    const int r = blockIdx.x; // 0..63
    const int tid = threadIdx.x;

    // Compute RMSNorm factors of `cur` and `msg`
    float sq_a = 0.0f;
    float sq_b = 0.0f;
    for (int i = tid; i < kD; i += 256) {
        float va = cur[i];
        float vb = msg[i];
        sq_a += va * va;
        sq_b += vb * vb;
    }
    sq_a = warp_sum(sq_a);
    if ((tid & 31) == 0) smem_red[tid >> 5] = sq_a;
    __syncthreads();
    if (tid < 8) sq_a = smem_red[tid]; else sq_a = 0.0f;
    sq_a = warp_sum(sq_a);
    if (tid == 0) inv_a = ptx_rsqrt(sq_a * (1.0f / static_cast<float>(kD)) + 1e-6f);
    __syncthreads();

    sq_b = warp_sum(sq_b);
    if ((tid & 31) == 0) smem_red[tid >> 5] = sq_b;
    __syncthreads();
    if (tid < 8) sq_b = smem_red[tid]; else sq_b = 0.0f;
    sq_b = warp_sum(sq_b);
    if (tid == 0) inv_b = ptx_rsqrt(sq_b * (1.0f / static_cast<float>(kD)) + 1e-6f);
    __syncthreads();

    const uint4* w_row = down_bf16_u128 + static_cast<size_t>(r) * 576u;
    const float ka = inv_a;
    const float kb = inv_b;
    float dot_acc = 0.0f;
    for (int c8 = tid; c8 < 576; c8 += 256) {
        uint4 pack = w_row[c8];
        uint32_t w_u32[4] = {pack.x, pack.y, pack.z, pack.w};
        int base = c8 * 8;
        const float* src = (base < kD) ? (cur + base) : (msg + (base - kD));
        float k_scale = (base < kD) ? ka : kb;
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            float w0 = bf16_to_f32(static_cast<uint16_t>(w_u32[p] & 0xFFFFu));
            float w1 = bf16_to_f32(static_cast<uint16_t>(w_u32[p] >> 16));
            dot_acc += (w0 * src[p * 2] + w1 * src[p * 2 + 1]) * k_scale;
        }
    }
    dot_acc = warp_sum(dot_acc);
    if ((tid & 31) == 0) smem_red[tid >> 5] = dot_acc;
    __syncthreads();
    if (tid < 32) {
        float s = (tid < 8) ? smem_red[tid] : 0.0f;
        s = warp_sum(s);
        if (tid == 0) g_out[r] = s;
    }
}

extern "C" __global__ void ink2_ctrl_apply(
    float* __restrict__ cur_inout,            // [D]
    const float* __restrict__ msg,            // [D]
    const float* __restrict__ g_vec,          // [64]
    const uint4* __restrict__ gain_bf16_u128, // [D, 64 / 8 = 8]
    const uint4* __restrict__ forget_bf16_u128) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kD) return;
    float gain_acc = 0.0f;
    float forget_acc = 0.0f;
    const uint4* g_row = gain_bf16_u128 + static_cast<size_t>(i) * 8u;
    const uint4* f_row = forget_bf16_u128 + static_cast<size_t>(i) * 8u;
    #pragma unroll
    for (int c8 = 0; c8 < 8; ++c8) {
        uint4 gp = g_row[c8];
        uint4 fp = f_row[c8];
        uint32_t gu[4] = {gp.x, gp.y, gp.z, gp.w};
        uint32_t fu[4] = {fp.x, fp.y, fp.z, fp.w};
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            float gv0 = g_vec[c8 * 8 + p * 2];
            float gv1 = g_vec[c8 * 8 + p * 2 + 1];
            gain_acc += bf16_to_f32(static_cast<uint16_t>(gu[p] & 0xFFFFu)) * gv0
                      + bf16_to_f32(static_cast<uint16_t>(gu[p] >> 16)) * gv1;
            forget_acc += bf16_to_f32(static_cast<uint16_t>(fu[p] & 0xFFFFu)) * gv0
                        + bf16_to_f32(static_cast<uint16_t>(fu[p] >> 16)) * gv1;
        }
    }
    float c = cur_inout[i];
    cur_inout[i] = c - ptx_tanh(forget_acc) * c + ptx_tanh(gain_acc) * msg[i];
}

// ----------------------------------------------------------------------------
// 5. Coalesced 128-Bit BF16 GEMV (`8` warps = `256` threads/CTA -> `8` rows/CTA)
//    Optional `softcap > 0.0f` applies `softcap * tanh(y / softcap)` (for `unembed`!)
// ----------------------------------------------------------------------------
extern "C" __global__ void ink2_bf16_gemv(
    const uint4* __restrict__ w_bf16_u128, // [rows, cols / 8]
    const float* __restrict__ x,           // [cols]
    float* __restrict__ y,                 // [rows]
    int32_t rows,
    int32_t cols,
    float softcap) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp_id;
    if (row >= rows) return;

    const int cols8 = cols >> 3;
    const uint4* w_row = w_bf16_u128 + static_cast<size_t>(row) * cols8;
    const float4* x4 = reinterpret_cast<const float4*>(x);

    float acc = 0.0f;
    for (int c8 = lane; c8 < cols8; c8 += 32) {
        uint4 pack = w_row[c8];
        float4 xv0 = x4[c8 * 2];
        float4 xv1 = x4[c8 * 2 + 1];
        acc += bf16_to_f32(static_cast<uint16_t>(pack.x & 0xFFFFu)) * xv0.x;
        acc += bf16_to_f32(static_cast<uint16_t>(pack.x >> 16)) * xv0.y;
        acc += bf16_to_f32(static_cast<uint16_t>(pack.y & 0xFFFFu)) * xv0.z;
        acc += bf16_to_f32(static_cast<uint16_t>(pack.y >> 16)) * xv0.w;
        acc += bf16_to_f32(static_cast<uint16_t>(pack.z & 0xFFFFu)) * xv1.x;
        acc += bf16_to_f32(static_cast<uint16_t>(pack.z >> 16)) * xv1.y;
        acc += bf16_to_f32(static_cast<uint16_t>(pack.w & 0xFFFFu)) * xv1.z;
        acc += bf16_to_f32(static_cast<uint16_t>(pack.w >> 16)) * xv1.w;
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        if (softcap > 0.0f) {
            acc = softcap * ptx_tanh(acc * ptx_rcp(softcap));
        }
        y[row] = acc;
    }
}

// ----------------------------------------------------------------------------
// 6. Causal 4-Tap Short-Convolution (`sconv`) + Optional Residual Add
//    `w_conv4` is `[channels, 4]` (`f32`), `hist` is `[channels, 3]` (`t-3, t-2, t-1`).
// ----------------------------------------------------------------------------
extern "C" __global__ void ink2_short_conv4(
    const float* __restrict__ cur,      // [channels]
    const float4* __restrict__ w_conv4, // [channels] (`x=t-3, y=t-2, z=t-1, w=t`)
    const float4* __restrict__ dyn_tap, // nullable (`has_dyn == 1`: `[8, 4]` lag0..3)
    float* __restrict__ hist3,          // [channels * 3] updated in-place
    float* __restrict__ out,            // [channels]
    const float* __restrict__ res_add,  // nullable (`do_res_add == 1`: `out += res_add`)
    int32_t channels,
    int32_t has_dyn,
    int32_t do_res_add) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    const float x = cur[c];
    float* h = hist3 + c * 3;
    const float h0 = h[0];
    const float h1 = h[1];
    const float h2 = h[2];
    const float4 w = w_conv4[c];

    float conv_o = w.x * h0 + w.y * h1 + w.z * h2 + w.w * x;
    float inv_scale = 0.4472135954999579f; // 1 / sqrt(5.0)
    if (has_dyn) {
        int head = c >> 7; // channels = 1024, 8 heads -> 128 ch/head
        float4 d = dyn_tap[head]; // lag 0=x, 1=h2, 2=h1, 3=h0
        conv_o += d.x * x + d.y * h2 + d.z * h1 + d.w * h0;
        inv_scale = 0.3333333333333333f; // 1 / sqrt(9.0)
    }

    h[0] = h1;
    h[1] = h2;
    h[2] = x;

    float y = (x + conv_o) * inv_scale;
    if (do_res_add) {
        y += res_add[c];
    }
    out[c] = y;
}

// ----------------------------------------------------------------------------
// 7. Q/R/K Clip + V Head-RMSNorm + Log-Scaling (`tau`) + FP16 KV Cache Write
//    Grid: `40` warps (`5` CTAs × `256` threads = `40` warps):
//      - Warps `0..23` : Q heads `0..23` (`clip` `22.627417` + `rel` `clip` `8.0` + `tau`)
//      - Warps `24..31`: K heads `0..7`  (`clip` `22.627417` -> write `k_cache_fp16`)
//      - Warps `32..39`: V heads `0..7`  (`rmsnorm` no-gamma -> write `v_cache_fp16`)
// ----------------------------------------------------------------------------
extern "C" __global__ void ink2_prep_qkv_and_cache(
    float* __restrict__ q_buf,                 // [24 * 128]
    float* __restrict__ r_buf,                 // [24 * 16]
    const float* __restrict__ k_in,            // [8 * 128]
    const float* __restrict__ v_in,            // [8 * 128]
    const uint32_t* __restrict__ q_clip_bf16,  // [24 * 128 / 2]
    const uint32_t* __restrict__ r_clip_bf16,  // [24 * 16 / 2]
    uint16_t* __restrict__ k_cache_fp16,       // [kv_cap, 8 * 128]
    uint16_t* __restrict__ v_cache_fp16,       // [kv_cap, 8 * 128]
    const int32_t* __restrict__ pos_ptr,
    int32_t kv_cap,
    int32_t is_global) {
    const int g_warp = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int pos = *pos_ptr;
    const int cache_row = is_global ? pos : (pos % kv_cap);

    float tau = 1.0f;
    if (is_global) {
        float ratio = static_cast<float>(pos + 1) * (1.0f / 8192.0f);
        if (ratio > 1.0f) {
            tau = 1.0f + 0.1f * ptx_ln(ratio);
        }
    }

    if (g_warp < 24) {
        // Q head (`128` dims = `4` per lane) + R head (`16` dims)
        const int h = g_warp;
        float* qh = q_buf + h * 128;
        const uint32_t* qc = q_clip_bf16 + h * 64;
        float qv[4];
        float q_sq = 0.0f;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int pair_idx = lane * 2 + j;
            uint32_t cp = qc[pair_idx];
            float s0 = bf16_to_f32(static_cast<uint16_t>(cp & 0xFFFFu));
            float s1 = bf16_to_f32(static_cast<uint16_t>(cp >> 16));
            qv[j * 2] = qh[pair_idx * 2] * s0;
            qv[j * 2 + 1] = qh[pair_idx * 2 + 1] * s1;
            q_sq += qv[j * 2] * qv[j * 2] + qv[j * 2 + 1] * qv[j * 2 + 1];
        }
        q_sq = warp_sum(q_sq);
        float q_norm = ptx_sqrt(q_sq) + 1e-6f;
        float q_f = 22.627416997969522f * ptx_rcp(q_norm);
        if (q_f > 1.0f) q_f = 1.0f;
        q_f *= tau;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            qh[lane * 4 + i] = qv[i] * q_f;
        }

        // Relative query `r_h` (`16` dims -> lanes `0..7` hold 2 dims each)
        float* rh = r_buf + h * 16;
        const uint32_t* rc = r_clip_bf16 + h * 8;
        float rv0 = 0.0f, rv1 = 0.0f;
        if (lane < 8) {
            uint32_t rp = rc[lane];
            rv0 = rh[lane * 2] * bf16_to_f32(static_cast<uint16_t>(rp & 0xFFFFu));
            rv1 = rh[lane * 2 + 1] * bf16_to_f32(static_cast<uint16_t>(rp >> 16));
        }
        float r_sq = warp_sum(rv0 * rv0 + rv1 * rv1);
        float r_f = 8.0f * ptx_rcp(ptx_sqrt(r_sq) + 1e-6f);
        if (r_f > 1.0f) r_f = 1.0f;
        r_f *= tau;
        if (lane < 8) {
            rh[lane * 2] = rv0 * r_f;
            rh[lane * 2 + 1] = rv1 * r_f;
        }
    } else if (g_warp < 32) {
        // K head (`128` dims = `4` per lane)
        const int kh = g_warp - 24;
        const float* ks = k_in + kh * 128;
        float kv[4];
        float k_sq = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            kv[i] = ks[lane * 4 + i];
            k_sq += kv[i] * kv[i];
        }
        k_sq = warp_sum(k_sq);
        float k_f = 22.627416997969522f * ptx_rcp(ptx_sqrt(k_sq) + 1e-6f);
        if (k_f > 1.0f) k_f = 1.0f;
        uint16_t* dst = k_cache_fp16 + static_cast<size_t>(cache_row) * 1024u + kh * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            dst[lane * 4 + i] = f32_to_fp16_bits(kv[i] * k_f);
        }
    } else if (g_warp < 40) {
        // V head (`128` dims = `4` per lane, no-gamma RMSNorm)
        const int vh = g_warp - 32;
        const float* vs = v_in + vh * 128;
        float vv[4];
        float v_sq = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            vv[i] = vs[lane * 4 + i];
            v_sq += vv[i] * vv[i];
        }
        v_sq = warp_sum(v_sq);
        float v_inv = ptx_rsqrt(v_sq * (1.0f / 128.0f) + 1e-6f);
        uint16_t* dst = v_cache_fp16 + static_cast<size_t>(cache_row) * 1024u + vh * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            dst[lane * 4 + i] = f32_to_fp16_bits(vv[i] * v_inv);
        }
    }
}

// ----------------------------------------------------------------------------
// 8. GQA Attention (`24` Q-heads = `24` CTAs, `128` threads = `4` warps/CTA)
//    Online Softmax initialized with `m = sinks[h], l = 1.0f, o = 0.0f` +
//    1024-extent relative position bias + sigmoid `gate_attn`!
// ----------------------------------------------------------------------------
extern "C" __global__ void ink2_gqa_rel_sink_attn(
    const float* __restrict__ q_buf,            // [24, 128]
    const float* __restrict__ r_buf,            // [24, 16]
    const float* __restrict__ gate_buf,         // [24, 128]
    const float* __restrict__ sinks_buf,        // [24]
    const uint32_t* __restrict__ rel_proj_bf16, // [16, 1024 / 2 = 512]
    const uint16_t* __restrict__ k_cache_fp16,  // [kv_cap, 8, 128]
    const uint16_t* __restrict__ v_cache_fp16,  // [kv_cap, 8, 128]
    const int32_t* __restrict__ pos_ptr,
    int32_t kv_cap,
    int32_t is_global,
    float* __restrict__ attn_out) {             // [24, 128]
    __shared__ float smem_rel[kExtent];
    __shared__ float smem_q[kHd];
    __shared__ float smem_r[kRel];
    __shared__ float smem_m[4];
    __shared__ float smem_l[4];
    __shared__ float smem_o[4][kHd];

    const int h = blockIdx.x; // 0..23
    const int kvh = h / 3;
    const int tid = threadIdx.x; // 0..127
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int pos = *pos_ptr;
    const int n = pos + 1;
    const int first = is_global ? 0 : ((n > kExtent) ? (n - kExtent) : 0);

    smem_q[tid] = q_buf[h * kHd + tid];
    if (tid < kRel) {
        smem_r[tid] = r_buf[h * kRel + tid];
    }
    __syncthreads();

    // Precompute `smem_rel[0..1024]` (`128` threads compute `8` distances each = `4` `u32` pairs)
    const int max_dist = (n < kExtent) ? n : kExtent;
    for (int pair_d = tid; pair_d < (max_dist + 1) / 2; pair_d += 128) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        #pragma unroll
        for (int i = 0; i < kRel; ++i) {
            uint32_t p = rel_proj_bf16[i * 512 + pair_d];
            float r_val = smem_r[i];
            acc0 += r_val * bf16_to_f32(static_cast<uint16_t>(p & 0xFFFFu));
            acc1 += r_val * bf16_to_f32(static_cast<uint16_t>(p >> 16));
        }
        smem_rel[pair_d * 2] = acc0;
        if (pair_d * 2 + 1 < kExtent) {
            smem_rel[pair_d * 2 + 1] = acc1;
        }
    }
    __syncthreads();

    // Split `j in [first, n)` across the `4` warps; each lane holds `4` dims of `q` & `acc_o`!
    const float q0 = smem_q[lane * 4];
    const float q1 = smem_q[lane * 4 + 1];
    const float q2 = smem_q[lane * 4 + 2];
    const float q3 = smem_q[lane * 4 + 3];
    const float scale = 0.08838834764831845f; // 1 / sqrt(128)

    float m_w = (warp_id == 0) ? sinks_buf[h] : -1e30f;
    float l_w = (warp_id == 0) ? 1.0f : 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    for (int j = first + warp_id; j < n; j += 4) {
        const int slot_row = is_global ? j : (j % kv_cap);
        const uint64_t* kp = reinterpret_cast<const uint64_t*>(
            k_cache_fp16 + static_cast<size_t>(slot_row) * 1024u + kvh * 128);
        uint64_t kw = kp[lane];
        float k0 = fp16_bits_to_f32(static_cast<uint16_t>(kw & 0xFFFFu));
        float k1 = fp16_bits_to_f32(static_cast<uint16_t>((kw >> 16) & 0xFFFFu));
        float k2 = fp16_bits_to_f32(static_cast<uint16_t>((kw >> 32) & 0xFFFFu));
        float k3 = fp16_bits_to_f32(static_cast<uint16_t>((kw >> 48) & 0xFFFFu));

        float dot = warp_sum(q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3) * scale;
        int d = pos - j;
        if (d < kExtent) {
            dot += smem_rel[d];
        }

        float m_new = (dot > m_w) ? dot : m_w;
        float alpha = ptx_exp(m_w - m_new);
        float p = ptx_exp(dot - m_new);
        l_w = l_w * alpha + p;
        m_w = m_new;

        const uint64_t* vp = reinterpret_cast<const uint64_t*>(
            v_cache_fp16 + static_cast<size_t>(slot_row) * 1024u + kvh * 128);
        uint64_t vw = vp[lane];
        o0 = o0 * alpha + p * fp16_bits_to_f32(static_cast<uint16_t>(vw & 0xFFFFu));
        o1 = o1 * alpha + p * fp16_bits_to_f32(static_cast<uint16_t>((vw >> 16) & 0xFFFFu));
        o2 = o2 * alpha + p * fp16_bits_to_f32(static_cast<uint16_t>((vw >> 32) & 0xFFFFu));
        o3 = o3 * alpha + p * fp16_bits_to_f32(static_cast<uint16_t>((vw >> 48) & 0xFFFFu));
    }

    if (lane == 0) {
        smem_m[warp_id] = m_w;
        smem_l[warp_id] = l_w;
    }
    smem_o[warp_id][lane * 4] = o0;
    smem_o[warp_id][lane * 4 + 1] = o1;
    smem_o[warp_id][lane * 4 + 2] = o2;
    smem_o[warp_id][lane * 4 + 3] = o3;
    __syncthreads();

    float m_max = smem_m[0];
    #pragma unroll
    for (int w = 1; w < 4; ++w) {
        if (smem_m[w] > m_max) m_max = smem_m[w];
    }
    float denom = 0.0f;
    float sum_d = 0.0f;
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        float s = ptx_exp(smem_m[w] - m_max);
        denom += smem_l[w] * s;
        sum_d += smem_o[w][tid] * s;
    }
    float g = ptx_sigmoid(gate_buf[h * kHd + tid]);
    attn_out[h * kHd + tid] = (sum_d * ptx_rcp(fmaxf(denom, 1e-20f))) * g;
}

// ----------------------------------------------------------------------------
// 9. Router Sigmoid + Bias Top-8 Selection (`1` CTA, `128` threads)
// ----------------------------------------------------------------------------
extern "C" __global__ void ink2_router_topk8(
    const float* __restrict__ logits128, // [128]
    const float* __restrict__ bias128,   // [128]
    int32_t* __restrict__ sel_ids,       // [8]
    float* __restrict__ sel_wts) {       // [8]
    __shared__ float smem_p[128];
    __shared__ float smem_score[128];

    const int e = threadIdx.x; // 0..127
    float p = ptx_sigmoid(logits128[e]);
    smem_p[e] = p;
    smem_score[e] = p + bias128[e];
    __syncthreads();

    if (e == 0) {
        float sum_p = 1e-20f;
        int chosen[8];
        for (int k = 0; k < 8; ++k) {
            int best_e = 0;
            float best_s = -1e30f;
            for (int i = 0; i < 128; ++i) {
                if (smem_score[i] > best_s) {
                    best_s = smem_score[i];
                    best_e = i;
                }
            }
            chosen[k] = best_e;
            sum_p += smem_p[best_e];
            smem_score[best_e] = -1e30f;
        }
        float inv = 8.0f * ptx_rcp(sum_p);
        for (int k = 0; k < 8; ++k) {
            sel_ids[k] = chosen[k];
            sel_wts[k] = smem_p[chosen[k]] * inv;
        }
    }
}

// ----------------------------------------------------------------------------
// 10. Full-GPU NVFP4 (`layers 0..19`) MoE Up+SITU & Down Kernels (`0` scalar LDG8!)
//     - `ink2_moe_nvfp4_up_situ`: Grid `(1152 / 4 = 288, 8)`, `256` threads (`8` warps)
//       4 warp-pairs (`2` warps = `gate` + `up`) compute 4 `act[k, m]` outputs per CTA!
//     - `ink2_moe_nvfp4_down`   : Grid `2304 / 8 = 288`, `256` threads (`8` warps)
//       Each warp computes row `d` summed over all `k = 0..8` selected experts!
// ----------------------------------------------------------------------------
__device__ __forceinline__ float warp_dot_nvfp4_row(
    const uint2* __restrict__ row_packed_u64,  // [n_blocks16] (8 bytes = 16 nibbles/blk)
    const uint32_t* __restrict__ row_scale_u32,// [n_blocks16 / 4]
    const float4* __restrict__ x4,             // [cols / 4]
    int n_blocks16,
    int lane) {
    float acc = 0.0f;
    for (int blk = lane; blk < n_blocks16; blk += 32) {
        uint2 pk = row_packed_u64[blk];
        uint32_t sc4 = row_scale_u32[blk >> 2];
        uint32_t sc_byte = (sc4 >> ((blk & 3) * 8)) & 0xFFu;
        float sc = e4m3_byte_to_f32(sc_byte) * 0.5f; // `* 0.5f` cancels `2x` E2M1 LUT!

        uint32_t words[2] = {pk.x, pk.y};
        float blk_dot = 0.0f;
        #pragma unroll
        for (int u = 0; u < 2; ++u) {
            uint32_t raw = words[u];
            uint32_t lo4 = decode_4x_e2m1_nibbles16_to_2x_s8(raw & 0xFFFFu);
            uint32_t hi4 = decode_4x_e2m1_nibbles16_to_2x_s8(raw >> 16);
            float4 xa = x4[blk * 4 + u * 2];
            float4 xb = x4[blk * 4 + u * 2 + 1];

            blk_dot += static_cast<float>(static_cast<int8_t>(lo4 & 0xFFu)) * xa.x;
            blk_dot += static_cast<float>(static_cast<int8_t>((lo4 >> 8) & 0xFFu)) * xa.y;
            blk_dot += static_cast<float>(static_cast<int8_t>((lo4 >> 16) & 0xFFu)) * xa.z;
            blk_dot += static_cast<float>(static_cast<int8_t>((lo4 >> 24) & 0xFFu)) * xa.w;
            blk_dot += static_cast<float>(static_cast<int8_t>(hi4 & 0xFFu)) * xb.x;
            blk_dot += static_cast<float>(static_cast<int8_t>((hi4 >> 8) & 0xFFu)) * xb.y;
            blk_dot += static_cast<float>(static_cast<int8_t>((hi4 >> 16) & 0xFFu)) * xb.z;
            blk_dot += static_cast<float>(static_cast<int8_t>((hi4 >> 24) & 0xFFu)) * xb.w;
        }
        acc += blk_dot * sc;
    }
    return warp_sum(acc);
}

extern "C" __global__ void ink2_moe_nvfp4_up_situ(
    const uint2* __restrict__ w13_u64,     // [128, 2304, 144]
    const uint32_t* __restrict__ s13_u32,  // [128, 2304, 36]
    const float* __restrict__ a13,         // [128]
    const float4* __restrict__ x4,         // [2304 / 4]
    const int32_t* __restrict__ sel_ids,   // [8]
    float* __restrict__ act_out) {         // [8, 1152]
    __shared__ float smem_gu[8];
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int k = blockIdx.y; // 0..7
    const int m = blockIdx.x * 4 + (warp_id >> 1); // 0..1151
    const int is_up = warp_id & 1;
    const int r = m * 2 + is_up;
    const int ex = sel_ids[k];

    const size_t row_idx = static_cast<size_t>(ex) * (2u * kFf) + r;
    float dot = warp_dot_nvfp4_row(
        w13_u64 + row_idx * 144u,
        s13_u32 + row_idx * 36u,
        x4,
        144,
        lane);
    if (lane == 0) {
        smem_gu[warp_id] = dot * (a13[ex] * (1.0f / 2688.0f));
    }
    __syncthreads();
    if (lane == 0 && is_up == 0) {
        float g = smem_gu[warp_id];
        float u = smem_gu[warp_id + 1];
        float situ = 4.0f * ptx_tanh(g * 0.25f) * ptx_sigmoid(g) * 25.0f * ptx_tanh(u * 0.04f);
        act_out[k * kFf + m] = situ;
    }
}

extern "C" __global__ void ink2_moe_nvfp4_down(
    const uint2* __restrict__ w2_u64,      // [128, 2304, 72]
    const uint32_t* __restrict__ s2_u32,   // [128, 2304, 18]
    const float* __restrict__ a2,          // [128]
    const float4* __restrict__ act4,       // [8, 1152 / 4 = 288]
    const int32_t* __restrict__ sel_ids,   // [8]
    const float* __restrict__ sel_wts,     // [8]
    float* __restrict__ moe_out) {         // [2304]
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp_id;
    if (row >= kD) return;

    float sum_y = 0.0f;
    for (int k = 0; k < kTopK; ++k) {
        const int ex = sel_ids[k];
        const size_t r_idx = static_cast<size_t>(ex) * kD + row;
        float dot = warp_dot_nvfp4_row(
            w2_u64 + r_idx * 72u,
            s2_u32 + r_idx * 18u,
            act4 + k * 288,
            72,
            lane);
        sum_y += dot * (a2[ex] * (1.0f / 2688.0f)) * sel_wts[k];
    }
    if (lane == 0) {
        moe_out[row] = sum_y;
    }
}

// ----------------------------------------------------------------------------
// 11. Full-GPU MXFP8 (`layers 20..23`) MoE Up+SITU & Down Kernels (`0` CPU sync!)
// ----------------------------------------------------------------------------
__device__ __forceinline__ float warp_dot_mxfp8_row(
    const uint4* __restrict__ row_fp8_u128,     // [n_blocks32 * 2] (32 E4M3 bytes = 2x uint4)
    const uint32_t* __restrict__ row_e8m0_u32,  // [n_blocks32 / 4]
    const float4* __restrict__ x4,              // [cols / 4]
    int n_blocks32,
    int lane) {
    float acc = 0.0f;
    for (int blk = lane; blk < n_blocks32; blk += 32) {
        uint32_t s4 = row_e8m0_u32[blk >> 2];
        uint32_t s_e8m0 = (s4 >> ((blk & 3) * 8)) & 0xFFu;
        float exp_scale = __uint_as_float(s_e8m0 << 23);

        uint4 p0 = row_fp8_u128[blk * 2];
        uint4 p1 = row_fp8_u128[blk * 2 + 1];
        uint32_t words[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
        float blk_sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            uint32_t packed = words[w];
            float4 xv = x4[blk * 8 + w];
            blk_sum += e4m3_byte_to_f32(packed & 0xFFu) * xv.x;
            blk_sum += e4m3_byte_to_f32((packed >> 8) & 0xFFu) * xv.y;
            blk_sum += e4m3_byte_to_f32((packed >> 16) & 0xFFu) * xv.z;
            blk_sum += e4m3_byte_to_f32((packed >> 24) & 0xFFu) * xv.w;
        }
        acc += blk_sum * exp_scale;
    }
    return warp_sum(acc);
}

extern "C" __global__ void ink2_moe_mxfp8_up_situ(
    const uint4* __restrict__ w13_u128,    // [128, 2304, 144]
    const uint32_t* __restrict__ s13_u32,  // [128, 2304, 18]
    const float4* __restrict__ x4,         // [2304 / 4]
    const int32_t* __restrict__ sel_ids,   // [8]
    float* __restrict__ act_out) {         // [8, 1152]
    __shared__ float smem_gu[8];
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int k = blockIdx.y;
    const int m = blockIdx.x * 4 + (warp_id >> 1);
    const int is_up = warp_id & 1;
    const int r = m * 2 + is_up;
    const int ex = sel_ids[k];

    const size_t row_idx = static_cast<size_t>(ex) * (2u * kFf) + r;
    float dot = warp_dot_mxfp8_row(
        w13_u128 + row_idx * 144u,
        s13_u32 + row_idx * 18u,
        x4,
        72,
        lane);
    if (lane == 0) {
        smem_gu[warp_id] = dot;
    }
    __syncthreads();
    if (lane == 0 && is_up == 0) {
        float g = smem_gu[warp_id];
        float u = smem_gu[warp_id + 1];
        float situ = 4.0f * ptx_tanh(g * 0.25f) * ptx_sigmoid(g) * 25.0f * ptx_tanh(u * 0.04f);
        act_out[k * kFf + m] = situ;
    }
}

extern "C" __global__ void ink2_moe_mxfp8_down(
    const uint4* __restrict__ w2_u128,     // [128, 2304, 72]
    const uint32_t* __restrict__ s2_u32,   // [128, 2304, 9]
    const float4* __restrict__ act4,       // [8, 288]
    const int32_t* __restrict__ sel_ids,   // [8]
    const float* __restrict__ sel_wts,     // [8]
    float* __restrict__ moe_out) {         // [2304]
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp_id;
    if (row >= kD) return;

    float sum_y = 0.0f;
    for (int k = 0; k < kTopK; ++k) {
        const int ex = sel_ids[k];
        const size_t r_idx = static_cast<size_t>(ex) * kD + row;
        float dot = warp_dot_mxfp8_row(
            w2_u128 + r_idx * 72u,
            s2_u32 + r_idx * 9u,
            act4 + k * 288,
            36,
            lane);
        sum_y += dot * sel_wts[k];
    }
    if (lane == 0) {
        moe_out[row] = sum_y;
    }
}

// ----------------------------------------------------------------------------
// 12. Utility Device Copy & Greedy ArgMax (`[VOCAB = 200058]`) + Increment `pos`
// ----------------------------------------------------------------------------
extern "C" __global__ void ink2_copy_d(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

extern "C" __global__ void ink2_argmax_and_inc_pos(
    const float* __restrict__ logits,
    int32_t* __restrict__ out_tok,
    int32_t* __restrict__ pos_ptr) {
    __shared__ float smem_val[8];
    __shared__ int smem_idx[8];

    const int tid = threadIdx.x; // 256
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    float best_v = -1e30f;
    int best_i = 0;
    for (int i = tid; i < kVocab; i += 256) {
        float v = logits[i];
        if (v > best_v) {
            best_v = v;
            best_i = i;
        }
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        float ov = __shfl_xor_sync(0xFFFFFFFFu, best_v, mask);
        int oi = __shfl_xor_sync(0xFFFFFFFFu, best_i, mask);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }
    if (lane == 0) {
        smem_val[warp_id] = best_v;
        smem_idx[warp_id] = best_i;
    }
    __syncthreads();
    if (tid == 0) {
        float fv = smem_val[0];
        int fi = smem_idx[0];
        for (int w = 1; w < 8; ++w) {
            if (smem_val[w] > fv || (smem_val[w] == fv && smem_idx[w] < fi)) {
                fv = smem_val[w];
                fi = smem_idx[w];
            }
        }
        *out_tok = fi;
        *pos_ptr += 1;
    }
}
