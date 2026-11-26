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
[[maybe_unused]] static constexpr int kHead = 24;
[[maybe_unused]] static constexpr int kKv = 8;
static constexpr int kHd = 128;
static constexpr int kRel = 16;
static constexpr int kExtent = 1024;
[[maybe_unused]] static constexpr int kExp = 128;
static constexpr int kTopK = 8;
static constexpr int kFf = 1152;
[[maybe_unused]] static constexpr int kRank = 64;
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
    // Byte i selects pos[i] or neg[i]. Each nibble's sign becomes bit 2
    // of its permutation selector; the low two bits identify the byte.
    const uint32_t select_sign = ((nibbles_lo16 >> 1) & 0x4444u) | 0x3210u;
    return __byte_perm(pos, neg, select_sign);
}

} // namespace leansonic::inkling2

using namespace leansonic::inkling2;

// ----------------------------------------------------------------------------
// 1. Token Embedding Lookup (from BF16 row) + RMSNorm (`[D = 2304]`)
// ----------------------------------------------------------------------------
extern "C" __global__ void ink2_embed_rmsnorm(
    const uint32_t* __restrict__ embed_bf16_pairs, // [vocab_padded, D / 2]
    const uint32_t* __restrict__ norm_bf16_pairs,  // [D / 2]
    const int32_t* __restrict__ token_buf_ptr,
    const int32_t* __restrict__ pos_idx_ptr,       // used when `use_pos_idx == 1`
    int32_t use_pos_idx,
    float* __restrict__ out_x) {
    __shared__ float smem_red[8];
    const int tid = threadIdx.x; // 256 threads
    const int32_t token_id = use_pos_idx ? token_buf_ptr[*pos_idx_ptr] : token_buf_ptr[0];
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

// Fused Batched RMSNorm (`[count, 2304]` with BF16 gamma) + Planar `Q8_64` Quantizer!
// Grid: `(count, 1)`, Block: `256` threads (`8` warps; `1` warp quantizes `1` 64-elem block!).
extern "C" __global__ void ink2_rmsnorm_quant_q8_64(
    const float* __restrict__ inp_base,            // [count, 2304]
    const uint32_t* __restrict__ gamma_bf16_pairs, // [count, 1152]
    uint32_t* __restrict__ out_q8_u32,             // [count, 576]
    float* __restrict__ out_sc64) {                // [count, 36]
    __shared__ float smem_red[8];
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const float* inp = inp_base + static_cast<size_t>(b) * kD;
    const uint32_t* gamma = gamma_bf16_pairs + static_cast<size_t>(b) * (kD / 2);
    uint32_t* dst_q8 = out_q8_u32 + static_cast<size_t>(b) * (kD / 4);
    float* dst_sc = out_sc64 + static_cast<size_t>(b) * (kD / 64);

    float sq = 0.0f;
    for (int i = tid; i < kD; i += 256) {
        float v = inp[i];
        sq += v * v;
    }
    sq = warp_sum(sq);
    if (lane == 0) smem_red[warp_id] = sq;
    __syncthreads();
    if (tid < 32) {
        float bsum = (tid < 8) ? smem_red[tid] : 0.0f;
        bsum = warp_sum(bsum);
        if (tid == 0) smem_red[0] = ptx_rsqrt(bsum * (1.0f / static_cast<float>(kD)) + 1e-6f);
    }
    __syncthreads();
    const float k = smem_red[0];

    // Each warp (`32` lanes) normalizes & quantizes one `64`-element block (`2` elems/lane)!
    for (int blk = warp_id; blk < (kD / 64); blk += 8) {
        uint32_t gw = gamma[blk * 32 + lane];
        int base = blk * 64 + lane * 2;
        float y0 = inp[base] * k * bf16_to_f32(static_cast<uint16_t>(gw & 0xFFFFu));
        float y1 = inp[base + 1] * k * bf16_to_f32(static_cast<uint16_t>(gw >> 16));
        float amax = fmaxf(fabsf(y0), fabsf(y1));
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask));
        }
        float inv = (amax > 0.0f) ? (127.0f * ptx_rcp(amax)) : 0.0f;
        int q0 = __float2int_rn(y0 * inv);
        int q1 = __float2int_rn(y1 * inv);
        if (q0 < -127) q0 = -127;
        if (q0 > 127) q0 = 127;
        if (q1 < -127) q1 = -127;
        if (q1 > 127) q1 = 127;
        uint32_t pair16 = static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(q0)))
            | (static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(q1))) << 8);
        uint32_t pack32 = pair16 | (__shfl_xor_sync(0xFFFFFFFFu, pair16, 1) << 16);
        if ((lane & 1) == 0) {
            dst_q8[blk * 16 + (lane >> 1)] = pack32;
        }
        if (lane == 0) {
            dst_sc[blk] = amax * (1.0f / 127.0f);
        }
    }
}

// ----------------------------------------------------------------------------
// 3. Depthwise Attention-Residual `mix(states, queries)` (`n_states <= 6`, `n_q <= 4`)
//    Fused dual `(sq, dq)` reduction per state + optional `broadcast3`!
// ----------------------------------------------------------------------------
extern "C" __global__ void ink2_depth_mix(
    const float* __restrict__ blocks_buf, // [max_blocks, D]
    int32_t n_blocks,
    const float* __restrict__ cur_opt,    // [D] if `append_cur == 1`
    int32_t append_cur,
    const float* __restrict__ queries,    // [n_q, D]
    float* __restrict__ out_mixed,        // [n_q, D] (or `[3, D]` if `broadcast3 == 1`)
    int32_t broadcast3) {
    __shared__ float smem_sq[8];
    __shared__ float smem_dq[8];
    __shared__ float smem_inv_rms[6];
    __shared__ float smem_scores[6];
    __shared__ float smem_probs[6];

    const int q_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int total_states = n_blocks + append_cur;
    const float* q = queries + static_cast<size_t>(q_idx) * kD;
    const float inv_sqrt_d = 0.020833333333333332f; // 1.0 / 48.0

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
        if (lane == 0) {
            smem_sq[warp_id] = sq;
            smem_dq[warp_id] = dq;
        }
        __syncthreads();
        if (warp_id == 0) {
            float total_sq = (lane < 8) ? smem_sq[lane] : 0.0f;
            float total_dq = (lane < 8) ? smem_dq[lane] : 0.0f;
            total_sq = warp_sum(total_sq);
            total_dq = warp_sum(total_dq);
            if (lane == 0) {
                float inv = ptx_rsqrt(total_sq * (1.0f / static_cast<float>(kD)) + 1e-6f);
                smem_inv_rms[s] = inv;
                smem_scores[s] = total_dq * inv * inv_sqrt_d;
            }
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
        if (broadcast3) {
            dst[kD + i] = acc;
            dst[2 * kD + i] = acc;
        }
    }
}

// ----------------------------------------------------------------------------
// 4. Batched Loop Controller Blend (`n_ctrls = 1` or `3` via `blockIdx.y`!)
//    - `ink2_ctrl_down`: Grid `(64, n_ctrls)`, Block `256` (`2` barriers instead of `4`!)
//    - `ink2_ctrl_apply`: Grid `(9, n_ctrls)`, Block `256`
// ----------------------------------------------------------------------------
extern "C" __global__ void ink2_ctrl_down(
    const float* __restrict__ cur_base,       // [n_ctrls, D]
    const float* __restrict__ msg_base,       // [n_ctrls, D]
    const uint4* __restrict__ down_bf16_u128, // [n_ctrls, 64, 576]
    float* __restrict__ g_out_base) {         // [n_ctrls, 64]
    __shared__ float smem_red_a[8];
    __shared__ float smem_red_b[8];
    __shared__ float inv_ab[2];

    const int r = blockIdx.x;     // 0..63
    const int c_idx = blockIdx.y; // 0..n_ctrls-1
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const float* cur = cur_base + static_cast<size_t>(c_idx) * kD;
    const float* msg = msg_base + static_cast<size_t>(c_idx) * kD;

    float sq_a = 0.0f;
    float sq_b = 0.0f;
    for (int i = tid; i < kD; i += 256) {
        float va = cur[i];
        float vb = msg[i];
        sq_a += va * va;
        sq_b += vb * vb;
    }
    sq_a = warp_sum(sq_a);
    sq_b = warp_sum(sq_b);
    if (lane == 0) {
        smem_red_a[warp_id] = sq_a;
        smem_red_b[warp_id] = sq_b;
    }
    __syncthreads();
    if (warp_id == 0) {
        float sa = (lane < 8) ? smem_red_a[lane] : 0.0f;
        float sb = (lane < 8) ? smem_red_b[lane] : 0.0f;
        sa = warp_sum(sa);
        sb = warp_sum(sb);
        if (lane == 0) {
            inv_ab[0] = ptx_rsqrt(sa * (1.0f / static_cast<float>(kD)) + 1e-6f);
            inv_ab[1] = ptx_rsqrt(sb * (1.0f / static_cast<float>(kD)) + 1e-6f);
        }
    }
    __syncthreads();

    const uint4* w_row = down_bf16_u128 + (static_cast<size_t>(c_idx) * 64u + r) * 576u;
    const float ka = inv_ab[0];
    const float kb = inv_ab[1];
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
    if (lane == 0) smem_red_a[warp_id] = dot_acc;
    __syncthreads();
    if (warp_id == 0) {
        float s = (lane < 8) ? smem_red_a[lane] : 0.0f;
        s = warp_sum(s);
        if (lane == 0) g_out_base[c_idx * 64 + r] = s;
    }
}

extern "C" __global__ void ink2_ctrl_apply(
    float* __restrict__ cur_inout_base,       // [n_ctrls, D]
    const float* __restrict__ msg_base,       // [n_ctrls, D]
    const float* __restrict__ g_vec_base,     // [n_ctrls, 64]
    const uint4* __restrict__ gain_bf16_u128, // [n_ctrls, D, 8]
    const uint4* __restrict__ forget_bf16_u128) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int c_idx = blockIdx.y;
    if (i >= kD) return;
    float* cur_inout = cur_inout_base + static_cast<size_t>(c_idx) * kD;
    const float* msg = msg_base + static_cast<size_t>(c_idx) * kD;
    const float* g_vec = g_vec_base + c_idx * 64;
    const size_t row_idx = static_cast<size_t>(c_idx) * kD + i;

    float gain_acc = 0.0f;
    float forget_acc = 0.0f;
    const uint4* g_row = gain_bf16_u128 + row_idx * 8u;
    const uint4* f_row = forget_bf16_u128 + row_idx * 8u;
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
    float* out,                         // [channels] (may alias `res_add`)
    const float* res_add,               // nullable (`do_res_add == 1`: `out += res_add`)
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
// 8. GQA Attention (`24` Q-heads = `24` CTAs, `256` threads = `8` warps/CTA!)
//    8-way Split-KV Online Softmax (`m = sinks[h], l = 1.0f`) + 1024 rel bias!
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
    __shared__ float smem_m[8];
    __shared__ float smem_l[8];
    __shared__ float smem_o[8][kHd];

    const int h = blockIdx.x; // 0..23
    const int kvh = h / 3;
    const int tid = threadIdx.x; // 0..255
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int pos = *pos_ptr;
    const int n = pos + 1;
    const int first = is_global ? 0 : ((n > kExtent) ? (n - kExtent) : 0);

    if (tid < kHd) {
        smem_q[tid] = q_buf[h * kHd + tid];
    }
    if (tid < kRel) {
        smem_r[tid] = r_buf[h * kRel + tid];
    }
    __syncthreads();

    const int max_dist = (n < kExtent) ? n : kExtent;
    for (int pair_d = tid; pair_d < (max_dist + 1) / 2; pair_d += 256) {
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

    const float q0 = smem_q[lane * 4];
    const float q1 = smem_q[lane * 4 + 1];
    const float q2 = smem_q[lane * 4 + 2];
    const float q3 = smem_q[lane * 4 + 3];
    const float scale = 0.08838834764831845f; // 1 / sqrt(128)

    float m_w = (warp_id == 0) ? sinks_buf[h] : -1e30f;
    float l_w = (warp_id == 0) ? 1.0f : 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    for (int j = first + warp_id; j < n; j += 8) {
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

    if (tid < kHd) {
        float m_max = smem_m[0];
        #pragma unroll
        for (int w = 1; w < 8; ++w) {
            if (smem_m[w] > m_max) m_max = smem_m[w];
        }
        float denom = 0.0f;
        float sum_d = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            float s = ptx_exp(smem_m[w] - m_max);
            denom += smem_l[w] * s;
            sum_d += smem_o[w][tid] * s;
        }
        float g = ptx_sigmoid(gate_buf[h * kHd + tid]);
        attn_out[h * kHd + tid] = (sum_d * ptx_rcp(fmaxf(denom, 1e-20f))) * g;
    }
}

// ----------------------------------------------------------------------------
// 9. Single-Warp Register-Butterfly Router Top-8 Selection (`1` CTA, `32` threads)
// ----------------------------------------------------------------------------
extern "C" __global__ void ink2_router_topk8(
    const float* __restrict__ logits128, // [128]
    const float* __restrict__ bias128,   // [128]
    int32_t* __restrict__ sel_ids,       // [8]
    float* __restrict__ sel_wts) {       // [8]
    __shared__ int smem_ids[8];
    __shared__ float smem_ps[8];

    const int lane = threadIdx.x & 31;
    const int base = lane * 4;
    float p0 = ptx_sigmoid(logits128[base + 0]);
    float p1 = ptx_sigmoid(logits128[base + 1]);
    float p2 = ptx_sigmoid(logits128[base + 2]);
    float p3 = ptx_sigmoid(logits128[base + 3]);
    float s0 = p0 + bias128[base + 0];
    float s1 = p1 + bias128[base + 1];
    float s2 = p2 + bias128[base + 2];
    float s3 = p3 + bias128[base + 3];

    float sum_p = 1e-20f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float best_s = s0;
        int best_e = base + 0;
        float best_p = p0;
        if (s1 > best_s) { best_s = s1; best_e = base + 1; best_p = p1; }
        if (s2 > best_s) { best_s = s2; best_e = base + 2; best_p = p2; }
        if (s3 > best_s) { best_s = s3; best_e = base + 3; best_p = p3; }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            float os = __shfl_xor_sync(0xFFFFFFFFu, best_s, mask);
            int oe = __shfl_xor_sync(0xFFFFFFFFu, best_e, mask);
            float op = __shfl_xor_sync(0xFFFFFFFFu, best_p, mask);
            if (os > best_s || (os == best_s && oe < best_e)) {
                best_s = os;
                best_e = oe;
                best_p = op;
            }
        }
        int win_e = __shfl_sync(0xFFFFFFFFu, best_e, 0);
        float win_p = __shfl_sync(0xFFFFFFFFu, best_p, 0);
        sum_p += win_p;
        if (lane == 0) {
            smem_ids[k] = win_e;
            smem_ps[k] = win_p;
        }
        if (win_e == base + 0) s0 = -1e30f;
        if (win_e == base + 1) s1 = -1e30f;
        if (win_e == base + 2) s2 = -1e30f;
        if (win_e == base + 3) s3 = -1e30f;
    }
    __syncwarp();
    if (lane < 8) {
        float inv = 8.0f * ptx_rcp(sum_p);
        sel_ids[lane] = smem_ids[lane];
        sel_wts[lane] = smem_ps[lane] * inv;
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
// 11. Unit-Stride `512B` Coalesced MXFP8 (`layers 20..23`) MoE (`<= 42` regs!)
//     `Grid (288, 8)` & `Grid 288` (`100%` SM120 occupancy across all SMs + `512B` PCIe burst!)
// ----------------------------------------------------------------------------
__device__ __forceinline__ float warp_dot_mxfp8_u128_row(
    const uint4* __restrict__ row_fp8_u128,     // [n_u128]
    const uint32_t* __restrict__ row_e8m0_u32,  // [n_u128 / 8]
    const float4* __restrict__ x4,              // [n_u128 * 4]
    int n_u128,
    int lane) {
    float acc = 0.0f;
    for (int u16 = lane; u16 < n_u128; u16 += 32) {
        const int blk32 = u16 >> 1;
        uint32_t s4 = row_e8m0_u32[blk32 >> 2];
        uint32_t s_e8m0 = (s4 >> ((blk32 & 3) * 8)) & 0xFFu;
        float exp_scale = __uint_as_float(s_e8m0 << 23);

        uint4 p = row_fp8_u128[u16];
        uint32_t words[4] = {p.x, p.y, p.z, p.w};
        float sub = 0.0f;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            uint32_t packed = words[w];
            float4 xv = x4[u16 * 4 + w];
            sub += e4m3_byte_to_f32(packed & 0xFFu) * xv.x;
            sub += e4m3_byte_to_f32((packed >> 8) & 0xFFu) * xv.y;
            sub += e4m3_byte_to_f32((packed >> 16) & 0xFFu) * xv.z;
            sub += e4m3_byte_to_f32((packed >> 24) & 0xFFu) * xv.w;
        }
        acc += sub * exp_scale;
    }
    return warp_sum(acc);
}

extern "C" __global__ void ink2_moe_mxfp8_up_situ(
    const uint4* __restrict__ w13_u128,    // [128, 2304, 144]
    const uint32_t* __restrict__ s13_u32,  // [128, 2304, 18]
    const float4* __restrict__ x4,         // [576]
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
    float dot = warp_dot_mxfp8_u128_row(
        w13_u128 + row_idx * 144u,
        s13_u32 + row_idx * 18u,
        x4,
        144,
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
        float dot = warp_dot_mxfp8_u128_row(
            w2_u128 + r_idx * 72u,
            s2_u32 + r_idx * 9u,
            act4 + k * 288,
            72,
            lane);
        sum_y += dot * sel_wts[k];
    }
    if (lane == 0) {
        moe_out[row] = sum_y;
    }
}

// ----------------------------------------------------------------------------
// 12. Wave 1 Optimizations (`<= 42` Regs SM120 Full-Occupancy Verified!):
//     Planar `Q8_64` 2-Rows/Warp `__dp4a` GEMV (`38` regs -> `48` warps/SM!) &
//     Wide `uint4` (`16B` `LDG.E.128`) `__dp4a` `NVFP4` MoE (`Grid (288,8)` & `288`!)
// ----------------------------------------------------------------------------

extern "C" __global__ void ink2_quant_q8_64(
    const float* __restrict__ x,
    uint32_t* __restrict__ out_q8_u32, // [cols / 4]
    float* __restrict__ out_sc64) {    // [cols / 64]
    __shared__ float smem_max[2];
    __shared__ float smem_inv;
    const int blk = blockIdx.x;
    const int tid = threadIdx.x; // 0..63
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    float v = x[blk * 64 + tid];
    float amax = fabsf(v);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask));
    }
    if (lane == 0) smem_max[warp_id] = amax;
    __syncthreads();
    if (tid == 0) {
        float m = fmaxf(smem_max[0], smem_max[1]);
        out_sc64[blk] = m * (1.0f / 127.0f);
        smem_inv = (m > 0.0f) ? (127.0f * ptx_rcp(m)) : 0.0f;
    }
    __syncthreads();

    int q = __float2int_rn(v * smem_inv);
    if (q < -127) q = -127;
    if (q > 127) q = 127;
    uint32_t qb = static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(q)));
    uint32_t p4 = qb
        | (__shfl_sync(0xFFFFFFFFu, qb, lane + 1) << 8)
        | (__shfl_sync(0xFFFFFFFFu, qb, lane + 2) << 16)
        | (__shfl_sync(0xFFFFFFFFu, qb, lane + 3) << 24);
    if ((lane & 3) == 0) {
        out_q8_u32[blk * 16 + warp_id * 8 + (lane >> 2)] = p4;
    }
}

extern "C" __global__ void ink2_quant_q8_16(
    const float* __restrict__ x,       // [n_vecs, cols]
    uint32_t* __restrict__ out_q8_u32, // [n_vecs, cols / 4]
    float* __restrict__ out_sc16,      // [n_vecs, cols / 16]
    int32_t cols) {
    const int blk16 = blockIdx.x;
    const int vec_id = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int off = vec_id * cols + blk16 * 16;

    float v = (lane < 16) ? x[off + lane] : 0.0f;
    float amax = fabsf(v);
    #pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask));
    }
    float sc = amax * (1.0f / 127.0f);
    float inv = (amax > 0.0f) ? (127.0f * ptx_rcp(amax)) : 0.0f;
    int q = __float2int_rn(v * inv);
    if (q < -127) q = -127;
    if (q > 127) q = 127;
    uint32_t qb = static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(q)));
    uint32_t p4 = qb
        | (__shfl_sync(0xFFFFFFFFu, qb, lane + 1) << 8)
        | (__shfl_sync(0xFFFFFFFFu, qb, lane + 2) << 16)
        | (__shfl_sync(0xFFFFFFFFu, qb, lane + 3) << 24);
    if (lane < 16 && (lane & 3) == 0) {
        out_q8_u32[(vec_id * (cols >> 2)) + blk16 * 4 + (lane >> 2)] = p4;
    }
    if (lane == 0) {
        out_sc16[vec_id * (cols >> 4) + blk16] = sc;
    }
}

// Planar `Q8_64` SMEM-Staged 2-Rows-Per-Warp `__dp4a` GEMV (`16` rows/CTA, `38 <= 42` regs -> `48` warps/SM!)
extern "C" __global__ void ink2_q8_dp4a_gemv(
    const uint4* __restrict__ w_q8_u128,   // [rows, cols / 16]
    const float* __restrict__ w_sc64,      // [rows, cols / 64]
    const uint4* __restrict__ x_q8_u128,   // [cols / 16]
    const float* __restrict__ x_sc64,      // [cols / 64]
    float* __restrict__ y,                 // [rows]
    int32_t rows,
    int32_t cols,
    float softcap) {
    __shared__ uint4 smem_x[192];  // max `3072 / 16 = 192` (`3,072` B)
    __shared__ float smem_xsc[48]; // max `3072 / 64 = 48`  (`192` B)

    const int tid = threadIdx.x; // 256
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int cols16 = cols >> 4;
    const int cols64 = cols >> 6;

    for (int i = tid; i < cols16; i += 256) {
        smem_x[i] = x_q8_u128[i];
    }
    if (tid < cols64) {
        smem_xsc[tid] = x_sc64[tid];
    }
    __syncthreads();

    const int row0 = (blockIdx.x * 8 + warp_id) * 2;
    if (row0 >= rows) return;
    const bool has_row1 = (row0 + 1 < rows);

    const uint4* w_r0 = w_q8_u128 + static_cast<size_t>(row0) * cols16;
    const uint4* w_r1 = has_row1 ? (w_r0 + cols16) : w_r0;
    const float* s_r0 = w_sc64 + static_cast<size_t>(row0) * cols64;
    const float* s_r1 = has_row1 ? (s_r0 + cols64) : s_r0;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int c16 = lane; c16 < cols16; c16 += 32) {
        uint4 xq = smem_x[c16];
        float xsc = smem_xsc[c16 >> 2];

        uint4 w0 = w_r0[c16];
        int d0 = __dp4a(static_cast<int>(w0.x), static_cast<int>(xq.x), 0);
        d0 = __dp4a(static_cast<int>(w0.y), static_cast<int>(xq.y), d0);
        d0 = __dp4a(static_cast<int>(w0.z), static_cast<int>(xq.z), d0);
        d0 = __dp4a(static_cast<int>(w0.w), static_cast<int>(xq.w), d0);
        acc0 += static_cast<float>(d0) * (s_r0[c16 >> 2] * xsc);

        if (has_row1) {
            uint4 w1 = w_r1[c16];
            int d1 = __dp4a(static_cast<int>(w1.x), static_cast<int>(xq.x), 0);
            d1 = __dp4a(static_cast<int>(w1.y), static_cast<int>(xq.y), d1);
            d1 = __dp4a(static_cast<int>(w1.z), static_cast<int>(xq.z), d1);
            d1 = __dp4a(static_cast<int>(w1.w), static_cast<int>(xq.w), d1);
            acc1 += static_cast<float>(d1) * (s_r1[c16 >> 2] * xsc);
        }
    }

    acc0 = warp_sum(acc0);
    acc1 = warp_sum(acc1);
    if (lane == 0) {
        if (softcap > 0.0f) {
            float inv_sc = ptx_rcp(softcap);
            acc0 = softcap * ptx_tanh(acc0 * inv_sc);
            acc1 = softcap * ptx_tanh(acc1 * inv_sc);
        }
        y[row0] = acc0;
        if (has_row1) y[row0 + 1] = acc1;
    }
}

__device__ __forceinline__ float warp_dot_nvfp4_dp4a_u128_row(
    const uint4* __restrict__ row_u128,         // [n_u128] (`16B` = `32` nibbles = `2` blocks of 16!)
    const uint32_t* __restrict__ row_scale_u32, // [n_u128 / 2]
    const uint4* __restrict__ smem_xq16,        // [n_u128 * 2]
    const float* __restrict__ smem_xsc16,       // [n_u128 * 2]
    int n_u128,
    int lane) {
    float acc = 0.0f;
    for (int u16 = lane; u16 < n_u128; u16 += 32) {
        uint4 pk16 = row_u128[u16];
        uint32_t sc4 = row_scale_u32[u16 >> 1];
        int sh = (u16 & 1) * 16;
        float s0 = e4m3_byte_to_f32((sc4 >> sh) & 0xFFu) * 0.5f * smem_xsc16[u16 * 2];
        float s1 = e4m3_byte_to_f32((sc4 >> (sh + 8)) & 0xFFu) * 0.5f * smem_xsc16[u16 * 2 + 1];

        uint4 xq0 = smem_xq16[u16 * 2];
        uint4 xq1 = smem_xq16[u16 * 2 + 1];
        int d0 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.x & 0xFFFFu)), static_cast<int>(xq0.x), 0);
        d0 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.x >> 16)), static_cast<int>(xq0.y), d0);
        d0 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.y & 0xFFFFu)), static_cast<int>(xq0.z), d0);
        d0 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.y >> 16)), static_cast<int>(xq0.w), d0);
        int d1 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.z & 0xFFFFu)), static_cast<int>(xq1.x), 0);
        d1 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.z >> 16)), static_cast<int>(xq1.y), d1);
        d1 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.w & 0xFFFFu)), static_cast<int>(xq1.z), d1);
        d1 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.w >> 16)), static_cast<int>(xq1.w), d1);
        acc += static_cast<float>(d0) * s0 + static_cast<float>(d1) * s1;
    }
    return warp_sum(acc);
}

extern "C" __global__ void ink2_moe_nvfp4_dp4a_up_situ(
    const uint4* __restrict__ w13_u128,    // [128, 2304, 72]
    const uint32_t* __restrict__ s13_u32,  // [128, 2304, 36]
    const float* __restrict__ a13,         // [128]
    const uint4* __restrict__ x_q8_u128,   // [144]
    const float* __restrict__ x_sc16,      // [144]
    const int32_t* __restrict__ sel_ids,   // [8]
    float* __restrict__ act_out) {         // [8, 1152]
    __shared__ uint4 smem_xq[144];
    __shared__ float smem_xsc[144];
    __shared__ float smem_gu[8];

    const int tid = threadIdx.x;
    if (tid < 144) {
        smem_xq[tid] = x_q8_u128[tid];
        smem_xsc[tid] = x_sc16[tid];
    }
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int k = blockIdx.y;
    const int m = blockIdx.x * 4 + (warp_id >> 1);
    const int is_up = warp_id & 1;
    const int r = m * 2 + is_up;
    const int ex = sel_ids[k];

    const size_t row_idx = static_cast<size_t>(ex) * (2u * kFf) + r;
    float dot = warp_dot_nvfp4_dp4a_u128_row(
        w13_u128 + row_idx * 72u,
        s13_u32 + row_idx * 36u,
        smem_xq,
        smem_xsc,
        72,
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

extern "C" __global__ void ink2_moe_nvfp4_dp4a_down(
    const uint4* __restrict__ w2_u128,     // [128, 2304, 36]
    const uint32_t* __restrict__ s2_u32,   // [128, 2304, 18]
    const float* __restrict__ a2,          // [128]
    const uint4* __restrict__ act_q8_u128, // [8, 72]
    const float* __restrict__ act_sc16,    // [8, 72]
    const int32_t* __restrict__ sel_ids,   // [8]
    const float* __restrict__ sel_wts,     // [8]
    float* __restrict__ moe_out) {         // [2304]
    __shared__ uint4 smem_aq[8 * 72];      // 9,216 B
    __shared__ float smem_asc[8 * 72];     // 2,304 B

    const int tid = threadIdx.x;
    for (int i = tid; i < 8 * 72; i += 256) {
        smem_aq[i] = act_q8_u128[i];
        smem_asc[i] = act_sc16[i];
    }
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + warp_id;
    if (row >= kD) return;

    float sum_y = 0.0f;
    for (int k = 0; k < kTopK; ++k) {
        const int ex = sel_ids[k];
        const size_t r_idx = static_cast<size_t>(ex) * kD + row;
        float dot = warp_dot_nvfp4_dp4a_u128_row(
            w2_u128 + r_idx * 36u,
            s2_u32 + r_idx * 18u,
            smem_aq + k * 72,
            smem_asc + k * 72,
            36,
            lane);
        sum_y += dot * (a2[ex] * (1.0f / 2688.0f)) * sel_wts[k];
    }
    if (lane == 0) {
        moe_out[row] = sum_y;
    }
}

extern "C" __global__ void ink2_inc_pos(int32_t* __restrict__ pos_ptr) {
    if (threadIdx.x == 0) {
        *pos_ptr += 1;
    }
}

// ----------------------------------------------------------------------------
// 13. Utility Device Copy & Greedy ArgMax (`[VOCAB = 200058]`) + Increment `pos`
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

// ============================================================================
// 14. Cycle 1 (`v0.2.0`) & Cycle 2 (`v0.3.0`) Unified Multi-Token Batch Suite
//     (`1 <= B <= 64`):
//     - Planar `Q8_64` 2-Rows/Warp x 8-Tokens `__dp4a` GEMM (`ink2_q8_dp4a_gemm_b`)
//     - Register-Persistent Causal `short_conv4` across `0..B` (`0-ULP` exact!)
//     - Collision-Free Ring-Buffer Batched QKV Prep + Online-Softmax GQA (`SafeHybridStatePool`)
//     - Zero-Atomic Register-Resident Grouped-Expert `NVFP4` (`__dp4a`) & `MXFP8` MoE
// ============================================================================

extern "C" __global__ void ink2_embed_rmsnorm_b(
    const uint32_t* __restrict__ embed_bf16_pairs, // [vocab_padded, D / 2]
    const uint32_t* __restrict__ norm_bf16_pairs,  // [D / 2]
    const int32_t* __restrict__ token_buf_ptr,     // [B]
    float* __restrict__ out_x_base) {              // [B, D]
    __shared__ float smem_red[8];
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int32_t token_id = token_buf_ptr[b];
    const uint32_t* row = embed_bf16_pairs + static_cast<size_t>(token_id) * (kD / 2);
    float* out_x = out_x_base + static_cast<size_t>(b) * kD;

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

extern "C" __global__ void ink2_rmsnorm_d_b(
    const float* __restrict__ inp_base,             // [B, D]
    const uint32_t* __restrict__ gamma_bf16_pairs,  // [D / 2]
    float* __restrict__ out_base,                   // [B, D]
    int32_t has_gamma) {
    __shared__ float smem_red[8];
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const float* inp = inp_base + static_cast<size_t>(b) * kD;
    float* out = out_base + static_cast<size_t>(b) * kD;

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

// Grid: `(B_out, count)`, Block: `256` (`b = blockIdx.x`, `c_idx = blockIdx.y`)
extern "C" __global__ void ink2_rmsnorm_quant_q8_64_b(
    const float* __restrict__ inp_base,            // [count, B_out, 2304]
    const uint32_t* __restrict__ gamma_bf16_pairs, // [count, 1152]
    uint32_t* __restrict__ out_q8_u32,             // [count, B_out, 576]
    float* __restrict__ out_sc64,                  // [count, B_out, 36]
    int32_t B_out) {
    __shared__ float smem_red[8];
    const int b = blockIdx.x;
    const int c_idx = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const size_t vec_idx = static_cast<size_t>(c_idx) * B_out + b;
    const float* inp = inp_base + vec_idx * kD;
    const uint32_t* gamma = gamma_bf16_pairs + static_cast<size_t>(c_idx) * (kD / 2);
    uint32_t* dst_q8 = out_q8_u32 + vec_idx * (kD / 4);
    float* dst_sc = out_sc64 + vec_idx * (kD / 64);

    float sq = 0.0f;
    for (int i = tid; i < kD; i += 256) {
        float v = inp[i];
        sq += v * v;
    }
    sq = warp_sum(sq);
    if (lane == 0) smem_red[warp_id] = sq;
    __syncthreads();
    if (tid < 32) {
        float bsum = (tid < 8) ? smem_red[tid] : 0.0f;
        bsum = warp_sum(bsum);
        if (tid == 0) smem_red[0] = ptx_rsqrt(bsum * (1.0f / static_cast<float>(kD)) + 1e-6f);
    }
    __syncthreads();
    const float k = smem_red[0];

    for (int blk = warp_id; blk < (kD / 64); blk += 8) {
        uint32_t gw = gamma[blk * 32 + lane];
        int base = blk * 64 + lane * 2;
        float y0 = inp[base] * k * bf16_to_f32(static_cast<uint16_t>(gw & 0xFFFFu));
        float y1 = inp[base + 1] * k * bf16_to_f32(static_cast<uint16_t>(gw >> 16));
        float amax = fmaxf(fabsf(y0), fabsf(y1));
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask));
        }
        float inv = (amax > 0.0f) ? (127.0f * ptx_rcp(amax)) : 0.0f;
        int q0 = __float2int_rn(y0 * inv);
        int q1 = __float2int_rn(y1 * inv);
        if (q0 < -127) q0 = -127;
        if (q0 > 127) q0 = 127;
        if (q1 < -127) q1 = -127;
        if (q1 > 127) q1 = 127;
        uint32_t pair16 = static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(q0)))
            | (static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(q1))) << 8);
        uint32_t pack32 = pair16 | (__shfl_xor_sync(0xFFFFFFFFu, pair16, 1) << 16);
        if ((lane & 1) == 0) {
            dst_q8[blk * 16 + (lane >> 1)] = pack32;
        }
        if (lane == 0) {
            dst_sc[blk] = amax * (1.0f / 127.0f);
        }
    }
}

// Grid: `(cols / 64, B)`, Block: `64`
extern "C" __global__ void ink2_quant_q8_64_b(
    const float* __restrict__ x_base,   // [B, cols]
    uint32_t* __restrict__ out_q8_u32,  // [B, cols / 4]
    float* __restrict__ out_sc64,       // [B, cols / 64]
    int32_t cols) {
    __shared__ float smem_max[2];
    __shared__ float smem_inv;
    const int blk = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const float* x = x_base + static_cast<size_t>(b) * cols;
    float v = x[blk * 64 + tid];
    float amax = fabsf(v);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask));
    }
    if (lane == 0) smem_max[warp_id] = amax;
    __syncthreads();
    if (tid == 0) {
        float m = fmaxf(smem_max[0], smem_max[1]);
        out_sc64[static_cast<size_t>(b) * (cols >> 6) + blk] = m * (1.0f / 127.0f);
        smem_inv = (m > 0.0f) ? (127.0f * ptx_rcp(m)) : 0.0f;
    }
    __syncthreads();

    int q = __float2int_rn(v * smem_inv);
    if (q < -127) q = -127;
    if (q > 127) q = 127;
    uint32_t qb = static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(q)));
    uint32_t p4 = qb
        | (__shfl_sync(0xFFFFFFFFu, qb, lane + 1) << 8)
        | (__shfl_sync(0xFFFFFFFFu, qb, lane + 2) << 16)
        | (__shfl_sync(0xFFFFFFFFu, qb, lane + 3) << 24);
    if ((lane & 3) == 0) {
        out_q8_u32[static_cast<size_t>(b) * (cols >> 2) + blk * 16 + warp_id * 8 + (lane >> 2)] = p4;
    }
}

// Grid: `(n_q, B_out)`, Block: `256`
// Supports `stride_B` + `b_offset` so `LastOnly` (`b_offset = B - 1, B_out = 1`) reads directly!
extern "C" __global__ void ink2_depth_mix_b(
    const float* __restrict__ blocks_buf, // [max_blocks, stride_B, D]
    int32_t n_blocks,
    int32_t stride_B,
    int32_t b_offset,
    int32_t B_out,
    const float* __restrict__ cur_opt,    // [stride_B, D] if `append_cur == 1`
    int32_t append_cur,
    const float* __restrict__ queries,    // [n_q, D]
    float* __restrict__ out_mixed,        // [n_q, B_out, D] (or `[3, B_out, D]` if `broadcast3`)
    int32_t broadcast3) {
    __shared__ float smem_sq[8];
    __shared__ float smem_dq[8];
    __shared__ float smem_inv_rms[6];
    __shared__ float smem_scores[6];
    __shared__ float smem_probs[6];

    const int q_idx = blockIdx.x;
    const int b_out = blockIdx.y;
    const int b_src = b_offset + b_out;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int total_states = n_blocks + append_cur;
    const float* q = queries + static_cast<size_t>(q_idx) * kD;
    const float inv_sqrt_d = 0.020833333333333332f;

    for (int s = 0; s < total_states; ++s) {
        const float* st = (s < n_blocks)
            ? (blocks_buf + (static_cast<size_t>(s) * stride_B + b_src) * kD)
            : (cur_opt + static_cast<size_t>(b_src) * kD);
        float sq = 0.0f;
        float dq = 0.0f;
        for (int i = tid; i < kD; i += 256) {
            float v = st[i];
            sq += v * v;
            dq += v * q[i];
        }
        sq = warp_sum(sq);
        dq = warp_sum(dq);
        if (lane == 0) {
            smem_sq[warp_id] = sq;
            smem_dq[warp_id] = dq;
        }
        __syncthreads();
        if (warp_id == 0) {
            float total_sq = (lane < 8) ? smem_sq[lane] : 0.0f;
            float total_dq = (lane < 8) ? smem_dq[lane] : 0.0f;
            total_sq = warp_sum(total_sq);
            total_dq = warp_sum(total_dq);
            if (lane == 0) {
                float inv = ptx_rsqrt(total_sq * (1.0f / static_cast<float>(kD)) + 1e-6f);
                smem_inv_rms[s] = inv;
                smem_scores[s] = total_dq * inv * inv_sqrt_d;
            }
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

    float* dst = out_mixed + (static_cast<size_t>(q_idx) * B_out + b_out) * kD;
    const size_t plane_stride = static_cast<size_t>(B_out) * kD;
    for (int i = tid; i < kD; i += 256) {
        float acc = 0.0f;
        for (int s = 0; s < total_states; ++s) {
            const float* st = (s < n_blocks)
                ? (blocks_buf + (static_cast<size_t>(s) * stride_B + b_src) * kD)
                : (cur_opt + static_cast<size_t>(b_src) * kD);
            acc += smem_probs[s] * st[i];
        }
        dst[i] = acc;
        if (broadcast3) {
            dst[plane_stride + i] = acc;
            dst[2u * plane_stride + i] = acc;
        }
    }
}

// Grid: `(64, n_ctrls * B)`, Block: `256`
extern "C" __global__ void ink2_ctrl_down_b(
    const float* __restrict__ cur_base,       // [n_ctrls, B, D]
    const float* __restrict__ msg_base,       // [n_ctrls, B, D]
    const uint4* __restrict__ down_bf16_u128, // [n_ctrls, 64, 576]
    float* __restrict__ g_out_base,           // [n_ctrls, B, 64]
    int32_t B) {
    __shared__ float smem_red_a[8];
    __shared__ float smem_red_b[8];
    __shared__ float inv_ab[2];

    const int r = blockIdx.x;
    const int cb = blockIdx.y;
    const int c_idx = cb / B;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const float* cur = cur_base + static_cast<size_t>(cb) * kD;
    const float* msg = msg_base + static_cast<size_t>(cb) * kD;

    float sq_a = 0.0f;
    float sq_b = 0.0f;
    for (int i = tid; i < kD; i += 256) {
        float va = cur[i];
        float vb = msg[i];
        sq_a += va * va;
        sq_b += vb * vb;
    }
    sq_a = warp_sum(sq_a);
    sq_b = warp_sum(sq_b);
    if (lane == 0) {
        smem_red_a[warp_id] = sq_a;
        smem_red_b[warp_id] = sq_b;
    }
    __syncthreads();
    if (warp_id == 0) {
        float sa = (lane < 8) ? smem_red_a[lane] : 0.0f;
        float sb = (lane < 8) ? smem_red_b[lane] : 0.0f;
        sa = warp_sum(sa);
        sb = warp_sum(sb);
        if (lane == 0) {
            inv_ab[0] = ptx_rsqrt(sa * (1.0f / static_cast<float>(kD)) + 1e-6f);
            inv_ab[1] = ptx_rsqrt(sb * (1.0f / static_cast<float>(kD)) + 1e-6f);
        }
    }
    __syncthreads();

    const uint4* w_row = down_bf16_u128 + (static_cast<size_t>(c_idx) * 64u + r) * 576u;
    const float ka = inv_ab[0];
    const float kb = inv_ab[1];
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
    if (lane == 0) smem_red_a[warp_id] = dot_acc;
    __syncthreads();
    if (warp_id == 0) {
        float s = (lane < 8) ? smem_red_a[lane] : 0.0f;
        s = warp_sum(s);
        if (lane == 0) g_out_base[cb * 64 + r] = s;
    }
}

// Grid: `(9, n_ctrls * B)`, Block: `256`
extern "C" __global__ void ink2_ctrl_apply_b(
    float* __restrict__ cur_inout_base,       // [n_ctrls, B, D]
    const float* __restrict__ msg_base,       // [n_ctrls, B, D]
    const float* __restrict__ g_vec_base,     // [n_ctrls, B, 64]
    const uint4* __restrict__ gain_bf16_u128, // [n_ctrls, D, 8]
    const uint4* __restrict__ forget_bf16_u128,
    int32_t B) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int cb = blockIdx.y;
    if (i >= kD) return;
    const int c_idx = cb / B;
    float* cur_inout = cur_inout_base + static_cast<size_t>(cb) * kD;
    const float* msg = msg_base + static_cast<size_t>(cb) * kD;
    const float* g_vec = g_vec_base + cb * 64;
    const size_t row_idx = static_cast<size_t>(c_idx) * kD + i;

    float gain_acc = 0.0f;
    float forget_acc = 0.0f;
    const uint4* g_row = gain_bf16_u128 + row_idx * 8u;
    const uint4* f_row = forget_bf16_u128 + row_idx * 8u;
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

// Batched Planar `Q8_64` SMEM-Staged `__dp4a` GEMM (`16` rows x `8` tokens per CTA!):
// - `26,112` B SMEM (`2` CTAs/SM <= `64` KiB carveout -> `64` KiB L1 retained!)
// - Amortizes `W` DRAM reads `8x` inside SMEM and `64x` via Blackwell L2 across `blockIdx.y`!
// - `0-ULP` bit-identical to single-token `ink2_q8_dp4a_gemv`!
__device__ __forceinline__ int dp4a_u128(uint4 w, uint4 x) {
    int d = __dp4a(static_cast<int>(w.x), static_cast<int>(x.x), 0);
    d = __dp4a(static_cast<int>(w.y), static_cast<int>(x.y), d);
    d = __dp4a(static_cast<int>(w.z), static_cast<int>(x.z), d);
    return __dp4a(static_cast<int>(w.w), static_cast<int>(x.w), d);
}

extern "C" __global__ void ink2_q8_dp4a_gemm_b(
    const uint4* __restrict__ w_q8_u128,   // [rows, cols / 16]
    const float* __restrict__ w_sc64,      // [rows, cols / 64]
    const uint4* __restrict__ x_q8_u128,   // [B, cols / 16]
    const float* __restrict__ x_sc64,      // [B, cols / 64]
    float* __restrict__ y_base,            // [B, y_stride]
    int32_t rows,
    int32_t cols,
    int32_t B,
    int32_t y_stride,
    float softcap) {
    __shared__ uint4 smem_x[8][192];  // 24,576 B
    __shared__ float smem_xsc[8][48]; // 1,536 B

    const int tid = threadIdx.x; // 256
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int cols16 = cols >> 4;
    const int cols64 = cols >> 6;
    const int b_base = blockIdx.y * 8;
    const int b_rem = B - b_base;
    const int tile_b = (b_rem < 8) ? b_rem : 8;

    for (int t = 0; t < tile_b; ++t) {
        const uint4* src_x = x_q8_u128 + static_cast<size_t>(b_base + t) * cols16;
        const float* src_sc = x_sc64 + static_cast<size_t>(b_base + t) * cols64;
        for (int i = tid; i < cols16; i += 256) {
            smem_x[t][i] = src_x[i];
        }
        if (tid < cols64) {
            smem_xsc[t][tid] = src_sc[tid];
        }
    }
    __syncthreads();

    const int row0 = (blockIdx.x * 8 + warp_id) * 2;
    if (row0 >= rows) return;
    const bool has_row1 = (row0 + 1 < rows);

    const uint4* w_r0 = w_q8_u128 + static_cast<size_t>(row0) * cols16;
    const uint4* w_r1 = has_row1 ? (w_r0 + cols16) : w_r0;
    const float* s_r0 = w_sc64 + static_cast<size_t>(row0) * cols64;
    const float* s_r1 = has_row1 ? (s_r0 + cols64) : s_r0;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
    float b4 = 0.0f, b5 = 0.0f, b6 = 0.0f, b7 = 0.0f;

    for (int c16 = lane; c16 < cols16; c16 += 32) {
        const int c64 = c16 >> 2;
        const uint4 w0 = w_r0[c16];
        const float ws0 = s_r0[c64];
        const uint4 w1 = has_row1 ? w_r1[c16] : w0;
        const float ws1 = has_row1 ? s_r1[c64] : 0.0f;

        if (tile_b > 0) {
            uint4 xq = smem_x[0][c16]; float xs = smem_xsc[0][c64];
            a0 += static_cast<float>(dp4a_u128(w0, xq)) * (ws0 * xs);
            if (has_row1) b0 += static_cast<float>(dp4a_u128(w1, xq)) * (ws1 * xs);
        }
        if (tile_b > 1) {
            uint4 xq = smem_x[1][c16]; float xs = smem_xsc[1][c64];
            a1 += static_cast<float>(dp4a_u128(w0, xq)) * (ws0 * xs);
            if (has_row1) b1 += static_cast<float>(dp4a_u128(w1, xq)) * (ws1 * xs);
        }
        if (tile_b > 2) {
            uint4 xq = smem_x[2][c16]; float xs = smem_xsc[2][c64];
            a2 += static_cast<float>(dp4a_u128(w0, xq)) * (ws0 * xs);
            if (has_row1) b2 += static_cast<float>(dp4a_u128(w1, xq)) * (ws1 * xs);
        }
        if (tile_b > 3) {
            uint4 xq = smem_x[3][c16]; float xs = smem_xsc[3][c64];
            a3 += static_cast<float>(dp4a_u128(w0, xq)) * (ws0 * xs);
            if (has_row1) b3 += static_cast<float>(dp4a_u128(w1, xq)) * (ws1 * xs);
        }
        if (tile_b > 4) {
            uint4 xq = smem_x[4][c16]; float xs = smem_xsc[4][c64];
            a4 += static_cast<float>(dp4a_u128(w0, xq)) * (ws0 * xs);
            if (has_row1) b4 += static_cast<float>(dp4a_u128(w1, xq)) * (ws1 * xs);
        }
        if (tile_b > 5) {
            uint4 xq = smem_x[5][c16]; float xs = smem_xsc[5][c64];
            a5 += static_cast<float>(dp4a_u128(w0, xq)) * (ws0 * xs);
            if (has_row1) b5 += static_cast<float>(dp4a_u128(w1, xq)) * (ws1 * xs);
        }
        if (tile_b > 6) {
            uint4 xq = smem_x[6][c16]; float xs = smem_xsc[6][c64];
            a6 += static_cast<float>(dp4a_u128(w0, xq)) * (ws0 * xs);
            if (has_row1) b6 += static_cast<float>(dp4a_u128(w1, xq)) * (ws1 * xs);
        }
        if (tile_b > 7) {
            uint4 xq = smem_x[7][c16]; float xs = smem_xsc[7][c64];
            a7 += static_cast<float>(dp4a_u128(w0, xq)) * (ws0 * xs);
            if (has_row1) b7 += static_cast<float>(dp4a_u128(w1, xq)) * (ws1 * xs);
        }
    }

    a0 = warp_sum(a0); b0 = warp_sum(b0);
    a1 = warp_sum(a1); b1 = warp_sum(b1);
    a2 = warp_sum(a2); b2 = warp_sum(b2);
    a3 = warp_sum(a3); b3 = warp_sum(b3);
    a4 = warp_sum(a4); b4 = warp_sum(b4);
    a5 = warp_sum(a5); b5 = warp_sum(b5);
    a6 = warp_sum(a6); b6 = warp_sum(b6);
    a7 = warp_sum(a7); b7 = warp_sum(b7);

    if (lane == 0) {
        const float inv_sc = (softcap > 0.0f) ? ptx_rcp(softcap) : 0.0f;
        auto write_pair = [&](int t, float v0, float v1) {
            if (t < tile_b) {
                if (softcap > 0.0f) {
                    v0 = softcap * ptx_tanh(v0 * inv_sc);
                    v1 = softcap * ptx_tanh(v1 * inv_sc);
                }
                float* dst = y_base + static_cast<size_t>(b_base + t) * y_stride;
                dst[row0] = v0;
                if (has_row1) dst[row0 + 1] = v1;
            }
        };
        write_pair(0, a0, b0);
        write_pair(1, a1, b1);
        write_pair(2, a2, b2);
        write_pair(3, a3, b3);
        write_pair(4, a4, b4);
        write_pair(5, a5, b5);
        write_pair(6, a6, b6);
        write_pair(7, a7, b7);
    }
}

// Register-Persistent Causal `short_conv4` over `b = 0..B-1` (`0-ULP` bit-identical!)
extern "C" __global__ void ink2_short_conv4_b(
    const float* __restrict__ cur_base,      // [B, in_stride]
    int32_t in_stride,
    const float4* __restrict__ w_conv4,      // [channels]
    const float4* __restrict__ dyn_base,     // nullable (`[B, dyn_f4_stride]`)
    int32_t dyn_f4_stride,
    float* __restrict__ hist3,               // [channels * 3]
    float* out_base,                         // [B, out_stride]
    int32_t out_stride,
    const float* res_base,                   // nullable (`[B, res_stride]`)
    int32_t res_stride,
    int32_t channels,
    int32_t B,
    int32_t has_dyn,
    int32_t do_res_add) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    float* h = hist3 + c * 3;
    float h0 = h[0];
    float h1 = h[1];
    float h2 = h[2];
    const float4 w = w_conv4[c];
    const int head = c >> 7;
    const float inv_scale = has_dyn ? 0.3333333333333333f : 0.4472135954999579f;

    for (int b = 0; b < B; ++b) {
        const float x = cur_base[static_cast<size_t>(b) * in_stride + c];
        float conv_o = w.x * h0 + w.y * h1 + w.z * h2 + w.w * x;
        if (has_dyn) {
            float4 d = dyn_base[static_cast<size_t>(b) * dyn_f4_stride + head];
            conv_o += d.x * x + d.y * h2 + d.z * h1 + d.w * h0;
        }
        h0 = h1;
        h1 = h2;
        h2 = x;
        float y = (x + conv_o) * inv_scale;
        if (do_res_add) {
            y += res_base[static_cast<size_t>(b) * res_stride + c];
        }
        out_base[static_cast<size_t>(b) * out_stride + c] = y;
    }

    h[0] = h0;
    h[1] = h1;
    h[2] = h2;
}

// Grid: `(5, B)`, Block: `256`
extern "C" __global__ void ink2_prep_qkv_and_cache_b(
    float* __restrict__ qrgs_base,             // [B, qrgs_stride]
    int32_t qrgs_stride,
    const float* __restrict__ k_in_base,       // [B, 1024]
    const float* __restrict__ v_in_base,       // [B, 1024]
    const uint32_t* __restrict__ q_clip_bf16,  // [24 * 64]
    const uint32_t* __restrict__ r_clip_bf16,  // [24 * 8]
    uint16_t* __restrict__ k_cache_fp16,       // [kv_cap, 1024]
    uint16_t* __restrict__ v_cache_fp16,       // [kv_cap, 1024]
    const int32_t* __restrict__ pos_ptr,
    int32_t kv_cap,
    int32_t is_global) {
    const int g_warp = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int b = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int pos = (*pos_ptr) + b;
    const int cache_row = is_global ? pos : (pos % kv_cap);

    float* qrgs = qrgs_base + static_cast<size_t>(b) * qrgs_stride;
    float* q_buf = qrgs;
    float* r_buf = qrgs + 3072;
    const float* k_in = k_in_base + static_cast<size_t>(b) * 1024u;
    const float* v_in = v_in_base + static_cast<size_t>(b) * 1024u;

    float tau = 1.0f;
    if (is_global) {
        float ratio = static_cast<float>(pos + 1) * (1.0f / 8192.0f);
        if (ratio > 1.0f) {
            tau = 1.0f + 0.1f * ptx_ln(ratio);
        }
    }

    if (g_warp < 24) {
        const int h = g_warp;
        float* qh = q_buf + h * 128;
        const uint32_t* qc = q_clip_bf16 + h * 64;
        float qv0, qv1, qv2, qv3;
        {
            uint32_t cp0 = qc[lane * 2];
            uint32_t cp1 = qc[lane * 2 + 1];
            qv0 = qh[lane * 4] * bf16_to_f32(static_cast<uint16_t>(cp0 & 0xFFFFu));
            qv1 = qh[lane * 4 + 1] * bf16_to_f32(static_cast<uint16_t>(cp0 >> 16));
            qv2 = qh[lane * 4 + 2] * bf16_to_f32(static_cast<uint16_t>(cp1 & 0xFFFFu));
            qv3 = qh[lane * 4 + 3] * bf16_to_f32(static_cast<uint16_t>(cp1 >> 16));
        }
        float q_sq = warp_sum((qv0 * qv0 + qv1 * qv1) + (qv2 * qv2 + qv3 * qv3));
        float q_f = 22.627416997969522f * ptx_rcp(ptx_sqrt(q_sq) + 1e-6f);
        if (q_f > 1.0f) q_f = 1.0f;
        q_f *= tau;
        qh[lane * 4] = qv0 * q_f;
        qh[lane * 4 + 1] = qv1 * q_f;
        qh[lane * 4 + 2] = qv2 * q_f;
        qh[lane * 4 + 3] = qv3 * q_f;

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
        const int kh = g_warp - 24;
        const float* ks = k_in + kh * 128;
        float kv0 = ks[lane * 4], kv1 = ks[lane * 4 + 1], kv2 = ks[lane * 4 + 2], kv3 = ks[lane * 4 + 3];
        float k_sq = warp_sum(kv0 * kv0 + kv1 * kv1 + kv2 * kv2 + kv3 * kv3);
        float k_f = 22.627416997969522f * ptx_rcp(ptx_sqrt(k_sq) + 1e-6f);
        if (k_f > 1.0f) k_f = 1.0f;
        uint16_t* dst = k_cache_fp16 + static_cast<size_t>(cache_row) * 1024u + kh * 128;
        dst[lane * 4] = f32_to_fp16_bits(kv0 * k_f);
        dst[lane * 4 + 1] = f32_to_fp16_bits(kv1 * k_f);
        dst[lane * 4 + 2] = f32_to_fp16_bits(kv2 * k_f);
        dst[lane * 4 + 3] = f32_to_fp16_bits(kv3 * k_f);
    } else if (g_warp < 40) {
        const int vh = g_warp - 32;
        const float* vs = v_in + vh * 128;
        float vv0 = vs[lane * 4], vv1 = vs[lane * 4 + 1], vv2 = vs[lane * 4 + 2], vv3 = vs[lane * 4 + 3];
        float v_sq = warp_sum(vv0 * vv0 + vv1 * vv1 + vv2 * vv2 + vv3 * vv3);
        float v_inv = ptx_rsqrt(v_sq * (1.0f / 128.0f) + 1e-6f);
        uint16_t* dst = v_cache_fp16 + static_cast<size_t>(cache_row) * 1024u + vh * 128;
        dst[lane * 4] = f32_to_fp16_bits(vv0 * v_inv);
        dst[lane * 4 + 1] = f32_to_fp16_bits(vv1 * v_inv);
        dst[lane * 4 + 2] = f32_to_fp16_bits(vv2 * v_inv);
        dst[lane * 4 + 3] = f32_to_fp16_bits(vv3 * v_inv);
    }
}

// Grid: `(24, B)`, Block: `256`
extern "C" __global__ void ink2_gqa_rel_sink_attn_b(
    const float* __restrict__ qrgs_base,        // [B, qrgs_stride]
    int32_t qrgs_stride,
    const uint32_t* __restrict__ rel_proj_bf16, // [16, 512]
    const uint16_t* __restrict__ k_cache_fp16,  // [kv_cap, 1024]
    const uint16_t* __restrict__ v_cache_fp16,  // [kv_cap, 1024]
    const int32_t* __restrict__ pos_ptr,
    int32_t kv_cap,
    int32_t is_global,
    float* __restrict__ attn_out_base) {        // [B, 3072]
    __shared__ float smem_rel[kExtent];
    __shared__ float smem_q[kHd];
    __shared__ float smem_r[kRel];
    __shared__ float smem_m[8];
    __shared__ float smem_l[8];
    __shared__ float smem_o[8][kHd];

    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int kvh = h / 3;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int pos = (*pos_ptr) + b;
    const int n = pos + 1;
    const int first = is_global ? 0 : ((n > kExtent) ? (n - kExtent) : 0);

    const float* qrgs = qrgs_base + static_cast<size_t>(b) * qrgs_stride;
    const float* q_buf = qrgs;
    const float* r_buf = qrgs + 3072;
    const float* gate_buf = qrgs + 3456;
    const float* sinks_buf = qrgs + 6528;

    if (tid < kHd) smem_q[tid] = q_buf[h * kHd + tid];
    if (tid < kRel) smem_r[tid] = r_buf[h * kRel + tid];
    __syncthreads();

    const int max_dist = (n < kExtent) ? n : kExtent;
    for (int pair_d = tid; pair_d < (max_dist + 1) / 2; pair_d += 256) {
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

    const float q0 = smem_q[lane * 4];
    const float q1 = smem_q[lane * 4 + 1];
    const float q2 = smem_q[lane * 4 + 2];
    const float q3 = smem_q[lane * 4 + 3];
    const float scale = 0.08838834764831845f;

    float m_w = (warp_id == 0) ? sinks_buf[h] : -1e30f;
    float l_w = (warp_id == 0) ? 1.0f : 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    for (int j = first + warp_id; j < n; j += 8) {
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

    if (tid < kHd) {
        float m_max = smem_m[0];
        #pragma unroll
        for (int w = 1; w < 8; ++w) {
            if (smem_m[w] > m_max) m_max = smem_m[w];
        }
        float denom = 0.0f;
        float sum_d = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            float s = ptx_exp(smem_m[w] - m_max);
            denom += smem_l[w] * s;
            sum_d += smem_o[w][tid] * s;
        }
        float g = ptx_sigmoid(gate_buf[h * kHd + tid]);
        attn_out_base[static_cast<size_t>(b) * 3072u + h * kHd + tid] =
            (sum_d * ptx_rcp(fmaxf(denom, 1e-20f))) * g;
    }
}

// Batched Exact BF16 Router (`[128, 2304] x [B, 2304]^T -> [B, 128]`):
// Grid: `(16, ceil(B / 4))`, Block: `256` (`8` rows x `4` tokens, `36,864` B SMEM)
extern "C" __global__ void ink2_bf16_gemm_router_b(
    const uint4* __restrict__ w_bf16_u128, // [128, 288]
    const float4* __restrict__ x4_base,    // [B, 576]
    float* __restrict__ y_base,            // [B, 128]
    int32_t B) {
    __shared__ float4 smem_x[4][576];      // 36,864 B

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int b_base = blockIdx.y * 4;
    const int b_rem = B - b_base;
    const int tile_b = (b_rem < 4) ? b_rem : 4;

    for (int t = 0; t < tile_b; ++t) {
        const float4* src = x4_base + static_cast<size_t>(b_base + t) * 576u;
        for (int i = tid; i < 576; i += 256) {
            smem_x[t][i] = src[i];
        }
    }
    __syncthreads();

    const int row = blockIdx.x * 8 + warp_id;
    if (row >= kExp) return;

    const uint4* w_row = w_bf16_u128 + static_cast<size_t>(row) * 288u;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    for (int c8 = lane; c8 < 288; c8 += 32) {
        uint4 pack = w_row[c8];
        float w0 = bf16_to_f32(static_cast<uint16_t>(pack.x & 0xFFFFu));
        float w1 = bf16_to_f32(static_cast<uint16_t>(pack.x >> 16));
        float w2 = bf16_to_f32(static_cast<uint16_t>(pack.y & 0xFFFFu));
        float w3 = bf16_to_f32(static_cast<uint16_t>(pack.y >> 16));
        float w4 = bf16_to_f32(static_cast<uint16_t>(pack.z & 0xFFFFu));
        float w5 = bf16_to_f32(static_cast<uint16_t>(pack.z >> 16));
        float w6 = bf16_to_f32(static_cast<uint16_t>(pack.w & 0xFFFFu));
        float w7 = bf16_to_f32(static_cast<uint16_t>(pack.w >> 16));

        if (tile_b > 0) {
            float4 u = smem_x[0][c8 * 2]; float4 v = smem_x[0][c8 * 2 + 1];
            a0 += w0 * u.x; a0 += w1 * u.y; a0 += w2 * u.z; a0 += w3 * u.w;
            a0 += w4 * v.x; a0 += w5 * v.y; a0 += w6 * v.z; a0 += w7 * v.w;
        }
        if (tile_b > 1) {
            float4 u = smem_x[1][c8 * 2]; float4 v = smem_x[1][c8 * 2 + 1];
            a1 += w0 * u.x; a1 += w1 * u.y; a1 += w2 * u.z; a1 += w3 * u.w;
            a1 += w4 * v.x; a1 += w5 * v.y; a1 += w6 * v.z; a1 += w7 * v.w;
        }
        if (tile_b > 2) {
            float4 u = smem_x[2][c8 * 2]; float4 v = smem_x[2][c8 * 2 + 1];
            a2 += w0 * u.x; a2 += w1 * u.y; a2 += w2 * u.z; a2 += w3 * u.w;
            a2 += w4 * v.x; a2 += w5 * v.y; a2 += w6 * v.z; a2 += w7 * v.w;
        }
        if (tile_b > 3) {
            float4 u = smem_x[3][c8 * 2]; float4 v = smem_x[3][c8 * 2 + 1];
            a3 += w0 * u.x; a3 += w1 * u.y; a3 += w2 * u.z; a3 += w3 * u.w;
            a3 += w4 * v.x; a3 += w5 * v.y; a3 += w6 * v.z; a3 += w7 * v.w;
        }
    }
    a0 = warp_sum(a0);
    a1 = warp_sum(a1);
    a2 = warp_sum(a2);
    a3 = warp_sum(a3);
    if (lane == 0) {
        if (tile_b > 0) y_base[(b_base + 0) * kExp + row] = a0;
        if (tile_b > 1) y_base[(b_base + 1) * kExp + row] = a1;
        if (tile_b > 2) y_base[(b_base + 2) * kExp + row] = a2;
        if (tile_b > 3) y_base[(b_base + 3) * kExp + row] = a3;
    }
}

// Grid: `(B, 1)`, Block: `32`
extern "C" __global__ void ink2_router_topk8_b(
    const float* __restrict__ logits128_base, // [B, 128]
    const float* __restrict__ bias128,        // [128]
    int32_t* __restrict__ sel_ids_base,       // [B, 8]
    float* __restrict__ sel_wts_base) {       // [B, 8]
    __shared__ int smem_ids[8];
    __shared__ float smem_ps[8];

    const int b = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int base = lane * 4;
    const float* logits128 = logits128_base + static_cast<size_t>(b) * kExp;

    float p0 = ptx_sigmoid(logits128[base + 0]);
    float p1 = ptx_sigmoid(logits128[base + 1]);
    float p2 = ptx_sigmoid(logits128[base + 2]);
    float p3 = ptx_sigmoid(logits128[base + 3]);
    float s0 = p0 + bias128[base + 0];
    float s1 = p1 + bias128[base + 1];
    float s2 = p2 + bias128[base + 2];
    float s3 = p3 + bias128[base + 3];

    float sum_p = 1e-20f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float best_s = s0;
        int best_e = base + 0;
        float best_p = p0;
        if (s1 > best_s) { best_s = s1; best_e = base + 1; best_p = p1; }
        if (s2 > best_s) { best_s = s2; best_e = base + 2; best_p = p2; }
        if (s3 > best_s) { best_s = s3; best_e = base + 3; best_p = p3; }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            float os = __shfl_xor_sync(0xFFFFFFFFu, best_s, mask);
            int oe = __shfl_xor_sync(0xFFFFFFFFu, best_e, mask);
            float op = __shfl_xor_sync(0xFFFFFFFFu, best_p, mask);
            if (os > best_s || (os == best_s && oe < best_e)) {
                best_s = os;
                best_e = oe;
                best_p = op;
            }
        }
        int win_e = __shfl_sync(0xFFFFFFFFu, best_e, 0);
        float win_p = __shfl_sync(0xFFFFFFFFu, best_p, 0);
        sum_p += win_p;
        if (lane == 0) {
            smem_ids[k] = win_e;
            smem_ps[k] = win_p;
        }
        if (win_e == base + 0) s0 = -1e30f;
        if (win_e == base + 1) s1 = -1e30f;
        if (win_e == base + 2) s2 = -1e30f;
        if (win_e == base + 3) s3 = -1e30f;
    }
    __syncwarp();
    if (lane < 8) {
        float inv = 8.0f * ptx_rcp(sum_p);
        sel_ids_base[b * 8 + lane] = smem_ids[lane];
        sel_wts_base[b * 8 + lane] = smem_ps[lane] * inv;
    }
}

// Deterministic Zero-Atomic Expert Grouping (`1` CTA, `128` threads = `1` thread/expert!)
extern "C" __global__ void ink2_build_expert_groups_b(
    const int32_t* __restrict__ sel_ids, // [B * 8]
    int32_t* __restrict__ exp_counts,    // [128]
    int32_t* __restrict__ exp_toks,      // [128, 64]
    int32_t* __restrict__ exp_pairs,     // [128, 64]
    int32_t B) {
    __shared__ int32_t smem_sel[512];
    const int ex = threadIdx.x; // 0..127
    const int total_pairs = B * 8;
    for (int i = ex; i < total_pairs; i += 128) {
        smem_sel[i] = sel_ids[i];
    }
    __syncthreads();

    int cnt = 0;
    int32_t* dst_tok = exp_toks + ex * 64;
    int32_t* dst_pair = exp_pairs + ex * 64;
    for (int p = 0; p < total_pairs; ++p) {
        if (smem_sel[p] == ex) {
            dst_tok[cnt] = p >> 3;
            dst_pair[cnt] = p;
            ++cnt;
        }
    }
    exp_counts[ex] = cnt;
}

__device__ __forceinline__ int dp4a_nvfp4_lo(uint4 pk, uint4 xq) {
    int d = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk.x & 0xFFFFu)), static_cast<int>(xq.x), 0);
    d = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk.x >> 16)), static_cast<int>(xq.y), d);
    d = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk.y & 0xFFFFu)), static_cast<int>(xq.z), d);
    return __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk.y >> 16)), static_cast<int>(xq.w), d);
}

__device__ __forceinline__ int dp4a_nvfp4_hi(uint4 pk, uint4 xq) {
    int d = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk.z & 0xFFFFu)), static_cast<int>(xq.x), 0);
    d = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk.z >> 16)), static_cast<int>(xq.y), d);
    d = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk.w & 0xFFFFu)), static_cast<int>(xq.z), d);
    return __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk.w >> 16)), static_cast<int>(xq.w), d);
}

// Tile-8 Grouped-Expert NVFP4 `__dp4a` Up+SITU (`Grid (72, 128)`, `Block 256`):
// Stages up to `8` tokens in SMEM (`23,296` B <= `64` KiB carveout at `2` CTAs/SM) and
// processes `4` row-groups (`16` `m` outputs) per CTA with exact `u16 = lane` FMA parity!
extern "C" __global__ void ink2_moe_nvfp4_dp4a_grouped_up_situ(
    const uint4* __restrict__ w13_u128,    // [128, 2304, 72]
    const uint32_t* __restrict__ s13_u32,  // [128, 2304, 36]
    const float* __restrict__ a13,         // [128]
    const uint4* __restrict__ x_q8_u128,   // [B, 144]
    const float* __restrict__ x_sc16,      // [B, 144]
    const int32_t* __restrict__ exp_counts,// [128]
    const int32_t* __restrict__ exp_toks,  // [128, 64]
    const int32_t* __restrict__ exp_pairs, // [128, 64]
    float* __restrict__ act_out) {         // [B * 8, 1152]
    const int ex = blockIdx.y;
    const int cnt = exp_counts[ex];
    if (cnt == 0) return;

    __shared__ uint4 smem_xq[8][144];
    __shared__ float smem_xsc[8][144];
    __shared__ float smem_gu[8][8];
    __shared__ int32_t smem_pairs[8];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int is_up = warp_id & 1;
    const float amax_sc = a13[ex] * (1.0f / 2688.0f);

    for (int t_base = 0; t_base < cnt; t_base += 8) {
        const int rem = cnt - t_base;
        const int tile_t = (rem < 8) ? rem : 8;
        if (t_base > 0) __syncthreads();
        if (tid < tile_t) {
            smem_pairs[tid] = exp_pairs[ex * 64 + t_base + tid];
        }
        for (int t = 0; t < tile_t; ++t) {
            const int tok = exp_toks[ex * 64 + t_base + t];
            if (tid < 144) {
                smem_xq[t][tid] = x_q8_u128[static_cast<size_t>(tok) * 144u + tid];
                smem_xsc[t][tid] = x_sc16[static_cast<size_t>(tok) * 144u + tid];
            }
        }
        __syncthreads();

        for (int rg = 0; rg < 4; ++rg) {
            const int m = (blockIdx.x * 4 + rg) * 4 + (warp_id >> 1);
            const int r = m * 2 + is_up;
            const size_t row_idx = static_cast<size_t>(ex) * (2u * kFf) + r;
            const uint4* r_u128 = w13_u128 + row_idx * 72u;
            const uint32_t* r_sc = s13_u32 + row_idx * 36u;

            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;

            for (int u16 = lane; u16 < 72; u16 += 32) {
                const uint4 pk = r_u128[u16];
                const uint32_t sc4 = r_sc[u16 >> 1];
                const int sh = (u16 & 1) * 16;
                const float ws0 = e4m3_byte_to_f32((sc4 >> sh) & 0xFFu) * 0.5f;
                const float ws1 = e4m3_byte_to_f32((sc4 >> (sh + 8)) & 0xFFu) * 0.5f;
                const int c0 = u16 * 2;
                const int c1 = c0 + 1;

                if (tile_t > 0) a0 += static_cast<float>(dp4a_nvfp4_lo(pk, smem_xq[0][c0])) * (ws0 * smem_xsc[0][c0]) + static_cast<float>(dp4a_nvfp4_hi(pk, smem_xq[0][c1])) * (ws1 * smem_xsc[0][c1]);
                if (tile_t > 1) a1 += static_cast<float>(dp4a_nvfp4_lo(pk, smem_xq[1][c0])) * (ws0 * smem_xsc[1][c0]) + static_cast<float>(dp4a_nvfp4_hi(pk, smem_xq[1][c1])) * (ws1 * smem_xsc[1][c1]);
                if (tile_t > 2) a2 += static_cast<float>(dp4a_nvfp4_lo(pk, smem_xq[2][c0])) * (ws0 * smem_xsc[2][c0]) + static_cast<float>(dp4a_nvfp4_hi(pk, smem_xq[2][c1])) * (ws1 * smem_xsc[2][c1]);
                if (tile_t > 3) a3 += static_cast<float>(dp4a_nvfp4_lo(pk, smem_xq[3][c0])) * (ws0 * smem_xsc[3][c0]) + static_cast<float>(dp4a_nvfp4_hi(pk, smem_xq[3][c1])) * (ws1 * smem_xsc[3][c1]);
                if (tile_t > 4) a4 += static_cast<float>(dp4a_nvfp4_lo(pk, smem_xq[4][c0])) * (ws0 * smem_xsc[4][c0]) + static_cast<float>(dp4a_nvfp4_hi(pk, smem_xq[4][c1])) * (ws1 * smem_xsc[4][c1]);
                if (tile_t > 5) a5 += static_cast<float>(dp4a_nvfp4_lo(pk, smem_xq[5][c0])) * (ws0 * smem_xsc[5][c0]) + static_cast<float>(dp4a_nvfp4_hi(pk, smem_xq[5][c1])) * (ws1 * smem_xsc[5][c1]);
                if (tile_t > 6) a6 += static_cast<float>(dp4a_nvfp4_lo(pk, smem_xq[6][c0])) * (ws0 * smem_xsc[6][c0]) + static_cast<float>(dp4a_nvfp4_hi(pk, smem_xq[6][c1])) * (ws1 * smem_xsc[6][c1]);
                if (tile_t > 7) a7 += static_cast<float>(dp4a_nvfp4_lo(pk, smem_xq[7][c0])) * (ws0 * smem_xsc[7][c0]) + static_cast<float>(dp4a_nvfp4_hi(pk, smem_xq[7][c1])) * (ws1 * smem_xsc[7][c1]);
            }

            a0 = warp_sum(a0); a1 = warp_sum(a1); a2 = warp_sum(a2); a3 = warp_sum(a3);
            a4 = warp_sum(a4); a5 = warp_sum(a5); a6 = warp_sum(a6); a7 = warp_sum(a7);

            if (rg > 0) __syncthreads();
            if (lane == 0) {
                if (tile_t > 0) smem_gu[0][warp_id] = a0 * amax_sc;
                if (tile_t > 1) smem_gu[1][warp_id] = a1 * amax_sc;
                if (tile_t > 2) smem_gu[2][warp_id] = a2 * amax_sc;
                if (tile_t > 3) smem_gu[3][warp_id] = a3 * amax_sc;
                if (tile_t > 4) smem_gu[4][warp_id] = a4 * amax_sc;
                if (tile_t > 5) smem_gu[5][warp_id] = a5 * amax_sc;
                if (tile_t > 6) smem_gu[6][warp_id] = a6 * amax_sc;
                if (tile_t > 7) smem_gu[7][warp_id] = a7 * amax_sc;
            }
            __syncthreads();
            if (is_up == 0 && lane < tile_t) {
                float g = smem_gu[lane][warp_id];
                float u = smem_gu[lane][warp_id + 1];
                float situ = 4.0f * ptx_tanh(g * 0.25f) * ptx_sigmoid(g) * 25.0f * ptx_tanh(u * 0.04f);
                act_out[static_cast<size_t>(smem_pairs[lane]) * kFf + m] = situ;
            }
        }
    }
}

// Tile-8 Grouped-Expert NVFP4 `__dp4a` Down (`Grid (72, 128)`, `Block 256`, Zero Atomics!)
extern "C" __global__ void ink2_moe_nvfp4_dp4a_grouped_down(
    const uint4* __restrict__ w2_u128,     // [128, 2304, 36]
    const uint32_t* __restrict__ s2_u32,   // [128, 2304, 18]
    const float* __restrict__ a2,          // [128]
    const uint4* __restrict__ act_q8_u128, // [B * 8, 72]
    const float* __restrict__ act_sc16,    // [B * 8, 72]
    const int32_t* __restrict__ exp_counts,// [128]
    const int32_t* __restrict__ exp_pairs, // [128, 64]
    float* __restrict__ pair_y) {          // [B * 8, 2304]
    const int ex = blockIdx.y;
    const int cnt = exp_counts[ex];
    if (cnt == 0) return;

    __shared__ uint4 smem_aq[8][72];
    __shared__ float smem_asc[8][72];
    __shared__ int32_t smem_pairs[8];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const float amax_sc = a2[ex] * (1.0f / 2688.0f);

    for (int t_base = 0; t_base < cnt; t_base += 8) {
        const int rem = cnt - t_base;
        const int tile_t = (rem < 8) ? rem : 8;
        if (t_base > 0) __syncthreads();
        if (tid < tile_t) {
            smem_pairs[tid] = exp_pairs[ex * 64 + t_base + tid];
        }
        for (int t = 0; t < tile_t; ++t) {
            const int pair_idx = exp_pairs[ex * 64 + t_base + t];
            if (tid < 72) {
                smem_aq[t][tid] = act_q8_u128[static_cast<size_t>(pair_idx) * 72u + tid];
                smem_asc[t][tid] = act_sc16[static_cast<size_t>(pair_idx) * 72u + tid];
            }
        }
        __syncthreads();

        for (int rg = 0; rg < 4; ++rg) {
            const int row = (blockIdx.x * 4 + rg) * 8 + warp_id;
            if (row >= kD) continue;
            const size_t r_idx = static_cast<size_t>(ex) * kD + row;
            const uint4* r_u128 = w2_u128 + r_idx * 36u;
            const uint32_t* r_sc = s2_u32 + r_idx * 18u;

            float a0 = 0.0f, a1 = 0.0f, a2v = 0.0f, a3 = 0.0f;
            float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;

            for (int u16 = lane; u16 < 36; u16 += 32) {
                const uint4 pk = r_u128[u16];
                const uint32_t sc4 = r_sc[u16 >> 1];
                const int sh = (u16 & 1) * 16;
                const float ws0 = e4m3_byte_to_f32((sc4 >> sh) & 0xFFu) * 0.5f;
                const float ws1 = e4m3_byte_to_f32((sc4 >> (sh + 8)) & 0xFFu) * 0.5f;
                const int c0 = u16 * 2;
                const int c1 = c0 + 1;

                if (tile_t > 0) a0 += static_cast<float>(dp4a_nvfp4_lo(pk, smem_aq[0][c0])) * (ws0 * smem_asc[0][c0]) + static_cast<float>(dp4a_nvfp4_hi(pk, smem_aq[0][c1])) * (ws1 * smem_asc[0][c1]);
                if (tile_t > 1) a1 += static_cast<float>(dp4a_nvfp4_lo(pk, smem_aq[1][c0])) * (ws0 * smem_asc[1][c0]) + static_cast<float>(dp4a_nvfp4_hi(pk, smem_aq[1][c1])) * (ws1 * smem_asc[1][c1]);
                if (tile_t > 2) a2v += static_cast<float>(dp4a_nvfp4_lo(pk, smem_aq[2][c0])) * (ws0 * smem_asc[2][c0]) + static_cast<float>(dp4a_nvfp4_hi(pk, smem_aq[2][c1])) * (ws1 * smem_asc[2][c1]);
                if (tile_t > 3) a3 += static_cast<float>(dp4a_nvfp4_lo(pk, smem_aq[3][c0])) * (ws0 * smem_asc[3][c0]) + static_cast<float>(dp4a_nvfp4_hi(pk, smem_aq[3][c1])) * (ws1 * smem_asc[3][c1]);
                if (tile_t > 4) a4 += static_cast<float>(dp4a_nvfp4_lo(pk, smem_aq[4][c0])) * (ws0 * smem_asc[4][c0]) + static_cast<float>(dp4a_nvfp4_hi(pk, smem_aq[4][c1])) * (ws1 * smem_asc[4][c1]);
                if (tile_t > 5) a5 += static_cast<float>(dp4a_nvfp4_lo(pk, smem_aq[5][c0])) * (ws0 * smem_asc[5][c0]) + static_cast<float>(dp4a_nvfp4_hi(pk, smem_aq[5][c1])) * (ws1 * smem_asc[5][c1]);
                if (tile_t > 6) a6 += static_cast<float>(dp4a_nvfp4_lo(pk, smem_aq[6][c0])) * (ws0 * smem_asc[6][c0]) + static_cast<float>(dp4a_nvfp4_hi(pk, smem_aq[6][c1])) * (ws1 * smem_asc[6][c1]);
                if (tile_t > 7) a7 += static_cast<float>(dp4a_nvfp4_lo(pk, smem_aq[7][c0])) * (ws0 * smem_asc[7][c0]) + static_cast<float>(dp4a_nvfp4_hi(pk, smem_aq[7][c1])) * (ws1 * smem_asc[7][c1]);
            }

            a0 = warp_sum(a0); a1 = warp_sum(a1); a2v = warp_sum(a2v); a3 = warp_sum(a3);
            a4 = warp_sum(a4); a5 = warp_sum(a5); a6 = warp_sum(a6); a7 = warp_sum(a7);

            if (lane == 0) {
                if (tile_t > 0) pair_y[static_cast<size_t>(smem_pairs[0]) * kD + row] = a0 * amax_sc;
                if (tile_t > 1) pair_y[static_cast<size_t>(smem_pairs[1]) * kD + row] = a1 * amax_sc;
                if (tile_t > 2) pair_y[static_cast<size_t>(smem_pairs[2]) * kD + row] = a2v * amax_sc;
                if (tile_t > 3) pair_y[static_cast<size_t>(smem_pairs[3]) * kD + row] = a3 * amax_sc;
                if (tile_t > 4) pair_y[static_cast<size_t>(smem_pairs[4]) * kD + row] = a4 * amax_sc;
                if (tile_t > 5) pair_y[static_cast<size_t>(smem_pairs[5]) * kD + row] = a5 * amax_sc;
                if (tile_t > 6) pair_y[static_cast<size_t>(smem_pairs[6]) * kD + row] = a6 * amax_sc;
                if (tile_t > 7) pair_y[static_cast<size_t>(smem_pairs[7]) * kD + row] = a7 * amax_sc;
            }
        }
    }
}

__device__ __forceinline__ float dot_mxfp8_u128_raw(uint4 p, const float4* __restrict__ x4_sub) {
    float sub = 0.0f;
    float4 xv0 = x4_sub[0];
    sub += e4m3_byte_to_f32(p.x & 0xFFu) * xv0.x;
    sub += e4m3_byte_to_f32((p.x >> 8) & 0xFFu) * xv0.y;
    sub += e4m3_byte_to_f32((p.x >> 16) & 0xFFu) * xv0.z;
    sub += e4m3_byte_to_f32((p.x >> 24) & 0xFFu) * xv0.w;
    float4 xv1 = x4_sub[1];
    sub += e4m3_byte_to_f32(p.y & 0xFFu) * xv1.x;
    sub += e4m3_byte_to_f32((p.y >> 8) & 0xFFu) * xv1.y;
    sub += e4m3_byte_to_f32((p.y >> 16) & 0xFFu) * xv1.z;
    sub += e4m3_byte_to_f32((p.y >> 24) & 0xFFu) * xv1.w;
    float4 xv2 = x4_sub[2];
    sub += e4m3_byte_to_f32(p.z & 0xFFu) * xv2.x;
    sub += e4m3_byte_to_f32((p.z >> 8) & 0xFFu) * xv2.y;
    sub += e4m3_byte_to_f32((p.z >> 16) & 0xFFu) * xv2.z;
    sub += e4m3_byte_to_f32((p.z >> 24) & 0xFFu) * xv2.w;
    float4 xv3 = x4_sub[3];
    sub += e4m3_byte_to_f32(p.w & 0xFFu) * xv3.x;
    sub += e4m3_byte_to_f32((p.w >> 8) & 0xFFu) * xv3.y;
    sub += e4m3_byte_to_f32((p.w >> 16) & 0xFFu) * xv3.z;
    sub += e4m3_byte_to_f32((p.w >> 24) & 0xFFu) * xv3.w;
    return sub;
}

__device__ __forceinline__ float load_e8m0_scale(const uint32_t* __restrict__ row_e8m0_u32, int u16_idx) {
    const int blk32 = u16_idx >> 1;
    uint32_t s4 = row_e8m0_u32[blk32 >> 2];
    uint32_t s_e8m0 = (s4 >> ((blk32 & 3) * 8)) & 0xFFu;
    return __uint_as_float(s_e8m0 << 23);
}

// Tile-4 Grouped-Expert MXFP8 Up+SITU (`Grid (72, 128)`, `Block 256`, `<= 42` regs!)
extern "C" __global__ void ink2_moe_mxfp8_grouped_up_situ(
    const uint4* __restrict__ w13_u128,    // [128, 2304, 144]
    const uint32_t* __restrict__ s13_u32,  // [128, 2304, 18]
    const float4* __restrict__ x4_base,    // [B, 576]
    const int32_t* __restrict__ exp_counts,// [128]
    const int32_t* __restrict__ exp_toks,  // [128, 64]
    const int32_t* __restrict__ exp_pairs, // [128, 64]
    float* __restrict__ act_out) {         // [B * 8, 1152]
    const int ex = blockIdx.y;
    const int cnt = exp_counts[ex];
    if (cnt == 0) return;

    __shared__ float4 smem_x4[4][576];     // 36,864 B
    __shared__ float smem_gu[4][8];
    __shared__ int32_t smem_pairs[4];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int is_up = warp_id & 1;

    for (int t_base = 0; t_base < cnt; t_base += 4) {
        const int rem = cnt - t_base;
        const int tile_t = (rem < 4) ? rem : 4;
        if (t_base > 0) __syncthreads();
        if (tid < tile_t) {
            smem_pairs[tid] = exp_pairs[ex * 64 + t_base + tid];
        }
        for (int t = 0; t < tile_t; ++t) {
            const int tok = exp_toks[ex * 64 + t_base + t];
            const float4* src = x4_base + static_cast<size_t>(tok) * 576u;
            for (int i = tid; i < 576; i += 256) {
                smem_x4[t][i] = src[i];
            }
        }
        __syncthreads();

        for (int rg = 0; rg < 4; ++rg) {
            const int m = (blockIdx.x * 4 + rg) * 4 + (warp_id >> 1);
            const int r = m * 2 + is_up;
            const size_t row_idx = static_cast<size_t>(ex) * (2u * kFf) + r;
            const uint4* r_u128 = w13_u128 + row_idx * 144u;
            const uint32_t* r_sc = s13_u32 + row_idx * 18u;

            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (int u16 = lane; u16 < 144; u16 += 32) {
                const uint4 p = r_u128[u16];
                const float es = load_e8m0_scale(r_sc, u16);
                const int c4 = u16 * 4;
                if (tile_t > 0) a0 += dot_mxfp8_u128_raw(p, smem_x4[0] + c4) * es;
                if (tile_t > 1) a1 += dot_mxfp8_u128_raw(p, smem_x4[1] + c4) * es;
                if (tile_t > 2) a2 += dot_mxfp8_u128_raw(p, smem_x4[2] + c4) * es;
                if (tile_t > 3) a3 += dot_mxfp8_u128_raw(p, smem_x4[3] + c4) * es;
            }
            a0 = warp_sum(a0); a1 = warp_sum(a1); a2 = warp_sum(a2); a3 = warp_sum(a3);

            if (rg > 0) __syncthreads();
            if (lane == 0) {
                if (tile_t > 0) smem_gu[0][warp_id] = a0;
                if (tile_t > 1) smem_gu[1][warp_id] = a1;
                if (tile_t > 2) smem_gu[2][warp_id] = a2;
                if (tile_t > 3) smem_gu[3][warp_id] = a3;
            }
            __syncthreads();
            if (is_up == 0 && lane < tile_t) {
                float g = smem_gu[lane][warp_id];
                float u = smem_gu[lane][warp_id + 1];
                float situ = 4.0f * ptx_tanh(g * 0.25f) * ptx_sigmoid(g) * 25.0f * ptx_tanh(u * 0.04f);
                act_out[static_cast<size_t>(smem_pairs[lane]) * kFf + m] = situ;
            }
        }
    }
}

// Tile-8 Grouped-Expert MXFP8 Down (`Grid (72, 128)`, `Block 256`, Zero Atomics!)
extern "C" __global__ void ink2_moe_mxfp8_grouped_down(
    const uint4* __restrict__ w2_u128,     // [128, 2304, 72]
    const uint32_t* __restrict__ s2_u32,   // [128, 2304, 9]
    const float4* __restrict__ act4_base,  // [B * 8, 288]
    const int32_t* __restrict__ exp_counts,// [128]
    const int32_t* __restrict__ exp_pairs, // [128, 64]
    float* __restrict__ pair_y) {          // [B * 8, 2304]
    const int ex = blockIdx.y;
    const int cnt = exp_counts[ex];
    if (cnt == 0) return;

    __shared__ float4 smem_a4[8][288];     // 36,864 B
    __shared__ int32_t smem_pairs[8];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    for (int t_base = 0; t_base < cnt; t_base += 8) {
        const int rem = cnt - t_base;
        const int tile_t = (rem < 8) ? rem : 8;
        if (t_base > 0) __syncthreads();
        if (tid < tile_t) {
            smem_pairs[tid] = exp_pairs[ex * 64 + t_base + tid];
        }
        for (int t = 0; t < tile_t; ++t) {
            const int pair_idx = exp_pairs[ex * 64 + t_base + t];
            const float4* src = act4_base + static_cast<size_t>(pair_idx) * 288u;
            for (int i = tid; i < 288; i += 256) {
                smem_a4[t][i] = src[i];
            }
        }
        __syncthreads();

        for (int rg = 0; rg < 4; ++rg) {
            const int row = (blockIdx.x * 4 + rg) * 8 + warp_id;
            if (row >= kD) continue;
            const size_t r_idx = static_cast<size_t>(ex) * kD + row;
            const uint4* r_u128 = w2_u128 + r_idx * 72u;
            const uint32_t* r_sc = s2_u32 + r_idx * 9u;

            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;

            for (int u16 = lane; u16 < 72; u16 += 32) {
                const uint4 p = r_u128[u16];
                const float es = load_e8m0_scale(r_sc, u16);
                const int c4 = u16 * 4;
                if (tile_t > 0) a0 += dot_mxfp8_u128_raw(p, smem_a4[0] + c4) * es;
                if (tile_t > 1) a1 += dot_mxfp8_u128_raw(p, smem_a4[1] + c4) * es;
                if (tile_t > 2) a2 += dot_mxfp8_u128_raw(p, smem_a4[2] + c4) * es;
                if (tile_t > 3) a3 += dot_mxfp8_u128_raw(p, smem_a4[3] + c4) * es;
                if (tile_t > 4) a4 += dot_mxfp8_u128_raw(p, smem_a4[4] + c4) * es;
                if (tile_t > 5) a5 += dot_mxfp8_u128_raw(p, smem_a4[5] + c4) * es;
                if (tile_t > 6) a6 += dot_mxfp8_u128_raw(p, smem_a4[6] + c4) * es;
                if (tile_t > 7) a7 += dot_mxfp8_u128_raw(p, smem_a4[7] + c4) * es;
            }
            a0 = warp_sum(a0); a1 = warp_sum(a1); a2 = warp_sum(a2); a3 = warp_sum(a3);
            a4 = warp_sum(a4); a5 = warp_sum(a5); a6 = warp_sum(a6); a7 = warp_sum(a7);

            if (lane == 0) {
                if (tile_t > 0) pair_y[static_cast<size_t>(smem_pairs[0]) * kD + row] = a0;
                if (tile_t > 1) pair_y[static_cast<size_t>(smem_pairs[1]) * kD + row] = a1;
                if (tile_t > 2) pair_y[static_cast<size_t>(smem_pairs[2]) * kD + row] = a2;
                if (tile_t > 3) pair_y[static_cast<size_t>(smem_pairs[3]) * kD + row] = a3;
                if (tile_t > 4) pair_y[static_cast<size_t>(smem_pairs[4]) * kD + row] = a4;
                if (tile_t > 5) pair_y[static_cast<size_t>(smem_pairs[5]) * kD + row] = a5;
                if (tile_t > 6) pair_y[static_cast<size_t>(smem_pairs[6]) * kD + row] = a6;
                if (tile_t > 7) pair_y[static_cast<size_t>(smem_pairs[7]) * kD + row] = a7;
            }
        }
    }
}

// Deterministic `k = 0..7` Combine (`0-ULP` bit-identical to single-token MoE down!)
extern "C" __global__ void ink2_moe_combine_top8_b(
    const float* __restrict__ pair_y,   // [B * 8, 2304]
    const float* __restrict__ sel_wts,  // [B * 8]
    float* __restrict__ moe_out_base,   // [B, 2304]
    int32_t B) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (d >= kD || b >= B) return;

    float sum_y = 0.0f;
    const size_t base_pair = static_cast<size_t>(b) * 8u;
    for (int k = 0; k < 8; ++k) {
        sum_y += pair_y[(base_pair + k) * kD + d] * sel_wts[base_pair + k];
    }
    moe_out_base[static_cast<size_t>(b) * kD + d] = sum_y;
}

extern "C" __global__ void ink2_inc_pos_by(int32_t* __restrict__ pos_ptr, int32_t delta) {
    if (threadIdx.x == 0) {
        *pos_ptr += delta;
    }
}

// Multi-Token Greedy ArgMax (`Grid (B_out, 1)`, `Block 256`) for Speculative Verification!
extern "C" __global__ void ink2_argmax_b(
    const float* __restrict__ logits_base, // [B_out, VOCAB]
    int32_t* __restrict__ out_toks) {      // [B_out]
    __shared__ float smem_val[8];
    __shared__ int smem_idx[8];

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const float* logits = logits_base + static_cast<size_t>(b) * kVocab;

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
        out_toks[b] = fi;
    }
}
