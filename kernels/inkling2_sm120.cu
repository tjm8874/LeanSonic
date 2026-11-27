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

// ----------------------------------------------------------------------------
// Opt-in INT8 KV for Full-attention (global) layers (`LEANSONIC_KV=int8`, #14 / T011).
// A row holds `8 x 128` int8 values (1,024 B) plus one FP32 absmax scale per
// (row, KV head) in a separate `[kv_cap, 8]` array. `scale == nullptr` selects the
// original FP16 layout, so FP16 numerics are unchanged bit-for-bit.
// ----------------------------------------------------------------------------
__device__ __forceinline__ void kv_load4(
    const uint16_t* __restrict__ cache, const float* __restrict__ scale,
    int row, int kvh, int lane, float& x0, float& x1, float& x2, float& x3) {
    if (scale == nullptr) {
        const uint64_t w = reinterpret_cast<const uint64_t*>(
            cache + static_cast<size_t>(row) * 1024u + kvh * 128)[lane];
        x0 = fp16_bits_to_f32(static_cast<uint16_t>(w & 0xFFFFu));
        x1 = fp16_bits_to_f32(static_cast<uint16_t>((w >> 16) & 0xFFFFu));
        x2 = fp16_bits_to_f32(static_cast<uint16_t>((w >> 32) & 0xFFFFu));
        x3 = fp16_bits_to_f32(static_cast<uint16_t>((w >> 48) & 0xFFFFu));
    } else {
        const uint32_t w = reinterpret_cast<const uint32_t*>(
            reinterpret_cast<const int8_t*>(cache) + static_cast<size_t>(row) * 1024u + kvh * 128)[lane];
        const float sc = scale[static_cast<size_t>(row) * 8u + kvh];
        x0 = __fmul_rn(static_cast<float>(static_cast<int8_t>(w & 0xFFu)), sc);
        x1 = __fmul_rn(static_cast<float>(static_cast<int8_t>((w >> 8) & 0xFFu)), sc);
        x2 = __fmul_rn(static_cast<float>(static_cast<int8_t>((w >> 16) & 0xFFu)), sc);
        x3 = __fmul_rn(static_cast<float>(static_cast<int8_t>((w >> 24) & 0xFFu)), sc);
    }
}

// Compile-time KV format for kernels instantiated per format (keeps FP16 register use unchanged).
template <bool kInt8>
__device__ __forceinline__ void kv_load4_t(
    const uint16_t* __restrict__ cache, const float* __restrict__ scale,
    int row, int kvh, int lane, float& x0, float& x1, float& x2, float& x3) {
    kv_load4(cache, kInt8 ? scale : nullptr, row, kvh, lane, x0, x1, x2, x3);
}

// Raw INT8 KV word (4 values) + its per-row/head scale, for software-pipelined loops.
__device__ __forceinline__ void kv_raw_i8(
    const uint16_t* __restrict__ cache, const float* __restrict__ scale,
    int row, int kvh, int lane, uint32_t& w, float& sc) {
    w = reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const int8_t*>(cache) + static_cast<size_t>(row) * 1024u + kvh * 128)[lane];
    sc = scale[static_cast<size_t>(row) * 8u + kvh];
}

// Same dequantization as `kv_load4` (INT8 branch).
__device__ __forceinline__ void kv_dequant_i8(
    uint32_t w, float sc, float& x0, float& x1, float& x2, float& x3) {
    x0 = __fmul_rn(static_cast<float>(static_cast<int8_t>(w & 0xFFu)), sc);
    x1 = __fmul_rn(static_cast<float>(static_cast<int8_t>((w >> 8) & 0xFFu)), sc);
    x2 = __fmul_rn(static_cast<float>(static_cast<int8_t>((w >> 16) & 0xFFu)), sc);
    x3 = __fmul_rn(static_cast<float>(static_cast<int8_t>((w >> 24) & 0xFFu)), sc);
}

__device__ __forceinline__ uint32_t kv_q8(float x, float inv) {
    int q = __float2int_rn(__fmul_rn(x, inv));
    q = q > 127 ? 127 : (q < -127 ? -127 : q);
    return static_cast<uint32_t>(q) & 0xFFu;
}

// Whole warp stores one KV head row (4 values per lane).
__device__ __forceinline__ void kv_store4(
    uint16_t* __restrict__ cache, float* __restrict__ scale,
    int row, int kvh, int lane, float x0, float x1, float x2, float x3) {
    if (scale == nullptr) {
        uint16_t* dst = cache + static_cast<size_t>(row) * 1024u + kvh * 128;
        dst[lane * 4] = f32_to_fp16_bits(x0);
        dst[lane * 4 + 1] = f32_to_fp16_bits(x1);
        dst[lane * 4 + 2] = f32_to_fp16_bits(x2);
        dst[lane * 4 + 3] = f32_to_fp16_bits(x3);
    } else {
        float a = fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3)));
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, m));
        // `rcp.approx` (deterministic, no IEEE-division subroutine call in the hot path).
        const float inv = (a > 0.0f) ? __fmul_rn(127.0f, ptx_rcp(a)) : 0.0f;
        const uint32_t packed = kv_q8(x0, inv) | (kv_q8(x1, inv) << 8) |
                                (kv_q8(x2, inv) << 16) | (kv_q8(x3, inv) << 24);
        reinterpret_cast<uint32_t*>(
            reinterpret_cast<int8_t*>(cache) + static_cast<size_t>(row) * 1024u + kvh * 128)[lane] = packed;
        if (lane == 0) scale[static_cast<size_t>(row) * 8u + kvh] = __fmul_rn(a, 1.0f / 127.0f);
    }
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
    __shared__ float smem_sq[6][8];
    __shared__ float smem_dq[6][8];
    __shared__ float smem_probs[6];
    __shared__ float smem_inv[6];
    __shared__ float smem_scores[6];

    const int q_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int total_states = n_blocks + append_cur;
    const float* q = queries + static_cast<size_t>(q_idx) * kD;
    const float inv_sqrt_d = 0.020833333333333332f; // 1.0 / 48.0

    float q_reg[9];
    #pragma unroll
    for (int t = 0; t < 9; ++t) {
        q_reg[t] = q[tid + t * 256];
    }

    for (int s = 0; s < total_states; ++s) {
        const float* st = (s < n_blocks) ? (blocks_buf + static_cast<size_t>(s) * kD) : cur_opt;
        float sq = 0.0f;
        float dq = 0.0f;
        #pragma unroll
        for (int t = 0; t < 9; ++t) {
            float v = st[tid + t * 256];
            sq += v * v;
            dq += v * q_reg[t];
        }
        sq = warp_sum(sq);
        dq = warp_sum(dq);
        if (lane == 0) {
            smem_sq[s][warp_id] = sq;
            smem_dq[s][warp_id] = dq;
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        for (int s = 0; s < total_states; ++s) {
            float total_sq = (lane < 8) ? smem_sq[s][lane] : 0.0f;
            float total_dq = (lane < 8) ? smem_dq[s][lane] : 0.0f;
            total_sq = warp_sum(total_sq);
            total_dq = warp_sum(total_dq);
            if (lane == 0) {
                float inv = ptx_rsqrt(total_sq * (1.0f / static_cast<float>(kD)) + 1e-6f);
                smem_inv[s] = inv;
                smem_scores[s] = total_dq * inv * inv_sqrt_d;
            }
        }
        if (lane == 0) {
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
                smem_probs[s] = smem_probs[s] * inv_z * smem_inv[s];
            }
        }
    }
    __syncthreads();

    float* dst = out_mixed + static_cast<size_t>(q_idx) * kD;
    #pragma unroll
    for (int t = 0; t < 9; ++t) {
        const int i = tid + t * 256;
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
__device__ __forceinline__ float accum_bf16_8(float acc, uint4 pack, float4 u, float4 v) {
    acc = __fmaf_rn(bf16_to_f32(static_cast<uint16_t>(pack.x & 0xFFFFu)), u.x, acc);
    acc = __fmaf_rn(bf16_to_f32(static_cast<uint16_t>(pack.x >> 16)), u.y, acc);
    acc = __fmaf_rn(bf16_to_f32(static_cast<uint16_t>(pack.y & 0xFFFFu)), u.z, acc);
    acc = __fmaf_rn(bf16_to_f32(static_cast<uint16_t>(pack.y >> 16)), u.w, acc);
    acc = __fmaf_rn(bf16_to_f32(static_cast<uint16_t>(pack.z & 0xFFFFu)), v.x, acc);
    acc = __fmaf_rn(bf16_to_f32(static_cast<uint16_t>(pack.z >> 16)), v.y, acc);
    acc = __fmaf_rn(bf16_to_f32(static_cast<uint16_t>(pack.w & 0xFFFFu)), v.z, acc);
    acc = __fmaf_rn(bf16_to_f32(static_cast<uint16_t>(pack.w >> 16)), v.w, acc);
    return acc;
}

__device__ __forceinline__ float eval_conv4_step(
    float4 w, float h0, float h1, float h2, float x, int has_dyn, float4 d, int do_res_add, float res) {
    float conv_o = __fmaf_rn(w.w, x, __fmaf_rn(w.z, h2, __fmaf_rn(w.y, h1, __fmul_rn(w.x, h0))));
    float inv_scale = 0.4472135954999579f;
    if (has_dyn) {
        float dyn_o = __fmaf_rn(d.w, h0, __fmaf_rn(d.z, h1, __fmaf_rn(d.y, h2, __fmul_rn(d.x, x))));
        conv_o = __fadd_rn(conv_o, dyn_o);
        inv_scale = 0.3333333333333333f;
    }
    float y = __fmul_rn(__fadd_rn(x, conv_o), inv_scale);
    if (do_res_add) {
        y = __fadd_rn(y, res);
    }
    return y;
}

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
        acc = accum_bf16_8(acc, w_row[c8], x4[c8 * 2], x4[c8 * 2 + 1]);
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
    const float4 d = has_dyn ? dyn_tap[c >> 7] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float res = do_res_add ? res_add[c] : 0.0f;

    h[0] = h1;
    h[1] = h2;
    h[2] = x;

    out[c] = eval_conv4_step(w, h0, h1, h2, x, has_dyn, d, do_res_add, res);
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
    float* __restrict__ k_scale,               // [kv_cap, 8] INT8 KV scales, or nullptr (FP16)
    float* __restrict__ v_scale,               // [kv_cap, 8] INT8 KV scales, or nullptr (FP16)
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
        uint32_t cp0 = qc[lane * 2];
        uint32_t cp1 = qc[lane * 2 + 1];
        float qv0 = qh[lane * 4] * bf16_to_f32(static_cast<uint16_t>(cp0 & 0xFFFFu));
        float qv1 = qh[lane * 4 + 1] * bf16_to_f32(static_cast<uint16_t>(cp0 >> 16));
        float qv2 = qh[lane * 4 + 2] * bf16_to_f32(static_cast<uint16_t>(cp1 & 0xFFFFu));
        float qv3 = qh[lane * 4 + 3] * bf16_to_f32(static_cast<uint16_t>(cp1 >> 16));
        float q_sq = warp_sum(__fadd_rn(
            __fmaf_rn(qv1, qv1, __fmul_rn(qv0, qv0)),
            __fmaf_rn(qv3, qv3, __fmul_rn(qv2, qv2))));
        float q_norm = ptx_sqrt(q_sq) + 1e-6f;
        float q_f = 22.627416997969522f * ptx_rcp(q_norm);
        if (q_f > 1.0f) q_f = 1.0f;
        q_f *= tau;
        qh[lane * 4] = qv0 * q_f;
        qh[lane * 4 + 1] = qv1 * q_f;
        qh[lane * 4 + 2] = qv2 * q_f;
        qh[lane * 4 + 3] = qv3 * q_f;

        // Relative query `r_h` (`16` dims -> lanes `0..7` hold 2 dims each)
        float* rh = r_buf + h * 16;
        const uint32_t* rc = r_clip_bf16 + h * 8;
        float rv0 = 0.0f, rv1 = 0.0f;
        if (lane < 8) {
            uint32_t rp = rc[lane];
            rv0 = rh[lane * 2] * bf16_to_f32(static_cast<uint16_t>(rp & 0xFFFFu));
            rv1 = rh[lane * 2 + 1] * bf16_to_f32(static_cast<uint16_t>(rp >> 16));
        }
        float r_sq = warp_sum(__fmaf_rn(rv1, rv1, __fmul_rn(rv0, rv0)));
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
        float kv0 = ks[lane * 4], kv1 = ks[lane * 4 + 1], kv2 = ks[lane * 4 + 2], kv3 = ks[lane * 4 + 3];
        float k_sq = warp_sum(__fadd_rn(
            __fmaf_rn(kv1, kv1, __fmul_rn(kv0, kv0)),
            __fmaf_rn(kv3, kv3, __fmul_rn(kv2, kv2))));
        float k_f = 22.627416997969522f * ptx_rcp(ptx_sqrt(k_sq) + 1e-6f);
        if (k_f > 1.0f) k_f = 1.0f;
        kv_store4(k_cache_fp16, k_scale, cache_row, kh, lane,
                  kv0 * k_f, kv1 * k_f, kv2 * k_f, kv3 * k_f);
    } else if (g_warp < 40) {
        // V head (`128` dims = `4` per lane, no-gamma RMSNorm)
        const int vh = g_warp - 32;
        const float* vs = v_in + vh * 128;
        float vv0 = vs[lane * 4], vv1 = vs[lane * 4 + 1], vv2 = vs[lane * 4 + 2], vv3 = vs[lane * 4 + 3];
        float v_sq = warp_sum(__fadd_rn(
            __fmaf_rn(vv1, vv1, __fmul_rn(vv0, vv0)),
            __fmaf_rn(vv3, vv3, __fmul_rn(vv2, vv2))));
        float v_inv = ptx_rsqrt(v_sq * (1.0f / 128.0f) + 1e-6f);
        kv_store4(v_cache_fp16, v_scale, cache_row, vh, lane,
                  vv0 * v_inv, vv1 * v_inv, vv2 * v_inv, vv3 * v_inv);
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
    const float* __restrict__ k_scale,          // [kv_cap, 8] INT8 KV scales, or nullptr (FP16)
    const float* __restrict__ v_scale,          // [kv_cap, 8] INT8 KV scales, or nullptr (FP16)
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
        float k0, k1, k2, k3;
        kv_load4(k_cache_fp16, k_scale, slot_row, kvh, lane, k0, k1, k2, k3);

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

        float v0, v1, v2, v3;
        kv_load4(v_cache_fp16, v_scale, slot_row, kvh, lane, v0, v1, v2, v3);
        o0 = o0 * alpha + p * v0;
        o1 = o1 * alpha + p * v1;
        o2 = o2 * alpha + p * v2;
        o3 = o3 * alpha + p * v3;
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
// Keep the raw entry points compatible with existing hosts. Each packed entry
// reconstructs four original scale bytes before the existing arithmetic.
#define INK2_SCALE_KERNEL(name, params, args) \
    template<bool PackedScales> __device__ __forceinline__ void name##_impl params; \
    extern "C" __global__ void name params { name##_impl<false> args; } \
    extern "C" __global__ void name##_packed_scales params { name##_impl<true> args; } \
    template<bool PackedScales> __device__ __forceinline__ void name##_impl params

template<bool PackedScales, int Bits, unsigned Base>
__device__ __forceinline__ unsigned expert_scale_word(const unsigned* scales, unsigned index) {
    if constexpr (!PackedScales) return scales[index];
    // The final zero word makes the second aligned load valid at the end.
    const unsigned bit = index * (4u * Bits);
    const unsigned word = bit >> 5;
    const unsigned fields = __funnelshift_r(scales[word], scales[word + 1], bit & 31);
    constexpr unsigned mask = (1u << Bits) - 1;
    return Base * 0x01010101u | (fields & mask)
        | (((fields >> Bits) & mask) << 8)
        | (((fields >> (2 * Bits)) & mask) << 16)
        | (((fields >> (3 * Bits)) & mask) << 24);
}

template<bool PackedScales>
__device__ __forceinline__ float warp_dot_nvfp4_row(
    const uint2* __restrict__ row_packed_u64,  // [n_blocks16] (8 bytes = 16 nibbles/blk)
    const uint32_t* __restrict__ row_scale_u32, unsigned scale_start, // [n_blocks16 / 4]
    const float4* __restrict__ x4,             // [cols / 4]
    int n_blocks16,
    int lane) {
    float acc = 0.0f;
    for (int blk = lane; blk < n_blocks16; blk += 32) {
        uint2 pk = row_packed_u64[blk];
        uint32_t sc4 = expert_scale_word<PackedScales, 6, 64>(row_scale_u32, scale_start + (blk >> 2));
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

INK2_SCALE_KERNEL(ink2_moe_nvfp4_up_situ, (
    const uint2* __restrict__ w13_u64,     // [128, 2304, 144]
    const uint32_t* __restrict__ s13_u32,  // [128, 2304, 36]
    const float* __restrict__ a13,         // [128]
    const float4* __restrict__ x4,         // [2304 / 4]
    const int32_t* __restrict__ sel_ids,   // [8]
    float* __restrict__ act_out),
    (w13_u64, s13_u32, a13, x4, sel_ids, act_out)) {         // [8, 1152]
    __shared__ float smem_gu[8];
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int k = blockIdx.y; // 0..7
    const int m = blockIdx.x * 4 + (warp_id >> 1); // 0..1151
    const int is_up = warp_id & 1;
    const int r = m * 2 + is_up;
    const int ex = sel_ids[k];

    const size_t row_idx = static_cast<size_t>(ex) * (2u * kFf) + r;
    float dot = warp_dot_nvfp4_row<PackedScales>(
        w13_u64 + row_idx * 144u,
        s13_u32, row_idx * 36u,
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

INK2_SCALE_KERNEL(ink2_moe_nvfp4_down, (
    const uint2* __restrict__ w2_u64,      // [128, 2304, 72]
    const uint32_t* __restrict__ s2_u32,   // [128, 2304, 18]
    const float* __restrict__ a2,          // [128]
    const float4* __restrict__ act4,       // [8, 1152 / 4 = 288]
    const int32_t* __restrict__ sel_ids,   // [8]
    const float* __restrict__ sel_wts,     // [8]
    float* __restrict__ moe_out),
    (w2_u64, s2_u32, a2, act4, sel_ids, sel_wts, moe_out)) {         // [2304]
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp_id;
    if (row >= kD) return;

    float sum_y = 0.0f;
    for (int k = 0; k < kTopK; ++k) {
        const int ex = sel_ids[k];
        const size_t r_idx = static_cast<size_t>(ex) * kD + row;
        float dot = warp_dot_nvfp4_row<PackedScales>(
            w2_u64 + r_idx * 72u,
            s2_u32, r_idx * 18u,
            act4 + k * 288,
            72,
            lane);
        sum_y = __fmaf_rn(__fmul_rn(dot, a2[ex] * (1.0f / 2688.0f)), sel_wts[k], sum_y);
    }
    if (lane == 0) {
        moe_out[row] = sum_y;
    }
}

// ----------------------------------------------------------------------------
// 11. Unit-Stride `512B` Coalesced MXFP8 (`layers 20..23`) MoE (`<= 42` regs!)
//     `Grid (288, 8)` & `Grid 288` (`100%` SM120 occupancy across all SMs + `512B` PCIe burst!)
// ----------------------------------------------------------------------------
template <int Stride = 1>
__device__ __forceinline__ float dot_mxfp8_u128_exact(uint4 p, const float4* __restrict__ x4_sub) {
    float sub = 0.0f;
    float4 xv0 = x4_sub[0];
    sub = __fmaf_rn(e4m3_byte_to_f32(p.x & 0xFFu), xv0.x, sub);
    sub = __fmaf_rn(e4m3_byte_to_f32((p.x >> 8) & 0xFFu), xv0.y, sub);
    sub = __fmaf_rn(e4m3_byte_to_f32((p.x >> 16) & 0xFFu), xv0.z, sub);
    sub = __fmaf_rn(e4m3_byte_to_f32((p.x >> 24) & 0xFFu), xv0.w, sub);
    float4 xv1 = x4_sub[Stride];
    sub = __fmaf_rn(e4m3_byte_to_f32(p.y & 0xFFu), xv1.x, sub);
    sub = __fmaf_rn(e4m3_byte_to_f32((p.y >> 8) & 0xFFu), xv1.y, sub);
    sub = __fmaf_rn(e4m3_byte_to_f32((p.y >> 16) & 0xFFu), xv1.z, sub);
    sub = __fmaf_rn(e4m3_byte_to_f32((p.y >> 24) & 0xFFu), xv1.w, sub);
    float4 xv2 = x4_sub[2 * Stride];
    sub = __fmaf_rn(e4m3_byte_to_f32(p.z & 0xFFu), xv2.x, sub);
    sub = __fmaf_rn(e4m3_byte_to_f32((p.z >> 8) & 0xFFu), xv2.y, sub);
    sub = __fmaf_rn(e4m3_byte_to_f32((p.z >> 16) & 0xFFu), xv2.z, sub);
    sub = __fmaf_rn(e4m3_byte_to_f32((p.z >> 24) & 0xFFu), xv2.w, sub);
    float4 xv3 = x4_sub[3 * Stride];
    sub = __fmaf_rn(e4m3_byte_to_f32(p.w & 0xFFu), xv3.x, sub);
    sub = __fmaf_rn(e4m3_byte_to_f32((p.w >> 8) & 0xFFu), xv3.y, sub);
    sub = __fmaf_rn(e4m3_byte_to_f32((p.w >> 16) & 0xFFu), xv3.z, sub);
    sub = __fmaf_rn(e4m3_byte_to_f32((p.w >> 24) & 0xFFu), xv3.w, sub);
    return sub;
}

template<bool PackedScales>
__device__ __forceinline__ float warp_dot_mxfp8_u128_row(
    const uint4* __restrict__ row_fp8_u128,     // [n_u128]
    const uint32_t* __restrict__ row_e8m0_u32, unsigned scale_start, // [n_u128 / 8]
    const float4* __restrict__ x4,              // [n_u128 * 4]
    int n_u128,
    int lane) {
    float acc = 0.0f;
    for (int u16 = lane; u16 < n_u128; u16 += 32) {
        const int blk32 = u16 >> 1;
        uint32_t s4 = expert_scale_word<PackedScales, 3, 112>(row_e8m0_u32, scale_start + (blk32 >> 2));
        uint32_t s_e8m0 = (s4 >> ((blk32 & 3) * 8)) & 0xFFu;
        float exp_scale = __uint_as_float(s_e8m0 << 23);
        acc = __fmaf_rn(dot_mxfp8_u128_exact(row_fp8_u128[u16], x4 + u16 * 4), exp_scale, acc);
    }
    return warp_sum(acc);
}

INK2_SCALE_KERNEL(ink2_moe_mxfp8_up_situ, (
    const uint4* __restrict__ w13_u128,    // [128, 2304, 144]
    const uint32_t* __restrict__ s13_u32,  // [128, 2304, 18]
    const float4* __restrict__ x4,         // [576]
    const int32_t* __restrict__ sel_ids,   // [8]
    float* __restrict__ act_out),
    (w13_u128, s13_u32, x4, sel_ids, act_out)) {         // [8, 1152]
    __shared__ float smem_gu[8];
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int k = blockIdx.y;
    const int m = blockIdx.x * 4 + (warp_id >> 1);
    const int is_up = warp_id & 1;
    const int r = m * 2 + is_up;
    const int ex = sel_ids[k];

    const size_t row_idx = static_cast<size_t>(ex) * (2u * kFf) + r;
    float dot = warp_dot_mxfp8_u128_row<PackedScales>(
        w13_u128 + row_idx * 144u,
        s13_u32, row_idx * 18u,
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

INK2_SCALE_KERNEL(ink2_moe_mxfp8_down, (
    const uint4* __restrict__ w2_u128,     // [128, 2304, 72]
    const uint32_t* __restrict__ s2_u32,   // [128, 2304, 9]
    const float4* __restrict__ act4,       // [8, 288]
    const int32_t* __restrict__ sel_ids,   // [8]
    const float* __restrict__ sel_wts,     // [8]
    float* __restrict__ moe_out),
    (w2_u128, s2_u32, act4, sel_ids, sel_wts, moe_out)) {         // [2304]
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp_id;
    if (row >= kD) return;

    float sum_y = 0.0f;
    for (int k = 0; k < kTopK; ++k) {
        const int ex = sel_ids[k];
        const size_t r_idx = static_cast<size_t>(ex) * kD + row;
        float dot = warp_dot_mxfp8_u128_row<PackedScales>(
            w2_u128 + r_idx * 72u,
            s2_u32, r_idx * 9u,
            act4 + k * 288,
            72,
            lane);
        sum_y = __fmaf_rn(dot, sel_wts[k], sum_y);
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

extern "C" __global__ void ink2_quant_q8_16_packed(
    const float* __restrict__ x,       // [n_vecs, cols]
    uint32_t* __restrict__ out_q8_u32, // [n_vecs, cols / 4]
    float* __restrict__ out_sc16,      // [n_vecs, cols / 16]
    int32_t cols) {
    // Each half warp quantizes one independent 16-value group.
    const int blk16 = (blockIdx.x * blockDim.x + threadIdx.x) / 16;
    const int lane = threadIdx.x & 15;
    const int vec_id = blockIdx.y;
    const unsigned active = __ballot_sync(0xffffffffu, blk16 < cols / 16);
    if (blk16 >= cols / 16) return;
    float v = x[vec_id * cols + blk16 * 16 + lane];
    float amax = fabsf(v);
    #pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(active, amax, mask, 16));
    }
    float sc = amax * (1.0f / 127.0f);
    float inv = (amax > 0.0f) ? (127.0f * ptx_rcp(amax)) : 0.0f;
    int q = __float2int_rn(v * inv);
    if (q < -127) q = -127;
    if (q > 127) q = 127;
    uint32_t qb = static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(q)));
    uint32_t p4 = qb
        | (__shfl_sync(active, qb, lane + 1, 16) << 8)
        | (__shfl_sync(active, qb, lane + 2, 16) << 16)
        | (__shfl_sync(active, qb, lane + 3, 16) << 24);
    if ((lane & 3) == 0) {
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
        acc0 = __fmaf_rn(static_cast<float>(d0), __fmul_rn(s_r0[c16 >> 2], xsc), acc0);

        if (has_row1) {
            uint4 w1 = w_r1[c16];
            int d1 = __dp4a(static_cast<int>(w1.x), static_cast<int>(xq.x), 0);
            d1 = __dp4a(static_cast<int>(w1.y), static_cast<int>(xq.y), d1);
            d1 = __dp4a(static_cast<int>(w1.z), static_cast<int>(xq.z), d1);
            d1 = __dp4a(static_cast<int>(w1.w), static_cast<int>(xq.w), d1);
            acc1 = __fmaf_rn(static_cast<float>(d1), __fmul_rn(s_r1[c16 >> 2], xsc), acc1);
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

template<bool PackedScales>
__device__ __forceinline__ float warp_dot_nvfp4_dp4a_u128_row(
    const uint4* __restrict__ row_u128,         // [n_u128] (`16B` = `32` nibbles = `2` blocks of 16!)
    const uint32_t* __restrict__ row_scale_u32, unsigned scale_start, // [n_u128 / 2]
    const uint4* __restrict__ smem_xq16,        // [n_u128 * 2]
    const float* __restrict__ smem_xsc16,       // [n_u128 * 2]
    int n_u128,
    int lane) {
    float acc = 0.0f;
    for (int u16 = lane; u16 < n_u128; u16 += 32) {
        uint4 pk16 = row_u128[u16];
        uint32_t sc4 = expert_scale_word<PackedScales, 6, 64>(row_scale_u32, scale_start + (u16 >> 1));
        int sh = (u16 & 1) * 16;
        float s0 = __fmul_rn(e4m3_byte_to_f32((sc4 >> sh) & 0xFFu) * 0.5f, smem_xsc16[u16 * 2]);
        float s1 = __fmul_rn(e4m3_byte_to_f32((sc4 >> (sh + 8)) & 0xFFu) * 0.5f, smem_xsc16[u16 * 2 + 1]);

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
        acc = __fmaf_rn(static_cast<float>(d1), s1, __fmaf_rn(static_cast<float>(d0), s0, acc));
    }
    return warp_sum(acc);
}

INK2_SCALE_KERNEL(ink2_moe_nvfp4_dp4a_up_situ, (
    const uint4* __restrict__ w13_u128,    // [128, 2304, 72]
    const uint32_t* __restrict__ s13_u32,  // [128, 2304, 36]
    const float* __restrict__ a13,         // [128]
    const uint4* __restrict__ x_q8_u128,   // [144]
    const float* __restrict__ x_sc16,      // [144]
    const int32_t* __restrict__ sel_ids,   // [8]
    float* __restrict__ act_out),
    (w13_u128, s13_u32, a13, x_q8_u128, x_sc16, sel_ids, act_out)) {         // [8, 1152]
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
    float dot = warp_dot_nvfp4_dp4a_u128_row<PackedScales>(
        w13_u128 + row_idx * 72u,
        s13_u32, row_idx * 36u,
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

INK2_SCALE_KERNEL(ink2_moe_nvfp4_dp4a_down, (
    const uint4* __restrict__ w2_u128,     // [128, 2304, 36]
    const uint32_t* __restrict__ s2_u32,   // [128, 2304, 18]
    const float* __restrict__ a2,          // [128]
    const uint4* __restrict__ act_q8_u128, // [8, 72]
    const float* __restrict__ act_sc16,    // [8, 72]
    const int32_t* __restrict__ sel_ids,   // [8]
    const float* __restrict__ sel_wts,     // [8]
    float* __restrict__ moe_out),
    (w2_u128, s2_u32, a2, act_q8_u128, act_sc16, sel_ids, sel_wts, moe_out)) {         // [2304]
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
        float dot = warp_dot_nvfp4_dp4a_u128_row<PackedScales>(
            w2_u128 + r_idx * 36u,
            s2_u32, r_idx * 18u,
            smem_aq + k * 72,
            smem_asc + k * 72,
            36,
            lane);
        sum_y = __fmaf_rn(__fmul_rn(dot, a2[ex] * (1.0f / 2688.0f)), sel_wts[k], sum_y);
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
//     - Planar `Q8_64` 16-row x 8-token integer MMA GEMM (`ink2_q8_dp4a_gemm_b`)
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
    __shared__ float smem_sq[6][8];
    __shared__ float smem_dq[6][8];
    __shared__ float smem_probs[6];
    __shared__ float smem_inv[6];
    __shared__ float smem_scores[6];

    const int q_idx = blockIdx.x;
    const int b_out = blockIdx.y;
    const int b_src = b_offset + b_out;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int total_states = n_blocks + append_cur;
    const float* q = queries + static_cast<size_t>(q_idx) * kD;
    const float inv_sqrt_d = 0.020833333333333332f;

    float q_reg[9];
    #pragma unroll
    for (int t = 0; t < 9; ++t) {
        q_reg[t] = q[tid + t * 256];
    }

    for (int s = 0; s < total_states; ++s) {
        const float* st = (s < n_blocks)
            ? (blocks_buf + (static_cast<size_t>(s) * stride_B + b_src) * kD)
            : (cur_opt + static_cast<size_t>(b_src) * kD);
        float sq = 0.0f;
        float dq = 0.0f;
        #pragma unroll
        for (int t = 0; t < 9; ++t) {
            float v = st[tid + t * 256];
            sq += v * v;
            dq += v * q_reg[t];
        }
        sq = warp_sum(sq);
        dq = warp_sum(dq);
        if (lane == 0) {
            smem_sq[s][warp_id] = sq;
            smem_dq[s][warp_id] = dq;
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        for (int s = 0; s < total_states; ++s) {
            float total_sq = (lane < 8) ? smem_sq[s][lane] : 0.0f;
            float total_dq = (lane < 8) ? smem_dq[s][lane] : 0.0f;
            total_sq = warp_sum(total_sq);
            total_dq = warp_sum(total_dq);
            if (lane == 0) {
                float inv = ptx_rsqrt(total_sq * (1.0f / static_cast<float>(kD)) + 1e-6f);
                smem_inv[s] = inv;
                smem_scores[s] = total_dq * inv * inv_sqrt_d;
            }
        }
        if (lane == 0) {
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
                smem_probs[s] = smem_probs[s] * inv_z * smem_inv[s];
            }
        }
    }
    __syncthreads();

    float* dst = out_mixed + (static_cast<size_t>(q_idx) * B_out + b_out) * kD;
    const size_t plane_stride = static_cast<size_t>(B_out) * kD;
    #pragma unroll
    for (int t = 0; t < 9; ++t) {
        const int i = tid + t * 256;
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

template<int Rows>
__device__ __forceinline__ void ctrl_down_rows(
    const float* __restrict__ cur_base,       // [n_ctrls, B, D]
    const float* __restrict__ msg_base,       // [n_ctrls, B, D]
    const uint4* __restrict__ down_bf16_u128, // [n_ctrls, 64, 576]
    float* __restrict__ g_out_base,           // [n_ctrls, B, 64]
    int32_t B) {
    __shared__ float smem_red_a[8];
    __shared__ float smem_red_b[8];
    __shared__ float inv_ab[2];

    __shared__ float cur_tile[Rows == 1 ? 1 : kD];
    __shared__ float msg_tile[Rows == 1 ? 1 : kD];
    const int first_row = blockIdx.x * Rows;
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
        if (Rows > 1) {
            cur_tile[i] = va;
            msg_tile[i] = vb;
        }
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

    // Reuse one normalized input across Rows rank projections. Each row keeps
    // the scalar kernel's accumulation order and its eight-warp reduction tree.
    __shared__ float partial[Rows][8];
    float dot[Rows] = {};
    for (int c8 = tid; c8 < 576; c8 += 256) {
        const int offset = c8 * 8;
        const float* src = offset < kD
            ? (Rows > 1 ? cur_tile : cur) + offset
            : (Rows > 1 ? msg_tile : msg) + (offset - kD);
        const float scale = inv_ab[offset < kD ? 0 : 1];
        #pragma unroll
        for (int row = 0; row < Rows; ++row) {
            size_t rank = static_cast<size_t>(c_idx) * 64 + first_row + row;
            uint4 pack = down_bf16_u128[rank * 576 + c8];
            uint32_t words[4] = {pack.x, pack.y, pack.z, pack.w};
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                float w0 = bf16_to_f32(static_cast<uint16_t>(words[p]));
                float w1 = bf16_to_f32(static_cast<uint16_t>(words[p] >> 16));
                dot[row] += (w0 * src[2 * p] + w1 * src[2 * p + 1]) * scale;
            }
        }
    }
    #pragma unroll
    for (int row = 0; row < Rows; ++row) {
        float sum = warp_sum(dot[row]);
        if (lane == 0) partial[row][warp_id] = sum;
    }
    __syncthreads();
    if (warp_id < Rows) {
        float sum = lane < 8 ? partial[warp_id][lane] : 0.0f;
        sum = warp_sum(sum);
        if (lane == 0) g_out_base[cb * 64 + first_row + warp_id] = sum;
    }
}

extern "C" __global__ void ink2_ctrl_down_b(
    const float* __restrict__ cur_base,       // [n_ctrls, B, D]
    const float* __restrict__ msg_base,       // [n_ctrls, B, D]
    const uint4* __restrict__ down_bf16_u128, // [n_ctrls, 64, 576]
    float* __restrict__ g_out_base,           // [n_ctrls, B, 64]
    int32_t B) {
    ctrl_down_rows<1>(cur_base, msg_base, down_bf16_u128, g_out_base, B);
}

extern "C" __global__ void ink2_ctrl_down_r8_b(
    const float* __restrict__ cur_base,       // [n_ctrls, B, D]
    const float* __restrict__ msg_base,       // [n_ctrls, B, D]
    const uint4* __restrict__ down_bf16_u128, // [n_ctrls, 64, 576]
    float* __restrict__ g_out_base,           // [n_ctrls, B, 64]
    int32_t B) {
    ctrl_down_rows<8>(cur_base, msg_base, down_bf16_u128, g_out_base, B);
}

extern "C" __global__ void ink2_ctrl_down_r8_t2_b(
    const float* __restrict__ cur_base,       // [n_ctrls, B, D]
    const float* __restrict__ msg_base,       // [n_ctrls, B, D]
    const uint4* __restrict__ down_bf16_u128, // [n_ctrls, 64, 576]
    float* __restrict__ g_out_base,           // [n_ctrls, B, 64]
    int32_t B) {
    constexpr int Rows = 8, Tokens = 2;
    __shared__ float norm_sums[Tokens][2][8];
    __shared__ float inv[Tokens][2];
    __shared__ float cur_tile[Tokens][kD];
    __shared__ float msg_tile[Tokens][kD];
    __shared__ float partial[Tokens][Rows][8];
    const int tiles = (B + Tokens - 1) / Tokens;
    const int ctrl = blockIdx.y / tiles;
    const int first_token = (blockIdx.y % tiles) * Tokens;
    const int cb = ctrl * B + first_token;
    const int first_row = blockIdx.x * Rows;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    #pragma unroll
    for (int t = 0; t < Tokens; ++t) {
        // A partial tile reuses its first valid input for the unwritten lane.
        const size_t row = cb + (first_token + t < B ? t : 0);
        const float* cur = cur_base + row * kD;
        const float* msg = msg_base + row * kD;
        float sq_cur = 0.0f, sq_msg = 0.0f;
        for (int i = tid; i < kD; i += 256) {
            const float c = cur[i], m = msg[i];
            cur_tile[t][i] = c;
            msg_tile[t][i] = m;
            sq_cur += c * c;
            sq_msg += m * m;
        }
        sq_cur = warp_sum(sq_cur);
        sq_msg = warp_sum(sq_msg);
        if (lane == 0) {
            norm_sums[t][0][warp] = sq_cur;
            norm_sums[t][1][warp] = sq_msg;
        }
    }
    __syncthreads();
    if (warp < Tokens * 2) {
        const int t = warp / 2, source = warp % 2;
        float sum = warp_sum(lane < 8 ? norm_sums[t][source][lane] : 0.0f);
        if (lane == 0) inv[t][source] = ptx_rsqrt(sum * (1.0f / static_cast<float>(kD)) + 1e-6f);
    }
    __syncthreads();

    // Rank rows share the input. Tokens share the converted weights. The inner
    // product order and the two warp reductions match the single-token path.
    float dot[Tokens][Rows] = {};
    for (int c8 = tid; c8 < 576; c8 += 256) {
        const int source = c8 * 8 >= kD;
        const int offset = c8 * 8 - source * kD;
        #pragma unroll
        for (int row = 0; row < Rows; ++row) {
            const size_t rank = static_cast<size_t>(ctrl) * 64 + first_row + row;
            const uint4 pack = down_bf16_u128[rank * 576 + c8];
            const uint32_t words[4] = {pack.x, pack.y, pack.z, pack.w};
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                float w0 = bf16_to_f32(static_cast<uint16_t>(words[p]));
                float w1 = bf16_to_f32(static_cast<uint16_t>(words[p] >> 16));
                #pragma unroll
                for (int t = 0; t < Tokens; ++t) {
                    const float* src = (source ? msg_tile[t] : cur_tile[t]) + offset;
                    dot[t][row] += (w0 * src[2 * p] + w1 * src[2 * p + 1]) * inv[t][source];
                }
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < Tokens; ++t) {
        #pragma unroll
        for (int row = 0; row < Rows; ++row) {
            float sum = warp_sum(dot[t][row]);
            if (lane == 0) partial[t][row][warp] = sum;
        }
    }
    __syncthreads();
    if (warp < Rows) {
        #pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            float sum = warp_sum(lane < 8 ? partial[t][warp][lane] : 0.0f);
            if (lane == 0 && first_token + t < B) {
                g_out_base[(cb + t) * 64 + first_row + warp] = sum;
            }
        }
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

// Grid: (9, n_ctrls * ceil(B / 8)), Block: 256
extern "C" __global__ void ink2_ctrl_apply_t8_b(
    float* __restrict__ cur_inout_base,       // [n_ctrls, B, D]
    const float* __restrict__ msg_base,       // [n_ctrls, B, D]
    const float* __restrict__ g_vec_base,     // [n_ctrls, B, 64]
    const uint4* __restrict__ gain_bf16_u128, // [n_ctrls, D, 8]
    const uint4* __restrict__ forget_bf16_u128,
    int32_t B) {
    constexpr int Tokens = 8;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int tiles = (B + Tokens - 1) / Tokens;
    const int ctrl = blockIdx.y / tiles;
    const int first = (blockIdx.y % tiles) * Tokens;
    const int cb = ctrl * B + first;
    __shared__ float gates[Tokens * 64];
    for (int j = threadIdx.x; j < Tokens * 64; j += blockDim.x) {
        gates[j] = first + j / 64 < B ? g_vec_base[cb * 64 + j] : 0.0f;
    }
    __syncthreads();
    if (i >= kD) return;

    // Convert each BF16 weight pair once for all tokens in this tile.
    const size_t row_idx = static_cast<size_t>(ctrl) * kD + i;
    float gain[Tokens] = {};
    float forget[Tokens] = {};
    #pragma unroll 2
    for (int c8 = 0; c8 < 8; ++c8) {
        uint4 gp = gain_bf16_u128[row_idx * 8 + c8];
        uint4 fp = forget_bf16_u128[row_idx * 8 + c8];
        uint32_t gu[4] = {gp.x, gp.y, gp.z, gp.w};
        uint32_t fu[4] = {fp.x, fp.y, fp.z, fp.w};
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            float g0 = bf16_to_f32(static_cast<uint16_t>(gu[p]));
            float g1 = bf16_to_f32(static_cast<uint16_t>(gu[p] >> 16));
            float f0 = bf16_to_f32(static_cast<uint16_t>(fu[p]));
            float f1 = bf16_to_f32(static_cast<uint16_t>(fu[p] >> 16));
            #pragma unroll
            for (int t = 0; t < Tokens; ++t) {
                int j = t * 64 + c8 * 8 + 2 * p;
                // The scalar projection rounds the high-half product before
                // fusing the low-half product. Preserve that rounding for every token.
                gain[t] += __fmaf_rn(g0, gates[j], __fmul_rn(g1, gates[j + 1]));
                forget[t] += __fmaf_rn(f0, gates[j], __fmul_rn(f1, gates[j + 1]));
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < Tokens; ++t) {
        if (first + t < B) {
            size_t idx = static_cast<size_t>(cb + t) * kD + i;
            float c = cur_inout_base[idx];
            cur_inout_base[idx] = c - ptx_tanh(forget[t]) * c + ptx_tanh(gain[t]) * msg_base[idx];
        }
    }
}

// Four signed DP4A instructions cover one packed 16-element dot product.
__device__ __forceinline__ int dp4a_u128(uint4 w, uint4 x) {
    int d = __dp4a(static_cast<int>(w.x), static_cast<int>(x.x), 0);
    d = __dp4a(static_cast<int>(w.y), static_cast<int>(x.y), d);
    d = __dp4a(static_cast<int>(w.z), static_cast<int>(x.z), d);
    return __dp4a(static_cast<int>(w.w), static_cast<int>(x.w), d);
}

// Signed integer partials for two row fragments and two token columns.
__device__ __forceinline__ void mma_s8_k16(
    unsigned a0, unsigned a1, unsigned b, int& d0, int& d1, int& d2, int& d3) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3},{%4,%5},{%6},{0,0,0,0};"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(a0), "r"(a1), "r"(b));
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
    // A warp's four threads per row cover one K=16 operand fragment. Each
    // block computes 16 weight rows by 8 tokens, including partial edge tiles.
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int fragment = lane % 4;
    const int first_row = blockIdx.x * 16;
    const int first_token = blockIdx.y * 8;
    const int row0 = min(first_row + lane / 4, rows - 1);
    const int row1 = min(first_row + lane / 4 + 8, rows - 1);
    const int token = min(first_token + lane / 4, B - 1);
    const int cols16 = cols / 16;
    const int cols64 = cols / 64;
    const auto* top = reinterpret_cast<const uint32_t*>(w_q8_u128 + static_cast<size_t>(row0) * cols16);
    const auto* bottom = reinterpret_cast<const uint32_t*>(w_q8_u128 + static_cast<size_t>(row1) * cols16);
    const auto* inputs = reinterpret_cast<const uint32_t*>(x_q8_u128 + static_cast<size_t>(token) * cols16);
    const float* top_scales = w_sc64 + static_cast<size_t>(row0) * cols64;
    const float* bottom_scales = w_sc64 + static_cast<size_t>(row1) * cols64;
    const float* input_scales = x_sc64 + static_cast<size_t>(token) * cols64;
    __shared__ float partials[8][128];
    float sum[4] = {};
    float left[4] = {};

    // Keep this loop rolled to bound the instruction footprint. The four
    // virtual lanes reproduce the original DP4A warp's FP32 reduction order.
    #pragma unroll 1
    for (int step = 0; step < 4; ++step) {
        float acc[4] = {};
        const int k_lane = warp + (step % 2) * 16 + (step / 2) * 8;
        for (int k = k_lane; k < cols16; k += 32) {
            const unsigned a0 = top[k * 4 + fragment];
            const unsigned a1 = bottom[k * 4 + fragment];
            const unsigned b = inputs[k * 4 + fragment];
            int d0, d1, d2, d3;
            mma_s8_k16(a0, a1, b, d0, d1, d2, d3);
            const float w0 = top_scales[k / 4];
            const float w1 = bottom_scales[k / 4];
            const float xs = input_scales[k / 4];
            const float x0 = __shfl_sync(0xffffffffu, xs, 8 * fragment);
            const float x1 = __shfl_sync(0xffffffffu, xs, 8 * fragment + 4);
            acc[0] = __fmaf_rn(float(d0), __fmul_rn(w0, x0), acc[0]);
            acc[1] = __fmaf_rn(float(d1), __fmul_rn(w0, x1), acc[1]);
            acc[2] = __fmaf_rn(float(d2), __fmul_rn(w1, x0), acc[2]);
            acc[3] = __fmaf_rn(float(d3), __fmul_rn(w1, x1), acc[3]);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (step % 2 == 0) {
                left[i] = acc[i];
            } else if (step == 1) {
                sum[i] = left[i] + acc[i];
            } else {
                sum[i] += left[i] + acc[i];
            }
        }
    }
    const int dst = (lane / 4) * 8 + 2 * fragment;
    partials[warp][dst] = sum[0];
    partials[warp][dst + 1] = sum[1];
    partials[warp][dst + 64] = sum[2];
    partials[warp][dst + 65] = sum[3];
    __syncthreads();

    const int out_row = first_row + tid / 8;
    const int out_token = first_token + tid % 8;
    if (tid < 128 && out_row < rows && out_token < B) {
        const float s0 = partials[0][tid] + partials[4][tid];
        const float s1 = partials[1][tid] + partials[5][tid];
        const float s2 = partials[2][tid] + partials[6][tid];
        const float s3 = partials[3][tid] + partials[7][tid];
        float value = (s0 + s2) + (s1 + s3);
        if (softcap > 0.0f) value = softcap * ptx_tanh(value * ptx_rcp(softcap));
        y_base[static_cast<size_t>(out_token) * y_stride + out_row] = value;
    }
}

// Register-Persistent Causal `short_conv4` over `b = 0..B-1` (`0-ULP` bit-identical!)
// Optional `seq_out` (`[3 + B, channels]`) records `[hist3; x_0..x_{B-1}]` for `O(1)` speculative rewind!
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
    float* __restrict__ seq_out,             // nullable (`[3 + B, channels]`)
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
    if (seq_out != nullptr) {
        seq_out[c] = h0;
        seq_out[channels + c] = h1;
        seq_out[2 * channels + c] = h2;
    }
    const float4 w = w_conv4[c];
    const int head = c >> 7;

    for (int b = 0; b < B; ++b) {
        const float x = cur_base[static_cast<size_t>(b) * in_stride + c];
        if (seq_out != nullptr) {
            seq_out[static_cast<size_t>(3 + b) * channels + c] = x;
        }
        const float4 d = has_dyn ? dyn_base[static_cast<size_t>(b) * dyn_f4_stride + head] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float res = do_res_add ? res_base[static_cast<size_t>(b) * res_stride + c] : 0.0f;
        const float y = eval_conv4_step(w, h0, h1, h2, x, has_dyn, d, do_res_add, res);
        h0 = h1;
        h1 = h2;
        h2 = x;
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
    float* __restrict__ k_scale,               // [kv_cap, 8] INT8 KV scales, or nullptr (FP16)
    float* __restrict__ v_scale,               // [kv_cap, 8] INT8 KV scales, or nullptr (FP16)
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
        float q_sq = warp_sum(__fadd_rn(
            __fmaf_rn(qv1, qv1, __fmul_rn(qv0, qv0)),
            __fmaf_rn(qv3, qv3, __fmul_rn(qv2, qv2))));
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
        float r_sq = warp_sum(__fmaf_rn(rv1, rv1, __fmul_rn(rv0, rv0)));
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
        float k_sq = warp_sum(__fadd_rn(
            __fmaf_rn(kv1, kv1, __fmul_rn(kv0, kv0)),
            __fmaf_rn(kv3, kv3, __fmul_rn(kv2, kv2))));
        float k_f = 22.627416997969522f * ptx_rcp(ptx_sqrt(k_sq) + 1e-6f);
        if (k_f > 1.0f) k_f = 1.0f;
        kv_store4(k_cache_fp16, k_scale, cache_row, kh, lane,
                  kv0 * k_f, kv1 * k_f, kv2 * k_f, kv3 * k_f);
    } else if (g_warp < 40) {
        const int vh = g_warp - 32;
        const float* vs = v_in + vh * 128;
        float vv0 = vs[lane * 4], vv1 = vs[lane * 4 + 1], vv2 = vs[lane * 4 + 2], vv3 = vs[lane * 4 + 3];
        float v_sq = warp_sum(__fadd_rn(
            __fmaf_rn(vv1, vv1, __fmul_rn(vv0, vv0)),
            __fmaf_rn(vv3, vv3, __fmul_rn(vv2, vv2))));
        float v_inv = ptx_rsqrt(v_sq * (1.0f / 128.0f) + 1e-6f);
        kv_store4(v_cache_fp16, v_scale, cache_row, vh, lane,
                  vv0 * v_inv, vv1 * v_inv, vv2 * v_inv, vv3 * v_inv);
    }
}

// Grid: `(24, B)`, Block: `256`
extern "C" __global__ void ink2_gqa_rel_sink_attn_b(
    const float* __restrict__ qrgs_base,        // [B, qrgs_stride]
    int32_t qrgs_stride,
    const uint32_t* __restrict__ rel_proj_bf16, // [16, 512]
    const uint16_t* __restrict__ k_cache_fp16,  // [kv_cap, 1024]
    const uint16_t* __restrict__ v_cache_fp16,  // [kv_cap, 1024]
    const float* __restrict__ k_scale,          // [kv_cap, 8] or nullptr (FP16)
    const float* __restrict__ v_scale,          // [kv_cap, 8] or nullptr (FP16)
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
        float k0, k1, k2, k3;
        kv_load4(k_cache_fp16, k_scale, slot_row, kvh, lane, k0, k1, k2, k3);

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

        float v0, v1, v2, v3;
        kv_load4(v_cache_fp16, v_scale, slot_row, kvh, lane, v0, v1, v2, v3);
        o0 = o0 * alpha + p * v0;
        o1 = o1 * alpha + p * v1;
        o2 = o2 * alpha + p * v2;
        o3 = o3 * alpha + p * v3;
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
        if (tile_b > 0) a0 = accum_bf16_8(a0, pack, smem_x[0][c8 * 2], smem_x[0][c8 * 2 + 1]);
        if (tile_b > 1) a1 = accum_bf16_8(a1, pack, smem_x[1][c8 * 2], smem_x[1][c8 * 2 + 1]);
        if (tile_b > 2) a2 = accum_bf16_8(a2, pack, smem_x[2][c8 * 2], smem_x[2][c8 * 2 + 1]);
        if (tile_b > 3) a3 = accum_bf16_8(a3, pack, smem_x[3][c8 * 2], smem_x[3][c8 * 2 + 1]);
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

// Keep the first 64 occurrences per expert in the legacy table. Subsequent
// pages use the same [expert, 64] layout. A 4- or 8-token tile fits in one page.
__device__ __forceinline__ int expert_map_offset(int expert, int occurrence) {
    return ((occurrence / 64) * kExp + expert) * 64 + occurrence % 64;
}

// Grid: 32 CTAs of four warps. Each warp scans one expert's selections in
// input order. The ballot ranks the matching lanes without atomic appends.
extern "C" __global__ void ink2_build_expert_groups_wide_b(
    const int32_t* __restrict__ sel_ids,
    int32_t* __restrict__ exp_counts,
    int32_t* __restrict__ exp_toks,
    int32_t* __restrict__ exp_pairs,
    int32_t B) {
    const int expert = blockIdx.x * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    int count = 0;
    for (int first = 0; first < B * 8; first += 32) {
        const int pair = first + lane;
        const bool match = pair < B * 8 && sel_ids[pair] == expert;
        const unsigned hits = __ballot_sync(0xffffffffu, match);
        if (match) {
            const int rank = count + __popc(hits & ((1u << lane) - 1u));
            const int dst = expert_map_offset(expert, rank);
            exp_toks[dst] = pair / 8;
            exp_pairs[dst] = pair;
        }
        count += __popc(hits);
    }
    if (lane == 0) exp_counts[expert] = count;
}

// A CTA computes 32 weight rows for one expert. Each integer MMA tile covers
// 16 rows x 8 tokens x 16 input values, the width of one NVFP4 scale group.
// The gate/up projection consumes token rows and fuses SITU. The down
// projection reads and writes the routed pair rows.
template <bool GateUp, bool PackedScales>
__device__ __forceinline__ void grouped_nvfp4_mma(
    const uint4* weight_data, const uint32_t* weight_scales, const float* amax,
    const uint4* x_q8, const float* x_scales, const int32_t* exp_counts,
    const int32_t* input_ids, const int32_t* exp_pairs, float* output) {
    const int expert = blockIdx.y;
    const int count = exp_counts[expert];
    if (count == 0) return;

    constexpr int rows = 16;
    constexpr int tokens = 8;
    constexpr int groups = (GateUp ? kD : kFf) / 16;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int row = lane / 4;
    const int fragment = lane % 4;
    const float expert_scale = amax[expert] * (1.0f / 2688.0f);
    const auto* x_words = reinterpret_cast<const uint32_t*>(x_q8);

    // Pad the shared row strides so corresponding fragments of neighboring
    // weight rows do not all address the same shared-memory bank.
    __shared__ uint4 weights[rows][groups / 2 + 1];
    __shared__ float scales[rows][groups + 1];
    __shared__ float partials[8][rows * tokens];
    __shared__ float projections[rows][tokens];

    for (int tile = 0; tile < 2; ++tile) {
        const int first_row = blockIdx.x * 32 + tile * rows;
        const int offset = expert * (GateUp ? 2 * kFf : kD) + first_row;
        for (int i = tid; i < rows * (groups / 2); i += 256) {
            weights[i / (groups / 2)][i % (groups / 2)] = weight_data[offset * (groups / 2) + i];
        }
        for (int i = tid; i < rows * (groups / 4); i += 256) {
            const uint32_t packed = expert_scale_word<PackedScales, 6, 64>(weight_scales, offset * (groups / 4) + i);
            #pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                scales[i / (groups / 4)][4 * (i % (groups / 4)) + byte] =
                    e4m3_byte_to_f32((packed >> (8 * byte)) & 255u) * 0.5f;
            }
        }
        __syncthreads();

        for (int first = 0; first < count; first += tokens) {
            const int live = min(tokens, count - first);
            const int map_base = expert_map_offset(expert, first);
            // All lanes participate in MMA. Replicate the final valid column
            // in a short tile and suppress its extra outputs at the store.
            const int token = input_ids[map_base + min(row, live - 1)];
            float sum[4] = {};
            float left[4] = {};

            // The DP4A kernel assigned K pairs to 32 lanes. Each warp here
            // owns four of those lanes. Visit them in XOR-reduction order
            // (0,16,8,24 relative to warp), preserving each FP32 addition.
            #pragma unroll
            for (int step = 0; step < 4; ++step) {
                const int k_lane = warp + (step % 2) * 16 + (step / 2) * 8;
                float acc[4] = {};
                for (int pair = k_lane; pair < groups / 2; pair += 32) {
                    #pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        const int k = 2 * pair + half;
                        const int part = 4 * k + fragment;
                        const auto* top = reinterpret_cast<const uint16_t*>(weights[row]);
                        const auto* bottom = reinterpret_cast<const uint16_t*>(weights[row + 8]);
                        const unsigned a0 = decode_4x_e2m1_nibbles16_to_2x_s8(top[part]);
                        const unsigned a1 = decode_4x_e2m1_nibbles16_to_2x_s8(bottom[part]);
                        const unsigned b = x_words[(token * groups + k) * 4 + fragment];
                        int d0, d1, d2, d3;
                        mma_s8_k16(a0, a1, b, d0, d1, d2, d3);

                        const float w0 = scales[row][k];
                        const float w1 = scales[row + 8][k];
                        const float xs = x_scales[token * groups + k];
                        const float x0 = __shfl_sync(0xffffffffu, xs, 8 * fragment);
                        const float x1 = __shfl_sync(0xffffffffu, xs, 8 * fragment + 4);
                        acc[0] = __fmaf_rn(float(d0), __fmul_rn(w0, x0), acc[0]);
                        acc[1] = __fmaf_rn(float(d1), __fmul_rn(w0, x1), acc[1]);
                        acc[2] = __fmaf_rn(float(d2), __fmul_rn(w1, x0), acc[2]);
                        acc[3] = __fmaf_rn(float(d3), __fmul_rn(w1, x1), acc[3]);
                    }
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (step % 2 == 0) {
                        left[i] = acc[i];
                    } else if (step == 1) {
                        sum[i] = left[i] + acc[i];
                    } else {
                        sum[i] += left[i] + acc[i];
                    }
                }
            }
            const int dst = row * tokens + 2 * fragment;
            partials[warp][dst] = sum[0];
            partials[warp][dst + 1] = sum[1];
            partials[warp][dst + 64] = sum[2];
            partials[warp][dst + 65] = sum[3];
            __syncthreads();

            if (tid < rows * tokens) {
                const float s0 = partials[0][tid] + partials[4][tid];
                const float s1 = partials[1][tid] + partials[5][tid];
                const float s2 = partials[2][tid] + partials[6][tid];
                const float s3 = partials[3][tid] + partials[7][tid];
                const float value = ((s0 + s2) + (s1 + s3)) * expert_scale;
                const int t = tid % tokens;
                if constexpr (GateUp) {
                    projections[tid / tokens][t] = value;
                } else if (t < live) {
                    const int dest = exp_pairs[map_base + t];
                    output[dest * kD + first_row + tid / tokens] = value;
                }
            }
            if constexpr (GateUp) {
                __syncthreads();
                const int m = tid / tokens;
                const int t = tid % tokens;
                if (m < rows / 2 && t < live) {
                    const float g = projections[2 * m][t];
                    const float u = projections[2 * m + 1][t];
                    const float situ = 4.0f * ptx_tanh(g * 0.25f) * ptx_sigmoid(g) * 25.0f * ptx_tanh(u * 0.04f);
                    const int dest = exp_pairs[map_base + t];
                    output[dest * kFf + first_row / 2 + m] = situ;
                }
            }
            __syncthreads();
        }
    }
}

// Retain the entry points used by existing launchers and external cubins.
INK2_SCALE_KERNEL(ink2_moe_nvfp4_dp4a_grouped_up_situ, (
    const uint4* __restrict__ w13_u128,
    const uint32_t* __restrict__ s13_u32,
    const float* __restrict__ a13,
    const uint4* __restrict__ x_q8_u128,
    const float* __restrict__ x_sc16,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_toks,
    const int32_t* __restrict__ exp_pairs,
    float* __restrict__ act_out),
    (w13_u128, s13_u32, a13, x_q8_u128, x_sc16, exp_counts, exp_toks, exp_pairs, act_out)) {
    grouped_nvfp4_mma<true, PackedScales>(w13_u128, s13_u32, a13, x_q8_u128, x_sc16, exp_counts, exp_toks, exp_pairs, act_out);
}

INK2_SCALE_KERNEL(ink2_moe_nvfp4_dp4a_grouped_down, (
    const uint4* __restrict__ w2_u128,     // [128, 2304, 36]
    const uint32_t* __restrict__ s2_u32,   // [128, 2304, 18]
    const float* __restrict__ a2,          // [128]
    const uint4* __restrict__ act_q8_u128, // [B * 8, 72]
    const float* __restrict__ act_sc16,    // [B * 8, 72]
    const int32_t* __restrict__ exp_counts,// [128]
    const int32_t* __restrict__ exp_pairs, // [page, 128, 64]
    float* __restrict__ pair_y),
    (w2_u128, s2_u32, a2, act_q8_u128, act_sc16, exp_counts, exp_pairs, pair_y)) {          // [B * 8, 2304]
    grouped_nvfp4_mma<false, PackedScales>(w2_u128, s2_u32, a2, act_q8_u128, act_sc16, exp_counts, exp_pairs, exp_pairs, pair_y);
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

template<bool PackedScales>
__device__ __forceinline__ float load_e8m0_scale(const uint32_t* __restrict__ row_e8m0_u32, unsigned scale_start, int u16_idx) {
    const int blk32 = u16_idx >> 1;
    uint32_t s4 = expert_scale_word<PackedScales, 3, 112>(row_e8m0_u32, scale_start + (blk32 >> 2));
    uint32_t s_e8m0 = (s4 >> ((blk32 & 3) * 8)) & 0xFFu;
    return __uint_as_float(s_e8m0 << 23);
}

// Tile-4 Grouped-Expert MXFP8 Up+SITU (`Grid (72, 128)`, `Block 256`, `<= 42` regs!)
INK2_SCALE_KERNEL(ink2_moe_mxfp8_grouped_up_situ, (
    const uint4* __restrict__ w13_u128,    // [128, 2304, 144]
    const uint32_t* __restrict__ s13_u32,  // [128, 2304, 18]
    const float4* __restrict__ x4_base,    // [B, 576]
    const int32_t* __restrict__ exp_counts,// [128]
    const int32_t* __restrict__ exp_toks,  // [page, 128, 64]
    const int32_t* __restrict__ exp_pairs, // [page, 128, 64]
    float* __restrict__ act_out),
    (w13_u128, s13_u32, x4_base, exp_counts, exp_toks, exp_pairs, act_out)) {         // [B * 8, 1152]
    const int ex = blockIdx.y;
    const int cnt = exp_counts[ex];
    if (cnt == 0) return;

    // Stage four float4 planes so adjacent warp lanes read adjacent 16-byte vectors.
    constexpr int kPlaneStride = 144;
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
            smem_pairs[tid] = exp_pairs[expert_map_offset(ex, t_base) + tid];
        }
        for (int t = 0; t < tile_t; ++t) {
            const int tok = exp_toks[expert_map_offset(ex, t_base) + t];
            const float4* src = x4_base + static_cast<size_t>(tok) * 576u;
            for (int i = tid; i < 576; i += 256) {
                smem_x4[t][(i & 3) * kPlaneStride + (i >> 2)] = src[i];
            }
        }
        __syncthreads();

        for (int rg = 0; rg < 4; ++rg) {
            const int m = (blockIdx.x * 4 + rg) * 4 + (warp_id >> 1);
            const int r = m * 2 + is_up;
            const size_t row_idx = static_cast<size_t>(ex) * (2u * kFf) + r;
            const uint4* r_u128 = w13_u128 + row_idx * 144u;
            const unsigned scale_start = row_idx * 18u;

            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (int u16 = lane; u16 < 144; u16 += 32) {
                const uint4 p = r_u128[u16];
                const float es = load_e8m0_scale<PackedScales>(s13_u32, scale_start, u16);
                if (tile_t > 0) a0 = __fmaf_rn(dot_mxfp8_u128_exact<kPlaneStride>(p, smem_x4[0] + u16), es, a0);
                if (tile_t > 1) a1 = __fmaf_rn(dot_mxfp8_u128_exact<kPlaneStride>(p, smem_x4[1] + u16), es, a1);
                if (tile_t > 2) a2 = __fmaf_rn(dot_mxfp8_u128_exact<kPlaneStride>(p, smem_x4[2] + u16), es, a2);
                if (tile_t > 3) a3 = __fmaf_rn(dot_mxfp8_u128_exact<kPlaneStride>(p, smem_x4[3] + u16), es, a3);
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
INK2_SCALE_KERNEL(ink2_moe_mxfp8_grouped_down, (
    const uint4* __restrict__ w2_u128,     // [128, 2304, 72]
    const uint32_t* __restrict__ s2_u32,   // [128, 2304, 9]
    const float4* __restrict__ act4_base,  // [B * 8, 288]
    const int32_t* __restrict__ exp_counts,// [128]
    const int32_t* __restrict__ exp_pairs, // [page, 128, 64]
    float* __restrict__ pair_y),
    (w2_u128, s2_u32, act4_base, exp_counts, exp_pairs, pair_y)) {          // [B * 8, 2304]
    const int ex = blockIdx.y;
    const int cnt = exp_counts[ex];
    if (cnt == 0) return;

    // Stage four float4 planes so adjacent warp lanes read adjacent 16-byte vectors.
    constexpr int kPlaneStride = 72;
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
            smem_pairs[tid] = exp_pairs[expert_map_offset(ex, t_base) + tid];
        }
        for (int t = 0; t < tile_t; ++t) {
            const int pair_idx = exp_pairs[expert_map_offset(ex, t_base) + t];
            const float4* src = act4_base + static_cast<size_t>(pair_idx) * 288u;
            for (int i = tid; i < 288; i += 256) {
                smem_a4[t][(i & 3) * kPlaneStride + (i >> 2)] = src[i];
            }
        }
        __syncthreads();

        for (int rg = 0; rg < 4; ++rg) {
            const int row = (blockIdx.x * 4 + rg) * 8 + warp_id;
            if (row >= kD) continue;
            const size_t r_idx = static_cast<size_t>(ex) * kD + row;
            const uint4* r_u128 = w2_u128 + r_idx * 72u;
            const unsigned scale_start = r_idx * 9u;

            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;

            for (int u16 = lane; u16 < 72; u16 += 32) {
                const uint4 p = r_u128[u16];
                const float es = load_e8m0_scale<PackedScales>(s2_u32, scale_start, u16);
                if (tile_t > 0) a0 = __fmaf_rn(dot_mxfp8_u128_exact<kPlaneStride>(p, smem_a4[0] + u16), es, a0);
                if (tile_t > 1) a1 = __fmaf_rn(dot_mxfp8_u128_exact<kPlaneStride>(p, smem_a4[1] + u16), es, a1);
                if (tile_t > 2) a2 = __fmaf_rn(dot_mxfp8_u128_exact<kPlaneStride>(p, smem_a4[2] + u16), es, a2);
                if (tile_t > 3) a3 = __fmaf_rn(dot_mxfp8_u128_exact<kPlaneStride>(p, smem_a4[3] + u16), es, a3);
                if (tile_t > 4) a4 = __fmaf_rn(dot_mxfp8_u128_exact<kPlaneStride>(p, smem_a4[4] + u16), es, a4);
                if (tile_t > 5) a5 = __fmaf_rn(dot_mxfp8_u128_exact<kPlaneStride>(p, smem_a4[5] + u16), es, a5);
                if (tile_t > 6) a6 = __fmaf_rn(dot_mxfp8_u128_exact<kPlaneStride>(p, smem_a4[6] + u16), es, a6);
                if (tile_t > 7) a7 = __fmaf_rn(dot_mxfp8_u128_exact<kPlaneStride>(p, smem_a4[7] + u16), es, a7);
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
        sum_y = __fmaf_rn(pair_y[(base_pair + k) * kD + d], sel_wts[base_pair + k], sum_y);
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

// ============================================================================
// 15. Cycle 2 (`v0.3.0`) Speculative Decoding Suite (`DSpark` MTP + `O(1)` Rewind)
// ============================================================================

namespace leansonic::inkling2 {

__device__ __forceinline__ void ptx_sincos_reduced(float angle, float* s_out, float* c_out) {
    // Reduce angle into [-pi, pi] via hardware RNDM + FFMA, then single-cycle MUFU SIN/COS!
    float turns = floorf(angle * 0.15915494309189535f + 0.5f);
    float r = angle - turns * 6.283185307179586f;
    float sv, cv;
    asm("sin.approx.ftz.f32 %0, %1;\n" : "=f"(sv) : "f"(r));
    asm("cos.approx.ftz.f32 %0, %1;\n" : "=f"(cv) : "f"(r));
    *s_out = sv;
    *c_out = cv;
}

__device__ __forceinline__ float head_norm_neox_rope128(
    float x,
    const uint32_t* __restrict__ w_bf16_pairs, // [64] (`128` BF16)
    const float* __restrict__ inv_freq64,      // [64]
    int pos,
    float* smem_hd) {                          // [128]
    const int d = threadIdx.x; // 0..127 (`4` warps)
    const int warp_id = d >> 5;
    const int lane = d & 31;

    float sq = warp_sum(x * x);
    if (lane == 0) smem_hd[warp_id] = sq;
    __syncthreads();
    float bsum = smem_hd[0] + smem_hd[1] + smem_hd[2] + smem_hd[3];
    float inv_rms = ptx_rsqrt(bsum * (1.0f / 128.0f) + 1e-6f);
    __syncthreads();

    uint32_t wp = w_bf16_pairs[d >> 1];
    float g = ((d & 1) == 0)
        ? bf16_to_f32(static_cast<uint16_t>(wp & 0xFFFFu))
        : bf16_to_f32(static_cast<uint16_t>(wp >> 16));
    smem_hd[d] = x * inv_rms * g;
    __syncthreads();

    const int i = d & 63;
    float s, c;
    ptx_sincos_reduced(static_cast<float>(pos) * inv_freq64[i], &s, &c);
    float rot = (d < 64)
        ? (smem_hd[d] * c - smem_hd[d + 64] * s)
        : (smem_hd[d - 64] * s + smem_hd[d] * c);
    __syncthreads();
    return rot;
}

} // namespace leansonic::inkling2

// `O(1)` Speculative Rewind across all 48 slots x 4 `short_conv4` states + `*pos_ptr`!
// Grid: `(9, 192)`, Block: `256`
extern "C" __global__ void ink2_conv_rewind(
    const float* __restrict__ seq_base,       // [48, 6656 * 11]
    const uint64_t* __restrict__ hist_ptrs,   // [48 * 4]
    int32_t* __restrict__ pos_ptr,
    int32_t keep,
    int32_t B) {
    const int slot = blockIdx.y >> 2;
    const int kind = blockIdx.y & 3;
    const int c = blockIdx.x * 256 + threadIdx.x;

    const float* slot_seq = seq_base + static_cast<size_t>(slot) * (6656u * 11u);
    int ch;
    const float* seq;
    if (kind == 0) {
        ch = 1024;
        seq = slot_seq;
    } else if (kind == 1) {
        ch = 1024;
        seq = slot_seq + 1024u * 11u;
    } else if (kind == 2) {
        ch = 2304;
        seq = slot_seq + 2048u * 11u;
    } else {
        ch = 2304;
        seq = slot_seq + 4352u * 11u;
    }

    if (c < ch) {
        float* h = reinterpret_cast<float*>(hist_ptrs[blockIdx.y]) + c * 3;
        h[0] = seq[static_cast<size_t>(keep + 0) * ch + c];
        h[1] = seq[static_cast<size_t>(keep + 1) * ch + c];
        h[2] = seq[static_cast<size_t>(keep + 2) * ch + c];
    }
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        *pos_ptr -= (B - keep);
    }
}

// General K-Tiled Planar `Q8_64` `__dp4a` GEMM for `1 <= B <= 8` & ANY `cols % 64 == 0`
// (`cols = 256, 2304, 3072, 4608, 9216`): `8,704` B SMEM, `<= 48` regs!
// Grid: `(ceil(rows / 16), 1)`, Block: `256`
extern "C" __global__ void ink2_q8_dp4a_gemm_ktiled(
    const uint4* __restrict__ w_q8_u128,   // [rows, cols / 16]
    const float* __restrict__ w_sc64,      // [rows, cols / 64]
    const uint4* __restrict__ x_q8_u128,   // [B, cols / 16]
    const float* __restrict__ x_sc64,      // [B, cols / 64]
    float* __restrict__ y_base,            // [B, y_stride]
    int32_t rows,
    int32_t cols,
    int32_t B,
    int32_t y_stride) {
    __shared__ uint4 smem_x[8][64];   // 8,192 B (`K_TILE = 1024`)
    __shared__ float smem_xsc[8][16]; // 512 B

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int cols16 = cols >> 4;
    const int cols64 = cols >> 6;
    const int row0 = (blockIdx.x * 8 + warp_id) * 2;
    const bool has_row0 = (row0 < rows);
    const bool has_row1 = (row0 + 1 < rows);

    const uint4* w_r0 = has_row0 ? (w_q8_u128 + static_cast<size_t>(row0) * cols16) : w_q8_u128;
    const uint4* w_r1 = has_row1 ? (w_r0 + cols16) : w_r0;
    const float* s_r0 = has_row0 ? (w_sc64 + static_cast<size_t>(row0) * cols64) : w_sc64;
    const float* s_r1 = has_row1 ? (s_r0 + cols64) : s_r0;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
    float b4 = 0.0f, b5 = 0.0f, b6 = 0.0f, b7 = 0.0f;

    for (int k16_base = 0; k16_base < cols16; k16_base += 64) {
        const int rem16 = cols16 - k16_base;
        const int tile16 = (rem16 < 64) ? rem16 : 64;
        const int k64_base = k16_base >> 2;
        const int tile64 = tile16 >> 2;

        if (k16_base > 0) __syncthreads();
        for (int t = 0; t < B; ++t) {
            if (tid < tile16) {
                smem_x[t][tid] = x_q8_u128[static_cast<size_t>(t) * cols16 + k16_base + tid];
            }
            if (tid < tile64) {
                smem_xsc[t][tid] = x_sc64[static_cast<size_t>(t) * cols64 + k64_base + tid];
            }
        }
        __syncthreads();

        if (has_row0) {
            for (int c16 = lane; c16 < tile16; c16 += 32) {
                const int c64 = c16 >> 2;
                const uint4 w0 = w_r0[k16_base + c16];
                const float ws0 = s_r0[k64_base + c64];
                const uint4 w1 = has_row1 ? w_r1[k16_base + c16] : w0;
                const float ws1 = has_row1 ? s_r1[k64_base + c64] : 0.0f;

                if (B > 0) { uint4 xq = smem_x[0][c16]; float xs = smem_xsc[0][c64]; a0 += static_cast<float>(dp4a_u128(w0, xq)) * (ws0 * xs); if (has_row1) b0 += static_cast<float>(dp4a_u128(w1, xq)) * (ws1 * xs); }
                if (B > 1) { uint4 xq = smem_x[1][c16]; float xs = smem_xsc[1][c64]; a1 += static_cast<float>(dp4a_u128(w0, xq)) * (ws0 * xs); if (has_row1) b1 += static_cast<float>(dp4a_u128(w1, xq)) * (ws1 * xs); }
                if (B > 2) { uint4 xq = smem_x[2][c16]; float xs = smem_xsc[2][c64]; a2 += static_cast<float>(dp4a_u128(w0, xq)) * (ws0 * xs); if (has_row1) b2 += static_cast<float>(dp4a_u128(w1, xq)) * (ws1 * xs); }
                if (B > 3) { uint4 xq = smem_x[3][c16]; float xs = smem_xsc[3][c64]; a3 += static_cast<float>(dp4a_u128(w0, xq)) * (ws0 * xs); if (has_row1) b3 += static_cast<float>(dp4a_u128(w1, xq)) * (ws1 * xs); }
                if (B > 4) { uint4 xq = smem_x[4][c16]; float xs = smem_xsc[4][c64]; a4 += static_cast<float>(dp4a_u128(w0, xq)) * (ws0 * xs); if (has_row1) b4 += static_cast<float>(dp4a_u128(w1, xq)) * (ws1 * xs); }
                if (B > 5) { uint4 xq = smem_x[5][c16]; float xs = smem_xsc[5][c64]; a5 += static_cast<float>(dp4a_u128(w0, xq)) * (ws0 * xs); if (has_row1) b5 += static_cast<float>(dp4a_u128(w1, xq)) * (ws1 * xs); }
                if (B > 6) { uint4 xq = smem_x[6][c16]; float xs = smem_xsc[6][c64]; a6 += static_cast<float>(dp4a_u128(w0, xq)) * (ws0 * xs); if (has_row1) b6 += static_cast<float>(dp4a_u128(w1, xq)) * (ws1 * xs); }
                if (B > 7) { uint4 xq = smem_x[7][c16]; float xs = smem_xsc[7][c64]; a7 += static_cast<float>(dp4a_u128(w0, xq)) * (ws0 * xs); if (has_row1) b7 += static_cast<float>(dp4a_u128(w1, xq)) * (ws1 * xs); }
            }
        }
    }

    if (!has_row0) return;
    a0 = warp_sum(a0); b0 = warp_sum(b0);
    a1 = warp_sum(a1); b1 = warp_sum(b1);
    a2 = warp_sum(a2); b2 = warp_sum(b2);
    a3 = warp_sum(a3); b3 = warp_sum(b3);
    a4 = warp_sum(a4); b4 = warp_sum(b4);
    a5 = warp_sum(a5); b5 = warp_sum(b5);
    a6 = warp_sum(a6); b6 = warp_sum(b6);
    a7 = warp_sum(a7); b7 = warp_sum(b7);

    if (lane == 0) {
        auto write_pair = [&](int t, float v0, float v1) {
            if (t < B) {
                float* dst = y_base + static_cast<size_t>(t) * y_stride;
                dst[row0] = v0;
                if (has_row1) dst[row0 + 1] = v1;
            }
        };
        write_pair(0, a0, b0); write_pair(1, a1, b1);
        write_pair(2, a2, b2); write_pair(3, a3, b3);
        write_pair(4, a4, b4); write_pair(5, a5, b5);
        write_pair(6, a6, b6); write_pair(7, a7, b7);
    }
}

extern "C" __global__ void mtp_tapmix(
    const float* __restrict__ taps,
    int32_t tstride,
    float a0, float a1, float a2, float a3,
    int32_t n_floats,
    float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_floats) return;
    out[i] = a0 * taps[i]
           + a1 * taps[static_cast<size_t>(tstride) + i]
           + a2 * taps[2u * static_cast<size_t>(tstride) + i]
           + a3 * taps[3u * static_cast<size_t>(tstride) + i];
}

extern "C" __global__ void mtp_copy_anchor(
    const float* __restrict__ taps,   // [4, tap_cap, D]
    int32_t tap_cap,
    int32_t tok_idx,
    float* __restrict__ anchor_out) { // [4, D]
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y; // 0..3
    if (i >= kD) return;
    anchor_out[static_cast<size_t>(j) * kD + i] =
        taps[(static_cast<size_t>(j) * tap_cap + tok_idx) * kD + i];
}

// Grid: `(8, n_toks)`, Block: `128`
extern "C" __global__ void mtp_ctx_store(
    const float* __restrict__ kv,               // [n_toks, 2048]
    const uint32_t* __restrict__ knorm_bf16_p,  // [64]
    const float* __restrict__ inv_freq64,       // [64]
    int32_t pos0,
    int32_t ring,                               // `0` = linear (`max_ctx`), `>0` = ring
    uint16_t* __restrict__ kc_fp16,
    uint16_t* __restrict__ vc_fp16) {
    __shared__ float smem_hd[128];
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int d = threadIdx.x;
    const int pos = pos0 + t;

    float k_val = head_norm_neox_rope128(
        kv[static_cast<size_t>(t) * 2048u + h * 128 + d],
        knorm_bf16_p,
        inv_freq64,
        pos,
        smem_hd);
    float v_val = kv[static_cast<size_t>(t) * 2048u + 1024u + h * 128 + d];
    const int row = (ring > 0) ? (pos % ring) : pos;
    const size_t at = static_cast<size_t>(row) * 1024u + h * 128 + d;
    kc_fp16[at] = f32_to_fp16_bits(k_val);
    vc_fp16[at] = f32_to_fp16_bits(v_val);
}

// Initializes `dr_x [8, D]` (`row 0 = embed_rmsnorm(next_tok)`, `rows 1..7 = mask_embed`),
// zeroes `dr_state [256]`, and sets `dr_res[0..2] = (next_tok, 1.0f)` in ONE CTA!
extern "C" __global__ void mtp_init_draft(
    const uint32_t* __restrict__ embed_bf16_pairs,
    const uint32_t* __restrict__ embed_norm_bf16_pairs,
    const float* __restrict__ mask_embed_f32, // [D]
    int32_t next_tok,
    float* __restrict__ dr_x,                 // [8, D]
    float* __restrict__ dr_state,             // [256]
    uint32_t* __restrict__ dr_res) {          // [16]
    __shared__ float smem_red[8];
    const int tid = threadIdx.x;
    const uint32_t* row = embed_bf16_pairs + static_cast<size_t>(next_tok) * (kD / 2);

    dr_state[tid] = 0.0f;
    float sq = 0.0f;
    for (int idx = tid; idx < kD / 2; idx += 256) {
        uint32_t pair = row[idx];
        float v0 = bf16_to_f32(static_cast<uint16_t>(pair & 0xFFFFu));
        float v1 = bf16_to_f32(static_cast<uint16_t>(pair >> 16));
        dr_x[idx * 2] = v0;
        dr_x[idx * 2 + 1] = v1;
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
        uint32_t nw = embed_norm_bf16_pairs[idx];
        dr_x[idx * 2] *= scale * bf16_to_f32(static_cast<uint16_t>(nw & 0xFFFFu));
        dr_x[idx * 2 + 1] *= scale * bf16_to_f32(static_cast<uint16_t>(nw >> 16));
    }
    for (int i = tid; i < kD; i += 256) {
        float mv = mask_embed_f32[i];
        #pragma unroll
        for (int r = 1; r < 8; ++r) {
            dr_x[r * kD + i] = mv;
        }
    }
    if (tid == 0) {
        dr_res[0] = static_cast<uint32_t>(next_tok);
        dr_res[1] = __float_as_uint(1.0f);
    }
}

// Grid: `(32, 8)`, Block: `128`
extern "C" __global__ void mtp_qk_norm_rope(
    float* __restrict__ qkv,                  // [8, 5120]
    const uint32_t* __restrict__ qn_bf16_p,   // [64]
    const uint32_t* __restrict__ kn_bf16_p,   // [64]
    const float* __restrict__ inv_freq64,     // [64]
    int32_t pos0) {
    __shared__ float smem_hd[128];
    const int hh = blockIdx.x; // 0..31 (`0..23` = Q, `24..31` = K)
    const int t = blockIdx.y;  // 0..7
    const int d = threadIdx.x; // 0..127

    float* p = qkv + static_cast<size_t>(t) * 5120u + ((hh < 24) ? (hh * 128) : (3072 + (hh - 24) * 128));
    float r = head_norm_neox_rope128(
        p[d],
        (hh < 24) ? qn_bf16_p : kn_bf16_p,
        inv_freq64,
        pos0 + t,
        smem_hd);
    p[d] = r;
}

// Bidirectional Context (`[lo..p)`) + 8-Row Draft Block Online-Softmax GQA!
// Grid: `(24, 8)`, Block: `256` (`8` warps/CTA, `4,672` B SMEM, `<= 40` regs!)
extern "C" __global__ void mtp_attn(
    const float* __restrict__ qkv,            // [8, 5120]
    const uint16_t* __restrict__ kc_fp16,     // [cap, 1024]
    const uint16_t* __restrict__ vc_fp16,     // [cap, 1024]
    int32_t ring,
    int32_t lo,
    int32_t hi,                               // `p = ctx_end`
    float* __restrict__ att_out) {            // [8, 3072]
    __shared__ float smem_q[128];
    __shared__ float smem_m[8];
    __shared__ float smem_l[8];
    __shared__ float smem_o[8][128];

    const int h = blockIdx.x; // 0..23
    const int t = blockIdx.y; // 0..7
    const int kvh = h / 3;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    if (tid < 128) {
        smem_q[tid] = qkv[static_cast<size_t>(t) * 5120u + h * 128 + tid];
    }
    __syncthreads();

    const float q0 = smem_q[lane * 4];
    const float q1 = smem_q[lane * 4 + 1];
    const float q2 = smem_q[lane * 4 + 2];
    const float q3 = smem_q[lane * 4 + 3];
    const float scale = 0.08838834764831845f; // 1 / sqrt(128)

    const int nc = hi - lo;
    const int total_keys = nc + 8;

    float m_w = -1e30f;
    float l_w = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    for (int j = warp_id; j < total_keys; j += 8) {
        float k0, k1, k2, k3;
        float v0, v1, v2, v3;
        if (j < nc) {
            const int cpos = lo + j;
            const int row = (ring > 0) ? (cpos % ring) : cpos;
            const uint64_t* kp = reinterpret_cast<const uint64_t*>(
                kc_fp16 + static_cast<size_t>(row) * 1024u + kvh * 128);
            const uint64_t* vp = reinterpret_cast<const uint64_t*>(
                vc_fp16 + static_cast<size_t>(row) * 1024u + kvh * 128);
            uint64_t kw = kp[lane];
            uint64_t vw = vp[lane];
            k0 = fp16_bits_to_f32(static_cast<uint16_t>(kw & 0xFFFFu));
            k1 = fp16_bits_to_f32(static_cast<uint16_t>((kw >> 16) & 0xFFFFu));
            k2 = fp16_bits_to_f32(static_cast<uint16_t>((kw >> 32) & 0xFFFFu));
            k3 = fp16_bits_to_f32(static_cast<uint16_t>((kw >> 48) & 0xFFFFu));
            v0 = fp16_bits_to_f32(static_cast<uint16_t>(vw & 0xFFFFu));
            v1 = fp16_bits_to_f32(static_cast<uint16_t>((vw >> 16) & 0xFFFFu));
            v2 = fp16_bits_to_f32(static_cast<uint16_t>((vw >> 32) & 0xFFFFu));
            v3 = fp16_bits_to_f32(static_cast<uint16_t>((vw >> 48) & 0xFFFFu));
        } else {
            const float* b_row = qkv + static_cast<size_t>(j - nc) * 5120u;
            const float4 kv4 = reinterpret_cast<const float4*>(b_row + 3072 + kvh * 128)[lane];
            const float4 vv4 = reinterpret_cast<const float4*>(b_row + 4096 + kvh * 128)[lane];
            k0 = kv4.x; k1 = kv4.y; k2 = kv4.z; k3 = kv4.w;
            v0 = vv4.x; v1 = vv4.y; v2 = vv4.z; v3 = vv4.w;
        }

        float dot = warp_sum(q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3) * scale;
        float m_new = (dot > m_w) ? dot : m_w;
        float alpha = (l_w > 0.0f) ? ptx_exp(m_w - m_new) : 0.0f;
        float p = ptx_exp(dot - m_new);
        l_w = l_w * alpha + p;
        m_w = m_new;
        o0 = o0 * alpha + p * v0;
        o1 = o1 * alpha + p * v1;
        o2 = o2 * alpha + p * v2;
        o3 = o3 * alpha + p * v3;
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

    if (tid < 128) {
        float m_max = smem_m[0];
        #pragma unroll
        for (int w = 1; w < 8; ++w) {
            if (smem_m[w] > m_max) m_max = smem_m[w];
        }
        float denom = 0.0f;
        float sum_d = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            float s = (smem_l[w] > 0.0f) ? ptx_exp(smem_m[w] - m_max) : 0.0f;
            denom += smem_l[w] * s;
            sum_d += smem_o[w][tid] * s;
        }
        att_out[static_cast<size_t>(t) * 3072u + h * 128 + tid] = sum_d * ptx_rcp(fmaxf(denom, 1e-20f));
    }
}

extern "C" __global__ void mtp_add_inplace(
    float* __restrict__ x,
    const float* __restrict__ o,
    int32_t n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += o[i];
}

extern "C" __global__ void mtp_silu_mul(
    const float* __restrict__ gu, // [8, 18432]
    float* __restrict__ act) {    // [8, 9216]
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 8 * 9216) return;
    const int t = i / 9216;
    const int j = i - t * 9216;
    const float g = gu[static_cast<size_t>(t) * 18432u + j];
    const float u = gu[static_cast<size_t>(t) * 18432u + 9216u + j];
    act[i] = (g * ptx_sigmoid(g)) * u;
}

extern "C" __global__ void mtp_concat_frnn(
    const float* __restrict__ hidden_rows1_7, // [7, 2304]
    const float* __restrict__ frnn,           // [2304]
    float* __restrict__ x2_out) {             // [7, 4608]
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 7 * 4608) return;
    const int t = i / 4608;
    const int j = i - t * 4608;
    x2_out[i] = (j < kD) ? hidden_rows1_7[static_cast<size_t>(t) * kD + j] : frnn[j - kD];
}

extern "C" __global__ void mtp_add_bias_rows(
    float* __restrict__ y,
    const float* __restrict__ bias,
    int32_t cols,
    int32_t total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) y[i] += bias[i % cols];
}

// Markov RNN Step Part 1 (`Grid (96, 1)`, `Block 256`):
// `raw = ctxm + J01 @ [state; w1[prev]]` (`w1` read via 32-bit `uint32_t` pairs!)
extern "C" __global__ void mtp_markov_raw(
    const float* __restrict__ ctxm_row,           // [768]
    const float* __restrict__ j01,                // [768, 512]
    const uint32_t* __restrict__ w1_bf16_pairs,   // [201024, 128]
    const float* __restrict__ state,              // [256]
    const int32_t* __restrict__ prev_tok_ptr,
    float* __restrict__ raw_out,                  // [768]
    const float* __restrict__ cum_ptr) {
    if (*cum_ptr < 0.0f) return;
    __shared__ float smem_in[512];

    const int tid = threadIdx.x; // 0..255
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int o = blockIdx.x * 8 + warp_id;

    smem_in[tid] = state[tid];
    if (tid < 128) {
        const size_t base = static_cast<size_t>(*prev_tok_ptr) * 128u;
        uint32_t pair = w1_bf16_pairs[base + tid];
        smem_in[256 + tid * 2] = bf16_to_f32(static_cast<uint16_t>(pair & 0xFFFFu));
        smem_in[256 + tid * 2 + 1] = bf16_to_f32(static_cast<uint16_t>(pair >> 16));
    }
    __syncthreads();

    const float* j_row = j01 + static_cast<size_t>(o) * 512u;
    float s = 0.0f;
    #pragma unroll
    for (int k = lane; k < 512; k += 32) {
        s += j_row[k] * smem_in[k];
    }
    s = warp_sum(s);
    if (lane == 0) {
        raw_out[o] = s + ctxm_row[o];
    }
}

// Markov RNN Step Part 2 + Fused `Q8_64` Quantization of `lat [256]` (`Grid (1, 1)`, `Block 256`)!
extern "C" __global__ void mtp_markov_step_q8(
    const float* __restrict__ raw,     // [768]
    float* __restrict__ state,         // [256]
    uint32_t* __restrict__ lat_q8_u32, // [64] (`256` i8)
    float* __restrict__ lat_sc64,      // [4]
    const float* __restrict__ cum_ptr) {
    if (*cum_ptr < 0.0f) return;
    __shared__ float smem_lat[256];

    const int tid = threadIdx.x; // 0..255
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    float gate = ptx_sigmoid(raw[tid]);
    state[tid] = gate * state[tid] + (1.0f - gate) * ptx_tanh(raw[256 + tid]);
    smem_lat[tid] = ptx_tanh(raw[512 + tid]);
    __syncthreads();

    if (warp_id < 4) {
        float y0 = smem_lat[warp_id * 64 + lane * 2];
        float y1 = smem_lat[warp_id * 64 + lane * 2 + 1];
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
            lat_q8_u32[warp_id * 16 + (lane >> 1)] = pack32;
        }
        if (lane == 0) {
            lat_sc64[warp_id] = amax * (1.0f / 127.0f);
        }
    }
}

// Planar `Q8_64` `__dp4a` Markov Fold Head (`Grid (ceil(VOCAB / 64), 1)`, `Block 256`):
// 4 lanes per vocab row (`64` cols/lane x `4` = `256` cols), softcap `30.0`, block max + sum_exp!
extern "C" __global__ void mtp_fold_mv_q8(
    const uint4* __restrict__ fold_q8_u128, // [VOCAB, 16]
    const float* __restrict__ fold_sc64,    // [VOCAB, 4]
    const uint4* __restrict__ lat_q8_u128,  // [16]
    const float* __restrict__ lat_sc64,     // [4]
    const float* __restrict__ fold_b,       // [VOCAB]
    const float* __restrict__ base_logits,  // [VOCAB]
    float* __restrict__ part_out,           // [nb, 3]
    const float* __restrict__ cum_ptr) {
    if (*cum_ptr < 0.0f) return;
    __shared__ uint4 smem_lq[16];
    __shared__ float smem_lsc[4];
    __shared__ float smem_bv[8];
    __shared__ float smem_bz[8];
    __shared__ int smem_bi[8];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    if (tid < 16) smem_lq[tid] = lat_q8_u128[tid];
    if (tid < 4) smem_lsc[tid] = lat_sc64[tid];
    __syncthreads();

    const int q = lane & 3;
    const int r = (blockIdx.x * 8 + warp_id) * 8 + (lane >> 2);
    float s = 0.0f;
    if (r < kVocab) {
        const uint4* wr = fold_q8_u128 + static_cast<size_t>(r) * 16u + q * 4;
        int d = dp4a_u128(wr[0], smem_lq[q * 4]);
        d += dp4a_u128(wr[1], smem_lq[q * 4 + 1]);
        d += dp4a_u128(wr[2], smem_lq[q * 4 + 2]);
        d += dp4a_u128(wr[3], smem_lq[q * 4 + 3]);
        s = static_cast<float>(d) * (fold_sc64[static_cast<size_t>(r) * 4u + q] * smem_lsc[q]);
    }
    s += __shfl_xor_sync(0xFFFFFFFFu, s, 1);
    s += __shfl_xor_sync(0xFFFFFFFFu, s, 2);

    float v = (r < kVocab && q == 0)
        ? (30.0f * ptx_tanh((s + fold_b[r] + base_logits[r]) * (1.0f / 30.0f)))
        : -1e30f;
    float m = v;
    int mi = r;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        float v2 = __shfl_xor_sync(0xFFFFFFFFu, m, mask);
        int i2 = __shfl_xor_sync(0xFFFFFFFFu, mi, mask);
        if (v2 > m || (v2 == m && i2 < mi)) {
            m = v2;
            mi = i2;
        }
    }
    if (lane == 0) {
        smem_bv[warp_id] = m;
        smem_bi[warp_id] = mi;
    }
    __syncthreads();

    m = smem_bv[0];
    mi = smem_bi[0];
    #pragma unroll
    for (int k = 1; k < 8; ++k) {
        if (smem_bv[k] > m || (smem_bv[k] == m && smem_bi[k] < mi)) {
            m = smem_bv[k];
            mi = smem_bi[k];
        }
    }
    float z = (r < kVocab && q == 0) ? ptx_exp(v - m) : 0.0f;
    z = warp_sum(z);
    if (lane == 0) smem_bz[warp_id] = z;
    __syncthreads();

    if (tid == 0) {
        float zs = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) zs += smem_bz[k];
        part_out[blockIdx.x * 3] = m;
        part_out[blockIdx.x * 3 + 1] = __int_as_float(mi);
        part_out[blockIdx.x * 3 + 2] = zs;
    }
}

extern "C" __global__ void mtp_fold_fin(
    const float* __restrict__ part, // [nb, 3]
    int32_t nb,
    int32_t* __restrict__ best_out,
    float* __restrict__ sumexp_out,
    float* __restrict__ cum_inout,
    float tau,
    int32_t is_last) {
    if (*cum_inout < 0.0f) return;
    __shared__ float smem_bv[8];
    __shared__ int smem_bi[8];
    __shared__ float smem_z[8];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    float v = -1e30f;
    int idx = 0x7fffffff;
    for (int b = tid; b < nb; b += 256) {
        float v2 = part[b * 3];
        int i2 = __float_as_int(part[b * 3 + 1]);
        if (v2 > v || (v2 == v && i2 < idx)) {
            v = v2;
            idx = i2;
        }
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        float v2 = __shfl_xor_sync(0xFFFFFFFFu, v, mask);
        int i2 = __shfl_xor_sync(0xFFFFFFFFu, idx, mask);
        if (v2 > v || (v2 == v && i2 < idx)) {
            v = v2;
            idx = i2;
        }
    }
    if (lane == 0) {
        smem_bv[warp_id] = v;
        smem_bi[warp_id] = idx;
    }
    __syncthreads();

    v = smem_bv[0];
    idx = smem_bi[0];
    #pragma unroll
    for (int w = 1; w < 8; ++w) {
        if (smem_bv[w] > v || (smem_bv[w] == v && smem_bi[w] < idx)) {
            v = smem_bv[w];
            idx = smem_bi[w];
        }
    }
    float z = 0.0f;
    for (int b = tid; b < nb; b += 256) {
        z += part[b * 3 + 2] * ptx_exp(part[b * 3] - v);
    }
    z = warp_sum(z);
    if (lane == 0) smem_z[warp_id] = z;
    __syncthreads();

    if (tid == 0) {
        float total_z = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) total_z += smem_z[w];
        float c = (*cum_inout) * ptx_rcp(fmaxf(total_z, 1.0f));
        *best_out = idx;
        *sumexp_out = total_z;
        *cum_inout = (c < tau || is_last) ? -1.0f : c;
    }
}

// ============================================================================
// 16. Cycle 4 (`v0.5.0`) Unified Split-KV + GQA-Grouped Attention (`PR #9`)
//     Shared identically by `B = 1` CUDA-Graph Decode AND `1 <= B <= 64`
//     `DSpark` Speculative Verify / Prefill (`0.0000-ULP` Solo == Verify!).
//     - `ink2_gqa_attn_split`: Grid `(8 * B, G)`, Block `256` (`3` Q heads share `1` KV read!)
//     - `ink2_gqa_attn_combine`: Grid `(24, B)`, Block `128`
// ============================================================================
template <bool kInt8>
__device__ __forceinline__ void gqa_attn_split_body(
    const float* __restrict__ qrgs_base,        // [B, qrgs_stride] (`q@0, r@3072, gate@3456, sinks@6528`)
    int32_t qrgs_stride,
    const uint32_t* __restrict__ rel_proj_bf16, // [16, 512]
    const uint16_t* __restrict__ k_cache_fp16,  // [kv_cap, 8, 128]
    const uint16_t* __restrict__ v_cache_fp16,  // [kv_cap, 8, 128]
    const float* __restrict__ k_scale,          // [kv_cap, 8] INT8 KV scales, or nullptr (FP16)
    const float* __restrict__ v_scale,          // [kv_cap, 8] INT8 KV scales, or nullptr (FP16)
    const int32_t* __restrict__ pos_ptr,
    int32_t kv_cap,
    int32_t is_global,
    float* __restrict__ part_ml,                // [B, 24, G, 2]
    float* __restrict__ part_o) {               // [B, 24, G, 128]
    constexpr int kChunk = 64;
    __shared__ float smem_q[3][kHd];
    __shared__ float smem_r[3][kRel];
    __shared__ float smem_rel[3][kChunk];
    __shared__ float smem_m[8][3];
    __shared__ float smem_l[8][3];
    __shared__ float smem_o[8][3][kHd];

    const int b = blockIdx.x >> 3;
    const int kvh = blockIdx.x & 7;
    const int g = blockIdx.y;
    const int G = gridDim.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int pos = (*pos_ptr) + b;
    const int n = pos + 1;
    const int first = is_global ? 0 : ((n > kExtent) ? (n - kExtent) : 0);

    const float* qrgs = qrgs_base + static_cast<size_t>(b) * qrgs_stride;
    const float* q_buf = qrgs;
    const float* r_buf = qrgs + 3072;
    const float* sinks_buf = qrgs + 6528;

    for (int i = tid; i < 3 * kHd; i += 256) {
        smem_q[i / kHd][i % kHd] = q_buf[(kvh * 3) * kHd + i];
    }
    if (tid < 3 * kRel) {
        smem_r[tid / kRel][tid % kRel] = r_buf[(kvh * 3) * kRel + tid];
    }
    __syncthreads();

    float qa[3][4];
    float m_w[3], l_w[3], o[3][4];
    #pragma unroll
    for (int hh = 0; hh < 3; ++hh) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            qa[hh][k] = smem_q[hh][lane * 4 + k];
            o[hh][k] = 0.0f;
        }
        const bool own_sink = (g == 0 && warp_id == 0);
        m_w[hh] = own_sink ? sinks_buf[kvh * 3 + hh] : -1e30f;
        l_w[hh] = own_sink ? 1.0f : 0.0f;
    }
    const float scale = 0.08838834764831845f; // 1 / sqrt(128)

    // Ring row of key `j` without a per-key `%`: a SWA window spans at most kv_cap rows, so after
    // removing the kv_cap multiple below `first` each key needs at most one wrap. Global slots
    // have kv_cap >= n, so their row is `j`. Same rows as `is_global ? j : j % kv_cap`.
    const int ring_off = is_global ? 0 : first - first % kv_cap;
    auto ring_row = [&](int j) {
        const int r = j - ring_off;
        return r < kv_cap ? r : r - kv_cap;
    };
    const int total = n - first;
    const int n_chunks = (total + kChunk - 1) / kChunk;
    for (int c = g; c < n_chunks; c += G) {
        const int j0 = first + c * kChunk;
        if (tid < 3 * kChunk) {
            const int hh = tid / kChunk;
            const int jj = tid % kChunk;
            const int j = j0 + jj;
            const int d = pos - j;
            float acc = 0.0f;
            if (j < n && d < kExtent) {
                const int pair_d = d >> 1;
                const int hi = d & 1;
                #pragma unroll
                for (int i = 0; i < kRel; ++i) {
                    uint32_t p = rel_proj_bf16[i * 512 + pair_d];
                    acc += smem_r[hh][i] * bf16_to_f32(static_cast<uint16_t>(hi ? (p >> 16) : (p & 0xFFFFu)));
                }
            }
            smem_rel[hh][jj] = acc;
        }
        __syncthreads();

        if constexpr (kInt8) {
            // Depth-1 software pipeline on the raw INT8 words (+4 regs): row jj+8's K/V words and
            // scales are in flight while row jj is computed. Per-row arithmetic and row order are
            // identical to the generic loop below, so the result does not change.
            int jj = warp_id;
            int j = j0 + jj;
            uint32_t wk = 0u, wv = 0u;
            float sk = 0.0f, sv = 0.0f;
            if (jj < kChunk && j < n) {
                kv_raw_i8(k_cache_fp16, k_scale, ring_row(j), kvh, lane, wk, sk);
                kv_raw_i8(v_cache_fp16, v_scale, ring_row(j), kvh, lane, wv, sv);
            }
            while (jj < kChunk && j < n) {
                const int jn = jj + 8;
                const int jnx = j0 + jn;
                uint32_t nwk = 0u, nwv = 0u;
                float nsk = 0.0f, nsv = 0.0f;
                if (jn < kChunk && jnx < n) {
                    kv_raw_i8(k_cache_fp16, k_scale, ring_row(jnx), kvh, lane, nwk, nsk);
                    kv_raw_i8(v_cache_fp16, v_scale, ring_row(jnx), kvh, lane, nwv, nsv);
                }
                float k0, k1, k2, k3, v0, v1, v2, v3;
                kv_dequant_i8(wk, sk, k0, k1, k2, k3);
                kv_dequant_i8(wv, sv, v0, v1, v2, v3);
                float dots[3];
                #pragma unroll
                for (int hh = 0; hh < 3; ++hh) {
                    dots[hh] = qa[hh][0] * k0 + qa[hh][1] * k1 + qa[hh][2] * k2 + qa[hh][3] * k3;
                }
                #pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) {
                    #pragma unroll
                    for (int hh = 0; hh < 3; ++hh) {
                        dots[hh] += __shfl_xor_sync(0xFFFFFFFFu, dots[hh], mask);
                    }
                }
                #pragma unroll
                for (int hh = 0; hh < 3; ++hh) {
                    // Deterministic G-way online-softmax update shared identically across B=1 and B>=1.
                    float dot = __fmul_rn(dots[hh], scale);
                    if (pos - j < kExtent) dot = __fadd_rn(dot, smem_rel[hh][jj]);
                    const float m_new = (dot > m_w[hh]) ? dot : m_w[hh];
                    const float alpha = ptx_exp(m_w[hh] - m_new);
                    const float p = ptx_exp(dot - m_new);
                    l_w[hh] = l_w[hh] * alpha + p;
                    m_w[hh] = m_new;
                    o[hh][0] = o[hh][0] * alpha + p * v0;
                    o[hh][1] = o[hh][1] * alpha + p * v1;
                    o[hh][2] = o[hh][2] * alpha + p * v2;
                    o[hh][3] = o[hh][3] * alpha + p * v3;
                }
                wk = nwk; wv = nwv; sk = nsk; sv = nsv;
                jj = jn;
                j = jnx;
            }
        } else {
            for (int jj = warp_id; jj < kChunk; jj += 8) {
                const int j = j0 + jj;
                if (j >= n) break;
                const int slot_row = ring_row(j);
                float k0, k1, k2, k3, v0, v1, v2, v3;
                kv_load4_t<kInt8>(k_cache_fp16, k_scale, slot_row, kvh, lane, k0, k1, k2, k3);
                kv_load4_t<kInt8>(v_cache_fp16, v_scale, slot_row, kvh, lane, v0, v1, v2, v3);
                float dots[3];
                #pragma unroll
                for (int hh = 0; hh < 3; ++hh) {
                    dots[hh] = qa[hh][0] * k0 + qa[hh][1] * k1 + qa[hh][2] * k2 + qa[hh][3] * k3;
                }
                #pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) {
                    #pragma unroll
                    for (int hh = 0; hh < 3; ++hh) {
                        dots[hh] += __shfl_xor_sync(0xFFFFFFFFu, dots[hh], mask);
                    }
                }
                #pragma unroll
                for (int hh = 0; hh < 3; ++hh) {
                    // Deterministic G-way online-softmax update shared identically across B=1 and B>=1.
                    float dot = __fmul_rn(dots[hh], scale);
                    if (pos - j < kExtent) dot = __fadd_rn(dot, smem_rel[hh][jj]);
                    const float m_new = (dot > m_w[hh]) ? dot : m_w[hh];
                    const float alpha = ptx_exp(m_w[hh] - m_new);
                    const float p = ptx_exp(dot - m_new);
                    l_w[hh] = l_w[hh] * alpha + p;
                    m_w[hh] = m_new;
                    o[hh][0] = o[hh][0] * alpha + p * v0;
                    o[hh][1] = o[hh][1] * alpha + p * v1;
                    o[hh][2] = o[hh][2] * alpha + p * v2;
                    o[hh][3] = o[hh][3] * alpha + p * v3;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int hh = 0; hh < 3; ++hh) {
        if (lane == 0) {
            smem_m[warp_id][hh] = m_w[hh];
            smem_l[warp_id][hh] = l_w[hh];
        }
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            smem_o[warp_id][hh][lane * 4 + k] = o[hh][k];
        }
    }
    __syncthreads();

    for (int i = tid; i < 3 * kHd; i += 256) {
        const int hh = i / kHd;
        const int dd = i % kHd;
        float m_max = smem_m[0][hh];
        #pragma unroll
        for (int w = 1; w < 8; ++w) {
            m_max = fmaxf(m_max, smem_m[w][hh]);
        }
        float denom = 0.0f;
        float sum_d = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            const float s = ptx_exp(smem_m[w][hh] - m_max);
            denom += smem_l[w][hh] * s;
            sum_d += smem_o[w][hh][dd] * s;
        }
        const int h = kvh * 3 + hh;
        const size_t bh = static_cast<size_t>(b) * 24u + h;
        part_o[(bh * G + g) * kHd + dd] = sum_d;
        if (dd == 0) {
            part_ml[(bh * G + g) * 2] = m_max;
            part_ml[(bh * G + g) * 2 + 1] = denom;
        }
    }
}

extern "C" __global__ void ink2_gqa_attn_split(
    const float* __restrict__ qrgs_base,        // [B, qrgs_stride] (`q@0, r@3072, gate@3456, sinks@6528`)
    int32_t qrgs_stride,
    const uint32_t* __restrict__ rel_proj_bf16, // [16, 512]
    const uint16_t* __restrict__ k_cache_fp16,  // [kv_cap, 8, 128]
    const uint16_t* __restrict__ v_cache_fp16,  // [kv_cap, 8, 128]
    const float* __restrict__ k_scale,          // [kv_cap, 8] INT8 KV scales, or nullptr (FP16)
    const float* __restrict__ v_scale,          // [kv_cap, 8] INT8 KV scales, or nullptr (FP16)
    const int32_t* __restrict__ pos_ptr,
    int32_t kv_cap,
    int32_t is_global,
    float* __restrict__ part_ml,                // [B, 24, G, 2]
    float* __restrict__ part_o) {
    gqa_attn_split_body<false>(qrgs_base, qrgs_stride, rel_proj_bf16, k_cache_fp16, v_cache_fp16, k_scale, v_scale, pos_ptr, kv_cap, is_global, part_ml, part_o);
}

// INT8-KV instantiation (`LEANSONIC_KV=int8`, Full-attention layers only).
// `__launch_bounds__(256, 4)` keeps 4 CTAs/SM (<= 64 regs) like the FP16 instantiation.
extern "C" __global__ void __launch_bounds__(256, 4) ink2_gqa_attn_split_i8(
    const float* __restrict__ qrgs_base,        // [B, qrgs_stride] (`q@0, r@3072, gate@3456, sinks@6528`)
    int32_t qrgs_stride,
    const uint32_t* __restrict__ rel_proj_bf16, // [16, 512]
    const uint16_t* __restrict__ k_cache_fp16,  // [kv_cap, 8, 128]
    const uint16_t* __restrict__ v_cache_fp16,  // [kv_cap, 8, 128]
    const float* __restrict__ k_scale,          // [kv_cap, 8] INT8 KV scales, or nullptr (FP16)
    const float* __restrict__ v_scale,          // [kv_cap, 8] INT8 KV scales, or nullptr (FP16)
    const int32_t* __restrict__ pos_ptr,
    int32_t kv_cap,
    int32_t is_global,
    float* __restrict__ part_ml,                // [B, 24, G, 2]
    float* __restrict__ part_o) {
    gqa_attn_split_body<true>(qrgs_base, qrgs_stride, rel_proj_bf16, k_cache_fp16, v_cache_fp16, k_scale, v_scale, pos_ptr, kv_cap, is_global, part_ml, part_o);
}

// Merge `G` split partials per head + sigmoid gate. Grid: `(24, B)`, Block: `128`.
extern "C" __global__ void ink2_gqa_attn_combine(
    const float* __restrict__ part_ml,   // [B, 24, G, 2]
    const float* __restrict__ part_o,    // [B, 24, G, 128]
    const float* __restrict__ qrgs_base, // [B, qrgs_stride] (`gate @ 3456`)
    int32_t qrgs_stride,
    int32_t G,
    float* __restrict__ attn_out_base) { // [B, 24, 128]
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int dd = threadIdx.x;
    const size_t bh = static_cast<size_t>(b) * 24u + h;
    const float* gate_buf = qrgs_base + static_cast<size_t>(b) * qrgs_stride + 3456;

    float m_max = -1e30f;
    for (int g = 0; g < G; ++g) {
        m_max = fmaxf(m_max, part_ml[(bh * G + g) * 2]);
    }
    float denom = 0.0f;
    float sum_d = 0.0f;
    for (int g = 0; g < G; ++g) {
        const float s = ptx_exp(part_ml[(bh * G + g) * 2] - m_max);
        denom += part_ml[(bh * G + g) * 2 + 1] * s;
        sum_d += part_o[(bh * G + g) * kHd + dd] * s;
    }
    const float gt = ptx_sigmoid(gate_buf[h * kHd + dd]);
    attn_out_base[bh * kHd + dd] = (sum_d * ptx_rcp(fmaxf(denom, 1e-20f))) * gt;
}

// ============================================================================
// 17. Cycle 5 (`v0.6.0`) `0.0000-ULP` Structural Glue Fusion & Coalesced Suite
//     (`Issue #11`: Eliminates ~177 micro-kernel launches + DRAM round-trips per
//     token across BOTH `B = 1` CUDA-Graph Decode and `B = 1..64` `DSpark` Verify!)
// ============================================================================

namespace leansonic::inkling2 {

// Exact `0-ULP` in-register `Q8_64` quantizer for 1 attention head (`128` dims = `2` blocks of `64`).
// Caller MUST pass `2` `float[2]` + `float[2]` SMEM arrays and ensure all CTA threads hit `__syncthreads()`.
__device__ __forceinline__ void quant_head128_epilogue_q8_64(
    int tid,
    int h,
    int b,
    float out_val,
    float smem_qmax[2][2],
    float smem_qinv[2],
    uint32_t* __restrict__ out_q8_u32, // [B, 768]
    float* __restrict__ out_sc64) {    // [B, 48]
    const int sub_blk = tid >> 6;          // 0 or 1 for tid < 128
    const int sub_tid = tid & 63;
    const int sub_warp = (tid >> 5) & 1;
    const int lane = tid & 31;

    if (tid < 128) {
        float amax = fabsf(out_val);
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask));
        }
        if (lane == 0) {
            smem_qmax[sub_blk][sub_warp] = amax;
        }
    }
    __syncthreads();
    if (tid < 128 && sub_tid == 0) {
        float m = fmaxf(smem_qmax[sub_blk][0], smem_qmax[sub_blk][1]);
        out_sc64[static_cast<size_t>(b) * 48u + (h * 2 + sub_blk)] = m * (1.0f / 127.0f);
        smem_qinv[sub_blk] = (m > 0.0f) ? (127.0f * ptx_rcp(m)) : 0.0f;
    }
    __syncthreads();
    if (tid < 128) {
        int q = __float2int_rn(out_val * smem_qinv[sub_blk]);
        if (q < -127) q = -127;
        if (q > 127) q = 127;
        uint32_t qb = static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(q)));
        uint32_t p4 = qb
            | (__shfl_sync(0xFFFFFFFFu, qb, lane + 1) << 8)
            | (__shfl_sync(0xFFFFFFFFu, qb, lane + 2) << 16)
            | (__shfl_sync(0xFFFFFFFFu, qb, lane + 3) << 24);
        if ((lane & 3) == 0) {
            out_q8_u32[static_cast<size_t>(b) * 768u + (h * 2 + sub_blk) * 16 + sub_warp * 8 + (lane >> 2)] = p4;
        }
    }
}

} // namespace leansonic::inkling2

// Fused `depth_mix` + `rmsnorm_quant_q8_64` (`Grid: (n_q, B_out)`, `Block: 256`):
// Stages `mixed[2304]` in SMEM (`9,216` B) so `d_inputs` is never written to or read from DRAM!
extern "C" __global__ void ink2_depth_mix_rmsnorm_quant_q8_64_b(
    const float* __restrict__ blocks_buf,          // [max_blocks, stride_B, D]
    int32_t n_blocks,
    int32_t stride_B,
    int32_t b_offset,
    int32_t B_out,
    const float* __restrict__ cur_opt,             // [stride_B, D] if `append_cur == 1`
    int32_t append_cur,
    const float* __restrict__ queries,             // [n_q, D]
    const uint32_t* __restrict__ gamma_bf16_pairs, // [n_q, 1152]
    uint32_t* __restrict__ out_q8_u32,             // [n_q, B_out, 576]
    float* __restrict__ out_sc64) {                // [n_q, B_out, 36]
    __shared__ float smem_mix[kD];
    __shared__ float smem_sq[6][8];
    __shared__ float smem_dq[6][8];
    __shared__ float smem_probs[6];
    __shared__ float smem_inv[6];
    __shared__ float smem_scores[6];
    __shared__ float smem_red[8];

    const int q_idx = blockIdx.x;
    const int b_out = blockIdx.y;
    const int b_src = b_offset + b_out;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int total_states = n_blocks + append_cur;
    const float* q = queries + static_cast<size_t>(q_idx) * kD;
    const float inv_sqrt_d = 0.020833333333333332f;

    float q_reg[9];
    #pragma unroll
    for (int t = 0; t < 9; ++t) {
        q_reg[t] = q[tid + t * 256];
    }

    for (int s = 0; s < total_states; ++s) {
        const float* st = (s < n_blocks)
            ? (blocks_buf + (static_cast<size_t>(s) * stride_B + b_src) * kD)
            : (cur_opt + static_cast<size_t>(b_src) * kD);
        float sq = 0.0f;
        float dq = 0.0f;
        #pragma unroll
        for (int t = 0; t < 9; ++t) {
            float v = st[tid + t * 256];
            sq += v * v;
            dq += v * q_reg[t];
        }
        sq = warp_sum(sq);
        dq = warp_sum(dq);
        if (lane == 0) {
            smem_sq[s][warp_id] = sq;
            smem_dq[s][warp_id] = dq;
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        for (int s = 0; s < total_states; ++s) {
            float total_sq = (lane < 8) ? smem_sq[s][lane] : 0.0f;
            float total_dq = (lane < 8) ? smem_dq[s][lane] : 0.0f;
            total_sq = warp_sum(total_sq);
            total_dq = warp_sum(total_dq);
            if (lane == 0) {
                float inv = ptx_rsqrt(total_sq * (1.0f / static_cast<float>(kD)) + 1e-6f);
                smem_inv[s] = inv;
                smem_scores[s] = total_dq * inv * inv_sqrt_d;
            }
        }
        if (lane == 0) {
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
                smem_probs[s] = smem_probs[s] * inv_z * smem_inv[s];
            }
        }
    }
    __syncthreads();

    float sq_norm = 0.0f;
    #pragma unroll
    for (int t = 0; t < 9; ++t) {
        const int i = tid + t * 256;
        float acc = 0.0f;
        for (int s = 0; s < total_states; ++s) {
            const float* st = (s < n_blocks)
                ? (blocks_buf + (static_cast<size_t>(s) * stride_B + b_src) * kD)
                : (cur_opt + static_cast<size_t>(b_src) * kD);
            acc += smem_probs[s] * st[i];
        }
        smem_mix[i] = acc;
        sq_norm += acc * acc;
    }
    sq_norm = warp_sum(sq_norm);
    if (lane == 0) smem_red[warp_id] = sq_norm;
    __syncthreads();
    if (tid < 32) {
        float bsum = (tid < 8) ? smem_red[tid] : 0.0f;
        bsum = warp_sum(bsum);
        if (tid == 0) smem_red[0] = ptx_rsqrt(bsum * (1.0f / static_cast<float>(kD)) + 1e-6f);
    }
    __syncthreads();
    const float k = smem_red[0];

    const size_t vec_idx = static_cast<size_t>(q_idx) * B_out + b_out;
    const uint32_t* gamma = gamma_bf16_pairs + static_cast<size_t>(q_idx) * (kD / 2);
    uint32_t* dst_q8 = out_q8_u32 + vec_idx * (kD / 4);
    float* dst_sc = out_sc64 + vec_idx * (kD / 64);

    for (int blk = warp_id; blk < (kD / 64); blk += 8) {
        uint32_t gw = gamma[blk * 32 + lane];
        int base = blk * 64 + lane * 2;
        float y0 = smem_mix[base] * k * bf16_to_f32(static_cast<uint16_t>(gw & 0xFFFFu));
        float y1 = smem_mix[base + 1] * k * bf16_to_f32(static_cast<uint16_t>(gw >> 16));
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

// Fused `depth_mix` + `rmsnorm_d` + optional `quant_q8_16_packed` (`Grid: (1, B_out)`, `Block: 256`):
// Collapses 3 MLP prologue kernels (`depth_mix -> rmsnorm_d -> quant_q8_16`) into 1 (`0-ULP` exact)!
extern "C" __global__ void ink2_depth_mix_rmsnorm_d_quant_q8_16_b(
    const float* __restrict__ blocks_buf,          // [max_blocks, stride_B, D]
    int32_t n_blocks,
    int32_t stride_B,
    int32_t b_offset,
    const float* __restrict__ cur_opt,             // [stride_B, D] if `append_cur == 1`
    int32_t append_cur,
    const float* __restrict__ query,               // [D]
    const uint32_t* __restrict__ gamma_bf16_pairs, // [D / 2]
    float* __restrict__ out_normed,                // [B_out, D] (for BF16 router / tap3)
    int32_t do_q8_16,
    uint32_t* __restrict__ out_q8_u32,             // [B_out, D / 4] when `do_q8_16 == 1`
    float* __restrict__ out_sc16) {                // [B_out, D / 16] when `do_q8_16 == 1`
    __shared__ float smem_x[kD];
    __shared__ float smem_sq[6][8];
    __shared__ float smem_dq[6][8];
    __shared__ float smem_probs[6];
    __shared__ float smem_inv[6];
    __shared__ float smem_scores[6];
    __shared__ float smem_red[8];

    const int b_out = blockIdx.y;
    const int b_src = b_offset + b_out;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int total_states = n_blocks + append_cur;
    const float inv_sqrt_d = 0.020833333333333332f;

    float q_reg[9];
    #pragma unroll
    for (int t = 0; t < 9; ++t) {
        q_reg[t] = query[tid + t * 256];
    }

    for (int s = 0; s < total_states; ++s) {
        const float* st = (s < n_blocks)
            ? (blocks_buf + (static_cast<size_t>(s) * stride_B + b_src) * kD)
            : (cur_opt + static_cast<size_t>(b_src) * kD);
        float sq = 0.0f;
        float dq = 0.0f;
        #pragma unroll
        for (int t = 0; t < 9; ++t) {
            float v = st[tid + t * 256];
            sq += v * v;
            dq += v * q_reg[t];
        }
        sq = warp_sum(sq);
        dq = warp_sum(dq);
        if (lane == 0) {
            smem_sq[s][warp_id] = sq;
            smem_dq[s][warp_id] = dq;
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        for (int s = 0; s < total_states; ++s) {
            float total_sq = (lane < 8) ? smem_sq[s][lane] : 0.0f;
            float total_dq = (lane < 8) ? smem_dq[s][lane] : 0.0f;
            total_sq = warp_sum(total_sq);
            total_dq = warp_sum(total_dq);
            if (lane == 0) {
                float inv = ptx_rsqrt(total_sq * (1.0f / static_cast<float>(kD)) + 1e-6f);
                smem_inv[s] = inv;
                smem_scores[s] = total_dq * inv * inv_sqrt_d;
            }
        }
        if (lane == 0) {
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
                smem_probs[s] = smem_probs[s] * inv_z * smem_inv[s];
            }
        }
    }
    __syncthreads();

    float sq_norm = 0.0f;
    #pragma unroll
    for (int t = 0; t < 9; ++t) {
        const int i = tid + t * 256;
        float acc = 0.0f;
        for (int s = 0; s < total_states; ++s) {
            const float* st = (s < n_blocks)
                ? (blocks_buf + (static_cast<size_t>(s) * stride_B + b_src) * kD)
                : (cur_opt + static_cast<size_t>(b_src) * kD);
            acc += smem_probs[s] * st[i];
        }
        smem_x[i] = acc;
        sq_norm += acc * acc;
    }
    sq_norm = warp_sum(sq_norm);
    if (lane == 0) smem_red[warp_id] = sq_norm;
    __syncthreads();
    if (tid < 32) {
        float bsum = (tid < 8) ? smem_red[tid] : 0.0f;
        bsum = warp_sum(bsum);
        if (tid == 0) smem_red[0] = ptx_rsqrt(bsum * (1.0f / static_cast<float>(kD)) + 1e-6f);
    }
    __syncthreads();
    const float k = smem_red[0];

    float* dst_f32 = out_normed + static_cast<size_t>(b_out) * kD;
    for (int idx = tid; idx < kD / 2; idx += 256) {
        uint32_t gw = gamma_bf16_pairs[idx];
        float y0 = smem_x[idx * 2] * k * bf16_to_f32(static_cast<uint16_t>(gw & 0xFFFFu));
        float y1 = smem_x[idx * 2 + 1] * k * bf16_to_f32(static_cast<uint16_t>(gw >> 16));
        dst_f32[idx * 2] = y0;
        dst_f32[idx * 2 + 1] = y1;
        if (do_q8_16) {
            smem_x[idx * 2] = y0;
            smem_x[idx * 2 + 1] = y1;
        }
    }
    if (!do_q8_16) return;
    __syncthreads();

    const int hw = tid >> 4;     // 0..15 (16 half-warps)
    const int lane16 = tid & 15; // 0..15
    uint32_t* dst_q8 = out_q8_u32 + static_cast<size_t>(b_out) * (kD >> 2);
    float* dst_sc = out_sc16 + static_cast<size_t>(b_out) * (kD >> 4);
    for (int blk16 = hw; blk16 < (kD / 16); blk16 += 16) {
        float v = smem_x[blk16 * 16 + lane16];
        float amax = fabsf(v);
        #pragma unroll
        for (int mask = 8; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask, 16));
        }
        float sc = amax * (1.0f / 127.0f);
        float inv = (amax > 0.0f) ? (127.0f * ptx_rcp(amax)) : 0.0f;
        int qv = __float2int_rn(v * inv);
        if (qv < -127) qv = -127;
        if (qv > 127) qv = 127;
        uint32_t qb = static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(qv)));
        uint32_t p4 = qb
            | (__shfl_sync(0xFFFFFFFFu, qb, lane16 + 1, 16) << 8)
            | (__shfl_sync(0xFFFFFFFFu, qb, lane16 + 2, 16) << 16)
            | (__shfl_sync(0xFFFFFFFFu, qb, lane16 + 3, 16) << 24);
        if ((lane16 & 3) == 0) {
            dst_q8[blk16 * 4 + (lane16 >> 2)] = p4;
        }
        if (lane16 == 0) {
            dst_sc[blk16] = sc;
        }
    }
}

// Fused `rmsnorm_d` + `quant_q8_16_packed` (`Grid: (1, B)`, `Block: 256`) for `has_msgs` Loop Core MLP!
extern "C" __global__ void ink2_rmsnorm_d_quant_q8_16_b(
    const float* __restrict__ inp_base,            // [B, D]
    const uint32_t* __restrict__ gamma_bf16_pairs, // [D / 2]
    float* __restrict__ out_normed,                // [B, D] (may alias `inp_base`)
    uint32_t* __restrict__ out_q8_u32,             // [B, D / 4]
    float* __restrict__ out_sc16) {                // [B, D / 16]
    __shared__ float smem_x[kD];
    __shared__ float smem_red[8];

    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const float* inp = inp_base + static_cast<size_t>(b) * kD;
    float* dst_f32 = out_normed + static_cast<size_t>(b) * kD;

    float sq = 0.0f;
    for (int i = tid; i < kD; i += 256) {
        float v = inp[i];
        smem_x[i] = v;
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

    for (int idx = tid; idx < kD / 2; idx += 256) {
        uint32_t gw = gamma_bf16_pairs[idx];
        float y0 = smem_x[idx * 2] * k * bf16_to_f32(static_cast<uint16_t>(gw & 0xFFFFu));
        float y1 = smem_x[idx * 2 + 1] * k * bf16_to_f32(static_cast<uint16_t>(gw >> 16));
        dst_f32[idx * 2] = y0;
        dst_f32[idx * 2 + 1] = y1;
        smem_x[idx * 2] = y0;
        smem_x[idx * 2 + 1] = y1;
    }
    __syncthreads();

    const int hw = tid >> 4;
    const int lane16 = tid & 15;
    uint32_t* dst_q8 = out_q8_u32 + static_cast<size_t>(b) * (kD >> 2);
    float* dst_sc = out_sc16 + static_cast<size_t>(b) * (kD >> 4);
    for (int blk16 = hw; blk16 < (kD / 16); blk16 += 16) {
        float v = smem_x[blk16 * 16 + lane16];
        float amax = fabsf(v);
        #pragma unroll
        for (int mask = 8; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask, 16));
        }
        float sc = amax * (1.0f / 127.0f);
        float inv = (amax > 0.0f) ? (127.0f * ptx_rcp(amax)) : 0.0f;
        int qv = __float2int_rn(v * inv);
        if (qv < -127) qv = -127;
        if (qv > 127) qv = 127;
        uint32_t qb = static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(qv)));
        uint32_t p4 = qb
            | (__shfl_sync(0xFFFFFFFFu, qb, lane16 + 1, 16) << 8)
            | (__shfl_sync(0xFFFFFFFFu, qb, lane16 + 2, 16) << 16)
            | (__shfl_sync(0xFFFFFFFFu, qb, lane16 + 3, 16) << 24);
        if ((lane16 & 3) == 0) {
            dst_q8[blk16 * 4 + (lane16 >> 2)] = p4;
        }
        if (lane16 == 0) {
            dst_sc[blk16] = sc;
        }
    }
}

// Coalesced `512B`-Burst Controller Apply (`Grid: (36, n_ctrls * B)`, `Block: 64` = `2` warps):
// Replaces `9`-CTA `128B`-strided reads with `108`-CTA contiguous `LDG.E.128` SMEM bursts (`0-ULP` exact)!
extern "C" __global__ void ink2_ctrl_apply_coalesced_b(
    float* __restrict__ cur_inout_base,       // [n_ctrls, B, D]
    const float* __restrict__ msg_base,       // [n_ctrls, B, D]
    const float* __restrict__ g_vec_base,     // [n_ctrls, B, 64]
    const uint4* __restrict__ gain_bf16_u128, // [n_ctrls, D, 8]
    const uint4* __restrict__ forget_bf16_u128,
    int32_t B) {
    __shared__ float smem_gv[64];
    __shared__ uint4 smem_buf[2][32][9];

    const int tid = threadIdx.x; // 0..63
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int cb = blockIdx.y;
    const int c_idx = cb / B;
    const int row_base = (blockIdx.x * 2 + warp_id) * 32;
    const int row = row_base + lane;

    smem_gv[tid] = g_vec_base[cb * 64 + tid];
    __syncthreads();

    const size_t tile_row_offset = (static_cast<size_t>(c_idx) * kD + row_base) * 8u;
    const uint4* g_tile = gain_bf16_u128 + tile_row_offset;
    const uint4* f_tile = forget_bf16_u128 + tile_row_offset;

    #pragma unroll
    for (int it = 0; it < 8; ++it) {
        const int idx = it * 32 + lane;
        smem_buf[warp_id][idx >> 3][idx & 7] = g_tile[idx];
    }
    __syncwarp();

    float gain_acc = 0.0f;
    #pragma unroll
    for (int c8 = 0; c8 < 8; ++c8) {
        uint4 gp = smem_buf[warp_id][lane][c8];
        uint32_t gu[4] = {gp.x, gp.y, gp.z, gp.w};
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            float gv0 = smem_gv[c8 * 8 + p * 2];
            float gv1 = smem_gv[c8 * 8 + p * 2 + 1];
            gain_acc += bf16_to_f32(static_cast<uint16_t>(gu[p] & 0xFFFFu)) * gv0
                      + bf16_to_f32(static_cast<uint16_t>(gu[p] >> 16)) * gv1;
        }
    }
    __syncwarp();

    #pragma unroll
    for (int it = 0; it < 8; ++it) {
        const int idx = it * 32 + lane;
        smem_buf[warp_id][idx >> 3][idx & 7] = f_tile[idx];
    }
    __syncwarp();

    float forget_acc = 0.0f;
    #pragma unroll
    for (int c8 = 0; c8 < 8; ++c8) {
        uint4 fp = smem_buf[warp_id][lane][c8];
        uint32_t fu[4] = {fp.x, fp.y, fp.z, fp.w};
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            float gv0 = smem_gv[c8 * 8 + p * 2];
            float gv1 = smem_gv[c8 * 8 + p * 2 + 1];
            forget_acc += bf16_to_f32(static_cast<uint16_t>(fu[p] & 0xFFFFu)) * gv0
                        + bf16_to_f32(static_cast<uint16_t>(fu[p] >> 16)) * gv1;
        }
    }

    float* cur_inout = cur_inout_base + static_cast<size_t>(cb) * kD;
    const float* msg = msg_base + static_cast<size_t>(cb) * kD;
    float c = cur_inout[row];
    cur_inout[row] = c - ptx_tanh(forget_acc) * c + ptx_tanh(gain_acc) * msg[row];
}

// Unified `G=0` Attention + In-Register `Q8_64` Epilogue (`Grid: (24, B)`, `Block: 256`):
// Eliminates `d_attn_out` DRAM writes/reads and `48` `ink2_quant_q8_64` launches per token/step!
template<bool kInt8, bool PipelineDecode>
__device__ __forceinline__ void gqa_rel_sink_attn_q8(
    const float* __restrict__ qrgs_base,        // [B, qrgs_stride]
    int32_t qrgs_stride,
    const uint32_t* __restrict__ rel_proj_bf16, // [16, 512]
    const uint16_t* __restrict__ k_cache_fp16,  // [kv_cap, 1024]
    const uint16_t* __restrict__ v_cache_fp16,  // [kv_cap, 1024]
    const float* __restrict__ k_scale,          // [kv_cap, 8] INT8 KV scales (unused for FP16)
    const float* __restrict__ v_scale,          // [kv_cap, 8] INT8 KV scales (unused for FP16)
    const int32_t* __restrict__ pos_ptr,
    int32_t kv_cap,
    int32_t is_global,
    uint32_t* __restrict__ out_q8_u32,          // [B, 768]
    float* __restrict__ out_sc64) {             // [B, 48]
    __shared__ float smem_rel[kExtent];
    __shared__ float smem_q[kHd];
    __shared__ float smem_r[kRel];
    __shared__ float smem_m[8];
    __shared__ float smem_l[8];
    __shared__ float smem_o[8][kHd];
    __shared__ float smem_qmax[2][2];
    __shared__ float smem_qinv[2];

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

    auto score = [&](int j, uint64_t kw) {
        float k0 = fp16_bits_to_f32(static_cast<uint16_t>(kw & 0xFFFFu));
        float k1 = fp16_bits_to_f32(static_cast<uint16_t>((kw >> 16) & 0xFFFFu));
        float k2 = fp16_bits_to_f32(static_cast<uint16_t>((kw >> 32) & 0xFFFFu));
        float k3 = fp16_bits_to_f32(static_cast<uint16_t>((kw >> 48) & 0xFFFFu));

        float dot = warp_sum(q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3) * scale;
        int d = pos - j;
        if (d < kExtent) {
            dot += smem_rel[d];
        }
        return dot;
    };
    auto advance_softmax = [&](float dot, float& alpha, float& p) {
        float m_new = (dot > m_w) ? dot : m_w;
        alpha = ptx_exp(m_w - m_new);
        p = ptx_exp(dot - m_new);
        l_w = l_w * alpha + p;
        m_w = m_new;
    };
    auto add_value = [&](uint64_t vw, float alpha, float p) {
        o0 = o0 * alpha + p * fp16_bits_to_f32(static_cast<uint16_t>(vw & 0xFFFFu));
        o1 = o1 * alpha + p * fp16_bits_to_f32(static_cast<uint16_t>((vw >> 16) & 0xFFFFu));
        o2 = o2 * alpha + p * fp16_bits_to_f32(static_cast<uint16_t>((vw >> 32) & 0xFFFFu));
        o3 = o3 * alpha + p * fp16_bits_to_f32(static_cast<uint16_t>((vw >> 48) & 0xFFFFu));
    };
    int j = first + warp_id;
    if constexpr (kInt8) {
        // INT8 KV: same staging as the FP16 path, on raw INT8 words + per-row/head scales.
        // Ring rows without a per-key `%` (see `gqa_attn_split_body`).
        const int ring_off = is_global ? 0 : first - first % kv_cap;
        auto ring_row = [&](int key) {
            const int r = key - ring_off;
            return r < kv_cap ? r : r - kv_cap;
        };
        auto score_i8 = [&](int j, uint32_t kw, float ks) {
            float k0, k1, k2, k3;
            kv_dequant_i8(kw, ks, k0, k1, k2, k3);
            float dot = warp_sum(q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3) * scale;
            int d = pos - j;
            if (d < kExtent) {
                dot += smem_rel[d];
            }
            return dot;
        };
        auto add_value_i8 = [&](uint32_t vw, float vs, float alpha, float p) {
            float v0, v1, v2, v3;
            kv_dequant_i8(vw, vs, v0, v1, v2, v3);
            o0 = o0 * alpha + p * v0;
            o1 = o1 * alpha + p * v1;
            o2 = o2 * alpha + p * v2;
            o3 = o3 * alpha + p * v3;
        };
        if constexpr (PipelineDecode) {
            constexpr int steps = 8;
            for (; j + (steps - 1) * 8 < n; j += steps * 8) {
                uint32_t keys[steps], values[steps];
                float kscales[steps], vscales[steps], scores[steps];
                #pragma unroll
                for (int i = 0; i < steps; ++i) {
                    const int position = j + i * 8;
                    const int row = ring_row(position);
                    kv_raw_i8(k_cache_fp16, k_scale, row, kvh, lane, keys[i], kscales[i]);
                    kv_raw_i8(v_cache_fp16, v_scale, row, kvh, lane, values[i], vscales[i]);
                }
                #pragma unroll
                for (int i = 0; i < steps; ++i) {
                    scores[i] = score_i8(j + i * 8, keys[i], kscales[i]);
                }
                #pragma unroll
                for (int i = 0; i < steps; ++i) {
                    float alpha, p;
                    advance_softmax(scores[i], alpha, p);
                    add_value_i8(values[i], vscales[i], alpha, p);
                }
            }
        }
        for (; j < n; j += 8) {
            const int slot_row = ring_row(j);
            uint32_t kw, vw;
            float ks, vs;
            kv_raw_i8(k_cache_fp16, k_scale, slot_row, kvh, lane, kw, ks);
            const float dot = score_i8(j, kw, ks);
            float alpha, p;
            advance_softmax(dot, alpha, p);
            kv_raw_i8(v_cache_fp16, v_scale, slot_row, kvh, lane, vw, vs);
            add_value_i8(vw, vs, alpha, p);
        }
    } else {
        if constexpr (PipelineDecode) {
            // Only 24 CTAs serve one token. Issue independent K/V loads and
            // dot products ahead of the sequential online-softmax recurrence.
            constexpr int steps = 8;
            for (; j + (steps - 1) * 8 < n; j += steps * 8) {
                uint64_t keys[steps], values[steps];
                float scores[steps];
                #pragma unroll
                for (int i = 0; i < steps; ++i) {
                    const int position = j + i * 8;
                    const int row = is_global ? position : position % kv_cap;
                    const size_t index = static_cast<size_t>(row) * 1024 + kvh * kHd + lane * 4;
                    keys[i] = *reinterpret_cast<const uint64_t*>(k_cache_fp16 + index);
                    values[i] = *reinterpret_cast<const uint64_t*>(v_cache_fp16 + index);
                }
                #pragma unroll
                for (int i = 0; i < steps; ++i) {
                    scores[i] = score(j + i * 8, keys[i]);
                }
                #pragma unroll
                for (int i = 0; i < steps; ++i) {
                    float alpha, p;
                    advance_softmax(scores[i], alpha, p);
                    add_value(values[i], alpha, p);
                }
            }
        }
        auto attend = [&](int key, int slot_row) {
            const uint64_t* kp = reinterpret_cast<const uint64_t*>(
                k_cache_fp16 + static_cast<size_t>(slot_row) * 1024u + kvh * 128);
            uint64_t kw = kp[lane];
            const float dot = score(key, kw);
            float alpha, p;
            advance_softmax(dot, alpha, p);

            const uint64_t* vp = reinterpret_cast<const uint64_t*>(
                v_cache_fp16 + static_cast<size_t>(slot_row) * 1024u + kvh * 128);
            uint64_t vw = vp[lane];
            add_value(vw, alpha, p);
        };
        if constexpr (PipelineDecode) {
            for (; j < n; j += 8) attend(j, is_global ? j : j % kv_cap);
        } else if (is_global || n <= kv_cap) {
            for (; j < n; j += 8) attend(j, j);
        } else {
            // A sliding window spans at most kv_cap rows. Compute its ring
            // offset once, then each physical row needs at most one wrap.
            const int offset = first - first % kv_cap;
            for (; j < n; j += 8) {
                const int row = j - offset;
                attend(j, row < kv_cap ? row : row - kv_cap);
            }
        }
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

    float out_val = 0.0f;
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
        out_val = (sum_d * ptx_rcp(fmaxf(denom, 1e-20f))) * g;
    }
    quant_head128_epilogue_q8_64(tid, h, b, out_val, smem_qmax, smem_qinv, out_q8_u32, out_sc64);
}

extern "C" __global__ void ink2_gqa_rel_sink_attn_q8_b(
    const float* __restrict__ qrgs_base, int32_t qrgs_stride, const uint32_t* __restrict__ rel_proj_bf16,
    const uint16_t* __restrict__ k_cache_fp16, const uint16_t* __restrict__ v_cache_fp16,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    const int32_t* __restrict__ pos_ptr, int32_t kv_cap, int32_t is_global,
    uint32_t* __restrict__ out_q8_u32, float* __restrict__ out_sc64) {
    gqa_rel_sink_attn_q8<false, false>(qrgs_base, qrgs_stride, rel_proj_bf16,
        k_cache_fp16, v_cache_fp16, k_scale, v_scale, pos_ptr, kv_cap, is_global, out_q8_u32, out_sc64);
}

extern "C" __global__ void ink2_gqa_rel_sink_attn_q8_decode(
    const float* __restrict__ qrgs_base, int32_t qrgs_stride, const uint32_t* __restrict__ rel_proj_bf16,
    const uint16_t* __restrict__ k_cache_fp16, const uint16_t* __restrict__ v_cache_fp16,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    const int32_t* __restrict__ pos_ptr, int32_t kv_cap, int32_t is_global,
    uint32_t* __restrict__ out_q8_u32, float* __restrict__ out_sc64) {
    gqa_rel_sink_attn_q8<false, true>(qrgs_base, qrgs_stride, rel_proj_bf16,
        k_cache_fp16, v_cache_fp16, k_scale, v_scale, pos_ptr, kv_cap, is_global, out_q8_u32, out_sc64);
}

// INT8-KV instantiations (`LEANSONIC_KV=int8`); the FP16 kernels above keep main's codegen.
extern "C" __global__ void ink2_gqa_rel_sink_attn_q8_b_i8(
    const float* __restrict__ qrgs_base, int32_t qrgs_stride, const uint32_t* __restrict__ rel_proj_bf16,
    const uint16_t* __restrict__ k_cache_fp16, const uint16_t* __restrict__ v_cache_fp16,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    const int32_t* __restrict__ pos_ptr, int32_t kv_cap, int32_t is_global,
    uint32_t* __restrict__ out_q8_u32, float* __restrict__ out_sc64) {
    gqa_rel_sink_attn_q8<true, false>(qrgs_base, qrgs_stride, rel_proj_bf16,
        k_cache_fp16, v_cache_fp16, k_scale, v_scale, pos_ptr, kv_cap, is_global, out_q8_u32, out_sc64);
}

extern "C" __global__ void ink2_gqa_rel_sink_attn_q8_decode_i8(
    const float* __restrict__ qrgs_base, int32_t qrgs_stride, const uint32_t* __restrict__ rel_proj_bf16,
    const uint16_t* __restrict__ k_cache_fp16, const uint16_t* __restrict__ v_cache_fp16,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    const int32_t* __restrict__ pos_ptr, int32_t kv_cap, int32_t is_global,
    uint32_t* __restrict__ out_q8_u32, float* __restrict__ out_sc64) {
    gqa_rel_sink_attn_q8<true, true>(qrgs_base, qrgs_stride, rel_proj_bf16,
        k_cache_fp16, v_cache_fp16, k_scale, v_scale, pos_ptr, kv_cap, is_global, out_q8_u32, out_sc64);
}

// Split-KV Combine + In-Register `Q8_64` Epilogue (`Grid: (24, B)`, `Block: 128`):
extern "C" __global__ void ink2_gqa_attn_combine_q8(
    const float* __restrict__ part_ml,   // [B, 24, G, 2]
    const float* __restrict__ part_o,    // [B, 24, G, 128]
    const float* __restrict__ qrgs_base, // [B, qrgs_stride] (`gate @ 3456`)
    int32_t qrgs_stride,
    int32_t G,
    uint32_t* __restrict__ out_q8_u32,   // [B, 768]
    float* __restrict__ out_sc64) {      // [B, 48]
    __shared__ float smem_qmax[2][2];
    __shared__ float smem_qinv[2];

    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int dd = threadIdx.x; // 0..127
    const size_t bh = static_cast<size_t>(b) * 24u + h;
    const float* gate_buf = qrgs_base + static_cast<size_t>(b) * qrgs_stride + 3456;

    float m_max = -1e30f;
    for (int g = 0; g < G; ++g) {
        m_max = fmaxf(m_max, part_ml[(bh * G + g) * 2]);
    }
    float denom = 0.0f;
    float sum_d = 0.0f;
    for (int g = 0; g < G; ++g) {
        const float s = ptx_exp(part_ml[(bh * G + g) * 2] - m_max);
        denom += part_ml[(bh * G + g) * 2 + 1] * s;
        sum_d += part_o[(bh * G + g) * kHd + dd] * s;
    }
    const float gt = ptx_sigmoid(gate_buf[h * kHd + dd]);
    const float out_val = (sum_d * ptx_rcp(fmaxf(denom, 1e-20f))) * gt;
    quant_head128_epilogue_q8_64(dd, h, b, out_val, smem_qmax, smem_qinv, out_q8_u32, out_sc64);
}

// Causal 4-Tap Short-Conv + Fused Pre-Overwrite Block Save (`save_prev_out` nullable):
extern "C" __global__ void ink2_short_conv4_save(
    const float* __restrict__ cur,
    const float4* __restrict__ w_conv4,
    const float4* __restrict__ dyn_tap,
    float* __restrict__ hist3,
    float* out,
    const float* res_add,
    float* __restrict__ save_prev_out,
    int32_t channels,
    int32_t has_dyn,
    int32_t do_res_add) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    if (save_prev_out != nullptr) {
        save_prev_out[c] = out[c];
    }
    const float x = cur[c];
    float* h = hist3 + c * 3;
    const float h0 = h[0];
    const float h1 = h[1];
    const float h2 = h[2];
    const float4 w = w_conv4[c];
    const float4 d = has_dyn ? dyn_tap[c >> 7] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float res = do_res_add ? res_add[c] : 0.0f;

    h[0] = h1;
    h[1] = h2;
    h[2] = x;

    out[c] = eval_conv4_step(w, h0, h1, h2, x, has_dyn, d, do_res_add, res);
}

// Batched Causal 4-Tap Short-Conv + Fused Pre-Overwrite Block Save (`save_prev_base` nullable):
extern "C" __global__ void ink2_short_conv4_save_b(
    const float* __restrict__ cur_base,
    int32_t in_stride,
    const float4* __restrict__ w_conv4,
    const float4* __restrict__ dyn_base,
    int32_t dyn_f4_stride,
    float* __restrict__ hist3,
    float* out_base,
    int32_t out_stride,
    const float* res_base,
    int32_t res_stride,
    float* __restrict__ save_prev_base,
    float* __restrict__ seq_out,
    int32_t channels,
    int32_t B,
    int32_t has_dyn,
    int32_t do_res_add) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    if (save_prev_base != nullptr) {
        for (int b = 0; b < B; ++b) {
            save_prev_base[static_cast<size_t>(b) * out_stride + c] =
                out_base[static_cast<size_t>(b) * out_stride + c];
        }
    }
    float* h = hist3 + c * 3;
    float h0 = h[0];
    float h1 = h[1];
    float h2 = h[2];
    if (seq_out != nullptr) {
        seq_out[0 * channels + c] = h0;
        seq_out[1 * channels + c] = h1;
        seq_out[2 * channels + c] = h2;
    }
    const float4 w = w_conv4[c];
    const int head = c >> 7;

    for (int b = 0; b < B; ++b) {
        const float x = cur_base[static_cast<size_t>(b) * in_stride + c];
        if (seq_out != nullptr) {
            seq_out[static_cast<size_t>(3 + b) * channels + c] = x;
        }
        const float4 d = has_dyn ? dyn_base[static_cast<size_t>(b) * dyn_f4_stride + head] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float res = do_res_add ? res_base[static_cast<size_t>(b) * res_stride + c] : 0.0f;
        const float y = eval_conv4_step(w, h0, h1, h2, x, has_dyn, d, do_res_add, res);
        h0 = h1;
        h1 = h2;
        h2 = x;
        out_base[static_cast<size_t>(b) * out_stride + c] = y;
    }

    h[0] = h0;
    h[1] = h1;
    h[2] = h2;
}

// One CTA owns 32 channels and their history. Its eight warps handle disjoint
// token ranges, seeding each range from the three preceding raw input rows.
__device__ __forceinline__ void short_conv4_tiled(
    const float* cur, int in_stride, const float4* weights, const float4* dyn, int dyn_stride,
    float* hist3, float* out, int out_stride, const float* residual, int res_stride,
    float* save_prev, float* seq, int channels, int B, int has_dyn, int do_res_add) {
    __shared__ float initial[3][32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int c = blockIdx.x * 32 + lane;
    if (warp == 0 && c < channels) {
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            initial[j][lane] = hist3[3 * c + j];
            if (seq) seq[j * channels + c] = initial[j][lane];
        }
    }
    __syncthreads(); // capture incoming history before the last warp updates it

    const int span = (B + 7) / 8;
    const int first = warp * span;
    const int end = first + span < B ? first + span : B;
    if (c >= channels || first >= end) return;
    auto preceding = [&](int t) {
        return t < 0 ? initial[t + 3][lane] : cur[static_cast<size_t>(t) * in_stride + c];
    };
    float h0 = preceding(first - 3);
    float h1 = preceding(first - 2);
    float h2 = preceding(first - 1);
    const float4 w = weights[c];
    for (int b = first; b < end; ++b) {
        const size_t dst = static_cast<size_t>(b) * out_stride + c;
        const float x = cur[static_cast<size_t>(b) * in_stride + c];
        if (save_prev) save_prev[dst] = out[dst];
        if (seq) seq[static_cast<size_t>(3 + b) * channels + c] = x;
        const float4 d = has_dyn ? dyn[static_cast<size_t>(b) * dyn_stride + (c >> 7)] : make_float4(0.f, 0.f, 0.f, 0.f);
        const float res = do_res_add ? residual[static_cast<size_t>(b) * res_stride + c] : 0.f;
        out[dst] = eval_conv4_step(w, h0, h1, h2, x, has_dyn, d, do_res_add, res);
        h0 = h1;
        h1 = h2;
        h2 = x;
    }
    if (end == B) {
        hist3[3 * c] = h0;
        hist3[3 * c + 1] = h1;
        hist3[3 * c + 2] = h2;
    }
}

extern "C" __global__ void ink2_short_conv4_tiled_b(
    const float* cur, int in_stride, const float4* weights, const float4* dyn, int dyn_stride,
    float* hist3, float* out, int out_stride, const float* residual, int res_stride,
    float* seq, int channels, int B, int has_dyn, int do_res_add) {
    short_conv4_tiled(cur, in_stride, weights, dyn, dyn_stride, hist3, out, out_stride, residual, res_stride,
                      nullptr, seq, channels, B, has_dyn, do_res_add);
}

extern "C" __global__ void ink2_short_conv4_tiled_save_b(
    const float* cur, int in_stride, const float4* weights, const float4* dyn, int dyn_stride,
    float* hist3, float* out, int out_stride, const float* residual, int res_stride,
    float* save_prev, float* seq, int channels, int B, int has_dyn, int do_res_add) {
    short_conv4_tiled(cur, in_stride, weights, dyn, dyn_stride, hist3, out, out_stride, residual, res_stride,
                      save_prev, seq, channels, B, has_dyn, do_res_add);
}

// Fused Prelude Save (`mode == 0`) & Loop Pass Transition (`mode == 1`) (`Grid: (9, B)`, `Block: 256`):
extern "C" __global__ void ink2_loop_pass_setup_b(
    float* __restrict__ cur_buf,            // [B, D]
    float* __restrict__ blocks_buf,         // [6, B, D]
    float* __restrict__ prelude_x_buf,      // [B, D]
    float* __restrict__ prelude_block0_buf, // [B, D]
    float* __restrict__ bank_buf,           // [2, B, D]
    int32_t B,
    int32_t mode) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (d >= kD || b >= B) return;
    const size_t bd = static_cast<size_t>(b) * kD + d;
    const size_t plane = static_cast<size_t>(B) * kD;

    if (mode == 0) {
        prelude_x_buf[bd] = cur_buf[bd];
        prelude_block0_buf[bd] = blocks_buf[bd];
    } else {
        bank_buf[bd] = blocks_buf[2u * plane + bd];
        bank_buf[plane + bd] = cur_buf[bd];
        cur_buf[bd] = prelude_x_buf[bd];
        blocks_buf[bd] = prelude_block0_buf[bd];
    }
}

// SM-Wave-Balanced 4-Rows/CTA (`Block: 64` = `2` warps × `2` rows/warp) Planar `Q8_64` `__dp4a` GEMV:
// Eliminates `40%–73%` idle SMs on `wo` (`576` CTAs vs `144`) and `wq_all` (`2,150` CTAs vs `538`)!
extern "C" __global__ void ink2_q8_dp4a_gemv_4r(
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

    const int tid = threadIdx.x; // 0..63
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int cols16 = cols >> 4;
    const int cols64 = cols >> 6;

    for (int i = tid; i < cols16; i += 64) {
        smem_x[i] = x_q8_u128[i];
    }
    if (tid < cols64) {
        smem_xsc[tid] = x_sc64[tid];
    }
    __syncthreads();

    const int row0 = (blockIdx.x * 2 + warp_id) * 2;
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
        acc0 = __fmaf_rn(static_cast<float>(d0), __fmul_rn(s_r0[c16 >> 2], xsc), acc0);

        if (has_row1) {
            uint4 w1 = w_r1[c16];
            int d1 = __dp4a(static_cast<int>(w1.x), static_cast<int>(xq.x), 0);
            d1 = __dp4a(static_cast<int>(w1.y), static_cast<int>(xq.y), d1);
            d1 = __dp4a(static_cast<int>(w1.z), static_cast<int>(xq.z), d1);
            d1 = __dp4a(static_cast<int>(w1.w), static_cast<int>(xq.w), d1);
            acc1 = __fmaf_rn(static_cast<float>(d1), __fmul_rn(s_r1[c16 >> 2], xsc), acc1);
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

// Fused 3-Projection `(Q, K, V)` Planar `Q8_64` `__dp4a` GEMV (`cols = kD = 2304`, `Block: 256` = `16` rows/CTA):
// Collapses `3` separate GEMVs (`wq`, `wk`, `wv`) across `32` `distinct_k_v` visits into `1` `542`-CTA wave!
extern "C" __global__ void ink2_q8_dp4a_qkv3_gemv(
    const uint4* __restrict__ wq_u128,
    const float* __restrict__ wq_sc64,
    int32_t rows_q,
    float* __restrict__ y_q,
    const uint4* __restrict__ wk_u128,
    const float* __restrict__ wk_sc64,
    int32_t rows_k,
    float* __restrict__ y_k,
    const uint4* __restrict__ wv_u128,
    const float* __restrict__ wv_sc64,
    int32_t rows_v,
    float* __restrict__ y_v,
    const uint4* __restrict__ x_q8_3_u128, // [3, 144]
    const float* __restrict__ x_sc64_3) {  // [3, 36]
    __shared__ uint4 smem_x[144];
    __shared__ float smem_xsc[36];

    const int cta_q = (rows_q + 15) >> 4;
    const int cta_k = (rows_k + 15) >> 4;
    const int bx = blockIdx.x;

    int slot;
    int cta_local;
    int rows;
    const uint4* w_u128;
    const float* w_sc;
    float* y;
    if (bx < cta_q) {
        slot = 0;
        cta_local = bx;
        rows = rows_q;
        w_u128 = wq_u128;
        w_sc = wq_sc64;
        y = y_q;
    } else if (bx < cta_q + cta_k) {
        slot = 1;
        cta_local = bx - cta_q;
        rows = rows_k;
        w_u128 = wk_u128;
        w_sc = wk_sc64;
        y = y_k;
    } else {
        slot = 2;
        cta_local = bx - cta_q - cta_k;
        rows = rows_v;
        w_u128 = wv_u128;
        w_sc = wv_sc64;
        y = y_v;
    }

    const int tid = threadIdx.x; // 0..255
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const uint4* x_q8 = x_q8_3_u128 + slot * 144;
    const float* x_sc = x_sc64_3 + slot * 36;

    if (tid < 144) {
        smem_x[tid] = x_q8[tid];
    }
    if (tid < 36) {
        smem_xsc[tid] = x_sc[tid];
    }
    __syncthreads();

    const int row0 = (cta_local * 8 + warp_id) * 2;
    if (row0 >= rows) return;
    const bool has_row1 = (row0 + 1 < rows);

    const uint4* w_r0 = w_u128 + static_cast<size_t>(row0) * 144u;
    const uint4* w_r1 = has_row1 ? (w_r0 + 144u) : w_r0;
    const float* s_r0 = w_sc + static_cast<size_t>(row0) * 36u;
    const float* s_r1 = has_row1 ? (s_r0 + 36u) : s_r0;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int c16 = lane; c16 < 144; c16 += 32) {
        uint4 xq = smem_x[c16];
        float xsc = smem_xsc[c16 >> 2];

        uint4 w0 = w_r0[c16];
        int d0 = __dp4a(static_cast<int>(w0.x), static_cast<int>(xq.x), 0);
        d0 = __dp4a(static_cast<int>(w0.y), static_cast<int>(xq.y), d0);
        d0 = __dp4a(static_cast<int>(w0.z), static_cast<int>(xq.z), d0);
        d0 = __dp4a(static_cast<int>(w0.w), static_cast<int>(xq.w), d0);
        acc0 = __fmaf_rn(static_cast<float>(d0), __fmul_rn(s_r0[c16 >> 2], xsc), acc0);

        if (has_row1) {
            uint4 w1 = w_r1[c16];
            int d1 = __dp4a(static_cast<int>(w1.x), static_cast<int>(xq.x), 0);
            d1 = __dp4a(static_cast<int>(w1.y), static_cast<int>(xq.y), d1);
            d1 = __dp4a(static_cast<int>(w1.z), static_cast<int>(xq.z), d1);
            d1 = __dp4a(static_cast<int>(w1.w), static_cast<int>(xq.w), d1);
            acc1 = __fmaf_rn(static_cast<float>(d1), __fmul_rn(s_r1[c16 >> 2], xsc), acc1);
        }
    }

    acc0 = warp_sum(acc0);
    acc1 = warp_sum(acc1);
    if (lane == 0) {
        y[row0] = acc0;
        if (has_row1) y[row0 + 1] = acc1;
    }
}

// SM-Wave-Balanced 2-Warps/CTA (`Grid: 1152`, `Block: 64`) NVFP4 `__dp4a` MoE Down:
// Eliminates the `288`-CTA (`240 + 48`) 2-wave stall on 240-SM `RTX PRO 5000` (`5` CTAs/SM co-resident)!
extern "C" __global__ void ink2_moe_nvfp4_dp4a_down_2w(
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

    const int tid = threadIdx.x; // 0..63
    for (int i = tid; i < 8 * 72; i += 64) {
        smem_aq[i] = act_q8_u128[i];
        smem_asc[i] = act_sc16[i];
    }
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 2 + warp_id;
    if (row >= kD) return;

    float sum_y = 0.0f;
    for (int k = 0; k < kTopK; ++k) {
        const int ex = sel_ids[k];
        const size_t r_idx = static_cast<size_t>(ex) * kD + row;
        float dot = warp_dot_nvfp4_dp4a_u128_row<false>(
            w2_u128 + r_idx * 36u,
            s2_u32, r_idx * 18u,
            smem_aq + k * 72,
            smem_asc + k * 72,
            36,
            lane);
        sum_y = __fmaf_rn(__fmul_rn(dot, a2[ex] * (1.0f / 2688.0f)), sel_wts[k], sum_y);
    }
    if (lane == 0) {
        moe_out[row] = sum_y;
    }
}

// SM-Wave-Balanced 2-Warps/CTA (`Grid: 1152`, `Block: 64`) MXFP8 MoE Down:
extern "C" __global__ void ink2_moe_mxfp8_down_2w(
    const uint4* __restrict__ w2_u128,
    const uint32_t* __restrict__ s2_u32,
    const float4* __restrict__ act4,
    const int32_t* __restrict__ sel_ids,
    const float* __restrict__ sel_wts,
    float* __restrict__ moe_out) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 2 + warp_id;
    if (row >= kD) return;

    float sum_y = 0.0f;
    for (int k = 0; k < kTopK; ++k) {
        const int ex = sel_ids[k];
        const size_t r_idx = static_cast<size_t>(ex) * kD + row;
        float dot = warp_dot_mxfp8_u128_row<false>(
            w2_u128 + r_idx * 72u,
            s2_u32, r_idx * 9u,
            act4 + k * 288,
            72,
            lane);
        sum_y = __fmaf_rn(dot, sel_wts[k], sum_y);
    }
    if (lane == 0) {
        moe_out[row] = sum_y;
    }
}

// Fused Dual K+V Causal 4-Tap Short-Conv (`Grid: (4, 2)`, `Block: 256` -> `-48` launches/tok!):
extern "C" __global__ void ink2_short_conv4_kv2(
    const float* __restrict__ cur_k,
    const float4* __restrict__ w_k,
    const float4* __restrict__ dyn_k,
    float* __restrict__ hist_k,
    float* __restrict__ out_k,
    const float* __restrict__ cur_v,
    const float4* __restrict__ w_v,
    const float4* __restrict__ dyn_v,
    float* __restrict__ hist_v,
    float* __restrict__ out_v,
    int32_t has_dyn) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= 1024) return;
    const bool is_v = (blockIdx.y != 0);
    const float* cur = is_v ? cur_v : cur_k;
    const float4* w_conv4 = is_v ? w_v : w_k;
    const float4* dyn_tap = is_v ? dyn_v : dyn_k;
    float* hist3 = is_v ? hist_v : hist_k;
    float* out = is_v ? out_v : out_k;

    const float x = cur[c];
    float* h = hist3 + c * 3;
    const float h0 = h[0];
    const float h1 = h[1];
    const float h2 = h[2];
    const float4 w = w_conv4[c];
    const float4 d = has_dyn ? dyn_tap[c >> 7] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    h[0] = h1;
    h[1] = h2;
    h[2] = x;

    out[c] = eval_conv4_step(w, h0, h1, h2, x, has_dyn, d, 0, 0.0f);
}

#undef INK2_SCALE_KERNEL
