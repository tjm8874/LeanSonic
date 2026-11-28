// ============================================================================
// LeanSonic — Full-GPU `Qwen3.8-Flash-Next` / `Qwen4Exp` SM120 Kernel Suite
// Target: NVIDIA Blackwell (`sm_120a` — RTX PRO 5000 72GB, PRO 4500 32GB, 5070 Ti / 5060 Ti 16GB)
//
// Architecture (`48` layers, `D = 2560`, `hc_count = 4`, `512` experts `Top-10`):
//   - Group-32 4-bit affine (`uint4` `LDG.E.128` = 32 nibbles + BF16 / lossless 24-bit Z24 scale/bias)
//   - Bank-conflict-free `stride = 9` `float4` SMEM staging (`smem_x9[g * 9 + 8].x = sum_x`)
//   - Programmatic Dependent Launch (`pdl_wait` / `pdl_trigger`) across the 48-layer pipeline
//   - 4-Stream Hyper-Connection (`hc_norm4_vec` + fused `hc_down_inj_10k` + fused `hc_up_mix4` + `hc_inject4_vec`)
//   - Layer 1 (`ple_layer_ids = [2]`) 3-Gram Host `Engram`/`PLE` Async HtoD + Dilated-3 Conv4
//   - 36-Layer Gated DeltaNet (`linear_attn`: `16` K-heads, `48` V-heads, `128` dim, FP32 state)
//   - 12-Layer QSA (`self_attn` + `indexer`: `24` Q-heads, `2` KV-heads, `256` dim, Top-512 blocks)
//   - 512-Expert `Top-10` + Gated Shared Expert (`k = 0..10` unified SwiGLU MoE, Z24 `0-ULP` & raw BF16)
//
// Obeys all Gate 2 static SASS rules (`ci/audit_sass.py`):
//   - `STACK: 0` (zero local-memory spills)
//   - Zero `CALL.REL.NOINC` (inline PTX `.approx` transcendentals)
//   - Zero 8-bit scalar `LDG.E.[U/S]8` (all byte/16-bit tables read as `uint32_t` / `uint4`)
// ============================================================================

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

namespace leansonic::qwen38 {

constexpr int kD = 2560;
constexpr int kHcD = 10240;
constexpr int kExp = 512;
constexpr int kTopK = 10;
constexpr int kMoeFf = 640;
constexpr int kVocab = 248320;

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

__device__ __forceinline__ float bf16_to_f32(uint16_t b) {
    return __uint_as_float(static_cast<uint32_t>(b) << 16);
}

__device__ __forceinline__ uint16_t f32_to_fp16_bits(float x) {
    uint16_t out;
    asm("cvt.rn.f16.f32 %0, %1;\n" : "=h"(out) : "f"(x));
    return out;
}

__device__ __forceinline__ float fp16_bits_to_f32(uint16_t h) {
    float out;
    asm("cvt.f32.f16 %0, %1;\n" : "=f"(out) : "h"(h));
    return out;
}

__device__ __forceinline__ float ptx_exp(float x) {
    float y;
    float z = x * 1.4426950408889634f;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(z));
    return y;
}

__device__ __forceinline__ float ptx_ln(float x) {
    float lg2;
    asm("lg2.approx.ftz.f32 %0, %1;\n" : "=f"(lg2) : "f"(x));
    return lg2 * 0.6931471805599453f;
}

__device__ __forceinline__ float ptx_rsqrt(float x) {
    float y;
    asm("rsqrt.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ float ptx_sigmoid(float x) {
    return ptx_rcp(1.0f + ptx_exp(-x));
}

__device__ __forceinline__ float ptx_silu(float x) {
    return x * ptx_sigmoid(x);
}

__device__ __forceinline__ float ptx_softplus(float x) {
    return (x > 20.0f) ? x : ptx_ln(1.0f + ptx_exp(x));
}

__device__ __forceinline__ void ptx_sincos_reduced(float angle, float* s_out, float* c_out) {
    float turns = floorf(angle * 0.15915494309189535f + 0.5f);
    float r = angle - turns * 6.283185307179586f;
    float sv, cv;
    asm("sin.approx.ftz.f32 %0, %1;\n" : "=f"(sv) : "f"(r));
    asm("cos.approx.ftz.f32 %0, %1;\n" : "=f"(cv) : "f"(r));
    *s_out = sv;
    *c_out = cv;
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, mask);
    }
    return v;
}

__device__ __forceinline__ float load_bf16_from_u32_array(const uint32_t* __restrict__ p_u32, int idx) {
    uint32_t pair = p_u32[idx >> 1];
    uint16_t u = ((idx & 1) == 0)
        ? static_cast<uint16_t>(pair & 0xFFFFu)
        : static_cast<uint16_t>(pair >> 16);
    return bf16_to_f32(u);
}

// 100% lossless (`0-ULP`) 24-bit planar `(scale, bias)` reconstruction via two `LDG.E.U32` loads.
__device__ __forceinline__ void load_z24_sb_pair(
    const uint32_t* __restrict__ sb_lo_u32,
    const uint32_t* __restrict__ sb_hi_u32,
    size_t idx,
    uint32_t base_sc_m1,
    uint32_t base_bi_m1,
    float* sc_out,
    float* bi_out) {
    uint32_t lo_pair = sb_lo_u32[idx >> 1];
    uint32_t lo16 = (lo_pair >> ((idx & 1u) * 16u)) & 0xFFFFu;
    uint32_t hi_quad = sb_hi_u32[idx >> 2];
    uint32_t hi8 = (hi_quad >> ((idx & 3u) * 8u)) & 0xFFu;

    uint32_t sc_sm = lo16 & 0xFFu;
    uint32_t sc_c = hi8 & 0xFu;
    uint32_t sc_exp = (sc_c == 0u) ? 0u : (base_sc_m1 + sc_c);
    *sc_out = __uint_as_float(((sc_sm & 0x80u) << 24) | (sc_exp << 23) | ((sc_sm & 0x7Fu) << 16));

    uint32_t bi_sm = lo16 >> 8;
    uint32_t bi_c = hi8 >> 4;
    uint32_t bi_exp = (bi_c == 0u) ? 0u : (base_bi_m1 + bi_c);
    *bi_out = __uint_as_float(((bi_sm & 0x80u) << 24) | (bi_exp << 23) | ((bi_sm & 0x7Fu) << 16));
}

// Warp-convergent staging of `n_groups` (`32` floats = `8` `float4`s/group) into bank-conflict-free
// `stride = 9` `float4` SMEM (`36` banks/group), storing `sum_{j=0..31} x_j` in `smem_x9[g * 9 + 8].x`.
__device__ __forceinline__ void stage_x_groups_smem9(
    const float4* x4,
    float4* smem_x9,
    int n_groups,
    int tid,
    int block_threads = 256) {
    const int sub = tid & 7;
    const int groups_per_step = block_threads >> 3;
    const int n_steps = (n_groups + groups_per_step - 1) / groups_per_step;
    for (int step = 0; step < n_steps; ++step) {
        const int g = step * groups_per_step + (tid >> 3);
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (g < n_groups) {
            v = x4[g * 8 + sub];
            smem_x9[g * 9 + sub] = v;
        }
        float s = (v.x + v.y) + (v.z + v.w);
        s += __shfl_xor_sync(0xFFFFFFFFu, s, 4);
        s += __shfl_xor_sync(0xFFFFFFFFu, s, 2);
        s += __shfl_xor_sync(0xFFFFFFFFu, s, 1);
        if (g < n_groups && sub == 0) {
            smem_x9[g * 9 + 8] = make_float4(s, 0.0f, 0.0f, 0.0f);
        }
    }
}

// 4-chain factored FP32 group-32 dot product over bank-conflict-free `smem_grp9`.
__device__ __forceinline__ float dot_q4_group32_smem9(
    uint4 pk,
    float sc,
    float bi,
    const float4* smem_grp9) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    #pragma unroll
    for (int w_idx = 0; w_idx < 4; ++w_idx) {
        uint32_t w = (w_idx == 0) ? pk.x : ((w_idx == 1) ? pk.y : ((w_idx == 2) ? pk.z : pk.w));
        float4 x0 = smem_grp9[w_idx * 2];
        float4 x1 = smem_grp9[w_idx * 2 + 1];
        a0 = __fmaf_rn(static_cast<float>(w & 0xFu), x0.x, a0);
        a1 = __fmaf_rn(static_cast<float>((w >> 4) & 0xFu), x0.y, a1);
        a2 = __fmaf_rn(static_cast<float>((w >> 8) & 0xFu), x0.z, a2);
        a3 = __fmaf_rn(static_cast<float>((w >> 12) & 0xFu), x0.w, a3);
        a0 = __fmaf_rn(static_cast<float>((w >> 16) & 0xFu), x1.x, a0);
        a1 = __fmaf_rn(static_cast<float>((w >> 20) & 0xFu), x1.y, a1);
        a2 = __fmaf_rn(static_cast<float>((w >> 24) & 0xFu), x1.z, a2);
        a3 = __fmaf_rn(static_cast<float>((w >> 28) & 0xFu), x1.w, a3);
    }
    float dot_nx = (a0 + a1) + (a2 + a3);
    return __fmaf_rn(dot_nx, sc, bi * smem_grp9[8].x);
}

__device__ __forceinline__ float warp_dot_q4_row_smem9(
    const uint4* __restrict__ row_w,
    const uint32_t* __restrict__ row_sc_u32,
    const uint32_t* __restrict__ row_bi_u32,
    const float4* smem_x9,
    int n_groups,
    int lane) {
    float acc = 0.0f;
    for (int g = lane; g < n_groups; g += 32) {
        uint4 pk = row_w[g];
        float sc = load_bf16_from_u32_array(row_sc_u32, g);
        float bi = load_bf16_from_u32_array(row_bi_u32, g);
        acc += dot_q4_group32_smem9(pk, sc, bi, smem_x9 + g * 9);
    }
    return warp_sum(acc);
}

// A BF16 in [128,255] carries its integer offset directly in its fraction.
// Subtract 128 for nonnegative codes and 256 for negative two's-complement codes.
template<unsigned Select>
__device__ __forceinline__ uint32_t q38_scaled_q8_pair(uint32_t word, uint32_t scales) {
    const uint32_t lanes = __byte_perm(word, 0x43434343u, Select);
    const uint32_t biased = lanes & 0xff7fff7fu;
    const uint32_t zero = (lanes & 0x00800080u) | 0x43004300u;
    uint32_t codes, values;
    asm("sub.rn.bf16x2 %0, %1, %2;" : "=r"(codes) : "r"(biased), "r"(zero));
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(values) : "r"(codes), "r"(scales));
    return values;
}

__device__ __forceinline__ float4 q38_scaled_q8_word(uint32_t word, uint16_t scale) {
    const uint32_t scales = static_cast<uint32_t>(scale) * 0x10001u;
    const uint32_t lo = q38_scaled_q8_pair<0x4140>(word, scales);
    const uint32_t hi = q38_scaled_q8_pair<0x4342>(word, scales);
    return make_float4(__uint_as_float(lo << 16), __uint_as_float(lo & 0xffff0000u),
                       __uint_as_float(hi << 16), __uint_as_float(hi & 0xffff0000u));
}

// Q4 has a bias plane. Symmetric Q8 stores two signed-code vectors per group
// and rounds each scaled code to BF16 before the FP32 dot.
__device__ __forceinline__ float shared_group_dot(
    const uint4* weights, const uint32_t* scales, const uint32_t* biases,
    size_t group, const float4* x) {
    if (biases) return dot_q4_group32_smem9(weights[group], load_bf16_from_u32_array(scales, group),
                                           load_bf16_from_u32_array(biases, group), x);
    const uint32_t* codes = reinterpret_cast<const uint32_t*>(weights) + 8 * group;
    const uint16_t scale = reinterpret_cast<const uint16_t*>(scales)[group];
    float accum[4] = {};
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float4 w = q38_scaled_q8_word(codes[i], scale);
        const float4 v = x[i];
        accum[0] = __fmaf_rn(w.x, v.x, accum[0]);
        accum[1] = __fmaf_rn(w.y, v.y, accum[1]);
        accum[2] = __fmaf_rn(w.z, v.z, accum[2]);
        accum[3] = __fmaf_rn(w.w, v.w, accum[3]);
    }
    return (accum[0] + accum[1]) + (accum[2] + accum[3]);
}

__device__ __forceinline__ float shared_warp_dot(
    const uint4* weights, const uint32_t* scales, const uint32_t* biases,
    size_t row, int groups, const float4* x, int lane) {
    float acc = 0.0f;
    for (int g = lane; g < groups; g += 32)
        acc += shared_group_dot(weights, scales, biases, row * groups + g, x + g * 9);
    return warp_sum(acc);
}

// Standalone entry for numerical and memory tests of the shared-expert inner loop.
extern "C" __global__ void q38_shared_q8_projection(
    const uint4* weights, const uint32_t* scales, const float4* input,
    float* output, int rows, int cols) {
    extern __shared__ float4 staged[];
    const int groups = cols / 32;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    pdl_wait();
    stage_x_groups_smem9(input + static_cast<size_t>(blockIdx.y) * (cols / 4), staged, groups, threadIdx.x);
    __syncthreads();
    if (row < rows) {
        const float sum = shared_warp_dot(weights, scales, nullptr, row, groups, staged, threadIdx.x & 31);
        if ((threadIdx.x & 31) == 0) output[static_cast<size_t>(blockIdx.y) * rows + row] = sum;
    }
    pdl_trigger();
}

__device__ __forceinline__ float warp_dot_q4_row_z24_smem9(
    const uint4* __restrict__ row_w,
    const uint32_t* __restrict__ sb_lo_base,
    const uint32_t* __restrict__ sb_hi_base,
    size_t row_group_base,
    uint32_t base_sc_m1,
    uint32_t base_bi_m1,
    const float4* smem_x9,
    int n_groups,
    int lane) {
    float acc = 0.0f;
    for (int g = lane; g < n_groups; g += 32) {
        uint4 pk = row_w[g];
        float sc, bi;
        load_z24_sb_pair(
            sb_lo_base,
            sb_hi_base,
            row_group_base + static_cast<size_t>(g),
            base_sc_m1,
            base_bi_m1,
            &sc,
            &bi);
        acc += dot_q4_group32_smem9(pk, sc, bi, smem_x9 + g * 9);
    }
    return warp_sum(acc);
}

} // namespace leansonic::qwen38

using namespace leansonic::qwen38;

// ----------------------------------------------------------------------------
// 1. 4-Bit Group-32 Token Embedding Lookup + 4-Stream Hyper-Connection Tile
//    Grid: `(10, 1)`, Block: `256` (`2,560` dims -> tiled into `[4, 2560]`)
// ----------------------------------------------------------------------------
extern "C" __global__ void q38_embed_init_hc4(
    const uint4* __restrict__ w_u128,      // [vocab, 80]
    const uint32_t* __restrict__ sc_u32,   // [vocab, 40] (`80` BF16)
    const uint32_t* __restrict__ bi_u32,   // [vocab, 40] (`80` BF16)
    const int32_t* token_ptr,
    float* out_hc) {                       // [10240]
    const int d = blockIdx.x * 256 + threadIdx.x;
    pdl_wait();
    if (d < kD) {
        const int tok = *token_ptr;
        const int g = d >> 5;
        const int sub = d & 31;
        const int w_idx = sub >> 3;
        const int sh = (sub & 7) * 4;

        const uint4 pk = w_u128[static_cast<size_t>(tok) * 80u + g];
        uint32_t word = (w_idx == 0) ? pk.x : ((w_idx == 1) ? pk.y : ((w_idx == 2) ? pk.z : pk.w));
        float sc = load_bf16_from_u32_array(sc_u32 + static_cast<size_t>(tok) * 40u, g);
        float bi = load_bf16_from_u32_array(bi_u32 + static_cast<size_t>(tok) * 40u, g);
        float val = __fmaf_rn(static_cast<float>((word >> sh) & 0xFu), sc, bi);

        out_hc[d] = val;
        out_hc[kD + d] = val;
        out_hc[2 * kD + d] = val;
        out_hc[3 * kD + d] = val;
    }
    pdl_trigger();
}

// ----------------------------------------------------------------------------
// 2. SMEM-Staged Group-32 4-Bit Affine GEMV (`cols <= 6144`, `8` rows/CTA)
//    `post_mode`: `0` = linear, `1` = `silu(y * 0.25f)`, `2` = `sigmoid(y)`
// ----------------------------------------------------------------------------
extern "C" __global__ void q38_q4_gemv(
    const uint4* __restrict__ w_u128,      // [rows, cols / 32]
    const uint32_t* __restrict__ sc_u32,   // [rows, cols / 64]
    const uint32_t* __restrict__ bi_u32,   // [rows, cols / 64]
    const float4* x4,                      // [cols / 4]
    float* y,                              // [rows]
    int32_t rows,
    int32_t cols,
    int32_t post_mode) {
    __shared__ float4 smem_x9[192 * 9];    // max `6144 / 32 = 192` groups (`27,648` B SMEM)

    const int tid = threadIdx.x;
    const int n_groups = cols >> 5;
    pdl_wait();
    stage_x_groups_smem9(x4, smem_x9, n_groups, tid);
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + warp_id;
    if (row < rows) {
        const int sc_stride_u32 = cols >> 6;
        float v = warp_dot_q4_row_smem9(
            w_u128 + static_cast<size_t>(row) * n_groups,
            sc_u32 + static_cast<size_t>(row) * sc_stride_u32,
            bi_u32 + static_cast<size_t>(row) * sc_stride_u32,
            smem_x9,
            n_groups,
            lane);
        if (lane == 0) {
            if (post_mode == 1) {
                v = ptx_silu(v * 0.25f);
            } else if (post_mode == 2) {
                v = ptx_sigmoid(v);
            }
            y[row] = v;
        }
    }
    pdl_trigger();
}

// High-occupancy `cols = 2560` (`80` groups -> `11,520` B SMEM = `5` CTAs/SM under 64 KiB carveout!)
extern "C" __global__ void q38_q4_gemv_2560(
    const uint4* __restrict__ w_u128,      // [rows, 80]
    const uint32_t* __restrict__ sc_u32,   // [rows, 40]
    const uint32_t* __restrict__ bi_u32,   // [rows, 40]
    const float4* x4,                      // [640]
    float* y,                              // [rows]
    int32_t rows,
    int32_t post_mode) {
    __shared__ float4 smem_x9[80 * 9];     // `11,520` B SMEM -> `5` CTAs/SM (`40` warps/SM)!

    const int tid = threadIdx.x;
    pdl_wait();
    stage_x_groups_smem9(x4, smem_x9, 80, tid);
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + warp_id;
    if (row < rows) {
        float v = warp_dot_q4_row_smem9(
            w_u128 + static_cast<size_t>(row) * 80u,
            sc_u32 + static_cast<size_t>(row) * 40u,
            bi_u32 + static_cast<size_t>(row) * 40u,
            smem_x9,
            80,
            lane);
        if (lane == 0) {
            if (post_mode == 1) {
                v = ptx_silu(v * 0.25f);
            } else if (post_mode == 2) {
                v = ptx_sigmoid(v);
            }
            y[row] = v;
        }
    }
    pdl_trigger();
}

// Wide `cols = 10240` variant (`46,080` B SMEM) for `hc` `down` (`320` rows) & `inject` (`4` rows)
extern "C" __global__ void q38_q4_gemv_10k(
    const uint4* __restrict__ w_u128,      // [rows, 320]
    const uint32_t* __restrict__ sc_u32,   // [rows, 160]
    const uint32_t* __restrict__ bi_u32,   // [rows, 160]
    const float4* x4,                      // [2560]
    float* y,                              // [rows]
    int32_t rows,
    int32_t post_mode) {
    __shared__ float4 smem_x9[320 * 9];    // `46,080` B SMEM (fits default 48 KiB static limit!)

    const int tid = threadIdx.x;
    pdl_wait();
    stage_x_groups_smem9(x4, smem_x9, 320, tid);
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + warp_id;
    if (row < rows) {
        float v = warp_dot_q4_row_smem9(
            w_u128 + static_cast<size_t>(row) * 320u,
            sc_u32 + static_cast<size_t>(row) * 160u,
            bi_u32 + static_cast<size_t>(row) * 160u,
            smem_x9,
            320,
            lane);
        if (lane == 0) {
            if (post_mode == 1) {
                v = ptx_silu(v * 0.25f);
            } else if (post_mode == 2) {
                v = ptx_sigmoid(v);
            }
            y[row] = v;
        }
    }
    pdl_trigger();
}

// ----------------------------------------------------------------------------
// 3. Layer 1 (`ple_layer_ids = [2]`) Fused PLE Gate + Dilated-3 Conv4 + Inject
//    Grid: `(4, 1)`, Block: `256` (`1` CTA per Hyper-Connection stream `s = 0..3`)
// ----------------------------------------------------------------------------
extern "C" __global__ void q38_ple_inject(
    float* hc_inout,                          // [4, 2560]
    const float* key_raw,                     // [4, 2560] (`key_proj(ple_emb)`)
    const float* val_raw,                     // [2560]    (`value_proj(ple_emb)`)
    const float* __restrict__ norm_key_w,     // [4, 2560] (`f32` effective `1+w` or `w`)
    const float* __restrict__ norm_query_w,   // [4, 2560]
    const float* __restrict__ norm_conv_w,    // [4, 2560]
    const float4* __restrict__ conv_w4,       // [4, 2560] (`[w0, w1, w2, w3]`)
    float* ple_hist9,                         // [9, 10240] ring buffer
    const int32_t* pos_ptr) {
    __shared__ float smem_a[8];
    __shared__ float smem_b[8];
    __shared__ float smem_c[8];
    __shared__ float smem_scalars[4]; // [inv_k, inv_q, sig_gate, inv_gv]

    const int s = blockIdx.x; // 0..3
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const float* nk_s = norm_key_w + s * kD;
    const float* nq_s = norm_query_w + s * kD;
    const float* nc_s = norm_conv_w + s * kD;
    const float4* cw_s = conv_w4 + s * kD;

    pdl_wait();
    const int pos = *pos_ptr;
    float* hc_s = hc_inout + s * kD;
    const float* kr_s = key_raw + s * kD;

    float sq_k = 0.0f;
    float sq_q = 0.0f;
    float sq_v = 0.0f;
    for (int d = tid; d < kD; d += 256) {
        float kv = kr_s[d];
        float qv = hc_s[d];
        float vv = val_raw[d];
        sq_k += kv * kv;
        sq_q += qv * qv;
        sq_v += vv * vv;
    }
    sq_k = warp_sum(sq_k);
    sq_q = warp_sum(sq_q);
    sq_v = warp_sum(sq_v);
    if (lane == 0) {
        smem_a[warp_id] = sq_k;
        smem_b[warp_id] = sq_q;
        smem_c[warp_id] = sq_v;
    }
    __syncthreads();
    if (warp_id == 0) {
        float tk = (lane < 8) ? smem_a[lane] : 0.0f;
        float tq = (lane < 8) ? smem_b[lane] : 0.0f;
        float tv = (lane < 8) ? smem_c[lane] : 0.0f;
        tk = warp_sum(tk);
        tq = warp_sum(tq);
        tv = warp_sum(tv);
        if (lane == 0) {
            smem_scalars[0] = ptx_rsqrt(tk * (1.0f / 2560.0f) + 1e-6f);
            smem_scalars[1] = ptx_rsqrt(tq * (1.0f / 2560.0f) + 1e-6f);
            smem_scalars[3] = tv * (1.0f / 2560.0f);
        }
    }
    __syncthreads();

    const float inv_k = smem_scalars[0];
    const float inv_q = smem_scalars[1];
    float dot = 0.0f;
    for (int d = tid; d < kD; d += 256) {
        float kn = kr_s[d] * inv_k * nk_s[d];
        float qn = hc_s[d] * inv_q * nq_s[d];
        dot += kn * qn;
    }
    dot = warp_sum(dot);
    if (lane == 0) smem_a[warp_id] = dot;
    __syncthreads();
    if (warp_id == 0) {
        float td = (lane < 8) ? smem_a[lane] : 0.0f;
        td = warp_sum(td);
        if (lane == 0) {
            float g = td * 0.01976423537605237f;
            float g_abs = fabsf(g);
            float g_clamp = (g_abs < 1e-6f) ? 1e-6f : g_abs;
            float signed_sqrt = (g > 0.0f) ? ptx_sqrt(g_clamp) : ((g < 0.0f) ? -ptx_sqrt(g_clamp) : 0.0f);
            float sig = ptx_sigmoid(signed_sqrt);
            smem_scalars[2] = sig;
            smem_scalars[3] = ptx_rsqrt(sig * sig * smem_scalars[3] + 1e-6f);
        }
    }
    __syncthreads();

    const float sig_g = smem_scalars[2];
    const float inv_gv = smem_scalars[3];
    const int s_m9 = pos % 9;
    const int s_m6 = (pos + 3) % 9;
    const int s_m3 = (pos + 6) % 9;

    for (int d = tid; d < kD; d += 256) {
        const int ch = s * kD + d;
        float gv = sig_g * val_raw[d];
        float gvn = gv * inv_gv * nc_s[d];

        float x_m9 = ple_hist9[static_cast<size_t>(s_m9) * kHcD + ch];
        float x_m6 = ple_hist9[static_cast<size_t>(s_m6) * kHcD + ch];
        float x_m3 = ple_hist9[static_cast<size_t>(s_m3) * kHcD + ch];
        ple_hist9[static_cast<size_t>(s_m9) * kHcD + ch] = gvn;

        float4 cw = cw_s[d];
        float c_out = ptx_silu(cw.x * x_m9 + cw.y * x_m6 + cw.z * x_m3 + cw.w * gvn);
        hc_s[d] += gv + c_out;
    }
    pdl_trigger();
}

// ----------------------------------------------------------------------------
// 4. 4-Stream Hyper-Connection Norm, Fused Down+Inject, Fused Up+Mix, and Inject
// ----------------------------------------------------------------------------
// Original ABI preserved (`Grid (4, 1)`, `Block 256`), upgraded with register-cached `float4`!
extern "C" __global__ void q38_hc_norm4(
    const float* hc_in,                   // [4, 2560]
    const float* __restrict__ hc_gamma,   // [4, 2560]
    float* hc_normed) {                   // [4, 2560]
    __shared__ float smem_sq[8];
    __shared__ float smem_inv;

    const int s = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const float4* gam4 = reinterpret_cast<const float4*>(hc_gamma + s * kD);

    pdl_wait();
    const float4* src4 = reinterpret_cast<const float4*>(hc_in + s * kD);
    float4* dst4 = reinterpret_cast<float4*>(hc_normed + s * kD);

    float4 r0 = src4[tid];
    float4 r1 = src4[tid + 256];
    float4 r2 = (tid < 128) ? src4[tid + 512] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float sq = (r0.x * r0.x + r0.y * r0.y) + (r0.z * r0.z + r0.w * r0.w)
             + (r1.x * r1.x + r1.y * r1.y) + (r1.z * r1.z + r1.w * r1.w)
             + (r2.x * r2.x + r2.y * r2.y) + (r2.z * r2.z + r2.w * r2.w);
    sq = warp_sum(sq);
    if (lane == 0) smem_sq[warp_id] = sq;
    __syncthreads();
    if (warp_id == 0) {
        float tot = (lane < 8) ? smem_sq[lane] : 0.0f;
        tot = warp_sum(tot);
        if (lane == 0) smem_inv = ptx_rsqrt(tot * (1.0f / 2560.0f) + 1e-6f);
    }
    __syncthreads();

    const float inv = smem_inv;
    {
        float4 g0 = gam4[tid];
        dst4[tid] = make_float4(r0.x * inv * g0.x, r0.y * inv * g0.y, r0.z * inv * g0.z, r0.w * inv * g0.w);
        float4 g1 = gam4[tid + 256];
        dst4[tid + 256] = make_float4(r1.x * inv * g1.x, r1.y * inv * g1.y, r1.z * inv * g1.z, r1.w * inv * g1.w);
        if (tid < 128) {
            float4 g2 = gam4[tid + 512];
            dst4[tid + 512] = make_float4(r2.x * inv * g2.x, r2.y * inv * g2.y, r2.z * inv * g2.z, r2.w * inv * g2.w);
        }
    }
    pdl_trigger();
}

// Fused `hc.down [320, 10240]` (`silu(y*0.25)`) + optional `hc.inject [4, 10240]` (linear)
// Grid: `(81, 1)` when `has_inject == 1` (`80` when `0`), Block: `256` (`2` warps/row = `4` rows/CTA -> `81` SMs active!)
extern "C" __global__ void q38_hc_down_inj_10k(
    const uint4* __restrict__ down_w,     // [320, 320]
    const uint32_t* __restrict__ down_sc, // [320, 160]
    const uint32_t* __restrict__ down_bi, // [320, 160]
    const uint4* __restrict__ inj_w,      // [4, 320] (nullable when `has_inject == 0`)
    const uint32_t* __restrict__ inj_sc,  // [4, 160]
    const uint32_t* __restrict__ inj_bi,  // [4, 160]
    const float4* hc_normed4,             // [2560] (`10,240` floats)
    float* hc_low_out,                    // [320]
    float* hc_inj4_out,                   // [4]
    int32_t has_inject) {
    __shared__ float4 smem_x9[320 * 9];   // `46,080` B SMEM
    __shared__ float smem_pair[8];

    const int tid = threadIdx.x;
    pdl_wait();
    stage_x_groups_smem9(hc_normed4, smem_x9, 320, tid);
    __syncthreads();

    const int warp_id = tid >> 5;          // 0..7
    const int lane = tid & 31;
    const int row_local = warp_id >> 1;    // 0..3 (`2` warps collaborate on each row!)
    const int half_id = warp_id & 1;       // 0 or 1
    const int row = blockIdx.x * 4 + row_local;
    const int max_rows = has_inject ? 324 : 320;

    float part = 0.0f;
    if (row < max_rows) {
        const uint4* rw = (row < 320)
            ? (down_w + static_cast<size_t>(row) * 320u)
            : (inj_w + static_cast<size_t>(row - 320) * 320u);
        const uint32_t* rsc = (row < 320)
            ? (down_sc + static_cast<size_t>(row) * 160u)
            : (inj_sc + static_cast<size_t>(row - 320) * 160u);
        const uint32_t* rbi = (row < 320)
            ? (down_bi + static_cast<size_t>(row) * 160u)
            : (inj_bi + static_cast<size_t>(row - 320) * 160u);

        for (int g = half_id * 32 + lane; g < 320; g += 64) {
            uint4 pk = rw[g];
            float sc = load_bf16_from_u32_array(rsc, g);
            float bi = load_bf16_from_u32_array(rbi, g);
            part += dot_q4_group32_smem9(pk, sc, bi, smem_x9 + g * 9);
        }
        part = warp_sum(part);
        if (lane == 0) {
            smem_pair[warp_id] = part;
        }
    }
    __syncthreads();

    if (row < max_rows && half_id == 0 && lane == 0) {
        float v = smem_pair[warp_id] + smem_pair[warp_id + 1];
        if (row < 320) {
            hc_low_out[row] = ptx_silu(v * 0.25f);
        } else {
            hc_inj4_out[row - 320] = v;
        }
    }
    pdl_trigger();
}

// Fused `hc.up [10240, 320]` (`sigmoid`) + `hc_mix4` (`0.25 * sum_{s=0..3} sig(up[s,d]) * hc_normed[s,d]`)
// Grid: `(320, 1)`, Block: `256` (`1` warp computes ALL `4` streams `s = 0..3` of output `d`! `1.4 KiB` SMEM -> `6` CTAs/SM!)
extern "C" __global__ void q38_hc_up_mix4(
    const uint4* __restrict__ up_w,       // [10240, 10]
    const uint32_t* __restrict__ up_sc,   // [10240, 5]
    const uint32_t* __restrict__ up_bi,   // [10240, 5]
    const float4* hc_low4,                // [80] (`320` floats = `10` groups)
    const float* hc_normed,               // [4, 2560]
    float* mixed_out) {                   // [2560]
    __shared__ float4 smem_low9[10 * 9];  // `1,440` B SMEM -> `6` CTAs/SM (`48` warps/SM = `100%` occupancy!)

    const int tid = threadIdx.x;
    pdl_wait();
    stage_x_groups_smem9(hc_low4, smem_low9, 10, tid);
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int d = blockIdx.x * 8 + warp_id;
    if (d < kD) {
        const int s = lane >> 3;          // 0..3 (`8` lanes per stream `s`!)
        const int sub_lane = lane & 7;    // 0..7
        const size_t row = static_cast<size_t>(s) * kD + static_cast<size_t>(d);
        const uint4* rw = up_w + row * 10u;
        const uint32_t* rsc = up_sc + row * 5u;
        const uint32_t* rbi = up_bi + row * 5u;

        // Iteration 0: `g = sub_lane` (`0..7`), Iteration 1: `g = 8 + sub_lane` (`8..9` for `sub_lane < 2`)
        float acc = 0.0f;
        {
            const int g0 = sub_lane;
            uint4 pk0 = rw[g0];
            float sc0 = load_bf16_from_u32_array(rsc, g0);
            float bi0 = load_bf16_from_u32_array(rbi, g0);
            acc += dot_q4_group32_smem9(pk0, sc0, bi0, smem_low9 + g0 * 9);
        }
        if (sub_lane < 2) {
            const int g1 = 8 + sub_lane;
            uint4 pk1 = rw[g1];
            float sc1 = load_bf16_from_u32_array(rsc, g1);
            float bi1 = load_bf16_from_u32_array(rbi, g1);
            acc += dot_q4_group32_smem9(pk1, sc1, bi1, smem_low9 + g1 * 9);
        }
        acc += __shfl_xor_sync(0xFFFFFFFFu, acc, 4);
        acc += __shfl_xor_sync(0xFFFFFFFFu, acc, 2);
        acc += __shfl_xor_sync(0xFFFFFFFFu, acc, 1);

        float weighted = 0.0f;
        if (sub_lane == 0) {
            weighted = ptx_sigmoid(acc) * hc_normed[row];
        }
        weighted += __shfl_down_sync(0xFFFFFFFFu, weighted, 16);
        weighted += __shfl_down_sync(0xFFFFFFFFu, weighted, 8);
        if (lane == 0) {
            mixed_out[d] = weighted * 0.25f;
        }
    }
    pdl_trigger();
}

// Original ABI preserved (`Grid (10, 1)`, `Block 256`)
extern "C" __global__ void q38_hc_mix4(
    const float* hc_normed,               // [4, 2560]
    const float* hc_up_sig,               // [4, 2560] (`sigmoid(up)`)
    float* mixed_out) {                   // [2560]
    const int d = blockIdx.x * 256 + threadIdx.x;
    pdl_wait();
    if (d < kD) {
        float sum = hc_up_sig[d] * hc_normed[d]
                  + hc_up_sig[kD + d] * hc_normed[kD + d]
                  + hc_up_sig[2 * kD + d] * hc_normed[2 * kD + d]
                  + hc_up_sig[3 * kD + d] * hc_normed[3 * kD + d];
        mixed_out[d] = sum * 0.25f;
    }
    pdl_trigger();
}

// Original ABI preserved (`Grid (40, 1)`, `Block 256`)
extern "C" __global__ void q38_hc_inject4(
    float* hc_inout,                      // [4, 2560]
    const float* branch_out,              // [2560]
    const float* inj_raw4) {              // [4]
    const int idx = blockIdx.x * 256 + threadIdx.x;
    pdl_wait();
    if (idx < kHcD) {
        const int s = idx / kD;
        const int d = idx - s * kD;
        const float w = 2.0f * ptx_sigmoid(inj_raw4[s] * 0.25f);
        hc_inout[idx] += branch_out[d] * w;
    }
    pdl_trigger();
}

// `float4`-vectorized `hc_inject4` (`Grid (3, 1)`, `Block 256` -> `640` `float4`s across `4` streams in 1 pass!)
extern "C" __global__ void q38_hc_inject4_vec(
    float4* hc_inout4,                    // [4, 640]
    const float4* branch_out4,            // [640]
    const float4* inj_raw4) {             // [1] (`4` floats)
    const int d4 = blockIdx.x * 256 + threadIdx.x;
    pdl_wait();
    if (d4 < 640) {
        const float4 raw = *inj_raw4;
        const float w0 = 2.0f * ptx_sigmoid(raw.x * 0.25f);
        const float w1 = 2.0f * ptx_sigmoid(raw.y * 0.25f);
        const float w2 = 2.0f * ptx_sigmoid(raw.z * 0.25f);
        const float w3 = 2.0f * ptx_sigmoid(raw.w * 0.25f);
        const float4 b = branch_out4[d4];

        float4 h0 = hc_inout4[d4];
        hc_inout4[d4] = make_float4(h0.x + b.x * w0, h0.y + b.y * w0, h0.z + b.z * w0, h0.w + b.w * w0);
        float4 h1 = hc_inout4[640 + d4];
        hc_inout4[640 + d4] = make_float4(h1.x + b.x * w1, h1.y + b.y * w1, h1.z + b.z * w1, h1.w + b.w * w1);
        float4 h2 = hc_inout4[1280 + d4];
        hc_inout4[1280 + d4] = make_float4(h2.x + b.x * w2, h2.y + b.y * w2, h2.z + b.z * w2, h2.w + b.w * w2);
        float4 h3 = hc_inout4[1920 + d4];
        hc_inout4[1920 + d4] = make_float4(h3.x + b.x * w3, h3.y + b.y * w3, h3.z + b.z * w3, h3.w + b.w * w3);
    }
    pdl_trigger();
}

// ----------------------------------------------------------------------------
// 5. Gated DeltaNet (`linear_attn`, 36 Layers): Causal Conv4+SiLU & Recurrent Rule
// ----------------------------------------------------------------------------
// Grid: `(40, 1)`, Block: `256` (`10,240` channels)
extern "C" __global__ void q38_gdn_conv_silu(
    const float* qkv_raw,                 // [10240]
    const float4* __restrict__ conv_w4,   // [10240] (`[w0, w1, w2, w3]`)
    float* conv_hist3,                    // [10240 * 3]
    float* qkv_conv_out) {                // [10240]
    const int c = blockIdx.x * 256 + threadIdx.x;
    float4 w = (c < 10240) ? conv_w4[c] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    pdl_wait();
    if (c < 10240) {
        float* h = conv_hist3 + c * 3;
        float h0 = h[0], h1 = h[1], h2 = h[2];
        float x = qkv_raw[c];
        h[0] = h1;
        h[1] = h2;
        h[2] = x;
        qkv_conv_out[c] = ptx_silu(w.x * h0 + w.y * h1 + w.z * h2 + w.w * x);
    }
    pdl_trigger();
}

// Grid: `(48, 1)`, Block: `128` (`1` CTA per V-head `vh = 0..47`, `1` thread per `dv = 0..127`!)
extern "C" __global__ void q38_gdn_recurrent_step(
    const float* qkv_conv,                // [10240]: `Q[2048], K[2048], V[6144]`
    const float* zba_buf,                 // [6240]:  `Z[6144], B[48], A[48]`
    const float* __restrict__ a_log,      // [48] (`f32`)
    const float* __restrict__ dt_bias,    // [48] (`f32`)
    const float* __restrict__ norm_w,     // [128] (`f32`, 1-centered `RMSNormGated`)
    int32_t use_sigmoid_gate,             // `1` = `sigmoid(z)`, `0` = `silu(z)`
    float* state_f32,                     // [48 * 128 * 128]
    float* gdn_out) {                     // [6144]
    __shared__ float smem_q[128];
    __shared__ float smem_k[128];
    __shared__ float smem_red[4];
    __shared__ float smem_gate[3]; // [decay, beta, inv_rms_out]

    const int vh = blockIdx.x;     // 0..47
    const int kh = vh / 3;         // 0..15
    const int dv = threadIdx.x;    // 0..127
    const int warp_id = dv >> 5;
    const int lane = dv & 31;
    const float nw_dv = norm_w[dv];
    const float al_vh = (dv == 0) ? a_log[vh] : 0.0f;
    const float dt_vh = (dv == 0) ? dt_bias[vh] : 0.0f;

    pdl_wait();

    float qv = qkv_conv[kh * 128 + dv];
    float kv = qkv_conv[2048 + kh * 128 + dv];
    float q_sq = warp_sum(qv * qv);
    float k_sq = warp_sum(kv * kv);
    if (lane == 0) smem_red[warp_id] = q_sq;
    __syncthreads();
    float q_tot = smem_red[0] + smem_red[1] + smem_red[2] + smem_red[3];
    float q_inv = ptx_rsqrt(q_tot + 1e-6f) * 0.08838834764831845f;
    __syncthreads();
    if (lane == 0) smem_red[warp_id] = k_sq;
    __syncthreads();
    float k_tot = smem_red[0] + smem_red[1] + smem_red[2] + smem_red[3];
    float k_inv = ptx_rsqrt(k_tot + 1e-6f);

    smem_q[dv] = qv * q_inv;
    smem_k[dv] = kv * k_inv;
    if (dv == 0) {
        float b_raw = zba_buf[6144 + vh];
        float a_raw = zba_buf[6192 + vh];
        float g = -ptx_exp(al_vh) * ptx_softplus(a_raw + dt_vh);
        smem_gate[0] = ptx_exp(g);
        smem_gate[1] = ptx_sigmoid(b_raw);
    }
    __syncthreads();

    const float decay = smem_gate[0];
    const float beta = smem_gate[1];
    float* S_vh = state_f32 + static_cast<size_t>(vh) * 16384u;

    float kv_mem = 0.0f;
    #pragma unroll 4
    for (int dk = 0; dk < 128; ++dk) {
        kv_mem += (S_vh[dk * 128 + dv] * decay) * smem_k[dk];
    }
    const float v_val = qkv_conv[4096 + vh * 128 + dv];
    const float delta = (v_val - kv_mem) * beta;

    float out_v = 0.0f;
    #pragma unroll 4
    for (int dk = 0; dk < 128; ++dk) {
        float s_new = S_vh[dk * 128 + dv] * decay + smem_k[dk] * delta;
        S_vh[dk * 128 + dv] = s_new;
        out_v += s_new * smem_q[dk];
    }

    float o_sq = warp_sum(out_v * out_v);
    if (lane == 0) smem_red[warp_id] = o_sq;
    __syncthreads();
    if (dv == 0) {
        float o_tot = smem_red[0] + smem_red[1] + smem_red[2] + smem_red[3];
        smem_gate[2] = ptx_rsqrt(o_tot * (1.0f / 128.0f) + 1e-6f);
    }
    __syncthreads();

    float z_val = zba_buf[vh * 128 + dv];
    float gate_act = use_sigmoid_gate ? ptx_sigmoid(z_val) : ptx_silu(z_val);
    gdn_out[vh * 128 + dv] = out_v * smem_gate[2] * nw_dv * gate_act;
    pdl_trigger();
}

// ----------------------------------------------------------------------------
// 6. QSA (`self_attn` + `indexer`, 12 Layers): Prep/RoPE/Block-Pool + Sparse GQA
// ----------------------------------------------------------------------------
// Grid: `(31, 1)`, Block: `256`
extern "C" __global__ void q38_qsa_prep_and_cache(
    const float* qsa_proj,                     // [13952]
    const float* __restrict__ q_norm_w,        // [256]
    const float* __restrict__ k_norm_w,        // [256]
    const float* __restrict__ idx_q_norm_w,    // [128]
    const float* __restrict__ idx_k_norm_w,    // [128]
    const float* __restrict__ inv_freq32,      // [32]
    const int32_t* pos_ptr,
    float* q_rot_out,                          // [24, 256]
    float* q_gate_out,                         // [24, 256]
    uint16_t* k_cache_fp16,                    // [max_ctx, 512]
    uint16_t* v_cache_fp16,                    // [max_ctx, 512]
    float* idx_q_out,                          // [4, 128]
    float* idx_raw_k_cache,                    // [max_ctx, 128]
    float* idx_block_k_cache) {                // [max_ctx / 4, 128]
    __shared__ float smem_vec[256];
    __shared__ float smem_red[8];

    const int blk = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    pdl_wait();
    const int pos = *pos_ptr;

    if (blk < 24) {
        const int h = blk;
        const float* src = qsa_proj + h * 512;
        q_gate_out[h * 256 + tid] = src[256 + tid];
        float v = src[tid];
        float sq = warp_sum(v * v);
        if (lane == 0) smem_red[warp_id] = sq;
        __syncthreads();
        if (warp_id == 0) {
            float tot = (lane < 8) ? smem_red[lane] : 0.0f;
            tot = warp_sum(tot);
            if (lane == 0) smem_red[0] = ptx_rsqrt(tot * (1.0f / 256.0f) + 1e-6f);
        }
        __syncthreads();
        smem_vec[tid] = v * smem_red[0] * q_norm_w[tid];
        __syncthreads();

        float out_v = smem_vec[tid];
        if (tid < 64) {
            int f_idx = tid & 31;
            float s, c;
            ptx_sincos_reduced(static_cast<float>(pos) * inv_freq32[f_idx], &s, &c);
            out_v = (tid < 32)
                ? (smem_vec[tid] * c - smem_vec[tid + 32] * s)
                : (smem_vec[tid - 32] * s + smem_vec[tid] * c);
        }
        q_rot_out[h * 256 + tid] = out_v;
    } else if (blk < 26) {
        const int kh = blk - 24;
        const float kv = qsa_proj[12288 + kh * 256 + tid];
        const float vv = qsa_proj[12800 + kh * 256 + tid];
        v_cache_fp16[static_cast<size_t>(pos) * 512u + kh * 256 + tid] = f32_to_fp16_bits(vv);

        float sq = warp_sum(kv * kv);
        if (lane == 0) smem_red[warp_id] = sq;
        __syncthreads();
        if (warp_id == 0) {
            float tot = (lane < 8) ? smem_red[lane] : 0.0f;
            tot = warp_sum(tot);
            if (lane == 0) smem_red[0] = ptx_rsqrt(tot * (1.0f / 256.0f) + 1e-6f);
        }
        __syncthreads();
        smem_vec[tid] = kv * smem_red[0] * k_norm_w[tid];
        __syncthreads();

        float out_k = smem_vec[tid];
        if (tid < 64) {
            int f_idx = tid & 31;
            float s, c;
            ptx_sincos_reduced(static_cast<float>(pos) * inv_freq32[f_idx], &s, &c);
            out_k = (tid < 32)
                ? (smem_vec[tid] * c - smem_vec[tid + 32] * s)
                : (smem_vec[tid - 32] * s + smem_vec[tid] * c);
        }
        k_cache_fp16[static_cast<size_t>(pos) * 512u + kh * 256 + tid] = f32_to_fp16_bits(out_k);
    } else if (blk < 30) {
        if (tid < 128) {
            const int ih = blk - 26;
            float qv = qsa_proj[13312 + ih * 128 + tid];
            float sq = warp_sum(qv * qv);
            if (lane == 0) smem_red[warp_id] = sq;
            __syncthreads();
            float tot = smem_red[0] + smem_red[1] + smem_red[2] + smem_red[3];
            float inv = ptx_rsqrt(tot * (1.0f / 128.0f) + 1e-6f);
            smem_vec[tid] = qv * inv * idx_q_norm_w[tid];
            __syncthreads();

            float out_q = smem_vec[tid];
            if (tid < 64) {
                int f_idx = tid & 31;
                float s, c;
                ptx_sincos_reduced(static_cast<float>(pos) * inv_freq32[f_idx], &s, &c);
                out_q = (tid < 32)
                    ? (smem_vec[tid] * c - smem_vec[tid + 32] * s)
                    : (smem_vec[tid - 32] * s + smem_vec[tid] * c);
            }
            idx_q_out[ih * 128 + tid] = out_q;
        }
    } else {
        if (tid < 128) {
            float rk = qsa_proj[13824 + tid];
            idx_raw_k_cache[static_cast<size_t>(pos) * 128u + tid] = rk;
            if ((pos & 3) == 3) {
                const int b_idx = pos >> 2;
                const int g_start = pos - 3;
                float pooled = 0.25f * (
                    idx_raw_k_cache[static_cast<size_t>(g_start) * 128u + tid] +
                    idx_raw_k_cache[static_cast<size_t>(g_start + 1) * 128u + tid] +
                    idx_raw_k_cache[static_cast<size_t>(g_start + 2) * 128u + tid] +
                    rk);
                float sq = warp_sum(pooled * pooled);
                if (lane == 0) smem_red[warp_id] = sq;
                __syncthreads();
                float tot = smem_red[0] + smem_red[1] + smem_red[2] + smem_red[3];
                float inv = ptx_rsqrt(tot * (1.0f / 128.0f) + 1e-6f);
                smem_vec[tid] = pooled * inv * idx_k_norm_w[tid];
                __syncthreads();

                float out_bk = smem_vec[tid];
                if (tid < 64) {
                    int f_idx = tid & 31;
                    float s, c;
                    ptx_sincos_reduced(static_cast<float>(g_start) * inv_freq32[f_idx], &s, &c);
                    out_bk = (tid < 32)
                        ? (smem_vec[tid] * c - smem_vec[tid + 32] * s)
                        : (smem_vec[tid - 32] * s + smem_vec[tid] * c);
                }
                idx_block_k_cache[static_cast<size_t>(b_idx) * 128u + tid] = out_bk;
            }
        }
    }
    pdl_trigger();
}

// Long-context (`n_blocks > 512`) QSA Block Scorer + Top-512 Mask Builder
extern "C" __global__ void q38_qsa_score_blocks(
    const float4* idx_q4,                       // [4, 32]
    const float4* block_k4_cache,               // [n_blocks, 32]
    int32_t n_blocks,
    float* block_scores) {                      // [n_blocks]
    __shared__ float4 smem_q4[4][33];           // bank-padded!
    const int tid = threadIdx.x;
    pdl_wait();
    if (tid < 128) {
        smem_q4[tid >> 5][tid & 31] = idx_q4[tid];
    }
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int b = blockIdx.x * 8 + warp_id;
    if (b < n_blocks) {
        float4 bk = block_k4_cache[static_cast<size_t>(b) * 32u + lane];
        float s_tot = 0.0f;
        #pragma unroll
        for (int ih = 0; ih < 4; ++ih) {
            float4 qv = smem_q4[ih][lane];
            float d = warp_sum(qv.x * bk.x + qv.y * bk.y + qv.z * bk.z + qv.w * bk.w);
            if (d > 0.0f) s_tot += d;
        }
        if (lane == 0) {
            block_scores[b] = s_tot * 0.08838834764831845f;
        }
    }
    pdl_trigger();
}

extern "C" __global__ void q38_qsa_select_top512(
    float* scores_scratch, // [n_blocks]
    int32_t n_blocks,
    int32_t* block_mask) { // [n_blocks]
    __shared__ float smem_val[8];
    __shared__ int smem_idx[8];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    pdl_wait();
    for (int i = tid; i < n_blocks; i += 256) {
        block_mask[i] = 0;
    }
    __syncthreads();

    for (int sel = 0; sel < 512; ++sel) {
        float best_v = -1e30f;
        int best_i = 0;
        for (int i = tid; i < n_blocks; i += 256) {
            float v = scores_scratch[i];
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
            block_mask[fi] = 1;
            scores_scratch[fi] = -1e30f;
        }
        __syncthreads();
    }
    pdl_trigger();
}

// 8-Warp Split-KV QSA GQA (`Grid (24, 1)`, `Block 256`, `head_dim = 256`, `128`-bit `uint4` FP16 loads!)
extern "C" __global__ void q38_qsa_attn(
    const float* q_rot,                         // [24, 256]
    const float* q_gate,                        // [24, 256]
    const uint4* k_cache_u128,                  // [max_ctx, 2, 32]
    const uint4* v_cache_u128,                  // [max_ctx, 2, 32]
    const int32_t* block_mask,                  // nullable when `use_mask == 0`
    const int32_t* pos_ptr,
    int32_t use_mask,
    float* attn_out) {                          // [24, 256]
    __shared__ float smem_m[8];
    __shared__ float smem_l[8];
    __shared__ float smem_o[8][256];

    const int h = blockIdx.x;   // 0..23
    const int kvh = h / 12;     // 0..1 (`12:1` GQA!)
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    pdl_wait();
    const int pos = *pos_ptr;
    const int n = pos + 1;
    const int complete_toks = (n >> 2) << 2;

    const float* qh = q_rot + h * 256 + lane * 8;
    const float scale = 0.0625f; // 1 / sqrt(256)
    const float q0 = qh[0] * scale, q1 = qh[1] * scale, q2 = qh[2] * scale, q3 = qh[3] * scale;
    const float q4 = qh[4] * scale, q5 = qh[5] * scale, q6 = qh[6] * scale, q7 = qh[7] * scale;

    float m_w = -1e30f;
    float l_w = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    float o4 = 0.0f, o5 = 0.0f, o6 = 0.0f, o7 = 0.0f;

    for (int j = warp_id; j < n; j += 8) {
        if (use_mask && j < complete_toks && block_mask[j >> 2] == 0) {
            continue;
        }
        const size_t kv_off = (static_cast<size_t>(j) * 2u + kvh) * 32u + lane;
        uint4 kp = k_cache_u128[kv_off];
        float dot = q0 * fp16_bits_to_f32(static_cast<uint16_t>(kp.x & 0xFFFFu))
                  + q1 * fp16_bits_to_f32(static_cast<uint16_t>(kp.x >> 16))
                  + q2 * fp16_bits_to_f32(static_cast<uint16_t>(kp.y & 0xFFFFu))
                  + q3 * fp16_bits_to_f32(static_cast<uint16_t>(kp.y >> 16))
                  + q4 * fp16_bits_to_f32(static_cast<uint16_t>(kp.z & 0xFFFFu))
                  + q5 * fp16_bits_to_f32(static_cast<uint16_t>(kp.z >> 16))
                  + q6 * fp16_bits_to_f32(static_cast<uint16_t>(kp.w & 0xFFFFu))
                  + q7 * fp16_bits_to_f32(static_cast<uint16_t>(kp.w >> 16));
        dot = warp_sum(dot);

        float m_new = (dot > m_w) ? dot : m_w;
        float alpha = (l_w == 0.0f) ? 0.0f : ptx_exp(m_w - m_new);
        float p = ptx_exp(dot - m_new);
        l_w = l_w * alpha + p;
        m_w = m_new;

        uint4 vp = v_cache_u128[kv_off];
        o0 = o0 * alpha + p * fp16_bits_to_f32(static_cast<uint16_t>(vp.x & 0xFFFFu));
        o1 = o1 * alpha + p * fp16_bits_to_f32(static_cast<uint16_t>(vp.x >> 16));
        o2 = o2 * alpha + p * fp16_bits_to_f32(static_cast<uint16_t>(vp.y & 0xFFFFu));
        o3 = o3 * alpha + p * fp16_bits_to_f32(static_cast<uint16_t>(vp.y >> 16));
        o4 = o4 * alpha + p * fp16_bits_to_f32(static_cast<uint16_t>(vp.z & 0xFFFFu));
        o5 = o5 * alpha + p * fp16_bits_to_f32(static_cast<uint16_t>(vp.z >> 16));
        o6 = o6 * alpha + p * fp16_bits_to_f32(static_cast<uint16_t>(vp.w & 0xFFFFu));
        o7 = o7 * alpha + p * fp16_bits_to_f32(static_cast<uint16_t>(vp.w >> 16));
    }

    if (lane == 0) {
        smem_m[warp_id] = m_w;
        smem_l[warp_id] = l_w;
    }
    const int base_d = lane * 8;
    smem_o[warp_id][base_d + 0] = o0;
    smem_o[warp_id][base_d + 1] = o1;
    smem_o[warp_id][base_d + 2] = o2;
    smem_o[warp_id][base_d + 3] = o3;
    smem_o[warp_id][base_d + 4] = o4;
    smem_o[warp_id][base_d + 5] = o5;
    smem_o[warp_id][base_d + 6] = o6;
    smem_o[warp_id][base_d + 7] = o7;
    __syncthreads();

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
    float g = ptx_sigmoid(q_gate[h * 256 + tid]);
    attn_out[h * 256 + tid] = (sum_d * ptx_rcp(fmaxf(denom, 1e-20f))) * g;
    pdl_trigger();
}

// ----------------------------------------------------------------------------
// 7. `512`-Expert BF16 Router + Exact Softmax Top-10 + Unified `k = 0..10` MoE
// ----------------------------------------------------------------------------
// Grid: `(64, 1)`, Block: `256` (`8` rows/CTA -> `512` BF16 router logits, bank-conflict-free!)
extern "C" __global__ void q38_bf16_router_512(
    const uint4* __restrict__ w_bf16_u128, // [512, 320]
    const float4* x4,                      // [640]
    float* logits512) {                    // [512]
    __shared__ float4 smem_x9[80 * 9];     // `11,520` B SMEM -> `5` CTAs/SM, `0` bank conflict!
    const int tid = threadIdx.x;
    pdl_wait();
    for (int i = tid; i < 640; i += 256) {
        smem_x9[(i >> 3) * 9 + (i & 7)] = x4[i];
    }
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + warp_id;
    if (row < kExp) {
        const uint4* w_row = w_bf16_u128 + static_cast<size_t>(row) * 320u;
        float a0 = 0.0f, a1 = 0.0f;
        for (int c8 = lane; c8 < 320; c8 += 32) {
            uint4 pk = w_row[c8];
            const int idx4 = c8 * 2;
            const int base9 = (idx4 >> 3) * 9 + (idx4 & 7);
            float4 u = smem_x9[base9];
            float4 v = smem_x9[base9 + 1];
            a0 += bf16_to_f32(static_cast<uint16_t>(pk.x & 0xFFFFu)) * u.x
                + bf16_to_f32(static_cast<uint16_t>(pk.y & 0xFFFFu)) * u.z
                + bf16_to_f32(static_cast<uint16_t>(pk.z & 0xFFFFu)) * v.x
                + bf16_to_f32(static_cast<uint16_t>(pk.w & 0xFFFFu)) * v.z;
            a1 += bf16_to_f32(static_cast<uint16_t>(pk.x >> 16)) * u.y
                + bf16_to_f32(static_cast<uint16_t>(pk.y >> 16)) * u.w
                + bf16_to_f32(static_cast<uint16_t>(pk.z >> 16)) * v.y
                + bf16_to_f32(static_cast<uint16_t>(pk.w >> 16)) * v.w;
        }
        float acc = warp_sum(a0 + a1);
        if (lane == 0) logits512[row] = acc;
    }
    pdl_trigger();
}

// Grid: `(1, 1)`, Block: `256` (`2` experts/thread -> Softmax + Top-10 + `norm_topk_prob`)
extern "C" __global__ void q38_softmax_topk10(
    const float* logits512,
    int32_t* sel_ids10,  // [10]
    float* sel_wts11) {  // [11]
    __shared__ float smem_p[512];
    __shared__ float smem_red[8];
    __shared__ int smem_idx[8];
    __shared__ float smem_topw[10];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    pdl_wait();
    float v0 = logits512[tid * 2];
    float v1 = logits512[tid * 2 + 1];
    float m = fmaxf(v0, v1);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, mask));
    }
    if (lane == 0) smem_red[warp_id] = m;
    __syncthreads();
    if (warp_id == 0) {
        float bm = (lane < 8) ? smem_red[lane] : -1e30f;
        #pragma unroll
        for (int mask = 4; mask > 0; mask >>= 1) {
            bm = fmaxf(bm, __shfl_xor_sync(0xFFFFFFFFu, bm, mask));
        }
        if (lane == 0) smem_red[0] = bm;
    }
    __syncthreads();
    const float max_l = smem_red[0];

    float e0 = ptx_exp(v0 - max_l);
    float e1 = ptx_exp(v1 - max_l);
    float es = warp_sum(e0 + e1);
    if (lane == 0) smem_red[warp_id] = es;
    __syncthreads();
    if (warp_id == 0) {
        float tot = (lane < 8) ? smem_red[lane] : 0.0f;
        tot = warp_sum(tot);
        if (lane == 0) smem_red[0] = ptx_rcp(tot);
    }
    __syncthreads();
    const float inv_z = smem_red[0];
    smem_p[tid * 2] = e0 * inv_z;
    smem_p[tid * 2 + 1] = e1 * inv_z;
    __syncthreads();

    float sum_top = 0.0f;
    for (int k = 0; k < kTopK; ++k) {
        float p0 = smem_p[tid * 2];
        float p1 = smem_p[tid * 2 + 1];
        float bw = p0;
        int bi = tid * 2;
        if (p1 > bw) {
            bw = p1;
            bi = tid * 2 + 1;
        }
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            float ow = __shfl_xor_sync(0xFFFFFFFFu, bw, mask);
            int oi = __shfl_xor_sync(0xFFFFFFFFu, bi, mask);
            if (ow > bw || (ow == bw && oi < bi)) {
                bw = ow;
                bi = oi;
            }
        }
        if (lane == 0) {
            smem_red[warp_id] = bw;
            smem_idx[warp_id] = bi;
        }
        __syncthreads();
        if (tid == 0) {
            float fw = smem_red[0];
            int fi = smem_idx[0];
            for (int w = 1; w < 8; ++w) {
                if (smem_red[w] > fw || (smem_red[w] == fw && smem_idx[w] < fi)) {
                    fw = smem_red[w];
                    fi = smem_idx[w];
                }
            }
            sel_ids10[k] = fi;
            smem_topw[k] = fw;
            smem_p[fi] = -1.0f;
        }
        __syncthreads();
        sum_top += (tid == 0) ? smem_topw[k] : 0.0f;
    }
    if (tid == 0) {
        float inv_top = ptx_rcp(fmaxf(sum_top, 1e-20f));
        for (int k = 0; k < kTopK; ++k) {
            sel_wts11[k] = smem_topw[k] * inv_top;
        }
    }
    pdl_trigger();
}

// Original ABI preserved (`Grid (160, 11)`, `Block 256`), upgraded with bank-conflict-free `smem_x9` (`5` CTAs/SM!)
extern "C" __global__ void q38_moe_up_swiglu11(
    const uint4* __restrict__ sw_gate_w,     // [512, 640, 80]
    const uint32_t* __restrict__ sw_gate_sc, // [512, 640, 40]
    const uint32_t* __restrict__ sw_gate_bi, // [512, 640, 40]
    const uint4* __restrict__ sw_up_w,       // [512, 640, 80]
    const uint32_t* __restrict__ sw_up_sc,   // [512, 640, 40]
    const uint32_t* __restrict__ sw_up_bi,   // [512, 640, 40]
    const uint4* __restrict__ sh_gu_gate_w,  // [1281, 80]
    const uint32_t* __restrict__ sh_gu_sc,   // [1281, 40]
    const uint32_t* __restrict__ sh_gu_bi,   // [1281, 40]
    const float4* x4,                        // [640]
    const int32_t* sel_ids10,                // [10]
    int32_t is_compact,
    float* sel_wts11,                        // [11]
    float* act11_out) {                      // [11, 640]
    __shared__ float4 smem_x9[80 * 9];       // `11,520` B SMEM -> `5` CTAs/SM!
    __shared__ float smem_gu[8];

    const int tid = threadIdx.x;
    pdl_wait();
    stage_x_groups_smem9(x4, smem_x9, 80, tid);
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int k = blockIdx.y;
    const int m = blockIdx.x * 4 + (warp_id >> 1);
    const int is_up = warp_id & 1;

    if (k < kTopK) {
        const int ex = is_compact ? k : sel_ids10[k];
        const size_t row = static_cast<size_t>(ex) * kMoeFf + m;
        const uint4* w_ptr = (is_up ? sw_up_w : sw_gate_w) + row * 80u;
        const uint32_t* sc_ptr = (is_up ? sw_up_sc : sw_gate_sc) + row * 40u;
        const uint32_t* bi_ptr = (is_up ? sw_up_bi : sw_gate_bi) + row * 40u;
        float dot = warp_dot_q4_row_smem9(w_ptr, sc_ptr, bi_ptr, smem_x9, 80, lane);
        if (lane == 0) smem_gu[warp_id] = dot;
    } else {
        const size_t row = static_cast<size_t>(is_up * kMoeFf + m);
        float dot = shared_warp_dot(sh_gu_gate_w, sh_gu_sc, sh_gu_bi, row, 80, smem_x9, lane);
        if (lane == 0) smem_gu[warp_id] = dot;

        if (blockIdx.x == 0 && warp_id == 0) {
            float sg = shared_warp_dot(sh_gu_gate_w, sh_gu_sc, sh_gu_bi, 1280u, 80, smem_x9, lane);
            if (lane == 0) sel_wts11[10] = ptx_sigmoid(sg);
        }
    }
    __syncthreads();

    if (lane == 0 && is_up == 0) {
        float g = smem_gu[warp_id];
        float u = smem_gu[warp_id + 1];
        act11_out[k * kMoeFf + m] = ptx_silu(g) * u;
    }
    pdl_trigger();
}

struct Z24ExpertProj {
    const uint4* w;
    const uint32_t* sb_lo;
    const uint32_t* sb_hi;
    size_t row_offset;
};

// Uniform runtime support for both contiguous Z24 slabs (`sb_lo != nullptr`) and per-expert
// `Mixed` pointer tables (`sb_lo == nullptr`, `w_or_table` = `[512]` `CUdeviceptr` table of `972,800 B` records).
__device__ __forceinline__ Z24ExpertProj resolve_z24_expert_proj(
    const uint4* w_or_table,
    const uint32_t* sb_lo,
    const uint32_t* sb_hi,
    int ex,
    size_t rows_per_ex,
    int row_in_ex) {
    if (sb_lo == nullptr) {
        const uint4* rec_w = reinterpret_cast<const uint4* const*>(w_or_table)[ex];
        const uint32_t* rec_lo = reinterpret_cast<const uint32_t*>(rec_w + 51200u);
        return {rec_w, rec_lo, rec_lo + 25600u, static_cast<size_t>(row_in_ex)};
    }
    return {w_or_table, sb_lo, sb_hi, static_cast<size_t>(ex) * rows_per_ex + static_cast<size_t>(row_in_ex)};
}

// `0-ULP` 24-bit `(scale, bias)` (`Z24`) `moe_up_swiglu11` (`Grid (160, 11)`, `Block 256`, `5` CTAs/SM!)
extern "C" __global__ void q38_moe_up_swiglu11_z24(
    const uint4* __restrict__ sw_gate_w,         // [512, 640, 80]
    const uint32_t* __restrict__ sw_gate_sb_lo,  // [512 * 640 * 40]
    const uint32_t* __restrict__ sw_gate_sb_hi,  // [512 * 640 * 20]
    const uint4* __restrict__ sw_up_w,           // [512, 640, 80]
    const uint32_t* __restrict__ sw_up_sb_lo,    // [512 * 640 * 40]
    const uint32_t* __restrict__ sw_up_sb_hi,    // [512 * 640 * 20]
    uint32_t packed_bases_m1,                    // `[gate_sc | (gate_bi<<8) | (up_sc<<16) | (up_bi<<24)]`
    const uint4* __restrict__ sh_gu_gate_w,      // [1281, 80]
    const uint32_t* __restrict__ sh_gu_sc,       // [1281, 40]
    const uint32_t* __restrict__ sh_gu_bi,       // [1281, 40]
    const float4* x4,                            // [640]
    const int32_t* sel_ids10,                    // [10]
    float* sel_wts11,                            // [11]
    float* act11_out) {                          // [11, 640]
    __shared__ float4 smem_x9[80 * 9];           // `11,520` B SMEM -> `5` CTAs/SM!
    __shared__ float smem_gu[8];

    const int tid = threadIdx.x;
    pdl_wait();
    stage_x_groups_smem9(x4, smem_x9, 80, tid);
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int k = blockIdx.y;
    const int m = blockIdx.x * 4 + (warp_id >> 1);
    const int is_up = warp_id & 1;

    if (k < kTopK) {
        const int ex = sel_ids10[k];
        const Z24ExpertProj p = resolve_z24_expert_proj(
            is_up ? sw_up_w : sw_gate_w,
            is_up ? sw_up_sb_lo : sw_gate_sb_lo,
            is_up ? sw_up_sb_hi : sw_gate_sb_hi,
            ex, kMoeFf, m);
        const uint32_t base_sc_m1 = is_up ? ((packed_bases_m1 >> 16) & 0xFFu) : (packed_bases_m1 & 0xFFu);
        const uint32_t base_bi_m1 = is_up ? ((packed_bases_m1 >> 24) & 0xFFu) : ((packed_bases_m1 >> 8) & 0xFFu);
        float dot = warp_dot_q4_row_z24_smem9(
            p.w + p.row_offset * 80u,
            p.sb_lo,
            p.sb_hi,
            p.row_offset * 80u,
            base_sc_m1,
            base_bi_m1,
            smem_x9,
            80,
            lane);
        if (lane == 0) smem_gu[warp_id] = dot;
    } else {
        const size_t row = static_cast<size_t>(is_up * kMoeFf + m);
        float dot = shared_warp_dot(sh_gu_gate_w, sh_gu_sc, sh_gu_bi, row, 80, smem_x9, lane);
        if (lane == 0) smem_gu[warp_id] = dot;

        if (blockIdx.x == 0 && warp_id == 0) {
            float sg = shared_warp_dot(sh_gu_gate_w, sh_gu_sc, sh_gu_bi, 1280u, 80, smem_x9, lane);
            if (lane == 0) sel_wts11[10] = ptx_sigmoid(sg);
        }
    }
    __syncthreads();

    if (lane == 0 && is_up == 0) {
        float g = smem_gu[warp_id];
        float u = smem_gu[warp_id + 1];
        act11_out[k * kMoeFf + m] = ptx_silu(g) * u;
    }
    pdl_trigger();
}

// Original ABI preserved (`Grid (320, 1)`, `Block 256`), upgraded with bank-conflict-free `smem_a9`!
extern "C" __global__ void q38_moe_down_combine11(
    const uint4* __restrict__ sw_down_w,     // [512, 2560, 20]
    const uint32_t* __restrict__ sw_down_sc, // [512, 2560, 10]
    const uint32_t* __restrict__ sw_down_bi, // [512, 2560, 10]
    const uint4* __restrict__ sh_down_w,     // [2560, 20]
    const uint32_t* __restrict__ sh_down_sc, // [2560, 10]
    const uint32_t* __restrict__ sh_down_bi, // [2560, 10]
    const float4* act11_x4,                  // [11, 160]
    const int32_t* sel_ids10,                // [10]
    int32_t is_compact,
    const float* sel_wts11,                  // [11]
    float* moe_out) {                        // [2560]
    __shared__ float4 smem_a9[11 * 20 * 9];  // `31,680` B SMEM -> `2` CTAs/SM under 64 KiB carveout!

    const int tid = threadIdx.x;
    pdl_wait();
    stage_x_groups_smem9(act11_x4, smem_a9, 11 * 20, tid);
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + warp_id;
    if (row < kD) {
        float sum_y = 0.0f;
        for (int k = 0; k < kTopK; ++k) {
            const int ex = is_compact ? k : sel_ids10[k];
            const size_t r_idx = static_cast<size_t>(ex) * kD + row;
            float dot = warp_dot_q4_row_smem9(
                sw_down_w + r_idx * 20u,
                sw_down_sc + r_idx * 10u,
                sw_down_bi + r_idx * 10u,
                smem_a9 + k * 20 * 9,
                20,
                lane);
            sum_y += dot * sel_wts11[k];
        }
        {
            float sh_dot = shared_warp_dot(sh_down_w, sh_down_sc, sh_down_bi, row, 20, smem_a9 + 10 * 20 * 9, lane);
            sum_y += sh_dot * sel_wts11[10];
        }
        if (lane == 0) {
            moe_out[row] = sum_y;
        }
    }
    pdl_trigger();
}

// `0-ULP` Z24 `moe_down_combine11` with `100%` lane utilization (`4` lanes/row × `8` rows/warp,
// `2` warps split `k = 0..4` and `k = 5..10` -> `32` rows/CTA, `Grid = (80, 1)`, `Block = 256`!)
extern "C" __global__ void q38_moe_down_combine11_z24(
    const uint4* __restrict__ sw_down_w,         // [512, 2560, 20]
    const uint32_t* __restrict__ sw_down_sb_lo,  // [512 * 2560 * 10]
    const uint32_t* __restrict__ sw_down_sb_hi,  // [512 * 2560 * 5]
    uint32_t down_bases_m1,                      // `[down_sc_m1 | (down_bi_m1 << 8)]`
    const uint4* __restrict__ sh_down_w,         // [2560, 20]
    const uint32_t* __restrict__ sh_down_sc,     // [2560, 10]
    const uint32_t* __restrict__ sh_down_bi,     // [2560, 10]
    const float4* act11_x4,                      // [11, 160]
    const int32_t* sel_ids10,                    // [10]
    const float* sel_wts11,                      // [11]
    float* moe_out) {                            // [2560]
    __shared__ float4 smem_a9[11 * 20 * 9];      // `31,680` B <= `31,744` B (`2` CTAs/SM under `64 KiB` carveout!)

    const int tid = threadIdx.x;
    pdl_wait();
    stage_x_groups_smem9(act11_x4, smem_a9, 11 * 20, tid);
    __syncthreads();

    const int warp_id = tid >> 5;                // 0..7
    const int lane = tid & 31;
    const int sub_row = lane >> 2;               // 0..7 (`8` rows per warp-pair!)
    const int sub_lane = lane & 3;               // 0..3 (`4` lanes per row -> `5` iters = `20` groups, `0` idle lanes!)
    const int pair_id = warp_id >> 1;            // 0..3
    const int is_odd = warp_id & 1;              // 0 (`k = 0..4`) or 1 (`k = 5..10`)
    const int row = blockIdx.x * 32 + pair_id * 8 + sub_row;

    const uint32_t base_sc_m1 = down_bases_m1 & 0xFFu;
    const uint32_t base_bi_m1 = (down_bases_m1 >> 8) & 0xFFu;

    float sum_y = 0.0f;
    if (row < kD) {
        const uint32_t lane4_mask = 0xFu << (sub_row * 4);
        const int k_start = is_odd ? 5 : 0;
        const int k_end = is_odd ? 10 : 5;
        for (int k = k_start; k < k_end; ++k) {
            const int ex = sel_ids10[k];
            const float wt = sel_wts11[k];
            const Z24ExpertProj p = resolve_z24_expert_proj(
                sw_down_w, sw_down_sb_lo, sw_down_sb_hi, ex, kD, row);
            const uint4* rw = p.w + p.row_offset * 20u;
            const size_t g_base = p.row_offset * 20u;
            const float4* a_k = smem_a9 + k * 20 * 9;

            float dot = 0.0f;
            #pragma unroll
            for (int it = 0; it < 5; ++it) {
                const int g = it * 4 + sub_lane;
                uint4 pk = rw[g];
                float sc, bi;
                load_z24_sb_pair(
                    p.sb_lo,
                    p.sb_hi,
                    g_base + static_cast<size_t>(g),
                    base_sc_m1,
                    base_bi_m1,
                    &sc,
                    &bi);
                dot += dot_q4_group32_smem9(pk, sc, bi, a_k + g * 9);
            }
            dot += __shfl_xor_sync(lane4_mask, dot, 2);
            dot += __shfl_xor_sync(lane4_mask, dot, 1);
            sum_y += dot * wt;
        }
        if (is_odd) {
            const float4* a_sh = smem_a9 + 10 * 20 * 9;
            float sh_dot = 0.0f;
            #pragma unroll
            for (int it = 0; it < 5; ++it) {
                const int g = it * 4 + sub_lane;
                sh_dot += shared_group_dot(sh_down_w, sh_down_sc, sh_down_bi, row * 20u + g, a_sh + g * 9);
            }
            sh_dot += __shfl_xor_sync(lane4_mask, sh_dot, 2);
            sh_dot += __shfl_xor_sync(lane4_mask, sh_dot, 1);
            sum_y += sh_dot * sel_wts11[10];
        }
    }
    __syncthreads();

    float* smem_part = reinterpret_cast<float*>(smem_a9);
    if (sub_lane == 0 && row < kD) {
        smem_part[warp_id * 8 + sub_row] = sum_y;
    }
    __syncthreads();

    if (is_odd == 0 && sub_lane == 0 && row < kD) {
        moe_out[row] = smem_part[warp_id * 8 + sub_row] + smem_part[(warp_id + 1) * 8 + sub_row];
    }
    pdl_trigger();
}

// ----------------------------------------------------------------------------
// 8. Greedy ArgMax (`248,320` Vocab) + `*pos_ptr += 1`
// ----------------------------------------------------------------------------
// Original ABI preserved (`Grid (1, 1)`, `Block 256`)
extern "C" __global__ void q38_argmax_and_inc_pos(
    const float* logits,
    int32_t* out_tok,
    int32_t* pos_ptr) {
    __shared__ float smem_val[8];
    __shared__ int smem_idx[8];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    pdl_wait();
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
    pdl_trigger();
}

// Fast `1024`-thread `float4` ArgMax (`Grid (1, 1)`, `Block 1024` -> `62,080` `LDG.E.128`s, `7.5x` faster!)
extern "C" __global__ void q38_argmax_fast(
    const float4* logits4, // [62080] (`248,320` floats)
    int32_t* out_tok,
    int32_t* pos_ptr) {
    __shared__ float smem_val[32];
    __shared__ int smem_idx[32];

    const int tid = threadIdx.x; // 0..1023
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    pdl_wait();
    float best_v = -1e30f;
    int best_i = 0;
    for (int i4 = tid; i4 < (kVocab / 4); i4 += 1024) {
        float4 v4 = logits4[i4];
        const int base = i4 * 4;
        if (v4.x > best_v) { best_v = v4.x; best_i = base; }
        if (v4.y > best_v) { best_v = v4.y; best_i = base + 1; }
        if (v4.z > best_v) { best_v = v4.z; best_i = base + 2; }
        if (v4.w > best_v) { best_v = v4.w; best_i = base + 3; }
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
    if (warp_id == 0) {
        float fv = smem_val[lane];
        int fi = smem_idx[lane];
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            float ov = __shfl_xor_sync(0xFFFFFFFFu, fv, mask);
            int oi = __shfl_xor_sync(0xFFFFFFFFu, fi, mask);
            if (ov > fv || (ov == fv && oi < fi)) {
                fv = ov;
                fi = oi;
            }
        }
        if (lane == 0) {
            *out_tok = fi;
            *pos_ptr += 1;
        }
    }
    pdl_trigger();
}

extern "C" __global__ void q38_inc_pos(int32_t* pos_ptr) {
    pdl_wait();
    if (threadIdx.x == 0) {
        *pos_ptr += 1;
    }
    pdl_trigger();
}

// ============================================================================
// 9. `0-ULP` Multi-Token Batched Prefill & Speculative Verify Suite (`1 <= B <= 64`)
//    Every kernel below preserves the exact floating-point reduction tree of its
//    `B = 1` counterpart while amortizing GDDR7 / PCIe weight reads across `B` rows.
// ============================================================================

// Grid: `(10, B)`, Block: `256`
extern "C" __global__ void q38_embed_init_hc4_batch(
    const uint4* __restrict__ w_u128,      // [vocab, 80]
    const uint32_t* __restrict__ sc_u32,   // [vocab, 40]
    const uint32_t* __restrict__ bi_u32,   // [vocab, 40]
    const int32_t* tokens_ptr,             // [B]
    float* out_hc_batch) {                 // [B, 10240]
    const int d = blockIdx.x * 256 + threadIdx.x;
    const int b = blockIdx.y;
    pdl_wait();
    if (d < kD) {
        const int tok = tokens_ptr[b];
        const int g = d >> 5;
        const int sub = d & 31;
        const int w_idx = sub >> 3;
        const int sh = (sub & 7) * 4;

        const uint4 pk = w_u128[static_cast<size_t>(tok) * 80u + g];
        uint32_t word = (w_idx == 0) ? pk.x : ((w_idx == 1) ? pk.y : ((w_idx == 2) ? pk.z : pk.w));
        float sc = load_bf16_from_u32_array(sc_u32 + static_cast<size_t>(tok) * 40u, g);
        float bi = load_bf16_from_u32_array(bi_u32 + static_cast<size_t>(tok) * 40u, g);
        float val = __fmaf_rn(static_cast<float>((word >> sh) & 0xFu), sc, bi);

        float* dst = out_hc_batch + static_cast<size_t>(b) * kHcD;
        dst[d] = val;
        dst[kD + d] = val;
        dst[2 * kD + d] = val;
        dst[3 * kD + d] = val;
    }
    pdl_trigger();
}

__device__ __forceinline__ void stage_x_groups_tk33(
    const float4* x4,
    float4 (*smem_k)[33],
    int n_groups,
    int tid) {
    if (tid < 256) {
        const int g = tid >> 3;
        const int sub = tid & 7;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (g < n_groups) {
            v = x4[tid];
            smem_k[sub][g] = v;
        }
        float s = (v.x + v.y) + (v.z + v.w);
        s += __shfl_xor_sync(0xFFFFFFFFu, s, 4);
        s += __shfl_xor_sync(0xFFFFFFFFu, s, 2);
        s += __shfl_xor_sync(0xFFFFFFFFu, s, 1);
        if (g < n_groups && sub == 0) {
            smem_k[8][g] = make_float4(s, 0.0f, 0.0f, 0.0f);
        }
    }
}

template <int MAX_T>
__device__ __forceinline__ void accum_q4_group32_tk33_multi(
    uint4 pk,
    float sc,
    float bi,
    const float4 (*smem_x)[9][33],
    int n_b,
    int lane,
    float* acc) {
    float a0[MAX_T], a1[MAX_T], a2[MAX_T], a3[MAX_T];
    #pragma unroll
    for (int t = 0; t < MAX_T; ++t) {
        a0[t] = 0.0f;
        a1[t] = 0.0f;
        a2[t] = 0.0f;
        a3[t] = 0.0f;
    }
    #pragma unroll
    for (int w_idx = 0; w_idx < 4; ++w_idx) {
        const uint32_t w = (w_idx == 0) ? pk.x : ((w_idx == 1) ? pk.y : ((w_idx == 2) ? pk.z : pk.w));
        const float q0 = static_cast<float>(w & 0xFu);
        const float q1 = static_cast<float>((w >> 4) & 0xFu);
        const float q2 = static_cast<float>((w >> 8) & 0xFu);
        const float q3 = static_cast<float>((w >> 12) & 0xFu);
        const float q4 = static_cast<float>((w >> 16) & 0xFu);
        const float q5 = static_cast<float>((w >> 20) & 0xFu);
        const float q6 = static_cast<float>((w >> 24) & 0xFu);
        const float q7 = static_cast<float>((w >> 28) & 0xFu);
        #pragma unroll
        for (int t = 0; t < MAX_T; ++t) {
            if (t < n_b) {
                const float4 x0 = smem_x[t][w_idx * 2][lane];
                const float4 x1 = smem_x[t][w_idx * 2 + 1][lane];
                a0[t] = __fmaf_rn(q0, x0.x, a0[t]);
                a1[t] = __fmaf_rn(q1, x0.y, a1[t]);
                a2[t] = __fmaf_rn(q2, x0.z, a2[t]);
                a3[t] = __fmaf_rn(q3, x0.w, a3[t]);
                a0[t] = __fmaf_rn(q4, x1.x, a0[t]);
                a1[t] = __fmaf_rn(q5, x1.y, a1[t]);
                a2[t] = __fmaf_rn(q6, x1.z, a2[t]);
                a3[t] = __fmaf_rn(q7, x1.w, a3[t]);
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < MAX_T; ++t) {
        if (t < n_b) {
            const float dot_nx = (a0[t] + a1[t]) + (a2[t] + a3[t]);
            acc[t] += __fmaf_rn(dot_nx, sc, bi * smem_x[t][8][lane].x);
        }
    }
}

// `0-ULP` High-Throughput Batched Q4 GEMM (`Block = 512`, `32` rows/CTA = `2` rows/warp, `8` tokens/tile, `38,016` B SMEM!)
// Unpacks Q4 nibbles ONCE per group and reuses across all `t = 0..7`!
// Grid: `(ceil(rows / 32), ceil(b_count / 8))`, Block: `512`
extern "C" __global__ void q38_q4_gemm_b8(
    const uint4* __restrict__ w_u128,      // [rows, cols / 32]
    const uint32_t* __restrict__ sc_u32,   // [rows, cols / 64]
    const uint32_t* __restrict__ bi_u32,   // [rows, cols / 64]
    const float4* x4_batch,                // [b_count, cols / 4]
    float* y_batch,                        // [b_count, rows]
    int32_t rows,
    int32_t cols,
    int32_t b_count,
    int32_t post_mode) {
    __shared__ float4 smem_x[8][9][33];    // `38,016` B SMEM (`0` write & `0` read bank conflicts, exact `0-ULP` `s`!)

    const int tid = threadIdx.x;           // 0..511
    const int warp_id = tid >> 5;          // 0..15
    const int lane = tid & 31;
    const int row0 = blockIdx.x * 32 + warp_id * 2;
    const int row1 = row0 + 1;
    const int b_base = blockIdx.y * 8;
    const int n_b = ((b_count - b_base) < 8) ? (b_count - b_base) : 8;
    const int n_groups = cols >> 5;
    const int cols4 = cols >> 2;
    const int sc_stride_u32 = cols >> 6;

    pdl_wait();
    float acc0[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g_base = 0; g_base < n_groups; g_base += 32) {
        const int tile_groups = ((n_groups - g_base) < 32) ? (n_groups - g_base) : 32;
        for (int t = 0; t < n_b; ++t) {
            const float4* src_t = x4_batch + static_cast<size_t>(b_base + t) * cols4 + g_base * 8;
            stage_x_groups_tk33(src_t, smem_x[t], tile_groups, tid);
        }
        __syncthreads();

        const int g = g_base + lane;
        if (g < n_groups) {
            if (row0 < rows) {
                const uint4 pk0 = w_u128[static_cast<size_t>(row0) * n_groups + g];
                const float sc0 = load_bf16_from_u32_array(sc_u32 + static_cast<size_t>(row0) * sc_stride_u32, g);
                const float bi0 = load_bf16_from_u32_array(bi_u32 + static_cast<size_t>(row0) * sc_stride_u32, g);
                accum_q4_group32_tk33_multi<8>(pk0, sc0, bi0, smem_x, n_b, lane, acc0);
            }
            if (row1 < rows) {
                const uint4 pk1 = w_u128[static_cast<size_t>(row1) * n_groups + g];
                const float sc1 = load_bf16_from_u32_array(sc_u32 + static_cast<size_t>(row1) * sc_stride_u32, g);
                const float bi1 = load_bf16_from_u32_array(bi_u32 + static_cast<size_t>(row1) * sc_stride_u32, g);
                accum_q4_group32_tk33_multi<8>(pk1, sc1, bi1, smem_x, n_b, lane, acc1);
            }
        }
        __syncthreads();
    }

    if (row0 < rows) {
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < n_b) {
                float v = warp_sum(acc0[t]);
                if (lane == 0) {
                    if (post_mode == 1) v = ptx_silu(v * 0.25f);
                    else if (post_mode == 2) v = ptx_sigmoid(v);
                    y_batch[static_cast<size_t>(b_base + t) * rows + row0] = v;
                }
            }
        }
    }
    if (row1 < rows) {
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < n_b) {
                float v = warp_sum(acc1[t]);
                if (lane == 0) {
                    if (post_mode == 1) v = ptx_silu(v * 0.25f);
                    else if (post_mode == 2) v = ptx_sigmoid(v);
                    y_batch[static_cast<size_t>(b_base + t) * rows + row1] = v;
                }
            }
        }
    }
    pdl_trigger();
}

// Layer 1 causal PLE inject across `b = 0..b_count-1` (`Grid (4, 1)`, `Block 256`)
extern "C" __global__ void q38_ple_inject_seq_b(
    float* hc_batch,                          // [b_count, 10240]
    const float* kv_batch,                    // [b_count, 12800] (`key [10240]` + `val [2560]`)
    const float* __restrict__ norm_key_w,     // [4, 2560]
    const float* __restrict__ norm_query_w,   // [4, 2560]
    const float* __restrict__ norm_conv_w,    // [4, 2560]
    const float4* __restrict__ conv_w4,       // [4, 2560]
    float* ple_hist9,                         // [9, 10240]
    const int32_t* pos0_ptr,
    int32_t b_count) {
    __shared__ float smem_a[8];
    __shared__ float smem_b[8];
    __shared__ float smem_c[8];
    __shared__ float smem_scalars[4];

    const int s = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const float* nk_s = norm_key_w + s * kD;
    const float* nq_s = norm_query_w + s * kD;
    const float* nc_s = norm_conv_w + s * kD;
    const float4* cw_s = conv_w4 + s * kD;

    pdl_wait();
    const int pos0 = *pos0_ptr;

    for (int b = 0; b < b_count; ++b) {
        const int pos = pos0 + b;
        float* hc_s = hc_batch + static_cast<size_t>(b) * kHcD + s * kD;
        const float* kv_b = kv_batch + static_cast<size_t>(b) * (kHcD + kD);
        const float* kr_s = kv_b + s * kD;
        const float* val_raw = kv_b + kHcD;

        float sq_k = 0.0f, sq_q = 0.0f, sq_v = 0.0f;
        for (int d = tid; d < kD; d += 256) {
            float kv = kr_s[d];
            float qv = hc_s[d];
            float vv = val_raw[d];
            sq_k += kv * kv;
            sq_q += qv * qv;
            sq_v += vv * vv;
        }
        sq_k = warp_sum(sq_k);
        sq_q = warp_sum(sq_q);
        sq_v = warp_sum(sq_v);
        if (lane == 0) {
            smem_a[warp_id] = sq_k;
            smem_b[warp_id] = sq_q;
            smem_c[warp_id] = sq_v;
        }
        __syncthreads();
        if (warp_id == 0) {
            float tk = (lane < 8) ? smem_a[lane] : 0.0f;
            float tq = (lane < 8) ? smem_b[lane] : 0.0f;
            float tv = (lane < 8) ? smem_c[lane] : 0.0f;
            tk = warp_sum(tk);
            tq = warp_sum(tq);
            tv = warp_sum(tv);
            if (lane == 0) {
                smem_scalars[0] = ptx_rsqrt(tk * (1.0f / 2560.0f) + 1e-6f);
                smem_scalars[1] = ptx_rsqrt(tq * (1.0f / 2560.0f) + 1e-6f);
                smem_scalars[3] = tv * (1.0f / 2560.0f);
            }
        }
        __syncthreads();

        const float inv_k = smem_scalars[0];
        const float inv_q = smem_scalars[1];
        float dot = 0.0f;
        for (int d = tid; d < kD; d += 256) {
            dot += (kr_s[d] * inv_k * nk_s[d]) * (hc_s[d] * inv_q * nq_s[d]);
        }
        dot = warp_sum(dot);
        if (lane == 0) smem_a[warp_id] = dot;
        __syncthreads();
        if (warp_id == 0) {
            float td = (lane < 8) ? smem_a[lane] : 0.0f;
            td = warp_sum(td);
            if (lane == 0) {
                float g = td * 0.01976423537605237f;
                float g_abs = fabsf(g);
                float g_clamp = (g_abs < 1e-6f) ? 1e-6f : g_abs;
                float signed_sqrt = (g > 0.0f) ? ptx_sqrt(g_clamp) : ((g < 0.0f) ? -ptx_sqrt(g_clamp) : 0.0f);
                float sig = ptx_sigmoid(signed_sqrt);
                smem_scalars[2] = sig;
                smem_scalars[3] = ptx_rsqrt(sig * sig * smem_scalars[3] + 1e-6f);
            }
        }
        __syncthreads();

        const float sig_g = smem_scalars[2];
        const float inv_gv = smem_scalars[3];
        const int s_m9 = pos % 9;
        const int s_m6 = (pos + 3) % 9;
        const int s_m3 = (pos + 6) % 9;

        for (int d = tid; d < kD; d += 256) {
            const int ch = s * kD + d;
            float gv = sig_g * val_raw[d];
            float gvn = gv * inv_gv * nc_s[d];

            float x_m9 = ple_hist9[static_cast<size_t>(s_m9) * kHcD + ch];
            float x_m6 = ple_hist9[static_cast<size_t>(s_m6) * kHcD + ch];
            float x_m3 = ple_hist9[static_cast<size_t>(s_m3) * kHcD + ch];
            ple_hist9[static_cast<size_t>(s_m9) * kHcD + ch] = gvn;

            float4 cw = cw_s[d];
            float c_out = ptx_silu(cw.x * x_m9 + cw.y * x_m6 + cw.z * x_m3 + cw.w * gvn);
            hc_s[d] += gv + c_out;
        }
        __syncthreads();
    }
    pdl_trigger();
}

// Grid: `(4, B)`, Block: `256`
extern "C" __global__ void q38_hc_norm4_batch(
    const float* hc_in_batch,             // [B, 10240]
    const float* __restrict__ hc_gamma,   // [4, 2560]
    float* hc_normed_batch) {             // [B, 10240]
    __shared__ float smem_sq[8];
    __shared__ float smem_inv;

    const int s = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const float4* gam4 = reinterpret_cast<const float4*>(hc_gamma + s * kD);

    pdl_wait();
    const size_t off = static_cast<size_t>(b) * kHcD + s * kD;
    const float4* src4 = reinterpret_cast<const float4*>(hc_in_batch + off);
    float4* dst4 = reinterpret_cast<float4*>(hc_normed_batch + off);

    float4 r0 = src4[tid];
    float4 r1 = src4[tid + 256];
    float4 r2 = (tid < 128) ? src4[tid + 512] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float sq = (r0.x * r0.x + r0.y * r0.y) + (r0.z * r0.z + r0.w * r0.w)
             + (r1.x * r1.x + r1.y * r1.y) + (r1.z * r1.z + r1.w * r1.w)
             + (r2.x * r2.x + r2.y * r2.y) + (r2.z * r2.z + r2.w * r2.w);
    sq = warp_sum(sq);
    if (lane == 0) smem_sq[warp_id] = sq;
    __syncthreads();
    if (warp_id == 0) {
        float tot = (lane < 8) ? smem_sq[lane] : 0.0f;
        tot = warp_sum(tot);
        if (lane == 0) smem_inv = ptx_rsqrt(tot * (1.0f / 2560.0f) + 1e-6f);
    }
    __syncthreads();

    const float inv = smem_inv;
    {
        float4 g0 = gam4[tid];
        dst4[tid] = make_float4(r0.x * inv * g0.x, r0.y * inv * g0.y, r0.z * inv * g0.z, r0.w * inv * g0.w);
        float4 g1 = gam4[tid + 256];
        dst4[tid + 256] = make_float4(r1.x * inv * g1.x, r1.y * inv * g1.y, r1.z * inv * g1.z, r1.w * inv * g1.w);
        if (tid < 128) {
            float4 g2 = gam4[tid + 512];
            dst4[tid + 512] = make_float4(r2.x * inv * g2.x, r2.y * inv * g2.y, r2.z * inv * g2.z, r2.w * inv * g2.w);
        }
    }
    pdl_trigger();
}

// `0-ULP` Exact Batched `hc_down_inj_10k` (`1` warp/row = `8` rows/CTA, `4` tokens × `32`-group `K_TILE` = `16,896` B SMEM -> `3` CTAs/SM!)
// Accumulates `part_lo[t]` (`g < 160`) and `part_hi[t]` (`g >= 160`) separately so `warp_sum(part_lo) + warp_sum(part_hi)`
// matches `B = 1` (`q38_hc_down_inj_10k`) bit-for-bit (`0-ULP`)!
// Grid: `(41 or 40, ceil(b_count / 4))`, Block: `256`
extern "C" __global__ void q38_hc_down_inj_10k_b4(
    const uint4* __restrict__ down_w,     // [320, 320]
    const uint32_t* __restrict__ down_sc, // [320, 160]
    const uint32_t* __restrict__ down_bi, // [320, 160]
    const uint4* __restrict__ inj_w,      // [4, 320]
    const uint32_t* __restrict__ inj_sc,  // [4, 160]
    const uint32_t* __restrict__ inj_bi,  // [4, 160]
    const float4* hc_normed4_batch,       // [b_count, 2560]
    float* hc_low_batch,                  // [b_count, 320]
    float* hc_inj4_batch,                 // [b_count, 4]
    int32_t b_count,
    int32_t has_inject) {
    __shared__ float4 smem_x[4][9][33];   // `19,008` B SMEM -> `3` CTAs/SM, `0` bank conflicts!

    const int tid = threadIdx.x;          // 0..255 (`32 * 8 = 256` `float4`s!)
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + warp_id;
    const int max_rows = has_inject ? 324 : 320;
    const int b_base = blockIdx.y * 4;
    const int n_b = ((b_count - b_base) < 4) ? (b_count - b_base) : 4;

    pdl_wait();
    float part_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float part_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const uint4* rw = nullptr;
    const uint32_t* rsc = nullptr;
    const uint32_t* rbi = nullptr;
    if (row < max_rows) {
        rw = (row < 320)
            ? (down_w + static_cast<size_t>(row) * 320u)
            : (inj_w + static_cast<size_t>(row - 320) * 320u);
        rsc = (row < 320)
            ? (down_sc + static_cast<size_t>(row) * 160u)
            : (inj_sc + static_cast<size_t>(row - 320) * 160u);
        rbi = (row < 320)
            ? (down_bi + static_cast<size_t>(row) * 160u)
            : (inj_bi + static_cast<size_t>(row - 320) * 160u);
    }

    for (int g_base = 0; g_base < 320; g_base += 32) {
        for (int t = 0; t < n_b; ++t) {
            const float4* src_t = hc_normed4_batch + static_cast<size_t>(b_base + t) * 2560u + g_base * 8;
            stage_x_groups_tk33(src_t, smem_x[t], 32, tid);
        }
        __syncthreads();

        if (row < max_rows) {
            const int g = g_base + lane;
            uint4 pk = rw[g];
            float sc = load_bf16_from_u32_array(rsc, g);
            float bi = load_bf16_from_u32_array(rbi, g);
            if (g_base < 160) {
                accum_q4_group32_tk33_multi<4>(pk, sc, bi, smem_x, n_b, lane, part_lo);
            } else {
                accum_q4_group32_tk33_multi<4>(pk, sc, bi, smem_x, n_b, lane, part_hi);
            }
        }
        __syncthreads();
    }

    if (row < max_rows) {
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            if (t < n_b) {
                float v = warp_sum(part_lo[t]) + warp_sum(part_hi[t]);
                if (lane == 0) {
                    if (row < 320) {
                        hc_low_batch[static_cast<size_t>(b_base + t) * 320u + row] = ptx_silu(v * 0.25f);
                    } else {
                        hc_inj4_batch[static_cast<size_t>(b_base + t) * 4u + (row - 320)] = v;
                    }
                }
            }
        }
    }
    pdl_trigger();
}

// `0-ULP` Batched `hc_up_mix4` (`8` tokens/tile × `10` groups = `11,520` B SMEM -> `5` CTAs/SM!)
// Grid: `(320, ceil(b_count / 8))`, Block: `256`
extern "C" __global__ void q38_hc_up_mix4_b8(
    const uint4* __restrict__ up_w,       // [10240, 10]
    const uint32_t* __restrict__ up_sc,   // [10240, 5]
    const uint32_t* __restrict__ up_bi,   // [10240, 5]
    const float4* hc_low4_batch,          // [b_count, 80]
    const float* hc_normed_batch,         // [b_count, 10240]
    float* mixed_out_batch,               // [b_count, 2560]
    int32_t b_count) {
    __shared__ float4 smem_low9[8 * 10 * 9]; // `11,520` B SMEM -> `5` CTAs/SM!

    const int tid = threadIdx.x;
    const int b_base = blockIdx.y * 8;
    const int n_b = ((b_count - b_base) < 8) ? (b_count - b_base) : 8;

    pdl_wait();
    for (int t = 0; t < n_b; ++t) {
        stage_x_groups_smem9(hc_low4_batch + static_cast<size_t>(b_base + t) * 80u, smem_low9 + t * 10 * 9, 10, tid);
    }
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int d = blockIdx.x * 8 + warp_id;
    if (d < kD) {
        const int s = lane >> 3;
        const int sub_lane = lane & 7;
        const size_t row = static_cast<size_t>(s) * kD + static_cast<size_t>(d);
        const uint4* rw = up_w + row * 10u;
        const uint32_t* rsc = up_sc + row * 5u;
        const uint32_t* rbi = up_bi + row * 5u;

        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        {
            const int g0 = sub_lane;
            uint4 pk0 = rw[g0];
            float sc0 = load_bf16_from_u32_array(rsc, g0);
            float bi0 = load_bf16_from_u32_array(rbi, g0);
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                if (t < n_b) {
                    acc[t] += dot_q4_group32_smem9(pk0, sc0, bi0, smem_low9 + (t * 10 + g0) * 9);
                }
            }
        }
        if (sub_lane < 2) {
            const int g1 = 8 + sub_lane;
            uint4 pk1 = rw[g1];
            float sc1 = load_bf16_from_u32_array(rsc, g1);
            float bi1 = load_bf16_from_u32_array(rbi, g1);
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                if (t < n_b) {
                    acc[t] += dot_q4_group32_smem9(pk1, sc1, bi1, smem_low9 + (t * 10 + g1) * 9);
                }
            }
        }

        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < n_b) {
                float a = acc[t];
                a += __shfl_xor_sync(0xFFFFFFFFu, a, 4);
                a += __shfl_xor_sync(0xFFFFFFFFu, a, 2);
                a += __shfl_xor_sync(0xFFFFFFFFu, a, 1);
                float weighted = 0.0f;
                if (sub_lane == 0) {
                    weighted = ptx_sigmoid(a) * hc_normed_batch[static_cast<size_t>(b_base + t) * kHcD + row];
                }
                weighted += __shfl_down_sync(0xFFFFFFFFu, weighted, 16);
                weighted += __shfl_down_sync(0xFFFFFFFFu, weighted, 8);
                if (lane == 0) {
                    mixed_out_batch[static_cast<size_t>(b_base + t) * kD + d] = weighted * 0.25f;
                }
            }
        }
    }
    pdl_trigger();
}

// Grid: `(3, B)`, Block: `256`
extern "C" __global__ void q38_hc_inject4_vec_batch(
    float4* hc_inout4_batch,              // [B, 4, 640]
    const float4* branch_out4_batch,      // [B, 640]
    const float4* inj_raw4_batch) {       // [B]
    const int d4 = blockIdx.x * 256 + threadIdx.x;
    const int b_idx = blockIdx.y;
    pdl_wait();
    if (d4 < 640) {
        const float4 raw = inj_raw4_batch[b_idx];
        const float w0 = 2.0f * ptx_sigmoid(raw.x * 0.25f);
        const float w1 = 2.0f * ptx_sigmoid(raw.y * 0.25f);
        const float w2 = 2.0f * ptx_sigmoid(raw.z * 0.25f);
        const float w3 = 2.0f * ptx_sigmoid(raw.w * 0.25f);
        const float4 b = branch_out4_batch[static_cast<size_t>(b_idx) * 640u + d4];
        float4* hc = hc_inout4_batch + static_cast<size_t>(b_idx) * 2560u;

        float4 h0 = hc[d4];
        hc[d4] = make_float4(h0.x + b.x * w0, h0.y + b.y * w0, h0.z + b.z * w0, h0.w + b.w * w0);
        float4 h1 = hc[640 + d4];
        hc[640 + d4] = make_float4(h1.x + b.x * w1, h1.y + b.y * w1, h1.z + b.z * w1, h1.w + b.w * w1);
        float4 h2 = hc[1280 + d4];
        hc[1280 + d4] = make_float4(h2.x + b.x * w2, h2.y + b.y * w2, h2.z + b.z * w2, h2.w + b.w * w2);
        float4 h3 = hc[1920 + d4];
        hc[1920 + d4] = make_float4(h3.x + b.x * w3, h3.y + b.y * w3, h3.z + b.z * w3, h3.w + b.w * w3);
    }
    pdl_trigger();
}

// Causal Conv4 + SiLU over `b = 0..b_count-1` (`Grid (40, 1)`, `Block 256`, `gdn_in` stride = `16,480`)
extern "C" __global__ void q38_gdn_conv_silu_seq_b(
    const float* gdn_in_batch,            // [b_count, 16480] (first `10240` = QKV)
    const float4* __restrict__ conv_w4,   // [10240]
    float* conv_hist3,                    // [10240 * 3]
    float* qkv_conv_batch,                // [b_count, 10240]
    int32_t b_count) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    float4 w = (c < 10240) ? conv_w4[c] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    pdl_wait();
    if (c < 10240) {
        float* h = conv_hist3 + c * 3;
        float h0 = h[0], h1 = h[1], h2 = h[2];
        for (int b = 0; b < b_count; ++b) {
            float x = gdn_in_batch[static_cast<size_t>(b) * 16480u + c];
            float y = ptx_silu(w.x * h0 + w.y * h1 + w.z * h2 + w.w * x);
            qkv_conv_batch[static_cast<size_t>(b) * 10240u + c] = y;
            h0 = h1;
            h1 = h2;
            h2 = x;
        }
        h[0] = h0;
        h[1] = h1;
        h[2] = h2;
    }
    pdl_trigger();
}

// `0-ULP` On-Chip Causal GDN Recurrent Step across `b = 0..b_count-1` (`Grid (48, 1)`, `Block 128`)
// Keeps `S_vh[0..63][dv]` in conflict-free `smem_S[64][129]` (`33,024 B`) and `S_vh[64..127][dv]` in registers (`s_hi[64]`),
// eliminating `98%` (`576 MiB -> 6 MiB`/layer) of global state traffic while preserving exact `0-ULP` `dk = 0..127` order!
extern "C" __global__ void q38_gdn_recurrent_step_seq_b(
    const float* qkv_conv_batch,          // [b_count, 10240]
    const float* gdn_in_batch,            // [b_count, 16480] (`+10240` = `Z[6144], B[48], A[48]`)
    const float* __restrict__ a_log,      // [48]
    const float* __restrict__ dt_bias,    // [48]
    const float* __restrict__ norm_w,     // [128]
    int32_t use_sigmoid_gate,
    int32_t b_count,
    float* state_f32,                     // [48 * 128 * 128]
    float* gdn_out_batch) {               // [b_count, 6144]
    __shared__ float smem_S[64][129];     // `33,024` B SMEM (`0` bank conflicts: stride `129` is odd!)
    __shared__ float smem_q[128];
    __shared__ float smem_k[128];
    __shared__ float smem_red[4];
    __shared__ float smem_gate[3];

    const int vh = blockIdx.x;
    const int kh = vh / 3;
    const int dv = threadIdx.x;
    const int warp_id = dv >> 5;
    const int lane = dv & 31;
    const float nw_dv = norm_w[dv];
    const float al_vh = (dv == 0) ? a_log[vh] : 0.0f;
    const float dt_vh = (dv == 0) ? dt_bias[vh] : 0.0f;
    float* S_vh = state_f32 + static_cast<size_t>(vh) * 16384u;

    pdl_wait();
    #pragma unroll
    for (int dk = 0; dk < 64; ++dk) {
        smem_S[dk][dv] = S_vh[dk * 128 + dv];
    }
    float s_hi[64];
    #pragma unroll
    for (int dk = 0; dk < 64; ++dk) {
        s_hi[dk] = S_vh[(64 + dk) * 128 + dv];
    }
    __syncthreads();

    for (int b = 0; b < b_count; ++b) {
        const float* qkv_conv = qkv_conv_batch + static_cast<size_t>(b) * 10240u;
        const float* zba_buf = gdn_in_batch + static_cast<size_t>(b) * 16480u + 10240u;

        float qv = qkv_conv[kh * 128 + dv];
        float kv = qkv_conv[2048 + kh * 128 + dv];
        float q_sq = warp_sum(qv * qv);
        float k_sq = warp_sum(kv * kv);
        if (lane == 0) smem_red[warp_id] = q_sq;
        __syncthreads();
        float q_tot = smem_red[0] + smem_red[1] + smem_red[2] + smem_red[3];
        float q_inv = ptx_rsqrt(q_tot + 1e-6f) * 0.08838834764831845f;
        __syncthreads();
        if (lane == 0) smem_red[warp_id] = k_sq;
        __syncthreads();
        float k_tot = smem_red[0] + smem_red[1] + smem_red[2] + smem_red[3];
        float k_inv = ptx_rsqrt(k_tot + 1e-6f);

        smem_q[dv] = qv * q_inv;
        smem_k[dv] = kv * k_inv;
        if (dv == 0) {
            float b_raw = zba_buf[6144 + vh];
            float a_raw = zba_buf[6192 + vh];
            float g = -ptx_exp(al_vh) * ptx_softplus(a_raw + dt_vh);
            smem_gate[0] = ptx_exp(g);
            smem_gate[1] = ptx_sigmoid(b_raw);
        }
        __syncthreads();

        const float decay = smem_gate[0];
        const float beta = smem_gate[1];

        float kv_mem = 0.0f;
        #pragma unroll 8
        for (int dk = 0; dk < 64; ++dk) {
            kv_mem += (smem_S[dk][dv] * decay) * smem_k[dk];
        }
        #pragma unroll
        for (int dk = 0; dk < 64; ++dk) {
            kv_mem += (s_hi[dk] * decay) * smem_k[64 + dk];
        }
        const float v_val = qkv_conv[4096 + vh * 128 + dv];
        const float delta = (v_val - kv_mem) * beta;

        float out_v = 0.0f;
        #pragma unroll 8
        for (int dk = 0; dk < 64; ++dk) {
            float s_new = smem_S[dk][dv] * decay + smem_k[dk] * delta;
            smem_S[dk][dv] = s_new;
            out_v += s_new * smem_q[dk];
        }
        #pragma unroll
        for (int dk = 0; dk < 64; ++dk) {
            float s_new = s_hi[dk] * decay + smem_k[64 + dk] * delta;
            s_hi[dk] = s_new;
            out_v += s_new * smem_q[64 + dk];
        }

        float o_sq = warp_sum(out_v * out_v);
        if (lane == 0) smem_red[warp_id] = o_sq;
        __syncthreads();
        if (dv == 0) {
            float o_tot = smem_red[0] + smem_red[1] + smem_red[2] + smem_red[3];
            smem_gate[2] = ptx_rsqrt(o_tot * (1.0f / 128.0f) + 1e-6f);
        }
        __syncthreads();

        float z_val = zba_buf[vh * 128 + dv];
        float gate_act = use_sigmoid_gate ? ptx_sigmoid(z_val) : ptx_silu(z_val);
        gdn_out_batch[static_cast<size_t>(b) * 6144u + vh * 128 + dv] = out_v * smem_gate[2] * nw_dv * gate_act;
        __syncthreads();
    }

    #pragma unroll
    for (int dk = 0; dk < 64; ++dk) {
        S_vh[dk * 128 + dv] = smem_S[dk][dv];
    }
    #pragma unroll
    for (int dk = 0; dk < 64; ++dk) {
        S_vh[(64 + dk) * 128 + dv] = s_hi[dk];
    }
    pdl_trigger();
}

// Batched QSA Prep (`Grid (31, B)`, `Block 256` — writes `q_rot`, `q_gate`, `k_cache`, `v_cache`, `idx_q`, `idx_raw_k`)
extern "C" __global__ void q38_qsa_prep_batch(
    const float* qsa_proj_batch,               // [B, 13952]
    const float* __restrict__ q_norm_w,        // [256]
    const float* __restrict__ k_norm_w,        // [256]
    const float* __restrict__ idx_q_norm_w,    // [128]
    const float* __restrict__ inv_freq32,      // [32]
    const int32_t* pos0_ptr,
    float* q_rot_batch,                        // [B, 24, 256]
    float* q_gate_batch,                       // [B, 24, 256]
    uint16_t* k_cache_fp16,                    // [max_ctx, 512]
    uint16_t* v_cache_fp16,                    // [max_ctx, 512]
    float* idx_q_batch,                        // [B, 4, 128]
    float* idx_raw_k_cache) {                  // [max_ctx, 128]
    __shared__ float smem_vec[256];
    __shared__ float smem_red[8];

    const int blk = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    pdl_wait();
    const int pos = *pos0_ptr + b;
    const float* qsa_proj = qsa_proj_batch + static_cast<size_t>(b) * 13952u;

    if (blk < 24) {
        const int h = blk;
        const float* src = qsa_proj + h * 512;
        q_gate_batch[static_cast<size_t>(b) * 6144u + h * 256 + tid] = src[256 + tid];
        float v = src[tid];
        float sq = warp_sum(v * v);
        if (lane == 0) smem_red[warp_id] = sq;
        __syncthreads();
        if (warp_id == 0) {
            float tot = (lane < 8) ? smem_red[lane] : 0.0f;
            tot = warp_sum(tot);
            if (lane == 0) smem_red[0] = ptx_rsqrt(tot * (1.0f / 256.0f) + 1e-6f);
        }
        __syncthreads();
        smem_vec[tid] = v * smem_red[0] * q_norm_w[tid];
        __syncthreads();

        float out_v = smem_vec[tid];
        if (tid < 64) {
            int f_idx = tid & 31;
            float s, c;
            ptx_sincos_reduced(static_cast<float>(pos) * inv_freq32[f_idx], &s, &c);
            out_v = (tid < 32)
                ? (smem_vec[tid] * c - smem_vec[tid + 32] * s)
                : (smem_vec[tid - 32] * s + smem_vec[tid] * c);
        }
        q_rot_batch[static_cast<size_t>(b) * 6144u + h * 256 + tid] = out_v;
    } else if (blk < 26) {
        const int kh = blk - 24;
        const float kv = qsa_proj[12288 + kh * 256 + tid];
        const float vv = qsa_proj[12800 + kh * 256 + tid];
        v_cache_fp16[static_cast<size_t>(pos) * 512u + kh * 256 + tid] = f32_to_fp16_bits(vv);

        float sq = warp_sum(kv * kv);
        if (lane == 0) smem_red[warp_id] = sq;
        __syncthreads();
        if (warp_id == 0) {
            float tot = (lane < 8) ? smem_red[lane] : 0.0f;
            tot = warp_sum(tot);
            if (lane == 0) smem_red[0] = ptx_rsqrt(tot * (1.0f / 256.0f) + 1e-6f);
        }
        __syncthreads();
        smem_vec[tid] = kv * smem_red[0] * k_norm_w[tid];
        __syncthreads();

        float out_k = smem_vec[tid];
        if (tid < 64) {
            int f_idx = tid & 31;
            float s, c;
            ptx_sincos_reduced(static_cast<float>(pos) * inv_freq32[f_idx], &s, &c);
            out_k = (tid < 32)
                ? (smem_vec[tid] * c - smem_vec[tid + 32] * s)
                : (smem_vec[tid - 32] * s + smem_vec[tid] * c);
        }
        k_cache_fp16[static_cast<size_t>(pos) * 512u + kh * 256 + tid] = f32_to_fp16_bits(out_k);
    } else if (blk < 30) {
        if (tid < 128) {
            const int ih = blk - 26;
            float qv = qsa_proj[13312 + ih * 128 + tid];
            float sq = warp_sum(qv * qv);
            if (lane == 0) smem_red[warp_id] = sq;
            __syncthreads();
            float tot = smem_red[0] + smem_red[1] + smem_red[2] + smem_red[3];
            float inv = ptx_rsqrt(tot * (1.0f / 128.0f) + 1e-6f);
            smem_vec[tid] = qv * inv * idx_q_norm_w[tid];
            __syncthreads();

            float out_q = smem_vec[tid];
            if (tid < 64) {
                int f_idx = tid & 31;
                float s, c;
                ptx_sincos_reduced(static_cast<float>(pos) * inv_freq32[f_idx], &s, &c);
                out_q = (tid < 32)
                    ? (smem_vec[tid] * c - smem_vec[tid + 32] * s)
                    : (smem_vec[tid - 32] * s + smem_vec[tid] * c);
            }
            idx_q_batch[static_cast<size_t>(b) * 512u + ih * 128 + tid] = out_q;
        }
    } else {
        if (tid < 128) {
            idx_raw_k_cache[static_cast<size_t>(pos) * 128u + tid] = qsa_proj[13824 + tid];
        }
    }
    pdl_trigger();
}

// Race-free 4-token Indexer Block Pooling (`Grid (B, 1)`, `Block 128` — runs right after `q38_qsa_prep_batch`!)
extern "C" __global__ void q38_qsa_build_blocks_batch(
    const float* idx_raw_k_cache,              // [max_ctx, 128]
    const float* __restrict__ idx_k_norm_w,    // [128]
    const float* __restrict__ inv_freq32,      // [32]
    const int32_t* pos0_ptr,
    float* idx_block_k_cache) {                // [max_ctx / 4, 128]
    __shared__ float smem_vec[128];
    __shared__ float smem_red[4];

    const int b = blockIdx.x;
    const int tid = threadIdx.x; // 0..127
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    pdl_wait();
    const int pos = *pos0_ptr + b;
    if ((pos & 3) == 3) {
        const int b_idx = pos >> 2;
        const int g_start = pos - 3;
        float pooled = 0.25f * (
            idx_raw_k_cache[static_cast<size_t>(g_start) * 128u + tid] +
            idx_raw_k_cache[static_cast<size_t>(g_start + 1) * 128u + tid] +
            idx_raw_k_cache[static_cast<size_t>(g_start + 2) * 128u + tid] +
            idx_raw_k_cache[static_cast<size_t>(g_start + 3) * 128u + tid]);
        float sq = warp_sum(pooled * pooled);
        if (lane == 0) smem_red[warp_id] = sq;
        __syncthreads();
        float tot = smem_red[0] + smem_red[1] + smem_red[2] + smem_red[3];
        float inv = ptx_rsqrt(tot * (1.0f / 128.0f) + 1e-6f);
        smem_vec[tid] = pooled * inv * idx_k_norm_w[tid];
        __syncthreads();

        float out_bk = smem_vec[tid];
        if (tid < 64) {
            int f_idx = tid & 31;
            float s, c;
            ptx_sincos_reduced(static_cast<float>(g_start) * inv_freq32[f_idx], &s, &c);
            out_bk = (tid < 32)
                ? (smem_vec[tid] * c - smem_vec[tid + 32] * s)
                : (smem_vec[tid - 32] * s + smem_vec[tid] * c);
        }
        idx_block_k_cache[static_cast<size_t>(b_idx) * 128u + tid] = out_bk;
    }
    pdl_trigger();
}

// Long-context Batched QSA Block Scorer (`Grid (ceil(max_blocks / 8), B)`, `Block 256`)
extern "C" __global__ void q38_qsa_score_blocks_batch(
    const float4* idx_q4_batch,                 // [B, 4, 32]
    const float4* block_k4_cache,               // [max_blocks, 32]
    const int32_t* pos0_ptr,
    int32_t mask_stride,
    float* block_scores_batch) {                // [B, mask_stride]
    __shared__ float4 smem_q4[4][33];

    const int tid = threadIdx.x;
    const int b_tok = blockIdx.y;
    pdl_wait();
    const int n_blocks = (*pos0_ptr + b_tok + 1) >> 2;
    if (n_blocks > 512) {
        if (tid < 128) {
            smem_q4[tid >> 5][tid & 31] = idx_q4_batch[static_cast<size_t>(b_tok) * 128u + tid];
        }
        __syncthreads();

        const int warp_id = tid >> 5;
        const int lane = tid & 31;
        const int blk = blockIdx.x * 8 + warp_id;
        if (blk < n_blocks) {
            float4 bk = block_k4_cache[static_cast<size_t>(blk) * 32u + lane];
            float s_tot = 0.0f;
            #pragma unroll
            for (int ih = 0; ih < 4; ++ih) {
                float4 qv = smem_q4[ih][lane];
                float d = warp_sum(qv.x * bk.x + qv.y * bk.y + qv.z * bk.z + qv.w * bk.w);
                if (d > 0.0f) s_tot += d;
            }
            if (lane == 0) {
                block_scores_batch[static_cast<size_t>(b_tok) * mask_stride + blk] = s_tot * 0.08838834764831845f;
            }
        }
    }
    pdl_trigger();
}

// Long-context Batched QSA Top-512 Selector (`Grid (1, B)`, `Block 256`)
extern "C" __global__ void q38_qsa_select_top512_batch(
    float* scores_batch,                        // [B, mask_stride]
    const int32_t* pos0_ptr,
    int32_t mask_stride,
    int32_t* block_mask_batch) {                // [B, mask_stride]
    __shared__ float smem_val[8];
    __shared__ int smem_idx[8];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int b_tok = blockIdx.y;

    pdl_wait();
    const int n_blocks = (*pos0_ptr + b_tok + 1) >> 2;
    if (n_blocks > 512) {
        float* scores_scratch = scores_batch + static_cast<size_t>(b_tok) * mask_stride;
        int32_t* block_mask = block_mask_batch + static_cast<size_t>(b_tok) * mask_stride;
        for (int i = tid; i < n_blocks; i += 256) {
            block_mask[i] = 0;
        }
        __syncthreads();

        for (int sel = 0; sel < 512; ++sel) {
            float best_v = -1e30f;
            int best_i = 0;
            for (int i = tid; i < n_blocks; i += 256) {
                float v = scores_scratch[i];
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
                block_mask[fi] = 1;
                scores_scratch[fi] = -1e30f;
            }
            __syncthreads();
        }
    }
    pdl_trigger();
}

// Batched Split-KV QSA GQA (`Grid (24, B)`, `Block 256`)
extern "C" __global__ void q38_qsa_attn_batch(
    const float* q_rot_batch,                   // [B, 24, 256]
    const float* q_gate_batch,                  // [B, 24, 256]
    const uint4* k_cache_u128,                  // [max_ctx, 2, 32]
    const uint4* v_cache_u128,                  // [max_ctx, 2, 32]
    const int32_t* block_mask_batch,            // [B, mask_stride]
    const int32_t* pos0_ptr,
    int32_t mask_stride,
    float* attn_out_batch) {                    // [B, 24, 256]
    __shared__ float smem_m[8];
    __shared__ float smem_l[8];
    __shared__ float smem_o[8][256];

    const int h = blockIdx.x;
    const int b_tok = blockIdx.y;
    const int kvh = h / 12;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    pdl_wait();
    const int pos = *pos0_ptr + b_tok;
    const int n = pos + 1;
    const int complete_toks = (n >> 2) << 2;
    const int use_mask = ((n >> 2) > 512) ? 1 : 0;
    const int32_t* block_mask = block_mask_batch + static_cast<size_t>(b_tok) * mask_stride;

    const float* qh = q_rot_batch + static_cast<size_t>(b_tok) * 6144u + h * 256 + lane * 8;
    const float scale = 0.0625f;
    const float q0 = qh[0] * scale, q1 = qh[1] * scale, q2 = qh[2] * scale, q3 = qh[3] * scale;
    const float q4 = qh[4] * scale, q5 = qh[5] * scale, q6 = qh[6] * scale, q7 = qh[7] * scale;

    float m_w = -1e30f;
    float l_w = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    float o4 = 0.0f, o5 = 0.0f, o6 = 0.0f, o7 = 0.0f;

    for (int j = warp_id; j < n; j += 8) {
        if (use_mask && j < complete_toks && block_mask[j >> 2] == 0) {
            continue;
        }
        const size_t kv_off = (static_cast<size_t>(j) * 2u + kvh) * 32u + lane;
        uint4 kp = k_cache_u128[kv_off];
        float dot = q0 * fp16_bits_to_f32(static_cast<uint16_t>(kp.x & 0xFFFFu))
                  + q1 * fp16_bits_to_f32(static_cast<uint16_t>(kp.x >> 16))
                  + q2 * fp16_bits_to_f32(static_cast<uint16_t>(kp.y & 0xFFFFu))
                  + q3 * fp16_bits_to_f32(static_cast<uint16_t>(kp.y >> 16))
                  + q4 * fp16_bits_to_f32(static_cast<uint16_t>(kp.z & 0xFFFFu))
                  + q5 * fp16_bits_to_f32(static_cast<uint16_t>(kp.z >> 16))
                  + q6 * fp16_bits_to_f32(static_cast<uint16_t>(kp.w & 0xFFFFu))
                  + q7 * fp16_bits_to_f32(static_cast<uint16_t>(kp.w >> 16));
        dot = warp_sum(dot);

        float m_new = (dot > m_w) ? dot : m_w;
        float alpha = (l_w == 0.0f) ? 0.0f : ptx_exp(m_w - m_new);
        float p = ptx_exp(dot - m_new);
        l_w = l_w * alpha + p;
        m_w = m_new;

        uint4 vp = v_cache_u128[kv_off];
        o0 = o0 * alpha + p * fp16_bits_to_f32(static_cast<uint16_t>(vp.x & 0xFFFFu));
        o1 = o1 * alpha + p * fp16_bits_to_f32(static_cast<uint16_t>(vp.x >> 16));
        o2 = o2 * alpha + p * fp16_bits_to_f32(static_cast<uint16_t>(vp.y & 0xFFFFu));
        o3 = o3 * alpha + p * fp16_bits_to_f32(static_cast<uint16_t>(vp.y >> 16));
        o4 = o4 * alpha + p * fp16_bits_to_f32(static_cast<uint16_t>(vp.z & 0xFFFFu));
        o5 = o5 * alpha + p * fp16_bits_to_f32(static_cast<uint16_t>(vp.z >> 16));
        o6 = o6 * alpha + p * fp16_bits_to_f32(static_cast<uint16_t>(vp.w & 0xFFFFu));
        o7 = o7 * alpha + p * fp16_bits_to_f32(static_cast<uint16_t>(vp.w >> 16));
    }

    if (lane == 0) {
        smem_m[warp_id] = m_w;
        smem_l[warp_id] = l_w;
    }
    const int base_d = lane * 8;
    smem_o[warp_id][base_d + 0] = o0;
    smem_o[warp_id][base_d + 1] = o1;
    smem_o[warp_id][base_d + 2] = o2;
    smem_o[warp_id][base_d + 3] = o3;
    smem_o[warp_id][base_d + 4] = o4;
    smem_o[warp_id][base_d + 5] = o5;
    smem_o[warp_id][base_d + 6] = o6;
    smem_o[warp_id][base_d + 7] = o7;
    __syncthreads();

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
    float g = ptx_sigmoid(q_gate_batch[static_cast<size_t>(b_tok) * 6144u + h * 256 + tid]);
    attn_out_batch[static_cast<size_t>(b_tok) * 6144u + h * 256 + tid] = (sum_d * ptx_rcp(fmaxf(denom, 1e-20f))) * g;
    pdl_trigger();
}

// `0-ULP` Batched BF16 Router (`4` tokens/tile × `80` groups = `46,080` B SMEM!)
// Grid: `(64, ceil(b_count / 4))`, Block: `256`
extern "C" __global__ void q38_bf16_router_512_b4(
    const uint4* __restrict__ w_bf16_u128, // [512, 320]
    const float4* x4_batch,                // [b_count, 640]
    float* logits512_batch,                // [b_count, 512]
    int32_t b_count) {
    __shared__ float4 smem_x9[4 * 80 * 9]; // `46,080` B SMEM
    const int tid = threadIdx.x;
    const int b_base = blockIdx.y * 4;
    const int n_b = ((b_count - b_base) < 4) ? (b_count - b_base) : 4;

    pdl_wait();
    for (int t = 0; t < n_b; ++t) {
        const float4* src = x4_batch + static_cast<size_t>(b_base + t) * 640u;
        float4* dst = smem_x9 + t * 80 * 9;
        for (int i = tid; i < 640; i += 256) {
            dst[(i >> 3) * 9 + (i & 7)] = src[i];
        }
    }
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + warp_id;
    if (row < kExp) {
        const uint4* w_row = w_bf16_u128 + static_cast<size_t>(row) * 320u;
        float a0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float a1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int c8 = lane; c8 < 320; c8 += 32) {
            uint4 pk = w_row[c8];
            const float w0 = bf16_to_f32(static_cast<uint16_t>(pk.x & 0xFFFFu));
            const float w1 = bf16_to_f32(static_cast<uint16_t>(pk.x >> 16));
            const float w2 = bf16_to_f32(static_cast<uint16_t>(pk.y & 0xFFFFu));
            const float w3 = bf16_to_f32(static_cast<uint16_t>(pk.y >> 16));
            const float w4 = bf16_to_f32(static_cast<uint16_t>(pk.z & 0xFFFFu));
            const float w5 = bf16_to_f32(static_cast<uint16_t>(pk.z >> 16));
            const float w6 = bf16_to_f32(static_cast<uint16_t>(pk.w & 0xFFFFu));
            const float w7 = bf16_to_f32(static_cast<uint16_t>(pk.w >> 16));
            const int idx4 = c8 * 2;
            const int base9 = (idx4 >> 3) * 9 + (idx4 & 7);
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                if (t < n_b) {
                    float4 u = smem_x9[t * 80 * 9 + base9];
                    float4 v = smem_x9[t * 80 * 9 + base9 + 1];
                    a0[t] += w0 * u.x + w2 * u.z + w4 * v.x + w6 * v.z;
                    a1[t] += w1 * u.y + w3 * u.w + w5 * v.y + w7 * v.w;
                }
            }
        }
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            if (t < n_b) {
                float acc = warp_sum(a0[t] + a1[t]);
                if (lane == 0) {
                    logits512_batch[static_cast<size_t>(b_base + t) * kExp + row] = acc;
                }
            }
        }
    }
    pdl_trigger();
}

// Batched Softmax Top-10 (`Grid (1, B)`, `Block 256`)
extern "C" __global__ void q38_softmax_topk10_batch(
    const float* logits512_batch,          // [B, 512]
    int32_t* sel_ids10_batch,              // [B, 10]
    float* sel_wts11_batch) {              // [B, 11]
    __shared__ float smem_p[512];
    __shared__ float smem_red[8];
    __shared__ int smem_idx[8];
    __shared__ float smem_topw[10];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int b = blockIdx.y;

    pdl_wait();
    const float* logits512 = logits512_batch + static_cast<size_t>(b) * kExp;
    int32_t* sel_ids10 = sel_ids10_batch + static_cast<size_t>(b) * kTopK;
    float* sel_wts11 = sel_wts11_batch + static_cast<size_t>(b) * 11u;

    float v0 = logits512[tid * 2];
    float v1 = logits512[tid * 2 + 1];
    float m = fmaxf(v0, v1);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, mask));
    }
    if (lane == 0) smem_red[warp_id] = m;
    __syncthreads();
    if (warp_id == 0) {
        float bm = (lane < 8) ? smem_red[lane] : -1e30f;
        #pragma unroll
        for (int mask = 4; mask > 0; mask >>= 1) {
            bm = fmaxf(bm, __shfl_xor_sync(0xFFFFFFFFu, bm, mask));
        }
        if (lane == 0) smem_red[0] = bm;
    }
    __syncthreads();
    const float max_l = smem_red[0];

    float e0 = ptx_exp(v0 - max_l);
    float e1 = ptx_exp(v1 - max_l);
    float es = warp_sum(e0 + e1);
    if (lane == 0) smem_red[warp_id] = es;
    __syncthreads();
    if (warp_id == 0) {
        float tot = (lane < 8) ? smem_red[lane] : 0.0f;
        tot = warp_sum(tot);
        if (lane == 0) smem_red[0] = ptx_rcp(tot);
    }
    __syncthreads();
    const float inv_z = smem_red[0];
    smem_p[tid * 2] = e0 * inv_z;
    smem_p[tid * 2 + 1] = e1 * inv_z;
    __syncthreads();

    float sum_top = 0.0f;
    for (int k = 0; k < kTopK; ++k) {
        float p0 = smem_p[tid * 2];
        float p1 = smem_p[tid * 2 + 1];
        float bw = p0;
        int bi = tid * 2;
        if (p1 > bw) {
            bw = p1;
            bi = tid * 2 + 1;
        }
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            float ow = __shfl_xor_sync(0xFFFFFFFFu, bw, mask);
            int oi = __shfl_xor_sync(0xFFFFFFFFu, bi, mask);
            if (ow > bw || (ow == bw && oi < bi)) {
                bw = ow;
                bi = oi;
            }
        }
        if (lane == 0) {
            smem_red[warp_id] = bw;
            smem_idx[warp_id] = bi;
        }
        __syncthreads();
        if (tid == 0) {
            float fw = smem_red[0];
            int fi = smem_idx[0];
            for (int w = 1; w < 8; ++w) {
                if (smem_red[w] > fw || (smem_red[w] == fw && smem_idx[w] < fi)) {
                    fw = smem_red[w];
                    fi = smem_idx[w];
                }
            }
            sel_ids10[k] = fi;
            smem_topw[k] = fw;
            smem_p[fi] = -1.0f;
        }
        __syncthreads();
        sum_top += (tid == 0) ? smem_topw[k] : 0.0f;
    }
    if (tid == 0) {
        float inv_top = ptx_rcp(fmaxf(sum_top, 1e-20f));
        for (int k = 0; k < kTopK; ++k) {
            sel_wts11[k] = smem_topw[k] * inv_top;
        }
    }
    pdl_trigger();
}

// Grouped Active-Expert Planner for `1 <= b_count <= 64` (`Grid (1, 1)`, `Block 512`, `< 2 us`!)
// Groups `(b, k)` routed pairs by expert `ex ∈ 0..511` so each active expert's weights are read ONCE!
extern "C" __global__ void q38_moe_build_plan_b64(
    const int32_t* sel_ids10_batch,        // [b_count, 10]
    int32_t b_count,
    int32_t* expert_offsets,               // [513] (`expert_offsets[512] == b_count * 10`)
    int32_t* expert_pairs) {               // [640] (`(b << 8) | k`)
    __shared__ int32_t smem_sel[640];
    __shared__ int32_t smem_warp_tot[16];

    const int tid = threadIdx.x; // 0..511 (`1` thread per expert!)
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int total_pairs = b_count * kTopK;

    pdl_wait();
    for (int i = tid; i < total_pairs; i += 512) {
        smem_sel[i] = sel_ids10_batch[i];
    }
    __syncthreads();

    int cnt = 0;
    for (int i = 0; i < total_pairs; ++i) {
        if (smem_sel[i] == tid) cnt += 1;
    }

    // Exclusive prefix sum across 512 threads (`16` warps)
    int prefix = cnt;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        int n = __shfl_up_sync(0xFFFFFFFFu, prefix, off);
        if (lane >= off) prefix += n;
    }
    if (lane == 31) smem_warp_tot[warp_id] = prefix;
    __syncthreads();

    if (warp_id == 0) {
        int wt = (lane < 16) ? smem_warp_tot[lane] : 0;
        #pragma unroll
        for (int off = 1; off < 16; off <<= 1) {
            int n = __shfl_up_sync(0xFFFFFFFFu, wt, off);
            if (lane >= off) wt += n;
        }
        if (lane < 16) smem_warp_tot[lane] = wt;
    }
    __syncthreads();

    const int warp_base = (warp_id > 0) ? smem_warp_tot[warp_id - 1] : 0;
    const int excl_start = warp_base + prefix - cnt;
    expert_offsets[tid] = excl_start;
    if (tid == 511) {
        expert_offsets[512] = excl_start + cnt;
    }

    if (cnt > 0) {
        int wr = excl_start;
        for (int i = 0; i < total_pairs; ++i) {
            if (smem_sel[i] == tid) {
                int b = i / kTopK;
                int k = i - b * kTopK;
                expert_pairs[wr] = (b << 8) | k;
                wr += 1;
            }
        }
    }
    pdl_trigger();
}

// `0-ULP` Grouped Active-Expert Z24 `moe_up_swiglu` (`Grid (80, 513)`, `Block 512` = `16` warps/CTA = `8` rows/CTA, `46,336` B SMEM!)
// `blockIdx.y ∈ 0..511`: routed expert `ex` (exits in `< 50 ns` if `cnt == 0`!); `blockIdx.y == 512`: Shared Expert (`b = 0..b_count-1`).
extern "C" __global__ void q38_moe_up_grouped_z24(
    const uint4* __restrict__ sw_gate_w,
    const uint32_t* __restrict__ sw_gate_sb_lo,
    const uint32_t* __restrict__ sw_gate_sb_hi,
    const uint4* __restrict__ sw_up_w,
    const uint32_t* __restrict__ sw_up_sb_lo,
    const uint32_t* __restrict__ sw_up_sb_hi,
    uint32_t packed_bases_m1,
    const uint4* __restrict__ sh_gu_gate_w,
    const uint32_t* __restrict__ sh_gu_sc,
    const uint32_t* __restrict__ sh_gu_bi,
    const float4* x4_batch,                // [b_count, 640]
    const int32_t* expert_offsets,         // [513]
    const int32_t* expert_pairs,           // [b_count * 10]
    int32_t b_count,
    float* sel_wts11_batch,                // [b_count, 11]
    float* act11_batch) {                  // [b_count, 11, 640]
    __shared__ float4 smem_x9[4 * 80 * 9]; // `46,080` B SMEM
    __shared__ float smem_gu[4][16];

    const int ex = blockIdx.y;
    pdl_wait();
    const int start = (ex < kExp) ? expert_offsets[ex] : 0;
    const int end = (ex < kExp) ? expert_offsets[ex + 1] : b_count;
    if (start >= end) {
        pdl_trigger();
        return;
    }

    const int tid = threadIdx.x;           // 0..511
    const int warp_id = tid >> 5;          // 0..15
    const int lane = tid & 31;
    const int m = blockIdx.x * 8 + (warp_id >> 1);
    const int is_up = warp_id & 1;

    for (int base_i = start; base_i < end; base_i += 4) {
        const int n_sub = ((end - base_i) < 4) ? (end - base_i) : 4;
        int pairs_t[4] = {0, 0, 0, 0};
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            if (t < n_sub) {
                int pr = (ex < kExp) ? expert_pairs[base_i + t] : (((base_i + t) << 8) | 10);
                pairs_t[t] = pr;
                int b_t = pr >> 8;
                const float4* src_x = x4_batch + static_cast<size_t>(b_t) * 640u;
                stage_x_groups_smem9(src_x, smem_x9 + t * 80 * 9, 80, tid, 512);
            }
        }
        __syncthreads();

        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (ex < kExp) {
            const Z24ExpertProj p = resolve_z24_expert_proj(
                is_up ? sw_up_w : sw_gate_w,
                is_up ? sw_up_sb_lo : sw_gate_sb_lo,
                is_up ? sw_up_sb_hi : sw_gate_sb_hi,
                ex, kMoeFf, m);
            const uint4* w_ptr = p.w + p.row_offset * 80u;
            const uint32_t base_sc_m1 = is_up ? ((packed_bases_m1 >> 16) & 0xFFu) : (packed_bases_m1 & 0xFFu);
            const uint32_t base_bi_m1 = is_up ? ((packed_bases_m1 >> 24) & 0xFFu) : ((packed_bases_m1 >> 8) & 0xFFu);

            for (int g = lane; g < 80; g += 32) {
                uint4 pk = w_ptr[g];
                float sc, bi;
                load_z24_sb_pair(p.sb_lo, p.sb_hi, p.row_offset * 80u + static_cast<size_t>(g), base_sc_m1, base_bi_m1, &sc, &bi);
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    if (t < n_sub) {
                        acc[t] += dot_q4_group32_smem9(pk, sc, bi, smem_x9 + (t * 80 + g) * 9);
                    }
                }
            }
        } else {
            const size_t row = static_cast<size_t>(is_up * kMoeFf + m);
            for (int g = lane; g < 80; g += 32) {
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    if (t < n_sub) {
                        acc[t] += shared_group_dot(sh_gu_gate_w, sh_gu_sc, sh_gu_bi, row * 80u + g, smem_x9 + (t * 80 + g) * 9);
                    }
                }
            }
            if (blockIdx.x == 0 && warp_id == 0) {
                float sg_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int g = lane; g < 80; g += 32) {
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        if (t < n_sub) {
                            sg_acc[t] += shared_group_dot(sh_gu_gate_w, sh_gu_sc, sh_gu_bi, 1280u * 80u + g, smem_x9 + (t * 80 + g) * 9);
                        }
                    }
                }
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    if (t < n_sub) {
                        float sg = warp_sum(sg_acc[t]);
                        if (lane == 0) {
                            int b_t = pairs_t[t] >> 8;
                            sel_wts11_batch[static_cast<size_t>(b_t) * 11u + 10u] = ptx_sigmoid(sg);
                        }
                    }
                }
            }
        }

        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            if (t < n_sub) {
                float dot = warp_sum(acc[t]);
                if (lane == 0) smem_gu[t][warp_id] = dot;
            }
        }
        __syncthreads();

        if (lane == 0 && is_up == 0) {
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                if (t < n_sub) {
                    int b_t = pairs_t[t] >> 8;
                    int k_t = pairs_t[t] & 0xFF;
                    float g = smem_gu[t][warp_id];
                    float u = smem_gu[t][warp_id + 1];
                    act11_batch[(static_cast<size_t>(b_t) * 11u + k_t) * kMoeFf + m] = ptx_silu(g) * u;
                }
            }
        }
        __syncthreads();
    }
    pdl_trigger();
}

// `0-ULP` Grouped Active-Expert Z24 `moe_down` (`Grid (40, 513)`, `Block 256`, `8` tokens/tile = `23,040` B SMEM -> `2` CTAs/SM!)
// Writes exact weighted partial `dot * wt` into `down_part [b_count, 11, 2560]` (`7.03 MiB` in L2 cache!).
extern "C" __global__ void q38_moe_down_grouped_z24(
    const uint4* __restrict__ sw_down_w,
    const uint32_t* __restrict__ sw_down_sb_lo,
    const uint32_t* __restrict__ sw_down_sb_hi,
    uint32_t down_bases_m1,
    const uint4* __restrict__ sh_down_w,
    const uint32_t* __restrict__ sh_down_sc,
    const uint32_t* __restrict__ sh_down_bi,
    const float4* act11_4_batch,           // [b_count * 11, 160]
    const int32_t* expert_offsets,         // [513]
    const int32_t* expert_pairs,           // [b_count * 10]
    const float* sel_wts11_batch,          // [b_count, 11]
    int32_t b_count,
    float* down_part_out) {                // [b_count, 11, 2560]
    __shared__ float4 smem_a9[8 * 20 * 9]; // `23,040` B SMEM (`<= 31,744 B` -> `2` CTAs/SM under 64 KiB carveout!)

    const int ex = blockIdx.y;
    pdl_wait();
    const int start = (ex < kExp) ? expert_offsets[ex] : 0;
    const int end = (ex < kExp) ? expert_offsets[ex + 1] : b_count;
    if (start >= end) {
        pdl_trigger();
        return;
    }

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;          // 0..7
    const int lane = tid & 31;
    const int sub_row = lane >> 2;         // 0..7
    const int sub_lane = lane & 3;         // 0..3 (`4` lanes/row -> `5` iters = `20` groups, `0` idle lanes!)
    const int row = blockIdx.x * 64 + warp_id * 8 + sub_row;
    const uint32_t lane4_mask = 0xFu << (sub_row * 4);
    const uint32_t base_sc_m1 = down_bases_m1 & 0xFFu;
    const uint32_t base_bi_m1 = (down_bases_m1 >> 8) & 0xFFu;

    for (int base_i = start; base_i < end; base_i += 8) {
        const int n_sub = ((end - base_i) < 8) ? (end - base_i) : 8;
        int pairs_t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < n_sub) {
                int pr = (ex < kExp) ? expert_pairs[base_i + t] : (((base_i + t) << 8) | 10);
                pairs_t[t] = pr;
                int b_t = pr >> 8;
                int k_t = pr & 0xFF;
                const float4* src_a = act11_4_batch + (static_cast<size_t>(b_t) * 11u + k_t) * 160u;
                stage_x_groups_smem9(src_a, smem_a9 + t * 20 * 9, 20, tid);
            }
        }
        __syncthreads();

        if (row < kD) {
            float dot[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            if (ex < kExp) {
                const Z24ExpertProj p = resolve_z24_expert_proj(
                    sw_down_w, sw_down_sb_lo, sw_down_sb_hi, ex, kD, row);
                const uint4* rw = p.w + p.row_offset * 20u;
                const size_t g_base = p.row_offset * 20u;
                #pragma unroll
                for (int it = 0; it < 5; ++it) {
                    const int g = it * 4 + sub_lane;
                    uint4 pk = rw[g];
                    float sc, bi;
                    load_z24_sb_pair(p.sb_lo, p.sb_hi, g_base + static_cast<size_t>(g), base_sc_m1, base_bi_m1, &sc, &bi);
                    #pragma unroll
                    for (int t = 0; t < 8; ++t) {
                        if (t < n_sub) {
                            dot[t] += dot_q4_group32_smem9(pk, sc, bi, smem_a9 + (t * 20 + g) * 9);
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int it = 0; it < 5; ++it) {
                    const int g = it * 4 + sub_lane;
                    #pragma unroll
                    for (int t = 0; t < 8; ++t) {
                        if (t < n_sub) {
                            dot[t] += shared_group_dot(sh_down_w, sh_down_sc, sh_down_bi, row * 20u + g, smem_a9 + (t * 20 + g) * 9);
                        }
                    }
                }
            }

            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                if (t < n_sub) {
                    float d_val = dot[t];
                    d_val += __shfl_xor_sync(lane4_mask, d_val, 2);
                    d_val += __shfl_xor_sync(lane4_mask, d_val, 1);
                    if (sub_lane == 0) {
                        int b_t = pairs_t[t] >> 8;
                        int k_t = pairs_t[t] & 0xFF;
                        size_t slot = static_cast<size_t>(b_t) * 11u + k_t;
                        down_part_out[slot * kD + row] = d_val * sel_wts11_batch[slot];
                    }
                }
            }
        }
        __syncthreads();
    }
    pdl_trigger();
}

// `0-ULP` Exact `(0..4) + (5..10)` MoE Down Combine (`Grid (10, B)`, `Block 256`)
extern "C" __global__ void q38_moe_combine11_batch(
    const float* down_part,                // [B, 11, 2560]
    float* moe_out_batch) {                // [B, 2560]
    const int d = blockIdx.x * 256 + threadIdx.x;
    const int b = blockIdx.y;
    pdl_wait();
    if (d < kD) {
        const float* p = down_part + static_cast<size_t>(b) * (11u * kD) + d;
        float even_sum = ((((0.0f + p[0 * kD]) + p[1 * kD]) + p[2 * kD]) + p[3 * kD]) + p[4 * kD];
        float odd_sum  = (((((0.0f + p[5 * kD]) + p[6 * kD]) + p[7 * kD]) + p[8 * kD]) + p[9 * kD]) + p[10 * kD];
        moe_out_batch[static_cast<size_t>(b) * kD + d] = even_sum + odd_sum;
    }
    pdl_trigger();
}

// Batched `1024`-thread `float4` ArgMax (`Grid (B, 1)`, `Block 1024`) + `*pos_ptr += b_count`
extern "C" __global__ void q38_argmax_fast_batch(
    const float4* logits4_batch,           // [B, 62080]
    int32_t* out_toks,                     // [B]
    int32_t* pos_ptr,
    int32_t b_count) {
    __shared__ float smem_val[32];
    __shared__ int smem_idx[32];

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    pdl_wait();
    const float4* logits4 = logits4_batch + static_cast<size_t>(b) * (kVocab / 4);
    float best_v = -1e30f;
    int best_i = 0;
    for (int i4 = tid; i4 < (kVocab / 4); i4 += 1024) {
        float4 v4 = logits4[i4];
        const int base = i4 * 4;
        if (v4.x > best_v) { best_v = v4.x; best_i = base; }
        if (v4.y > best_v) { best_v = v4.y; best_i = base + 1; }
        if (v4.z > best_v) { best_v = v4.z; best_i = base + 2; }
        if (v4.w > best_v) { best_v = v4.w; best_i = base + 3; }
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
    if (warp_id == 0) {
        float fv = smem_val[lane];
        int fi = smem_idx[lane];
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            float ov = __shfl_xor_sync(0xFFFFFFFFu, fv, mask);
            int oi = __shfl_xor_sync(0xFFFFFFFFu, fi, mask);
            if (ov > fv || (ov == fv && oi < fi)) {
                fv = ov;
                fi = oi;
            }
        }
        if (lane == 0) {
            out_toks[b] = fi;
            if (b == 0) {
                *pos_ptr += b_count;
            }
        }
    }
    pdl_trigger();
}

extern "C" __global__ void q38_add_pos(int32_t* pos_ptr, int32_t delta) {
    pdl_wait();
    if (threadIdx.x == 0) {
        *pos_ptr += delta;
    }
    pdl_trigger();
}

// ============================================================================
// 10. `v1.0.0` Low-Latency Multi-Row Speculative Verify (`1 <= B <= 8`) &
//     `O(1)` Per-Row Recurrent Checkpointing (`#248` Track 3)
// ============================================================================

template <int MAX_T>
__device__ __forceinline__ void warp_dot_q4_row_smem9_multi(
    const uint4* __restrict__ row_w,
    const uint32_t* __restrict__ row_sc_u32,
    const uint32_t* __restrict__ row_bi_u32,
    const float4* smem_x9_all,
    int n_b,
    int lane,
    float* out_acc) {
    #pragma unroll
    for (int t = 0; t < MAX_T; ++t) {
        out_acc[t] = 0.0f;
    }
    for (int g = lane; g < 80; g += 32) {
        const uint4 pk = row_w[g];
        const float sc = load_bf16_from_u32_array(row_sc_u32, g);
        const float bi = load_bf16_from_u32_array(row_bi_u32, g);
        float a0[MAX_T], a1[MAX_T], a2[MAX_T], a3[MAX_T];
        #pragma unroll
        for (int t = 0; t < MAX_T; ++t) {
            a0[t] = 0.0f; a1[t] = 0.0f; a2[t] = 0.0f; a3[t] = 0.0f;
        }
        #pragma unroll
        for (int w_idx = 0; w_idx < 4; ++w_idx) {
            const uint32_t w = (w_idx == 0) ? pk.x : ((w_idx == 1) ? pk.y : ((w_idx == 2) ? pk.z : pk.w));
            const float q0 = static_cast<float>(w & 0xFu);
            const float q1 = static_cast<float>((w >> 4) & 0xFu);
            const float q2 = static_cast<float>((w >> 8) & 0xFu);
            const float q3 = static_cast<float>((w >> 12) & 0xFu);
            const float q4 = static_cast<float>((w >> 16) & 0xFu);
            const float q5 = static_cast<float>((w >> 20) & 0xFu);
            const float q6 = static_cast<float>((w >> 24) & 0xFu);
            const float q7 = static_cast<float>((w >> 28) & 0xFu);
            #pragma unroll
            for (int t = 0; t < MAX_T; ++t) {
                if (t < n_b) {
                    const float4* grp9 = smem_x9_all + (t * 80 + g) * 9;
                    const float4 x0 = grp9[w_idx * 2];
                    const float4 x1 = grp9[w_idx * 2 + 1];
                    a0[t] = __fmaf_rn(q0, x0.x, a0[t]);
                    a1[t] = __fmaf_rn(q1, x0.y, a1[t]);
                    a2[t] = __fmaf_rn(q2, x0.z, a2[t]);
                    a3[t] = __fmaf_rn(q3, x0.w, a3[t]);
                    a0[t] = __fmaf_rn(q4, x1.x, a0[t]);
                    a1[t] = __fmaf_rn(q5, x1.y, a1[t]);
                    a2[t] = __fmaf_rn(q6, x1.z, a2[t]);
                    a3[t] = __fmaf_rn(q7, x1.w, a3[t]);
                }
            }
        }
        #pragma unroll
        for (int t = 0; t < MAX_T; ++t) {
            if (t < n_b) {
                const float dot_nx = (a0[t] + a1[t]) + (a2[t] + a3[t]);
                out_acc[t] += __fmaf_rn(dot_nx, sc, bi * smem_x9_all[(t * 80 + g) * 9 + 8].x);
            }
        }
    }
}

// Zero-loop `2`-CTAs/SM `cols == 2560` Q4 GEMV for `1 <= b_count <= 2` (`23,040 B <= 31,744 B` SMEM -> `2` CTAs/SM = `32` warps/SM!)
// Grid: `(ceil(rows / 32), 1)`, Block: `512` (`2` rows/warp = `32` rows/CTA)
extern "C" __global__ void q38_q4_gemv_2560_b2(
    const uint4* __restrict__ w_u128,      // [rows, 80]
    const uint32_t* __restrict__ sc_u32,   // [rows, 40]
    const uint32_t* __restrict__ bi_u32,   // [rows, 40]
    const float4* x4_batch,                // [b_count, 640]
    float* y_batch,                        // [b_count, rows]
    int32_t rows,
    int32_t b_count,
    int32_t post_mode) {
    __shared__ float4 smem_x9[2 * 80 * 9]; // `23,040` B <= `31,744` B (`2` CTAs/SM under `64 KiB` carveout!)

    const int tid = threadIdx.x;
    pdl_wait();
    for (int t = 0; t < b_count; ++t) {
        stage_x_groups_smem9(x4_batch + static_cast<size_t>(t) * 640u, smem_x9 + t * 80 * 9, 80, tid, 512);
    }
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row0 = blockIdx.x * 32 + warp_id * 2;
    const int row1 = row0 + 1;

    if (row0 < rows) {
        float acc0[2];
        warp_dot_q4_row_smem9_multi<2>(
            w_u128 + static_cast<size_t>(row0) * 80u,
            sc_u32 + static_cast<size_t>(row0) * 40u,
            bi_u32 + static_cast<size_t>(row0) * 40u,
            smem_x9, b_count, lane, acc0);
        #pragma unroll
        for (int t = 0; t < 2; ++t) {
            if (t < b_count) {
                float v = warp_sum(acc0[t]);
                if (lane == 0) {
                    if (post_mode == 1) v = ptx_silu(v * 0.25f);
                    else if (post_mode == 2) v = ptx_sigmoid(v);
                    y_batch[static_cast<size_t>(t) * rows + row0] = v;
                }
            }
        }
    }
    if (row1 < rows) {
        float acc1[2];
        warp_dot_q4_row_smem9_multi<2>(
            w_u128 + static_cast<size_t>(row1) * 80u,
            sc_u32 + static_cast<size_t>(row1) * 40u,
            bi_u32 + static_cast<size_t>(row1) * 40u,
            smem_x9, b_count, lane, acc1);
        #pragma unroll
        for (int t = 0; t < 2; ++t) {
            if (t < b_count) {
                float v = warp_sum(acc1[t]);
                if (lane == 0) {
                    if (post_mode == 1) v = ptx_silu(v * 0.25f);
                    else if (post_mode == 2) v = ptx_sigmoid(v);
                    y_batch[static_cast<size_t>(t) * rows + row1] = v;
                }
            }
        }
    }
    pdl_trigger();
}

// Zero-loop `512`-thread `cols == 2560` Q4 GEMV for `1 <= b_count <= 4` (`46,080` B SMEM, `16` warps/CTA!)
// Grid: `(ceil(rows / 32), 1)`, Block: `512` (`2` rows/warp = `32` rows/CTA)
extern "C" __global__ void q38_q4_gemv_2560_b4(
    const uint4* __restrict__ w_u128,      // [rows, 80]
    const uint32_t* __restrict__ sc_u32,   // [rows, 40]
    const uint32_t* __restrict__ bi_u32,   // [rows, 40]
    const float4* x4_batch,                // [b_count, 640]
    float* y_batch,                        // [b_count, rows]
    int32_t rows,
    int32_t b_count,
    int32_t post_mode) {
    __shared__ float4 smem_x9[4 * 80 * 9]; // `46,080` B SMEM

    const int tid = threadIdx.x;
    pdl_wait();
    for (int t = 0; t < b_count; ++t) {
        stage_x_groups_smem9(x4_batch + static_cast<size_t>(t) * 640u, smem_x9 + t * 80 * 9, 80, tid, 512);
    }
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row0 = blockIdx.x * 32 + warp_id * 2;
    const int row1 = row0 + 1;

    if (row0 < rows) {
        float acc0[4];
        warp_dot_q4_row_smem9_multi<4>(
            w_u128 + static_cast<size_t>(row0) * 80u,
            sc_u32 + static_cast<size_t>(row0) * 40u,
            bi_u32 + static_cast<size_t>(row0) * 40u,
            smem_x9, b_count, lane, acc0);
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            if (t < b_count) {
                float v = warp_sum(acc0[t]);
                if (lane == 0) {
                    if (post_mode == 1) v = ptx_silu(v * 0.25f);
                    else if (post_mode == 2) v = ptx_sigmoid(v);
                    y_batch[static_cast<size_t>(t) * rows + row0] = v;
                }
            }
        }
    }
    if (row1 < rows) {
        float acc1[4];
        warp_dot_q4_row_smem9_multi<4>(
            w_u128 + static_cast<size_t>(row1) * 80u,
            sc_u32 + static_cast<size_t>(row1) * 40u,
            bi_u32 + static_cast<size_t>(row1) * 40u,
            smem_x9, b_count, lane, acc1);
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            if (t < b_count) {
                float v = warp_sum(acc1[t]);
                if (lane == 0) {
                    if (post_mode == 1) v = ptx_silu(v * 0.25f);
                    else if (post_mode == 2) v = ptx_sigmoid(v);
                    y_batch[static_cast<size_t>(t) * rows + row1] = v;
                }
            }
        }
    }
    pdl_trigger();
}

// `0-ULP` `5`-CTAs/SM Small-Batch Verify `moe_up_swiglu11_z24_bn` (`Grid (160, 11 * b_count)`, `Block 256`!)
// Eliminates all `41,040`-CTA empty-expert & `build_plan` fixed overhead for `1 <= b_count <= 8`!
extern "C" __global__ void q38_moe_up_swiglu11_z24_bn(
    const uint4* __restrict__ sw_gate_w,
    const uint32_t* __restrict__ sw_gate_sb_lo,
    const uint32_t* __restrict__ sw_gate_sb_hi,
    const uint4* __restrict__ sw_up_w,
    const uint32_t* __restrict__ sw_up_sb_lo,
    const uint32_t* __restrict__ sw_up_sb_hi,
    uint32_t packed_bases_m1,
    const uint4* __restrict__ sh_gu_gate_w,
    const uint32_t* __restrict__ sh_gu_sc,
    const uint32_t* __restrict__ sh_gu_bi,
    const float4* x4_batch,                // [b_count, 640]
    const int32_t* sel_ids10_batch,        // [b_count, 10]
    float* sel_wts11_batch,                // [b_count, 11]
    float* act11_batch) {                  // [b_count, 11, 640]
    __shared__ float4 smem_x9[80 * 9];     // `11,520` B SMEM -> `5` CTAs/SM = `40` warps/SM!
    __shared__ float smem_gu[8];

    const int tid = threadIdx.x;
    const int b = blockIdx.y / 11;
    const int k = blockIdx.y - b * 11;

    pdl_wait();
    stage_x_groups_smem9(x4_batch + static_cast<size_t>(b) * 640u, smem_x9, 80, tid);
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int m = blockIdx.x * 4 + (warp_id >> 1);
    const int is_up = warp_id & 1;

    if (k < kTopK) {
        const int ex = sel_ids10_batch[static_cast<size_t>(b) * kTopK + k];
        const Z24ExpertProj p = resolve_z24_expert_proj(
            is_up ? sw_up_w : sw_gate_w,
            is_up ? sw_up_sb_lo : sw_gate_sb_lo,
            is_up ? sw_up_sb_hi : sw_gate_sb_hi,
            ex, kMoeFf, m);
        const uint32_t base_sc_m1 = is_up ? ((packed_bases_m1 >> 16) & 0xFFu) : (packed_bases_m1 & 0xFFu);
        const uint32_t base_bi_m1 = is_up ? ((packed_bases_m1 >> 24) & 0xFFu) : ((packed_bases_m1 >> 8) & 0xFFu);
        float dot = warp_dot_q4_row_z24_smem9(
            p.w + p.row_offset * 80u, p.sb_lo, p.sb_hi, p.row_offset * 80u, base_sc_m1, base_bi_m1, smem_x9, 80, lane);
        if (lane == 0) smem_gu[warp_id] = dot;
    } else {
        const size_t row = static_cast<size_t>(is_up * kMoeFf + m);
        float dot = shared_warp_dot(sh_gu_gate_w, sh_gu_sc, sh_gu_bi, row, 80, smem_x9, lane);
        if (lane == 0) smem_gu[warp_id] = dot;

        if (blockIdx.x == 0 && warp_id == 0) {
            float sg = shared_warp_dot(sh_gu_gate_w, sh_gu_sc, sh_gu_bi, 1280u, 80, smem_x9, lane);
            if (lane == 0) sel_wts11_batch[static_cast<size_t>(b) * 11u + 10u] = ptx_sigmoid(sg);
        }
    }
    __syncthreads();

    if (lane == 0 && is_up == 0) {
        float g = smem_gu[warp_id];
        float u = smem_gu[warp_id + 1];
        act11_batch[(static_cast<size_t>(b) * 11u + k) * kMoeFf + m] = ptx_silu(g) * u;
    }
    pdl_trigger();
}

// `0-ULP` `2`-CTAs/SM Small-Batch Verify `moe_down_combine11_z24_bn` (`Grid (80, b_count)`, `Block 256`!)
extern "C" __global__ void q38_moe_down_combine11_z24_bn(
    const uint4* __restrict__ sw_down_w,
    const uint32_t* __restrict__ sw_down_sb_lo,
    const uint32_t* __restrict__ sw_down_sb_hi,
    uint32_t down_bases_m1,
    const uint4* __restrict__ sh_down_w,
    const uint32_t* __restrict__ sh_down_sc,
    const uint32_t* __restrict__ sh_down_bi,
    const float4* act11_x4_batch,          // [b_count, 11, 160]
    const int32_t* sel_ids10_batch,        // [b_count, 10]
    const float* sel_wts11_batch,          // [b_count, 11]
    float* moe_out_batch) {                // [b_count, 2560]
    __shared__ float4 smem_a9[11 * 20 * 9]; // `31,680` B <= `31,744` B (`2` CTAs/SM under `64 KiB` carveout!)

    const int tid = threadIdx.x;
    const int b = blockIdx.y;
    const int32_t* sel_ids10 = sel_ids10_batch + static_cast<size_t>(b) * kTopK;
    const float* sel_wts11 = sel_wts11_batch + static_cast<size_t>(b) * 11u;

    pdl_wait();
    stage_x_groups_smem9(act11_x4_batch + static_cast<size_t>(b) * (11u * 160u), smem_a9, 11 * 20, tid);
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int sub_row = lane >> 2;
    const int sub_lane = lane & 3;
    const int pair_id = warp_id >> 1;
    const int is_odd = warp_id & 1;
    const int row = blockIdx.x * 32 + pair_id * 8 + sub_row;

    const uint32_t base_sc_m1 = down_bases_m1 & 0xFFu;
    const uint32_t base_bi_m1 = (down_bases_m1 >> 8) & 0xFFu;

    float sum_y = 0.0f;
    if (row < kD) {
        const uint32_t lane4_mask = 0xFu << (sub_row * 4);
        const int k_start = is_odd ? 5 : 0;
        const int k_end = is_odd ? 10 : 5;
        for (int k = k_start; k < k_end; ++k) {
            const int ex = sel_ids10[k];
            const float wt = sel_wts11[k];
            const Z24ExpertProj p = resolve_z24_expert_proj(
                sw_down_w, sw_down_sb_lo, sw_down_sb_hi, ex, kD, row);
            const uint4* rw = p.w + p.row_offset * 20u;
            const size_t g_base = p.row_offset * 20u;
            const float4* a_k = smem_a9 + k * 20 * 9;

            float dot = 0.0f;
            #pragma unroll
            for (int it = 0; it < 5; ++it) {
                const int g = it * 4 + sub_lane;
                uint4 pk = rw[g];
                float sc, bi;
                load_z24_sb_pair(p.sb_lo, p.sb_hi, g_base + static_cast<size_t>(g), base_sc_m1, base_bi_m1, &sc, &bi);
                dot += dot_q4_group32_smem9(pk, sc, bi, a_k + g * 9);
            }
            dot += __shfl_xor_sync(lane4_mask, dot, 2);
            dot += __shfl_xor_sync(lane4_mask, dot, 1);
            sum_y += dot * wt;
        }
        if (is_odd) {
            const float4* a_sh = smem_a9 + 10 * 20 * 9;
            float sh_dot = 0.0f;
            #pragma unroll
            for (int it = 0; it < 5; ++it) {
                const int g = it * 4 + sub_lane;
                sh_dot += shared_group_dot(sh_down_w, sh_down_sc, sh_down_bi, row * 20u + g, a_sh + g * 9);
            }
            sh_dot += __shfl_xor_sync(lane4_mask, sh_dot, 2);
            sh_dot += __shfl_xor_sync(lane4_mask, sh_dot, 1);
            sum_y += sh_dot * sel_wts11[10];
        }
    }
    __syncthreads();

    float* smem_part = reinterpret_cast<float*>(smem_a9);
    if (sub_lane == 0 && row < kD) {
        smem_part[warp_id * 8 + sub_row] = sum_y;
    }
    __syncthreads();

    if (is_odd == 0 && sub_lane == 0 && row < kD) {
        moe_out_batch[static_cast<size_t>(b) * kD + row] =
            smem_part[warp_id * 8 + sub_row] + smem_part[(warp_id + 1) * 8 + sub_row];
    }
    pdl_trigger();
}

// Layer 1 causal PLE inject with `O(1)` per-row `ple_hist9` checkpoint (`Grid (4, 1)`, `Block 256`)
extern "C" __global__ void q38_ple_inject_ckpt_b8(
    float* hc_batch,                          // [b_count, 10240]
    const float* kv_batch,                    // [b_count, 12800]
    const float* __restrict__ norm_key_w,     // [4, 2560]
    const float* __restrict__ norm_query_w,   // [4, 2560]
    const float* __restrict__ norm_conv_w,    // [4, 2560]
    const float4* __restrict__ conv_w4,       // [4, 2560]
    float* ple_hist9,                         // [9, 10240]
    float* ple_ckpt_out,                      // [(b_count - 1), 9, 10240] or nullptr
    const int32_t* pos0_ptr,
    int32_t b_count) {
    __shared__ float smem_a[8];
    __shared__ float smem_b[8];
    __shared__ float smem_c[8];
    __shared__ float smem_scalars[4];

    const int s = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const float* nk_s = norm_key_w + s * kD;
    const float* nq_s = norm_query_w + s * kD;
    const float* nc_s = norm_conv_w + s * kD;
    const float4* cw_s = conv_w4 + s * kD;

    pdl_wait();
    const int pos0 = *pos0_ptr;

    for (int b = 0; b < b_count; ++b) {
        const int pos = pos0 + b;
        float* hc_s = hc_batch + static_cast<size_t>(b) * kHcD + s * kD;
        const float* kv_b = kv_batch + static_cast<size_t>(b) * (kHcD + kD);
        const float* kr_s = kv_b + s * kD;
        const float* val_raw = kv_b + kHcD;

        float sq_k = 0.0f, sq_q = 0.0f, sq_v = 0.0f;
        for (int d = tid; d < kD; d += 256) {
            float kv = kr_s[d];
            float qv = hc_s[d];
            float vv = val_raw[d];
            sq_k += kv * kv;
            sq_q += qv * qv;
            sq_v += vv * vv;
        }
        sq_k = warp_sum(sq_k);
        sq_q = warp_sum(sq_q);
        sq_v = warp_sum(sq_v);
        if (lane == 0) {
            smem_a[warp_id] = sq_k;
            smem_b[warp_id] = sq_q;
            smem_c[warp_id] = sq_v;
        }
        __syncthreads();
        if (warp_id == 0) {
            float tk = (lane < 8) ? smem_a[lane] : 0.0f;
            float tq = (lane < 8) ? smem_b[lane] : 0.0f;
            float tv = (lane < 8) ? smem_c[lane] : 0.0f;
            tk = warp_sum(tk);
            tq = warp_sum(tq);
            tv = warp_sum(tv);
            if (lane == 0) {
                smem_scalars[0] = ptx_rsqrt(tk * (1.0f / 2560.0f) + 1e-6f);
                smem_scalars[1] = ptx_rsqrt(tq * (1.0f / 2560.0f) + 1e-6f);
                smem_scalars[3] = tv * (1.0f / 2560.0f);
            }
        }
        __syncthreads();

        const float inv_k = smem_scalars[0];
        const float inv_q = smem_scalars[1];
        float dot = 0.0f;
        for (int d = tid; d < kD; d += 256) {
            dot += (kr_s[d] * inv_k * nk_s[d]) * (hc_s[d] * inv_q * nq_s[d]);
        }
        dot = warp_sum(dot);
        if (lane == 0) smem_a[warp_id] = dot;
        __syncthreads();
        if (warp_id == 0) {
            float td = (lane < 8) ? smem_a[lane] : 0.0f;
            td = warp_sum(td);
            if (lane == 0) {
                float g = td * 0.01976423537605237f;
                float g_abs = fabsf(g);
                float g_clamp = (g_abs < 1e-6f) ? 1e-6f : g_abs;
                float signed_sqrt = (g > 0.0f) ? ptx_sqrt(g_clamp) : ((g < 0.0f) ? -ptx_sqrt(g_clamp) : 0.0f);
                float sig = ptx_sigmoid(signed_sqrt);
                smem_scalars[2] = sig;
                smem_scalars[3] = ptx_rsqrt(sig * sig * smem_scalars[3] + 1e-6f);
            }
        }
        __syncthreads();

        const float sig_g = smem_scalars[2];
        const float inv_gv = smem_scalars[3];
        const int s_m9 = pos % 9;
        const int s_m6 = (pos + 3) % 9;
        const int s_m3 = (pos + 6) % 9;

        for (int d = tid; d < kD; d += 256) {
            const int ch = s * kD + d;
            float gv = sig_g * val_raw[d];
            float gvn = gv * inv_gv * nc_s[d];

            float x_m9 = ple_hist9[static_cast<size_t>(s_m9) * kHcD + ch];
            float x_m6 = ple_hist9[static_cast<size_t>(s_m6) * kHcD + ch];
            float x_m3 = ple_hist9[static_cast<size_t>(s_m3) * kHcD + ch];
            ple_hist9[static_cast<size_t>(s_m9) * kHcD + ch] = gvn;

            float4 cw = cw_s[d];
            float c_out = ptx_silu(cw.x * x_m9 + cw.y * x_m6 + cw.z * x_m3 + cw.w * gvn);
            hc_s[d] += gv + c_out;

            if (ple_ckpt_out != nullptr && (b + 1) < b_count) {
                float* dst_ckpt = ple_ckpt_out + static_cast<size_t>(b) * (9u * kHcD);
                #pragma unroll
                for (int sl = 0; sl < 9; ++sl) {
                    dst_ckpt[static_cast<size_t>(sl) * kHcD + ch] = ple_hist9[static_cast<size_t>(sl) * kHcD + ch];
                }
            }
        }
        __syncthreads();
    }
    pdl_trigger();
}

// Causal Conv4 + SiLU with `O(1)` per-row `conv_hist3` checkpoint (`Grid (40, 1)`, `Block 256`)
extern "C" __global__ void q38_gdn_conv_silu_ckpt_b8(
    const float* gdn_in_batch,            // [b_count, 16480]
    const float4* __restrict__ conv_w4,   // [10240]
    float* conv_hist3,                    // [10240 * 3]
    float* conv_ckpt_out,                 // [(b_count - 1), 10240 * 3] or nullptr
    float* qkv_conv_batch,                // [b_count, 10240]
    int32_t b_count) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    float4 w = (c < 10240) ? conv_w4[c] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    pdl_wait();
    if (c < 10240) {
        float* h = conv_hist3 + c * 3;
        float h0 = h[0], h1 = h[1], h2 = h[2];
        for (int b = 0; b < b_count; ++b) {
            float x = gdn_in_batch[static_cast<size_t>(b) * 16480u + c];
            float y = ptx_silu(w.x * h0 + w.y * h1 + w.z * h2 + w.w * x);
            qkv_conv_batch[static_cast<size_t>(b) * 10240u + c] = y;
            h0 = h1;
            h1 = h2;
            h2 = x;
            if (conv_ckpt_out != nullptr && (b + 1) < b_count) {
                float* ck = conv_ckpt_out + static_cast<size_t>(b) * 30720u + c * 3;
                ck[0] = h0;
                ck[1] = h1;
                ck[2] = h2;
            }
        }
        h[0] = h0;
        h[1] = h1;
        h[2] = h2;
    }
    pdl_trigger();
}

// `0-ULP` On-Chip `smem_S + s_hi` Causal GDN Recurrent Step with `O(1)` per-row `state_f32` checkpoint (`Grid (48, 1)`, `Block 128`)
// Keeps `[128, 128]` FP32 recurrent state on-chip across all `b = 0..b_count-1` (`1` DRAM load + `1` final store instead of `b_count` R+W sweeps!).
extern "C" __global__ void q38_gdn_recurrent_step_ckpt_b8(
    const float* qkv_conv_batch,          // [b_count, 10240]
    const float* gdn_in_batch,            // [b_count, 16480]
    const float* __restrict__ a_log,      // [48]
    const float* __restrict__ dt_bias,    // [48]
    const float* __restrict__ norm_w,     // [128]
    int32_t use_sigmoid_gate,
    int32_t b_count,
    float* state_f32,                     // [48 * 128 * 128]
    float* state_ckpt_out,                // [(b_count - 1), 48 * 128 * 128] or nullptr
    float* gdn_out_batch) {               // [b_count, 6144]
    __shared__ float smem_S[64][129];     // `33,024` B SMEM (`0` bank conflicts: stride `129` is odd!)
    __shared__ float smem_q[128];
    __shared__ float smem_k[128];
    __shared__ float smem_red[4];
    __shared__ float smem_gate[3];

    const int vh = blockIdx.x;
    const int kh = vh / 3;
    const int dv = threadIdx.x;
    const int warp_id = dv >> 5;
    const int lane = dv & 31;
    const float nw_dv = norm_w[dv];
    const float al_vh = (dv == 0) ? a_log[vh] : 0.0f;
    const float dt_vh = (dv == 0) ? dt_bias[vh] : 0.0f;
    float* S_vh = state_f32 + static_cast<size_t>(vh) * 16384u;

    pdl_wait();
    #pragma unroll
    for (int dk = 0; dk < 64; ++dk) {
        smem_S[dk][dv] = S_vh[dk * 128 + dv];
    }
    float s_hi[64];
    #pragma unroll
    for (int dk = 0; dk < 64; ++dk) {
        s_hi[dk] = S_vh[(64 + dk) * 128 + dv];
    }
    __syncthreads();

    for (int b = 0; b < b_count; ++b) {
        const float* qkv_conv = qkv_conv_batch + static_cast<size_t>(b) * 10240u;
        const float* zba_buf = gdn_in_batch + static_cast<size_t>(b) * 16480u + 10240u;

        float qv = qkv_conv[kh * 128 + dv];
        float kv = qkv_conv[2048 + kh * 128 + dv];
        float q_sq = warp_sum(qv * qv);
        float k_sq = warp_sum(kv * kv);
        if (lane == 0) smem_red[warp_id] = q_sq;
        __syncthreads();
        float q_tot = smem_red[0] + smem_red[1] + smem_red[2] + smem_red[3];
        float q_inv = ptx_rsqrt(q_tot + 1e-6f) * 0.08838834764831845f;
        __syncthreads();
        if (lane == 0) smem_red[warp_id] = k_sq;
        __syncthreads();
        float k_tot = smem_red[0] + smem_red[1] + smem_red[2] + smem_red[3];
        float k_inv = ptx_rsqrt(k_tot + 1e-6f);

        smem_q[dv] = qv * q_inv;
        smem_k[dv] = kv * k_inv;
        if (dv == 0) {
            float b_raw = zba_buf[6144 + vh];
            float a_raw = zba_buf[6192 + vh];
            float g = -ptx_exp(al_vh) * ptx_softplus(a_raw + dt_vh);
            smem_gate[0] = ptx_exp(g);
            smem_gate[1] = ptx_sigmoid(b_raw);
        }
        __syncthreads();

        const float decay = smem_gate[0];
        const float beta = smem_gate[1];

        float kv_mem = 0.0f;
        #pragma unroll 8
        for (int dk = 0; dk < 64; ++dk) {
            kv_mem += (smem_S[dk][dv] * decay) * smem_k[dk];
        }
        #pragma unroll
        for (int dk = 0; dk < 64; ++dk) {
            kv_mem += (s_hi[dk] * decay) * smem_k[64 + dk];
        }
        const float v_val = qkv_conv[4096 + vh * 128 + dv];
        const float delta = (v_val - kv_mem) * beta;

        float* ck_vh = (state_ckpt_out != nullptr && (b + 1) < b_count)
            ? (state_ckpt_out + static_cast<size_t>(b) * 786432u + static_cast<size_t>(vh) * 16384u)
            : nullptr;

        float out_v = 0.0f;
        #pragma unroll 8
        for (int dk = 0; dk < 64; ++dk) {
            float s_new = smem_S[dk][dv] * decay + smem_k[dk] * delta;
            smem_S[dk][dv] = s_new;
            if (ck_vh != nullptr) {
                ck_vh[dk * 128 + dv] = s_new;
            }
            out_v += s_new * smem_q[dk];
        }
        #pragma unroll
        for (int dk = 0; dk < 64; ++dk) {
            float s_new = s_hi[dk] * decay + smem_k[64 + dk] * delta;
            s_hi[dk] = s_new;
            if (ck_vh != nullptr) {
                ck_vh[(64 + dk) * 128 + dv] = s_new;
            }
            out_v += s_new * smem_q[64 + dk];
        }

        float o_sq = warp_sum(out_v * out_v);
        if (lane == 0) smem_red[warp_id] = o_sq;
        __syncthreads();
        if (dv == 0) {
            float o_tot = smem_red[0] + smem_red[1] + smem_red[2] + smem_red[3];
            smem_gate[2] = ptx_rsqrt(o_tot * (1.0f / 128.0f) + 1e-6f);
        }
        __syncthreads();

        float z_val = zba_buf[vh * 128 + dv];
        float gate_act = use_sigmoid_gate ? ptx_sigmoid(z_val) : ptx_silu(z_val);
        gdn_out_batch[static_cast<size_t>(b) * 6144u + vh * 128 + dv] = out_v * smem_gate[2] * nw_dv * gate_act;
        __syncthreads();
    }

    #pragma unroll
    for (int dk = 0; dk < 64; ++dk) {
        S_vh[dk * 128 + dv] = smem_S[dk][dv];
    }
    #pragma unroll
    for (int dk = 0; dk < 64; ++dk) {
        S_vh[(64 + dk) * 128 + dv] = s_hi[dk];
    }
    pdl_trigger();
}

__device__ __forceinline__ void accum_q4_group32_smem9_b2(
    uint4 pk,
    float sc,
    float bi,
    const float4* grp0,
    const float4* grp1,
    float* acc) {
    float a00 = 0.0f, a01 = 0.0f, a02 = 0.0f, a03 = 0.0f;
    float a10 = 0.0f, a11 = 0.0f, a12 = 0.0f, a13 = 0.0f;
    #pragma unroll
    for (int w_idx = 0; w_idx < 4; ++w_idx) {
        const uint32_t w = (w_idx == 0) ? pk.x : ((w_idx == 1) ? pk.y : ((w_idx == 2) ? pk.z : pk.w));
        const float q0 = static_cast<float>(w & 0xFu);
        const float q1 = static_cast<float>((w >> 4) & 0xFu);
        const float q2 = static_cast<float>((w >> 8) & 0xFu);
        const float q3 = static_cast<float>((w >> 12) & 0xFu);
        const float q4 = static_cast<float>((w >> 16) & 0xFu);
        const float q5 = static_cast<float>((w >> 20) & 0xFu);
        const float q6 = static_cast<float>((w >> 24) & 0xFu);
        const float q7 = static_cast<float>((w >> 28) & 0xFu);

        const float4 x00 = grp0[w_idx * 2];
        const float4 x01 = grp0[w_idx * 2 + 1];
        a00 = __fmaf_rn(q0, x00.x, a00);
        a01 = __fmaf_rn(q1, x00.y, a01);
        a02 = __fmaf_rn(q2, x00.z, a02);
        a03 = __fmaf_rn(q3, x00.w, a03);
        a00 = __fmaf_rn(q4, x01.x, a00);
        a01 = __fmaf_rn(q5, x01.y, a01);
        a02 = __fmaf_rn(q6, x01.z, a02);
        a03 = __fmaf_rn(q7, x01.w, a03);

        const float4 x10 = grp1[w_idx * 2];
        const float4 x11 = grp1[w_idx * 2 + 1];
        a10 = __fmaf_rn(q0, x10.x, a10);
        a11 = __fmaf_rn(q1, x10.y, a11);
        a12 = __fmaf_rn(q2, x10.z, a12);
        a13 = __fmaf_rn(q3, x10.w, a13);
        a10 = __fmaf_rn(q4, x11.x, a10);
        a11 = __fmaf_rn(q5, x11.y, a11);
        a12 = __fmaf_rn(q6, x11.z, a12);
        a13 = __fmaf_rn(q7, x11.w, a13);
    }
    acc[0] += __fmaf_rn((a00 + a01) + (a02 + a03), sc, bi * grp0[8].x);
    acc[1] += __fmaf_rn((a10 + a11) + (a12 + a13), sc, bi * grp1[8].x);
}

// `0-ULP` `2`-CTAs/SM `B = 2` Q4 GEMV for `cols != 2560` (`out_proj [2560, 6144]`, `o_proj [2560, 4096]`, `ple.kv_proj`)
// Grid: `(ceil(rows / 8), 1)` = `320` CTAs across `110` SMs, Block: `256` (`27,648 B <= 31,744 B` SMEM -> `2` CTAs/SM!)
extern "C" __global__ void q38_q4_gemv_b2(
    const uint4* __restrict__ w_u128,      // [rows, cols/32]
    const uint32_t* __restrict__ sc_u32,   // [rows, cols/64]
    const uint32_t* __restrict__ bi_u32,   // [rows, cols/64]
    const float4* x4_batch,                // [2, cols/4]
    float* y_batch,                        // [2, rows]
    int32_t rows,
    int32_t cols,
    int32_t post_mode) {
    __shared__ float4 smem_x9[2 * 96 * 9]; // `27,648` B <= `31,744` B (`2` CTAs/SM under `64 KiB` carveout!)

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + warp_id;
    const int num_groups = cols >> 5;
    const int num_x4 = cols >> 2;

    float acc0 = 0.0f, acc1 = 0.0f;
    const uint4* rw = (row < rows) ? (w_u128 + static_cast<size_t>(row) * num_groups) : nullptr;
    const uint32_t* rsc = (row < rows) ? (sc_u32 + static_cast<size_t>(row) * (num_groups >> 1)) : nullptr;
    const uint32_t* rbi = (row < rows) ? (bi_u32 + static_cast<size_t>(row) * (num_groups >> 1)) : nullptr;

    pdl_wait();
    for (int g_base = 0; g_base < num_groups; g_base += 96) {
        const int tile_g = ((num_groups - g_base) < 96) ? (num_groups - g_base) : 96;
        stage_x_groups_smem9(x4_batch + (g_base * 8), smem_x9, tile_g, tid, 256);
        stage_x_groups_smem9(x4_batch + num_x4 + (g_base * 8), smem_x9 + 96 * 9, tile_g, tid, 256);
        __syncthreads();

        if (row < rows) {
            float a[2] = {acc0, acc1};
            for (int g = lane; g < tile_g; g += 32) {
                const int gg = g_base + g;
                uint4 pk = rw[gg];
                float sc = load_bf16_from_u32_array(rsc, gg);
                float bi = load_bf16_from_u32_array(rbi, gg);
                accum_q4_group32_smem9_b2(pk, sc, bi, smem_x9 + g * 9, smem_x9 + (96 + g) * 9, a);
            }
            acc0 = a[0];
            acc1 = a[1];
        }
        __syncthreads();
    }

    if (row < rows) {
        float v0 = warp_sum(acc0);
        float v1 = warp_sum(acc1);
        if (lane == 0) {
            if (post_mode == 1) {
                v0 = ptx_silu(v0 * 0.25f);
                v1 = ptx_silu(v1 * 0.25f);
            } else if (post_mode == 2) {
                v0 = ptx_sigmoid(v0);
                v1 = ptx_sigmoid(v1);
            }
            y_batch[row] = v0;
            y_batch[rows + row] = v1;
        }
    }
    pdl_trigger();
}

// `0-ULP` `B = 2` Fused `hc_down_inj_10k_b2` (`Grid (81, 1)` across `110` SMs, `Block 256`, only `2` 160-group tiles!)
extern "C" __global__ void q38_hc_down_inj_10k_b2(
    const uint4* __restrict__ down_w,     // [320, 320]
    const uint32_t* __restrict__ down_sc, // [320, 160]
    const uint32_t* __restrict__ down_bi, // [320, 160]
    const uint4* __restrict__ inj_w,      // [4, 320] or nullptr
    const uint32_t* __restrict__ inj_sc,  // [4, 160] or nullptr
    const uint32_t* __restrict__ inj_bi,  // [4, 160] or nullptr
    const float4* hc_normed4_batch,       // [2, 2560]
    float* hc_low_batch,                  // [2, 320]
    float* hc_inj4_batch,                 // [2, 4]
    int32_t has_inject) {
    __shared__ float4 smem_n9[2 * 160 * 9]; // `46,080` B SMEM
    __shared__ float smem_pair[2][8];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row_local = warp_id >> 1;    // `2` warps collaborate per row (`4` rows/CTA -> `81` CTAs!)
    const int half_id = warp_id & 1;
    const int row = blockIdx.x * 4 + row_local;
    const int max_rows = has_inject ? 324 : 320;

    float part[2] = {0.0f, 0.0f};
    const uint4* rw = nullptr;
    const uint32_t* rsc = nullptr;
    const uint32_t* rbi = nullptr;
    if (row < max_rows) {
        rw = (row < 320) ? (down_w + static_cast<size_t>(row) * 320u) : (inj_w + static_cast<size_t>(row - 320) * 320u);
        rsc = (row < 320) ? (down_sc + static_cast<size_t>(row) * 160u) : (inj_sc + static_cast<size_t>(row - 320) * 160u);
        rbi = (row < 320) ? (down_bi + static_cast<size_t>(row) * 160u) : (inj_bi + static_cast<size_t>(row - 320) * 160u);
    }

    pdl_wait();
    // Tile 0: `g = 0..159`
    stage_x_groups_smem9(hc_normed4_batch, smem_n9, 160, tid, 256);
    stage_x_groups_smem9(hc_normed4_batch + 2560u, smem_n9 + 160 * 9, 160, tid, 256);
    __syncthreads();
    if (row < max_rows) {
        for (int g = half_id * 32 + lane; g < 160; g += 64) {
            uint4 pk = rw[g];
            float sc = load_bf16_from_u32_array(rsc, g);
            float bi = load_bf16_from_u32_array(rbi, g);
            accum_q4_group32_smem9_b2(pk, sc, bi, smem_n9 + g * 9, smem_n9 + (160 + g) * 9, part);
        }
    }
    __syncthreads();

    // Tile 1: `g = 160..319` (`half_id == 0` starts at `192 + lane`, `half_id == 1` starts at `160 + lane` -> exact `0-ULP` order!)
    stage_x_groups_smem9(hc_normed4_batch + 1280u, smem_n9, 160, tid, 256);
    stage_x_groups_smem9(hc_normed4_batch + 2560u + 1280u, smem_n9 + 160 * 9, 160, tid, 256);
    __syncthreads();
    if (row < max_rows) {
        const int g_first = (half_id == 0) ? (32 + lane) : lane;
        for (int g_rel = g_first; g_rel < 160; g_rel += 64) {
            const int g = 160 + g_rel;
            uint4 pk = rw[g];
            float sc = load_bf16_from_u32_array(rsc, g);
            float bi = load_bf16_from_u32_array(rbi, g);
            accum_q4_group32_smem9_b2(pk, sc, bi, smem_n9 + g_rel * 9, smem_n9 + (160 + g_rel) * 9, part);
        }
        smem_pair[0][warp_id] = warp_sum(part[0]);
        smem_pair[1][warp_id] = warp_sum(part[1]);
    }
    __syncthreads();

    if (row < max_rows && half_id == 0 && lane == 0) {
        float v0 = smem_pair[0][warp_id] + smem_pair[0][warp_id + 1];
        float v1 = smem_pair[1][warp_id] + smem_pair[1][warp_id + 1];
        if (row < 320) {
            hc_low_batch[row] = ptx_silu(v0 * 0.25f);
            hc_low_batch[320u + row] = ptx_silu(v1 * 0.25f);
        } else {
            hc_inj4_batch[row - 320] = v0;
            hc_inj4_batch[4u + (row - 320)] = v1;
        }
    }
    pdl_trigger();
}

// `0-ULP` `6`-CTAs/SM (`48` warps/SM = `100%` occupancy!) `B = 2` `hc_up_mix4_b2` (`Grid (320, 1)`, `Block 256`)
extern "C" __global__ void q38_hc_up_mix4_b2(
    const uint4* __restrict__ up_w,       // [10240, 10]
    const uint32_t* __restrict__ up_sc,   // [10240, 5]
    const uint32_t* __restrict__ up_bi,   // [10240, 5]
    const float4* hc_low4_batch,          // [2, 80]
    const float* hc_normed_batch,         // [2, 10240]
    float* mixed_out_batch) {             // [2, 2560]
    __shared__ float4 smem_low9[2 * 10 * 9]; // `2,880` B SMEM -> `6` CTAs/SM = `48` warps/SM!

    const int tid = threadIdx.x;
    pdl_wait();
    stage_x_groups_smem9(hc_low4_batch, smem_low9, 10, tid, 256);
    stage_x_groups_smem9(hc_low4_batch + 80u, smem_low9 + 10 * 9, 10, tid, 256);
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int d = blockIdx.x * 8 + warp_id;
    if (d < kD) {
        const int s = lane >> 3;
        const int sub_lane = lane & 7;
        const size_t row = static_cast<size_t>(s) * kD + static_cast<size_t>(d);
        const uint4* rw = up_w + row * 10u;
        const uint32_t* rsc = up_sc + row * 5u;
        const uint32_t* rbi = up_bi + row * 5u;

        float acc[2] = {0.0f, 0.0f};
        {
            const int g0 = sub_lane;
            uint4 pk0 = rw[g0];
            float sc0 = load_bf16_from_u32_array(rsc, g0);
            float bi0 = load_bf16_from_u32_array(rbi, g0);
            accum_q4_group32_smem9_b2(pk0, sc0, bi0, smem_low9 + g0 * 9, smem_low9 + (10 + g0) * 9, acc);
        }
        if (sub_lane < 2) {
            const int g1 = 8 + sub_lane;
            uint4 pk1 = rw[g1];
            float sc1 = load_bf16_from_u32_array(rsc, g1);
            float bi1 = load_bf16_from_u32_array(rbi, g1);
            accum_q4_group32_smem9_b2(pk1, sc1, bi1, smem_low9 + g1 * 9, smem_low9 + (10 + g1) * 9, acc);
        }

        #pragma unroll
        for (int t = 0; t < 2; ++t) {
            float a = acc[t];
            a += __shfl_xor_sync(0xFFFFFFFFu, a, 4);
            a += __shfl_xor_sync(0xFFFFFFFFu, a, 2);
            a += __shfl_xor_sync(0xFFFFFFFFu, a, 1);
            float weighted = 0.0f;
            if (sub_lane == 0) {
                weighted = ptx_sigmoid(a) * hc_normed_batch[static_cast<size_t>(t) * kHcD + row];
            }
            weighted += __shfl_down_sync(0xFFFFFFFFu, weighted, 16);
            weighted += __shfl_down_sync(0xFFFFFFFFu, weighted, 8);
            if (lane == 0) {
                mixed_out_batch[static_cast<size_t>(t) * kD + d] = weighted * 0.25f;
            }
        }
    }
    pdl_trigger();
}

// All lanes of one warp evaluate a stored Q8 row. Reconstruct the BF16
// weight before the FP32 dot to match the independent packed-weight teacher.
__device__ __forceinline__ float q38_q8_row_dot(
    const uint32_t* weights, const uint16_t* scales, const float4* input,
    int row, int cols) {
    const int lane = threadIdx.x & 31;
    const int words = cols / 4;
    const uint32_t* w = weights + static_cast<size_t>(row) * words;
    const uint16_t* sc = scales + static_cast<size_t>(row) * (cols / 32);
    float acc = 0.0f;
    for (int i = lane; i < words; i += 32) {
        const float4 weights4 = q38_scaled_q8_word(w[i], sc[i / 8]);
        const float4 v = input[i];
        acc = fmaf(weights4.x, v.x, acc);
        acc = fmaf(weights4.y, v.y, acc);
        acc = fmaf(weights4.z, v.z, acc);
        acc = fmaf(weights4.w, v.w, acc);
    }
    return warp_sum(acc);
}

extern "C" __global__ void q38_dense_q8_gemv(
    const uint32_t* weights, const uint16_t* scales,
    const float4* input, float* output, int rows, int cols) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    pdl_wait();
    if (row < rows) {
        const int lane = threadIdx.x & 31;
        const int words = cols / 4;
        const uint32_t* w = weights + static_cast<size_t>(row) * words;
        const uint16_t* sc = scales + static_cast<size_t>(row) * (cols / 32);
        const float4* x = input + static_cast<size_t>(blockIdx.y) * words;
        float acc = 0.0f;
        for (int i = lane; i < words; i += 32) {
            const uint32_t codes = w[i];
            const float scale = bf16_to_f32(sc[i / 8]);
            const float4 v = x[i];
            const float elements[] = {v.x, v.y, v.z, v.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int code = static_cast<int8_t>(codes >> (8 * j));
                uint16_t rounded;
                asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(rounded) : "f"(code * scale));
                acc = fmaf(bf16_to_f32(rounded), elements[j], acc);
            }
        }
        const float sum = warp_sum(acc);
        if ((threadIdx.x & 31) == 0) output[static_cast<size_t>(blockIdx.y) * rows + row] = sum;
    }
    pdl_trigger();
}

// Four adjacent token rows share one group's codes, scales and reconstructed
// weights. Each token keeps the scalar lane-local and warp accumulation order.
extern "C" __global__ void q38_dense_q8_gemm_b4(
    const uint32_t* weights, const uint16_t* scales, const float4* input,
    float* output, int rows, int cols, int batch) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int token = blockIdx.y * 4;
    const int lane = threadIdx.x & 31;
    pdl_wait();
    if (row < rows) {
        const int words = cols / 4;
        const uint32_t* w = weights + static_cast<size_t>(row) * words;
        const uint16_t* sc = scales + static_cast<size_t>(row) * (cols / 32);
        float sums[4] = {};
        for (int i = lane; i < words; i += 32) {
            const float4 val = q38_scaled_q8_word(w[i], sc[i / 8]);
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                if (token + t < batch) {
                    const float4 v = input[static_cast<size_t>(token + t) * words + i];
                    float a = fmaf(val.x, v.x, sums[t]);
                    a = fmaf(val.y, v.y, a);
                    a = fmaf(val.z, v.z, a);
                    sums[t] = fmaf(val.w, v.w, a);
                }
            }
        }
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            if (token + t < batch) {
                const float v = warp_sum(sums[t]);
                if (lane == 0) output[static_cast<size_t>(token + t) * rows + row] = v;
            }
        }
    }
    pdl_trigger();
}

// The final mixer has 320 rows. Intermediate mixers append four linear injection rows.
extern "C" __global__ void q38_hc_q8_down_inject(
    const uint32_t* weights, const uint16_t* scales, const float4* normed,
    float* low, float* inject, int rows) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int token = blockIdx.y;
    pdl_wait();
    if (row < rows) {
        const float sum = q38_q8_row_dot(weights, scales, normed + token * kD, row, 4 * kD);
        if ((threadIdx.x & 31) == 0) {
            if (row < 320) low[token * 320 + row] = ptx_silu(0.25f * sum);
            else inject[token * 4 + row - 320] = sum;
        }
    }
    pdl_trigger();
}

extern "C" __global__ void q38_hc_q8_up_mix(
    const uint32_t* weights, const uint16_t* scales, const float4* low,
    const float* normed, float* mixed) {
    const int d = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int token = blockIdx.y;
    pdl_wait();
    if (d < kD) {
        float contributions[4];
        #pragma unroll
        for (int stream = 0; stream < 4; ++stream) {
            const int row = stream * kD + d;
            float dot = q38_q8_row_dot(weights, scales, low + token * 80, row, 320);
            if ((threadIdx.x & 31) == 0)
                contributions[stream] = ptx_sigmoid(dot) * normed[token * 4 * kD + row];
        }
        if ((threadIdx.x & 31) == 0) {
            const float a = contributions[0] + contributions[2];
            const float b = contributions[1] + contributions[3];
            mixed[token * kD + d] = 0.25f * (a + b);
        }
    }
    pdl_trigger();
}

// Same grid and argument order as the scalar/batched Q4 embedding. The unused
// bias pointer allows a typed loader to choose the entry when it loads the table.
extern "C" __global__ void q38_embed_q8_hc4(
    const uint32_t* weights, const uint16_t* scales, const void*,
    const int32_t* tokens, float* out) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    pdl_wait();
    if (d < kD) {
        const size_t offset = static_cast<size_t>(tokens[b]) * kD + d;
        const uint32_t word = weights[offset / 4];
        const int code = static_cast<int8_t>(word >> (8 * (d & 3)));
        const float value = static_cast<float>(code) * bf16_to_f32(scales[offset / 32]);
        uint16_t rounded;
        asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(rounded) : "f"(value));
        #pragma unroll
        for (int stream = 0; stream < 4; ++stream)
            out[(static_cast<size_t>(b) * 4 + stream) * kD + d] = bf16_to_f32(rounded);
    }
    pdl_trigger();
}
