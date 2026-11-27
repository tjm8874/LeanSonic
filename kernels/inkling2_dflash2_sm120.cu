// LeanSonic — `Inkling-2-Mini` DFlash 2 block-diffusion drafter kernels (`sm_120`, #103)
// ============================================================================
// A separate module from `inkling2_sm120.cu`: the drafter only proposes tokens (the target
// verifies every one), and keeping its kernels out of the target's translation unit leaves the
// target kernels' code generation byte-for-byte unchanged. The host compiles this file only when
// a DFlash 2 checkpoint is selected. The Q8_64 activation quantiser and DP4A GEMMs it needs are
// launched from the target module.
//
// Obeys the Gate 2 static SASS rules (`ci/audit_sass.py`): `STACK: 0`, no `CALL.REL.NOINC`
// (inline PTX `.approx` transcendentals and `__fdividef`), no scalar 8-bit global loads.
// PDL: every kernel calls `pdl_wait()` before touching memory a predecessor may write, then
// `pdl_trigger()` (the drafter chain is latency-bound).

#ifndef __CUDACC_RTC__
#include <stdint.h>
#include "include/leansonic_verified.cuh"
#else
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
#endif

namespace leansonic::inkling2_df2 {

static constexpr int kD = 2304;

// Helpers copied from `inkling2_sm120.cu` (same PTX).

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

__device__ __forceinline__ float ptx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
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

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, mask);
    }
    return v;
}

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

// ============================================================================
// DFlash 2 block-diffusion drafter (#103): five non-causal sliding-window layers wrapped in
// dynamic two-tap group-16 causal convolutions, target-head Top-16 candidates and a rank-256
// bilinear path selector. Draft rows only propose tokens, so this arithmetic never reaches
// the target's outputs. Block = 8 rows (anchor + 7 masks), D = 2304.
// ============================================================================
constexpr int kDf2Q = 4096;
constexpr int kDf2Kv = 1024;
constexpr int kDf2Groups = kD / 16;    // 144
constexpr int kDf2Dyn = 4 * kDf2Groups; // 576
constexpr int kDf2Rank = 256;
constexpr int kDf2Top = 16;

// x[t] = rms_norm(embed[tok_t]) * embed_norm with tok_0 = anchor, tok_t = mask_tok. The mask
// rows are constant: with `use_cache` they are copied from `mask_row` (written by a
// `use_cache = 0` launch), so only the anchor row is read from the (host-resident) table.
// Grid `8`, Block `256`.
extern "C" __global__ void df2_block_init(
    const uint4* __restrict__ embed_bf16,               // [vocab, D / 8]
    const uint32_t* __restrict__ embed_norm_bf16_pairs, // [D / 2]
    int32_t anchor,
    int32_t mask_tok,
    int32_t use_cache,
    float* __restrict__ mask_row,                       // [D]
    float* __restrict__ x) {                            // [8, D]
    pdl_wait();
    pdl_trigger();
    __shared__ float smem_red[8];
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    float* out = x + static_cast<size_t>(t) * kD;
    if (t > 0 && use_cache) {
        for (int c = tid; c < kD / 4; c += 256) {
            reinterpret_cast<float4*>(out)[c] = reinterpret_cast<const float4*>(mask_row)[c];
        }
        return;
    }
    const int tok = (t == 0) ? anchor : mask_tok;
    const uint4* row = embed_bf16 + static_cast<size_t>(tok) * (kD / 8);
    // 288 16-byte loads issued together: one round trip to the host-resident table.
    uint4 w0 = row[tid];
    uint4 w1 = (tid < kD / 8 - 256) ? row[256 + tid] : make_uint4(0u, 0u, 0u, 0u);
    float sq = 0.0f;
    auto sq8 = [&](const uint4& w) {
        const uint32_t u[4] = {w.x, w.y, w.z, w.w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float a = bf16_to_f32(static_cast<uint16_t>(u[j] & 0xFFFFu));
            const float b = bf16_to_f32(static_cast<uint16_t>(u[j] >> 16));
            sq += a * a + b * b;
        }
    };
    sq8(w0);
    sq8(w1);
    sq = warp_sum(sq);
    if ((tid & 31) == 0) smem_red[tid >> 5] = sq;
    __syncthreads();
    float tot = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) tot += smem_red[i];
    const float scale = ptx_rsqrt(tot * (1.0f / static_cast<float>(kD)) + 1e-6f);
    auto put8 = [&](const uint4& w, int c8) {
        const uint32_t u[4] = {w.x, w.y, w.z, w.w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t nw = embed_norm_bf16_pairs[c8 * 4 + j];
            const float a = bf16_to_f32(static_cast<uint16_t>(u[j] & 0xFFFFu)) * scale *
                            bf16_to_f32(static_cast<uint16_t>(nw & 0xFFFFu));
            const float b = bf16_to_f32(static_cast<uint16_t>(u[j] >> 16)) * scale *
                            bf16_to_f32(static_cast<uint16_t>(nw >> 16));
            out[c8 * 8 + j * 2] = a;
            out[c8 * 8 + j * 2 + 1] = b;
            if (t == 1) {
                mask_row[c8 * 8 + j * 2] = a;
                mask_row[c8 * 8 + j * 2 + 1] = b;
            }
        }
    };
    put8(w0, tid);
    if (tid < kD / 8 - 256) put8(w1, 256 + tid);
}

// ---------------------------------------------------------------------------------------------
// Q8_64 activations (`ink2_quant_q8_64_b` layout): per row `cols` int8 codes (`round(v * 127 /
// amax)`, clamped to `[-127, 127]`) and one FP32 scale `amax / 127` per 64 columns. One warp
// quantizes one group: lane `l` holds columns `2l` and `2l + 1`.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ int dp4a_u128(uint4 w, uint4 x) {
    int d = __dp4a(static_cast<int>(w.x), static_cast<int>(x.x), 0);
    d = __dp4a(static_cast<int>(w.y), static_cast<int>(x.y), d);
    d = __dp4a(static_cast<int>(w.z), static_cast<int>(x.z), d);
    return __dp4a(static_cast<int>(w.w), static_cast<int>(x.w), d);
}

__device__ __forceinline__ void df2_quant_group(
    float v0, float v1, int lane, uint32_t* q_row_u32, float* sc_row, int c64) {
    float amax = fmaxf(fabsf(v0), fabsf(v1));
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask));
    }
    const float inv = (amax > 0.0f) ? (127.0f * ptx_rcp(amax)) : 0.0f;
    const int q0 = min(127, max(-127, __float2int_rn(v0 * inv)));
    const int q1 = min(127, max(-127, __float2int_rn(v1 * inv)));
    const uint32_t pair16 = static_cast<uint32_t>(q0 & 0xFF) | (static_cast<uint32_t>(q1 & 0xFF) << 8);
    const uint32_t pack32 = pair16 | (__shfl_xor_sync(0xFFFFFFFFu, pair16, 1) << 16);
    if ((lane & 1) == 0) q_row_u32[c64 * 16 + (lane >> 1)] = pack32;
    if (lane == 0) sc_row[c64] = amax * (1.0f / 127.0f);
}

// `xn = rms_norm(x) * w` of the eight block rows, written as FP32 and as Q8_64 (for the conv
// projection). One warp per 64-column group recomputes its row's RMS from L2.
// Grid `36` (8 warps = 8 of the 288 row groups each), Block `256`.
extern "C" __global__ void df2_norm_quant8(
    const float* __restrict__ x,          // [8, D]
    const float* __restrict__ nw,         // [D]
    float* __restrict__ xn,               // [8, D]
    uint32_t* __restrict__ xq,            // [8, D / 4]
    float* __restrict__ xs) {             // [8, D / 64]
    pdl_wait();
    pdl_trigger();
    const int lane = threadIdx.x & 31;
    const int gw = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int t = gw / (kD / 64);
    const int c64 = gw - t * (kD / 64);
    const float4* row4 = reinterpret_cast<const float4*>(x + static_cast<size_t>(t) * kD);
    float sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < kD / 128; ++i) {
        const float4 v = row4[lane + 32 * i];
        sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    const float inv = ptx_rsqrt(warp_sum(sq) * (1.0f / static_cast<float>(kD)) + 1e-6f);
    const int c = c64 * 64 + lane * 2;
    const float2 xv = *reinterpret_cast<const float2*>(x + static_cast<size_t>(t) * kD + c);
    const float2 wv = *reinterpret_cast<const float2*>(nw + c);
    const float v0 = xv.x * inv * wv.x, v1 = xv.y * inv * wv.y;
    *reinterpret_cast<float2*>(xn + static_cast<size_t>(t) * kD + c) = make_float2(v0, v1);
    df2_quant_group(v0, v1, lane, xq + static_cast<size_t>(t) * (kD / 4), xs + t * (kD / 64), c64);
}

// Q8_64 of `rms_norm(sum_s x_s[r]) * w` for `rows` rows (`x` summed over `nsplit` partials
// `split_stride` floats apart). Grid `rows`, Block `256`.
extern "C" __global__ void df2_rmsnorm_quant(
    const float* __restrict__ x,          // [nsplit][rows, D]
    int32_t nsplit,
    int64_t split_stride,
    const float* __restrict__ w,          // [D]
    uint32_t* __restrict__ xq,            // [rows, D / 4]
    float* __restrict__ xs) {             // [rows, D / 64]
    pdl_wait();
    pdl_trigger();
    __shared__ float red[8];
    const int r = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const float* in = x + static_cast<size_t>(r) * kD;
    float sq = 0.0f;
    for (int c = threadIdx.x; c < kD; c += 256) {
        float v = 0.0f;
        for (int s = 0; s < nsplit; ++s) v += in[s * split_stride + c];
        sq += v * v;
    }
    sq = warp_sum(sq);
    if (lane == 0) red[warp] = sq;
    __syncthreads();
    float tot = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) tot += red[i];
    const float scale = ptx_rsqrt(tot * (1.0f / static_cast<float>(kD)) + 1e-6f);
    for (int c64 = warp; c64 < kD / 64; c64 += 8) {
        const int c = c64 * 64 + lane * 2;
        float v0 = 0.0f, v1 = 0.0f;
        for (int s = 0; s < nsplit; ++s) {
            const float2 p = *reinterpret_cast<const float2*>(in + s * split_stride + c);
            v0 += p.x;
            v1 += p.y;
        }
        df2_quant_group(v0 * scale * w[c], v1 * scale * w[c + 1], lane,
                        xq + static_cast<size_t>(r) * (kD / 4), xs + static_cast<size_t>(r) * (kD / 64), c64);
    }
}

// Pre-conv (taps `0, 1` of `base [4, D]`; `dyn` = sum of `ndyn` partials of `[8, 4, 144]`)
// straight into Q8_64:
//   a[t] = (base[0] + dyn[t][0][g]) * z[t] + (t > 0 ? (base[1] + dyn[t][1][g]) * z[t-1] : 0)
// Grid `36` (8 warps = 8 of the 288 row groups each), Block `256`.
extern "C" __global__ void df2_conv2_quant(
    const float* __restrict__ z,          // [8, D]
    const float* __restrict__ dyn,        // [ndyn][8, 576]
    int32_t ndyn,
    const float* __restrict__ base,       // [4, D]
    uint32_t* __restrict__ xq,            // [8, D / 4]
    float* __restrict__ xs) {             // [8, D / 64]
    pdl_wait();
    pdl_trigger();
    const int lane = threadIdx.x & 31;
    const int gw = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int t = gw / (kD / 64);
    const int c64 = gw - t * (kD / 64);
    const int c = c64 * 64 + lane * 2;
    const int g = c >> 4;
    float d0 = 0.0f, d1 = 0.0f;
    for (int s = 0; s < ndyn; ++s) {
        const float* dr = dyn + (s * 8 + t) * kDf2Dyn;
        d0 += dr[g];
        d1 += dr[kDf2Groups + g];
    }
    const float2 z0 = *reinterpret_cast<const float2*>(z + static_cast<size_t>(t) * kD + c);
    const float2 b0 = *reinterpret_cast<const float2*>(base + c);
    float v0 = (b0.x + d0) * z0.x;
    float v1 = (b0.y + d0) * z0.y;
    if (t > 0) {
        const float2 z1 = *reinterpret_cast<const float2*>(z + static_cast<size_t>(t - 1) * kD + c);
        const float2 b1 = *reinterpret_cast<const float2*>(base + kD + c);
        v0 += (b1.x + d1) * z1.x;
        v1 += (b1.y + d1) * z1.y;
    }
    df2_quant_group(v0, v1, lane, xq + static_cast<size_t>(t) * (kD / 4), xs + t * (kD / 64), c64);
}

// Post-conv residual (taps `2, 3`; `dyn` summed over `ndyn` partials) over `z = sum_s zp[s]`
// (split-K partials, `nsplit` copies of `[8, D]`): x[t] += (base[2] + dyn[t][2][g]) * z[t] +
// (t > 0 ? (base[3] + dyn[t][3][g]) * z[t-1] : 0). Grid `(9, 8)`, Block `256`.
extern "C" __global__ void df2_conv2_post(
    const float* __restrict__ zp,         // [nsplit][8, D]
    int32_t nsplit,
    const float* __restrict__ dyn,        // [ndyn][8, 576]
    int32_t ndyn,
    const float* __restrict__ base,       // [4, D]
    float* __restrict__ x) {              // [8, D]
    pdl_wait();
    pdl_trigger();
    const int c = blockIdx.x * 256 + threadIdx.x;
    const int t = blockIdx.y;
    const int g = c >> 4;
    const size_t ss = static_cast<size_t>(8) * kD;
    float z0 = 0.0f, z1 = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        z0 += zp[s * ss + static_cast<size_t>(t) * kD + c];
        if (t > 0) z1 += zp[s * ss + static_cast<size_t>(t - 1) * kD + c];
    }
    float d2 = 0.0f, d3 = 0.0f;
    for (int s = 0; s < ndyn; ++s) {
        const float* dr = dyn + (s * 8 + t) * kDf2Dyn;
        d2 += dr[2 * kDf2Groups + g];
        d3 += dr[3 * kDf2Groups + g];
    }
    float a = (base[2 * kD + c] + d2) * z0;
    if (t > 0) a += (base[3 * kD + c] + d3) * z1;
    x[static_cast<size_t>(t) * kD + c] += a;
}

// Split-K planar Q8_64 GEMM for the eight block rows: `part[blockIdx.y][t][r] = sum over
// column slice blockIdx.y of w[phys(r)] . x[t]` (`slice16 <= 128` uint4 columns, a multiple of
// 4), with `phys(r) = (r / seg) * seg_stride + seg_off + r % seg` (identity: `seg = rows`,
// `seg_off = 0`). Weights of the slice are loaded before `pdl_wait()`.
// Grid `(rows / 16, cols16 / slice16)`, Block `256` (8 warps x 2 rows).
extern "C" __global__ void __launch_bounds__(256) df2_gemm_sk(
    const uint4* __restrict__ w_q8,       // [rows, cols / 16]
    const float* __restrict__ w_sc64,     // [rows, cols / 64]
    const uint4* __restrict__ x_q8,       // [8, cols / 16]
    const float* __restrict__ x_sc64,     // [8, cols / 64]
    float* __restrict__ part,             // [nsplit][8, rows]
    int32_t rows,
    int32_t cols,
    int32_t slice16,
    int32_t seg,
    int32_t seg_stride,
    int32_t seg_off) {
    __shared__ uint4 sx[8][128];          // 16,384 B
    __shared__ float sxs[8][32];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int cols16 = cols >> 4, cols64 = cols >> 6;
    const int c0 = blockIdx.y * slice16;
    const int row0 = (blockIdx.x * 8 + warp) * 2;
    const int prow = (row0 / seg) * seg_stride + seg_off + row0 % seg; // `seg` is even
    const uint4* w_r0 = w_q8 + static_cast<size_t>(prow) * cols16 + c0;
    const uint4* w_r1 = w_r0 + cols16;
    const float* s_r0 = w_sc64 + static_cast<size_t>(prow) * cols64 + (c0 >> 2);
    const float* s_r1 = s_r0 + cols64;
    uint4 w0[4], w1[4];
    float ws0[4], ws1[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c16 = lane + 32 * i;
        if (c16 < slice16) {
            w0[i] = w_r0[c16];
            w1[i] = w_r1[c16];
            ws0[i] = s_r0[c16 >> 2];
            ws1[i] = s_r1[c16 >> 2];
        }
    }
    pdl_wait();
    pdl_trigger();
    for (int i = threadIdx.x; i < 8 * slice16; i += 256) {
        const int t = i / slice16, j = i - t * slice16;
        sx[t][j] = x_q8[static_cast<size_t>(t) * cols16 + c0 + j];
    }
    const int slice64 = slice16 >> 2;
    for (int i = threadIdx.x; i < 8 * slice64; i += 256) {
        const int t = i / slice64, j = i - t * slice64;
        sxs[t][j] = x_sc64[static_cast<size_t>(t) * cols64 + (c0 >> 2) + j];
    }
    __syncthreads();
    float a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float b[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c16 = lane + 32 * i;
        if (c16 < slice16) {
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                const uint4 xq = sx[t][c16];
                const float xsv = sxs[t][c16 >> 2];
                a[t] = __fmaf_rn(static_cast<float>(dp4a_u128(w0[i], xq)), ws0[i] * xsv, a[t]);
                b[t] = __fmaf_rn(static_cast<float>(dp4a_u128(w1[i], xq)), ws1[i] * xsv, b[t]);
            }
        }
    }
    float* dst = part + static_cast<size_t>(blockIdx.y) * 8 * rows;
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        const float v0 = warp_sum(a[t]);
        const float v1 = warp_sum(b[t]);
        if (lane == 0) {
            dst[static_cast<size_t>(t) * rows + row0] = v0;
            dst[static_cast<size_t>(t) * rows + row0 + 1] = v1;
        }
    }
}

// Per-head RMSNorm + NeoX RoPE (`theta = 1e4`) for `rows` rows at positions `pos0 + r` and
// `gridDim.z` layers, over split-K partials `qkv` (`nsplit` copies `split_stride` floats apart,
// row stride `row_stride`): layer `l`'s K / V start at columns `k_col + l * layer_cols` and
// `k_col + 1024 + l * layer_cols`; Q (layer 0 only, `do_q`) at column 0 goes to `q_out`.
// K (normed, rotated) and V go to ring cell `(pos0 + r) % cap` of layer `l` (`l * ring_layer`
// floats further; K norm `64 * l` BF16 pairs further).
// Grid `(do_q ? 40 : 8, rows, layers)`, Block `128`: `blockIdx.x < 8` = KV head, else Q head - 8.
extern "C" __global__ void df2_qk_prep(
    const float* __restrict__ qkv,            // [nsplit][rows, row_stride]
    int32_t nsplit,
    int32_t split_stride,
    int32_t row_stride,
    int32_t k_col,
    int32_t layer_cols,
    float* __restrict__ q_out,                // [rows, 4096]
    const uint32_t* __restrict__ qn_bf16_p,   // [64]
    const uint32_t* __restrict__ kn_bf16_p,   // [layers, 64]
    const float* __restrict__ inv_freq64,     // [64]
    int32_t pos0,
    int32_t cap,
    int64_t ring_layer,
    float* __restrict__ k_ring,               // [layers, cap, 1024]
    float* __restrict__ v_ring) {             // [layers, cap, 1024]
    pdl_wait();
    pdl_trigger();
    __shared__ float smem_hd[128];
    const int hh = blockIdx.x;
    const int r = blockIdx.y;
    const int l = blockIdx.z;
    const int d = threadIdx.x;
    const int pos = pos0 + r;
    const float* row = qkv + static_cast<size_t>(r) * row_stride;
    auto sum = [&](int col) {
        float v = 0.0f;
        for (int s = 0; s < nsplit; ++s) v += row[static_cast<size_t>(s) * split_stride + col];
        return v;
    };
    if (hh < 8) {
        const int col = k_col + l * layer_cols + hh * 128 + d;
        const float kr = head_norm_neox_rope128(sum(col), kn_bf16_p + l * 64, inv_freq64, pos, smem_hd);
        const size_t cell = l * ring_layer + static_cast<size_t>(pos % cap) * kDf2Kv + hh * 128 + d;
        k_ring[cell] = kr;
        v_ring[cell] = sum(col + kDf2Kv);
    } else {
        const int col = (hh - 8) * 128 + d;
        q_out[static_cast<size_t>(r) * kDf2Q + col] =
            head_norm_neox_rope128(sum(col), qn_bf16_p, inv_freq64, pos, smem_hd);
    }
}

// Non-causal block attention, quantized for `wo`: the 8 rows attend every ring key in `[lo, hi)`
// (cell `p % cap`); 32 query heads over 8 KV heads; the head's 128 outputs become two Q8_64
// groups of row `t`. Grid `(32, 8)`, Block `512` (16 warps split the keys).
extern "C" __global__ void __launch_bounds__(512) df2_attn_q8(
    const float* __restrict__ q,       // [8, 4096]
    const float* __restrict__ k_ring,  // [cap, 1024]
    const float* __restrict__ v_ring,  // [cap, 1024]
    int32_t lo,
    int32_t hi,
    int32_t cap,
    uint32_t* __restrict__ xq,         // [8, 4096 / 4]
    float* __restrict__ xs) {          // [8, 4096 / 64]
    pdl_wait();
    pdl_trigger();
    constexpr int kW = 16;
    __shared__ float smem_m[kW];
    __shared__ float smem_l[kW];
    __shared__ float smem_o[kW][128];
    __shared__ float smem_out[128];
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int kvh = h >> 2;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const float4 q4 = reinterpret_cast<const float4*>(q + static_cast<size_t>(t) * kDf2Q + h * 128)[lane];
    const float scale = 0.08838834764831845f; // 1 / sqrt(128)
    float m = -1e30f, l = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    for (int p = lo + warp; p < hi; p += kW) {
        const size_t cell = static_cast<size_t>(p % cap) * kDf2Kv + kvh * 128;
        const float4 k4 = reinterpret_cast<const float4*>(k_ring + cell)[lane];
        const float4 v4 = reinterpret_cast<const float4*>(v_ring + cell)[lane];
        const float dot = warp_sum(q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w) * scale;
        const float m_new = fmaxf(m, dot);
        const float alpha = ptx_exp(m - m_new);
        const float pr = ptx_exp(dot - m_new);
        l = l * alpha + pr;
        m = m_new;
        o0 = o0 * alpha + pr * v4.x;
        o1 = o1 * alpha + pr * v4.y;
        o2 = o2 * alpha + pr * v4.z;
        o3 = o3 * alpha + pr * v4.w;
    }
    if (lane == 0) {
        smem_m[warp] = m;
        smem_l[warp] = l;
    }
    smem_o[warp][lane * 4] = o0;
    smem_o[warp][lane * 4 + 1] = o1;
    smem_o[warp][lane * 4 + 2] = o2;
    smem_o[warp][lane * 4 + 3] = o3;
    __syncthreads();
    if (threadIdx.x < 128) {
        float m_max = smem_m[0];
        #pragma unroll
        for (int w = 1; w < kW; ++w) m_max = fmaxf(m_max, smem_m[w]);
        float den = 0.0f, num = 0.0f;
        #pragma unroll
        for (int w = 0; w < kW; ++w) {
            const float s = (smem_l[w] > 0.0f) ? ptx_exp(smem_m[w] - m_max) : 0.0f;
            den += smem_l[w] * s;
            num += smem_o[w][threadIdx.x] * s;
        }
        smem_out[threadIdx.x] = num * ptx_rcp(fmaxf(den, 1e-20f));
    }
    __syncthreads();
    if (warp < 2) {
        df2_quant_group(smem_out[warp * 64 + lane * 2], smem_out[warp * 64 + lane * 2 + 1], lane,
                        xq + static_cast<size_t>(t) * (kDf2Q / 4), xs + t * (kDf2Q / 64), h * 2 + warp);
    }
}

// act = silu(gate) * up of the eight block rows (`gu` = sum of `nsplit` partials of `[8, 2 *
// ff]`, gate first) straight into Q8_64. Grid `8 * ff / 512` (8 warps = 8 groups each),
// Block `256`.
extern "C" __global__ void df2_silu_quant(
    const float* __restrict__ gu,      // [nsplit][8, 2 * ff]
    int32_t nsplit,
    int32_t ff,
    uint32_t* __restrict__ xq,         // [8, ff / 4]
    float* __restrict__ xs) {          // [8, ff / 64]
    pdl_wait();
    pdl_trigger();
    const int lane = threadIdx.x & 31;
    const int groups = ff >> 6;
    const int gw = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int t = gw / groups;
    const int c64 = gw - t * groups;
    const int c = c64 * 64 + lane * 2;
    const size_t ss = static_cast<size_t>(8) * 2 * ff;
    float2 g = make_float2(0.0f, 0.0f), u = g;
    for (int s = 0; s < nsplit; ++s) {
        const float* row = gu + s * ss + static_cast<size_t>(t) * 2 * ff;
        const float2 gs = *reinterpret_cast<const float2*>(row + c);
        const float2 us = *reinterpret_cast<const float2*>(row + ff + c);
        g.x += gs.x;
        g.y += gs.y;
        u.x += us.x;
        u.y += us.y;
    }
    df2_quant_group((g.x * ptx_sigmoid(g.x)) * u.x, (g.y * ptx_sigmoid(g.y)) * u.y, lane,
                    xq + static_cast<size_t>(t) * (ff >> 2), xs + static_cast<size_t>(t) * groups, c64);
}

// Gathers target taps `[4, tap_stride / D, D]` rows `row0 .. row0 + rows` as Q8_64 rows of
// `4 * D` columns. Grid `(18, rows)` (8 warps = 8 of the 144 groups each), Block `256`.
extern "C" __global__ void df2_tap_quant(
    const float* __restrict__ taps,
    int64_t tap_stride,
    int32_t row0,
    uint32_t* __restrict__ xq,         // [rows, 4 * D / 4]
    float* __restrict__ xs) {          // [rows, 4 * D / 64]
    pdl_wait();
    pdl_trigger();
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.y;
    const int c64 = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int k = (c64 * 64) / kD;
    const int c = c64 * 64 - k * kD + lane * 2;
    const float2 v = *reinterpret_cast<const float2*>(
        taps + static_cast<size_t>(k) * tap_stride + static_cast<size_t>(row0 + r) * kD + c);
    df2_quant_group(v.x, v.y, lane, xq + static_cast<size_t>(r) * kD, xs + static_cast<size_t>(r) * (kD / 16), c64);
}

__device__ __forceinline__ bool df2_before(float va, int ia, float vb, int ib) {
    return va > vb || (va == vb && ia < ib);
}

constexpr int kDf2Chunks = 40;  // 5,002 tokens per chunk -> 40 KB of shared memory

// Block-wide pick of the first (value desc, id asc) element after `(last_v, last_i)` among the
// `n` (value, id) pairs in shared memory. All threads of the 256-thread block return the pick.
__device__ __forceinline__ void df2_block_pick(
    const float* sv, const int* si, int n, float last_v, int last_i, float* red_v, int* red_i,
    float& out_v, int& out_i) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    float bv = -3.0e38f;
    int bi = 0x7fffffff;
    for (int j = tid; j < n; j += blockDim.x) {
        const float v = sv[j];
        const int i = si[j];
        if (df2_before(last_v, last_i, v, i) && df2_before(v, i, bv, bi)) {
            bv = v;
            bi = i;
        }
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, mask);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, mask);
        if (df2_before(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) {
        red_v[warp] = bv;
        red_i[warp] = bi;
    }
    __syncthreads();
    bv = red_v[0];
    bi = red_i[0];
    for (int w = 1; w < static_cast<int>(blockDim.x >> 5); ++w) {
        if (df2_before(red_v[w], red_i[w], bv, bi)) {
            bv = red_v[w];
            bi = red_i[w];
        }
    }
    __syncthreads();
    out_v = bv;
    out_i = bi;
}

// Stage 1 of the Top-16: chunk `blockIdx.y` of row `blockIdx.x` is staged in shared memory, its
// sorted Top-16 `(value, id)` and its softcapped log-sum-exp `(m, l)` over ALL its tokens are
// written to `part`. Grid `(rows, 40)`, Block `256`.
extern "C" __global__ void df2_top16_part(
    const float* __restrict__ logits,  // [rows, vocab]
    int32_t vocab,
    float cap,
    float* __restrict__ part) {        // [rows, 40, 34]: 16 values, 16 ids (bits), m, l
    pdl_wait();
    pdl_trigger();
    const float inv_cap = __fdividef(1.0f, cap); // fast reciprocal: no IEEE-division call
    constexpr int kMaxChunk = 5056;
    __shared__ float sv[kMaxChunk];
    __shared__ int si[kMaxChunk];
    __shared__ float red_v[8];
    __shared__ int red_i[8];
    const int r = blockIdx.x;
    const int ch = blockIdx.y;
    const int per = (vocab + kDf2Chunks - 1) / kDf2Chunks;
    const int i0 = ch * per;
    const int n = max(0, min(vocab, i0 + per) - i0);
    const float* row = logits + static_cast<size_t>(r) * vocab;
    float m = -3.0e38f, l = 0.0f;
    for (int j = threadIdx.x; j < n; j += 256) {
        const float v = row[i0 + j];
        sv[j] = v;
        si[j] = i0 + j;
        const float c = cap * ptx_tanh(v * inv_cap);
        if (c > m) {
            l = l * ptx_exp(m - c) + 1.0f;
            m = c;
        } else {
            l += ptx_exp(c - m);
        }
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        const float om = __shfl_xor_sync(0xffffffffu, m, mask);
        const float ol = __shfl_xor_sync(0xffffffffu, l, mask);
        const float nm = fmaxf(m, om);
        l = l * ptx_exp(m - nm) + ol * ptx_exp(om - nm);
        m = nm;
    }
    if ((threadIdx.x & 31) == 0) {
        red_v[threadIdx.x >> 5] = m;
        red_i[threadIdx.x >> 5] = __float_as_int(l);
    }
    __syncthreads();
    float* out = part + (static_cast<size_t>(r) * kDf2Chunks + ch) * 34;
    if (threadIdx.x == 0) {
        float bm = red_v[0], bl = __int_as_float(red_i[0]);
        for (int w = 1; w < 8; ++w) {
            const float om = red_v[w], ol = __int_as_float(red_i[w]);
            const float nm = fmaxf(bm, om);
            bl = bl * ptx_exp(bm - nm) + ol * ptx_exp(om - nm);
            bm = nm;
        }
        out[32] = bm;
        out[33] = bl;
    }
    __syncthreads();
    float last_v = 3.0e38f;
    int last_i = -1;
    for (int k = 0; k < kDf2Top; ++k) {
        df2_block_pick(sv, si, n, last_v, last_i, red_v, red_i, last_v, last_i);
        if (threadIdx.x == 0) {
            out[k] = last_v;
            out[16 + k] = __int_as_float(last_i);
        }
    }
}

// Stage 2: merges the 40 chunk Top-16 lists into the row's sorted Top-16 (ties to the lower id)
// and the chunk LSEs into the full-vocabulary softcapped `(m, l)`. Grid `rows`, Block `256`.
extern "C" __global__ void df2_top16_merge(
    const float* __restrict__ part,  // [rows, 40, 34]
    int32_t* __restrict__ cand,      // [rows, 32]: 16 ids, then 16 raw logit bit patterns
    float* __restrict__ lse) {       // [rows, 2]: full-vocabulary softcapped (m, l)
    pdl_wait();
    pdl_trigger();
    __shared__ float sv[kDf2Chunks * kDf2Top];
    __shared__ int si[kDf2Chunks * kDf2Top];
    __shared__ float red_v[8];
    __shared__ int red_i[8];
    const int r = blockIdx.x;
    const float* in = part + static_cast<size_t>(r) * kDf2Chunks * 34;
    for (int j = threadIdx.x; j < kDf2Chunks * kDf2Top; j += 256) {
        const int ch = j / kDf2Top, k = j % kDf2Top;
        sv[j] = in[ch * 34 + k];
        si[j] = __float_as_int(in[ch * 34 + 16 + k]);
    }
    if (threadIdx.x == 0) {
        float bm = -3.0e38f, bl = 0.0f;
        for (int ch = 0; ch < kDf2Chunks; ++ch) {
            const float om = in[ch * 34 + 32], ol = in[ch * 34 + 33];
            const float nm = fmaxf(bm, om);
            bl = bl * ptx_exp(bm - nm) + ol * ptx_exp(om - nm);
            bm = nm;
        }
        lse[r * 2] = bm;
        lse[r * 2 + 1] = bl;
    }
    __syncthreads();
    float last_v = 3.0e38f;
    int last_i = -1;
    for (int k = 0; k < kDf2Top; ++k) {
        df2_block_pick(sv, si, kDf2Chunks * kDf2Top, last_v, last_i, red_v, red_i, last_v, last_i);
        if (threadIdx.x == 0) {
            cand[r * 32 + k] = last_i;
            cand[r * 32 + 16 + k] = __float_as_int(last_v);
        }
    }
}

// Bilinear candidate selector: step d scores candidate c as
//   raw_logit(c) + sum_i (pred[prev][i] * g[d][i]) * succ[c][i]
// (ties to the lower id), then `prev = best`; `conf[d]` is the full-vocabulary probability of
// the pick with `cap * tanh(. / cap)` applied after the correction. Grid `1`, Block `512`
// (one warp per candidate).
extern "C" __global__ void df2_select(
    int32_t anchor,
    const int32_t* __restrict__ cand,     // [n, 32]
    const float* __restrict__ tail,       // [n, 2]: full-vocabulary softcapped (m, l)
    const float* __restrict__ g,          // [n, 256]
    const uint32_t* __restrict__ pred,    // [vocab, 128] BF16 pairs
    const uint32_t* __restrict__ succ,    // [vocab, 128] BF16 pairs
    int32_t n,
    float cap,
    int32_t* __restrict__ ids,            // [n]
    float* __restrict__ conf) {           // [n]
    pdl_wait();
    pdl_trigger();
    const float inv_cap = __fdividef(1.0f, cap); // fast reciprocal: no IEEE-division call
    __shared__ float s_sc[kDf2Top];
    __shared__ int s_id[kDf2Top];
    __shared__ int s_prev;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (threadIdx.x == 0) s_prev = anchor;
    __syncthreads();
    for (int d = 0; d < n; ++d) {
        const int prev = s_prev;
        const int cid = cand[d * 32 + warp];
        const float raw = __int_as_float(cand[d * 32 + 16 + warp]);
        float acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int pi = lane * 4 + j;  // pair index 0..127
            const uint32_t pp = pred[static_cast<size_t>(prev) * 128 + pi];
            const uint32_t sp = succ[static_cast<size_t>(cid) * 128 + pi];
            const float g0 = g[d * kDf2Rank + 2 * pi];
            const float g1 = g[d * kDf2Rank + 2 * pi + 1];
            acc += (bf16_to_f32(static_cast<uint16_t>(pp & 0xFFFFu)) * g0) *
                   bf16_to_f32(static_cast<uint16_t>(sp & 0xFFFFu));
            acc += (bf16_to_f32(static_cast<uint16_t>(pp >> 16)) * g1) *
                   bf16_to_f32(static_cast<uint16_t>(sp >> 16));
        }
        acc = warp_sum(acc);
        if (lane == 0) {
            s_sc[warp] = raw + acc;
            s_id[warp] = cid;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int best = 0;
            for (int c = 1; c < kDf2Top; ++c) {
                if (df2_before(s_sc[c], s_id[c], s_sc[best], s_id[best])) best = c;
            }
            // `tail = (m, l)` over the full vocabulary: swap the 16 candidates' target-only terms
            // for their corrected ones.
            const float top = cap * ptx_tanh(s_sc[best] * inv_cap);
            const float m_t = tail[d * 2], l_t = tail[d * 2 + 1];
            const float m_ref = fmaxf(top, m_t);
            float z = l_t * ptx_exp(m_t - m_ref);
            for (int c = 0; c < kDf2Top; ++c) {
                const float raw_c = __int_as_float(cand[d * 32 + 16 + c]);
                z += ptx_exp(cap * ptx_tanh(s_sc[c] * inv_cap) - m_ref) -
                     ptx_exp(cap * ptx_tanh(raw_c * inv_cap) - m_ref);
            }
            z = fmaxf(z, ptx_exp(top - m_ref));
            ids[d] = s_id[best];
            conf[d] = __fdividef(ptx_exp(top - m_ref), z);
            s_prev = s_id[best];
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------------------------
// Q4 copy of the target's output head for drafting (`LEANSONIC_DF2_HEAD=q4`, default): half the
// bytes of the Q8_64 head that dominates the draft. Each 64-column group of row `r` is requantized
// from Q8_64 to signed 4-bit codes `[-7, 7]` with one FP32 scale. A 32-column `uint4` packs byte
// `b` of word `k` as column `8k + b` (low nibble) and column `8k + 4 + b` (high nibble), so the
// two nibble planes of word `k` line up with Q8 activation words `2k` and `2k + 1`.
// ---------------------------------------------------------------------------------------------

// Grid `(rows, D / 64)`, Block `32` (one warp per group, lane = two columns of the group).
extern "C" __global__ void df2_q8_to_q4(
    const uint32_t* __restrict__ w_q8_u32,   // [rows, D / 4]
    const float* __restrict__ w_sc64,        // [rows, D / 64]
    uint32_t* __restrict__ w_q4_u32,         // [rows, D / 8]
    float* __restrict__ w_q4_sc64) {         // [rows, D / 64]
    pdl_wait();
    pdl_trigger();
    const size_t r = blockIdx.x;
    const int g = blockIdx.y;
    const int lane = threadIdx.x;
    // Lane `l` holds columns `2l, 2l + 1` of the group.
    const uint32_t word = w_q8_u32[r * (kD / 4) + g * 16 + (lane >> 1)];
    const int sh = (lane & 1) * 16;
    const int q0 = static_cast<int8_t>((word >> sh) & 0xFFu);
    const int q1 = static_cast<int8_t>((word >> (sh + 8)) & 0xFFu);
    int amax = max(abs(q0), abs(q1));
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) amax = max(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask));
    const float inv = (amax > 0) ? 7.0f * ptx_rcp(static_cast<float>(amax)) : 0.0f;
    const int n0 = min(7, max(-7, __float2int_rn(static_cast<float>(q0) * inv))) & 0xF;
    const int n1 = min(7, max(-7, __float2int_rn(static_cast<float>(q1) * inv))) & 0xF;
    // Column `c = 2 * lane + j` of the group: 32-column half `h = c / 32`, word `k = (c % 32) / 8`,
    // nibble plane `p = (c % 8) / 4`, byte `b = c % 4`.
    uint32_t* out = w_q4_u32 + r * (kD / 8) + g * 8;
    auto put = [&](int c, int n) { return static_cast<uint32_t>(n) << ((((c & 3) * 8) + ((c >> 2) & 1) * 4)); };
    const int c0 = 2 * lane, c1 = 2 * lane + 1;
    // Both columns of a lane share a word (c0 even, c1 = c0 + 1 within the same 4-column byte run).
    uint32_t v = put(c0, n0) | put(c1, n1);
    // OR-reduce the lanes that share word `(c / 8)`: lanes `4m .. 4m + 3`.
    v |= __shfl_xor_sync(0xFFFFFFFFu, v, 1);
    v |= __shfl_xor_sync(0xFFFFFFFFu, v, 2);
    if ((lane & 3) == 0) out[lane >> 2] = v;
    if (lane == 0) w_q4_sc64[r * (kD / 64) + g] = w_sc64[r * (kD / 64) + g] * static_cast<float>(amax) * (1.0f / 7.0f);
}

// Sign-extends the two nibble planes of a packed word into int8 words (`lo`: columns `8k..8k+3`,
// `hi`: `8k+4..8k+7`).
__device__ __forceinline__ void df2_unpack_q4(uint32_t w, int& lo, int& hi) {
    const uint32_t m = 0x0F0F0F0Fu, b8 = 0x08080808u;
    lo = static_cast<int>(__vsub4((w & m) ^ b8, b8));
    hi = static_cast<int>(__vsub4(((w >> 4) & m) ^ b8, b8));
}

// `y[t][r] = w_q4[r] . x_q8[t]` for `B <= 8` rows of `D` columns (the draft head). Grid-stride
// over 16-row tiles (8 warps x 2 rows) so each block stages `x` in shared memory once; the next
// tile's weights are loaded while the current one is computed. Grid `<= ceil(rows / 16)`,
// Block `256`.
extern "C" __global__ void __launch_bounds__(256) df2_q4_head(
    const uint4* __restrict__ w_q4,          // [rows_padded, D / 32]
    const float* __restrict__ w_sc64,        // [rows_padded, D / 64]
    const uint4* __restrict__ x_q8,          // [B, D / 16]
    const float* __restrict__ x_sc64,        // [B, D / 64]
    float* __restrict__ y,                   // [B, y_stride]
    int32_t rows,
    int32_t B,
    int32_t y_stride) {
    __shared__ uint4 sx[8][kD / 16];         // 18,432 B
    __shared__ float sxs[8][kD / 64];
    constexpr int kC32 = kD / 32, kC64 = kD / 64; // 72, 36
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tiles = (rows + 15) >> 4;
    uint4 w0[3], w1[3];
    float ws0[3], ws1[3];
    auto load = [&](int tile) {
        const int row0 = (tile * 8 + warp) * 2;
        const uint4* w0p = w_q4 + static_cast<size_t>(row0) * kC32;
        const float* s0p = w_sc64 + static_cast<size_t>(row0) * kC64;
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            const int c32 = lane + 32 * i;
            if (c32 < kC32) {
                w0[i] = w0p[c32];
                w1[i] = w0p[kC32 + c32];
                ws0[i] = s0p[c32 >> 1];
                ws1[i] = s0p[kC64 + (c32 >> 1)];
            }
        }
    };
    int tile = blockIdx.x;
    if (tile < tiles) load(tile);
    pdl_wait();
    pdl_trigger();
    for (int i = threadIdx.x; i < B * (kD / 16); i += 256) {
        const int t = i / (kD / 16), j = i - t * (kD / 16);
        sx[t][j] = x_q8[i];
    }
    for (int i = threadIdx.x; i < B * kC64; i += 256) {
        const int t = i / kC64, j = i - t * kC64;
        sxs[t][j] = x_sc64[i];
    }
    __syncthreads();
    for (; tile < tiles; tile += gridDim.x) {
        const int row0 = (tile * 8 + warp) * 2;
        int l0[3][8], l1[3][8];
        float s0[3], s1[3];
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            const uint32_t a4[4] = {w0[i].x, w0[i].y, w0[i].z, w0[i].w};
            const uint32_t b4[4] = {w1[i].x, w1[i].y, w1[i].z, w1[i].w};
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                df2_unpack_q4(a4[k], l0[i][2 * k], l0[i][2 * k + 1]);
                df2_unpack_q4(b4[k], l1[i][2 * k], l1[i][2 * k + 1]);
            }
            s0[i] = ws0[i];
            s1[i] = ws1[i];
        }
        if (tile + static_cast<int>(gridDim.x) < tiles) load(tile + gridDim.x);
        float a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float b[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            const int c32 = lane + 32 * i;
            if (c32 < kC32) {
                #pragma unroll
                for (int t = 0; t < 8; ++t) {
                    if (t < B) {
                        const uint4 xa = sx[t][2 * c32], xb = sx[t][2 * c32 + 1];
                        const int xw[8] = {static_cast<int>(xa.x), static_cast<int>(xa.y),
                                           static_cast<int>(xa.z), static_cast<int>(xa.w),
                                           static_cast<int>(xb.x), static_cast<int>(xb.y),
                                           static_cast<int>(xb.z), static_cast<int>(xb.w)};
                        int d0 = 0, d1 = 0;
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            d0 = __dp4a(l0[i][k], xw[k], d0);
                            d1 = __dp4a(l1[i][k], xw[k], d1);
                        }
                        const float xs = sxs[t][c32 >> 1];
                        a[t] = __fmaf_rn(static_cast<float>(d0), s0[i] * xs, a[t]);
                        b[t] = __fmaf_rn(static_cast<float>(d1), s1[i] * xs, b[t]);
                    }
                }
            }
        }
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < B) {
                const float v0 = warp_sum(a[t]);
                const float v1 = warp_sum(b[t]);
                if (lane == 0) {
                    if (row0 < rows) y[static_cast<size_t>(t) * y_stride + row0] = v0;
                    if (row0 + 1 < rows) y[static_cast<size_t>(t) * y_stride + row0 + 1] = v1;
                }
            }
        }
    }
}

} // namespace leansonic::inkling2_df2
