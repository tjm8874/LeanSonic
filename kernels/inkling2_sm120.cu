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

// PDL contract (#152 P0-B): every `extern "C"` kernel below executes `pdl_wait()` before its
// first access to global memory that a preceding kernel may write (reads of constant weights
// may come first).  The host therefore launches *every* kernel with programmatic stream
// serialization when the cubin exports `ink2_pdl_all_marker`, hiding the launch gap between
// dependent kernels in the verify, drafter and prefill paths as well as single-sequence decode.
// `crates/leansonic-core` has a unit test that rejects a kernel without `pdl_wait()`.
//
// The speculative-verify and drafter kernels also call `pdl_trigger()` right after the wait, so
// their successor becomes resident early and loads its constant weights before its own wait
// (`ink2_q8_dp4a_gemm_b8_fast`, `ink2_bf16_gemv_router_b`, `ctrl_down_rows`).  Kernels shared
// with prefill, C=64 rounds and single-sequence decode keep their existing trigger points: an
// early trigger there measured 1-2% slower.
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

// Exported only so the host can tell that this cubin honours the PDL contract above.
extern "C" __global__ void ink2_pdl_all_marker() {
    pdl_wait();
}

// Breaks `ptxas` parameter-space provenance (`c[0x0][...]`) on mutable input pointers after `pdl_wait()`,
// guaranteeing (1) coherent `LDG.E` instead of non-coherent `LDG.E.CONSTANT`, and (2) strict SSA ordering
// after `ACQBULK` (`griddepcontrol.wait;`) so mutable loads are never hoisted into the PDL overlap window.
template <typename T>
__device__ __forceinline__ const T* pdl_in(const T* p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("" : "+l"(p) :: "memory");
#endif
    return p;
}

template <typename T>
__device__ __forceinline__ T* pdl_in_mut(T* p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("" : "+l"(p) :: "memory");
#endif
    return p;
}

__device__ __forceinline__ int32_t ld_ca_i32(const int32_t* p) {
#if defined(__CUDA_ARCH__)
    int32_t v;
    asm volatile("ld.global.ca.s32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
#else
    return *p;
#endif
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

// Raw FP16 KV word (4 values) and its unpacking, matching `kv_load4`'s FP16 branch.
__device__ __forceinline__ uint64_t kv_raw_f16(
    const uint16_t* __restrict__ cache, int row, int kvh, int lane) {
    return reinterpret_cast<const uint64_t*>(
        cache + static_cast<size_t>(row) * 1024u + kvh * 128)[lane];
}

__device__ __forceinline__ void kv_unpack_f16(
    uint64_t w, float& x0, float& x1, float& x2, float& x3) {
    x0 = fp16_bits_to_f32(static_cast<uint16_t>(w & 0xFFFFu));
    x1 = fp16_bits_to_f32(static_cast<uint16_t>((w >> 16) & 0xFFFFu));
    x2 = fp16_bits_to_f32(static_cast<uint16_t>((w >> 32) & 0xFFFFu));
    x3 = fp16_bits_to_f32(static_cast<uint16_t>((w >> 48) & 0xFFFFu));
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
    uint32_t nw_reg[5];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        nw_reg[t] = norm_bf16_pairs[tid + t * 256];
    }
    nw_reg[4] = (tid < 128) ? norm_bf16_pairs[tid + 1024] : 0u;

    pdl_wait();
    pdl_trigger();

    token_buf_ptr = pdl_in(token_buf_ptr);
    pos_idx_ptr = pdl_in(pos_idx_ptr);
    const int32_t token_id = use_pos_idx ? token_buf_ptr[*pos_idx_ptr] : token_buf_ptr[0];
    const uint32_t* row = embed_bf16_pairs + static_cast<size_t>(token_id) * (kD / 2);

    float v0_reg[5], v1_reg[5];
    float sq = 0.0f;
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        uint32_t pair = row[tid + t * 256];
        float v0 = bf16_to_f32(static_cast<uint16_t>(pair & 0xFFFFu));
        float v1 = bf16_to_f32(static_cast<uint16_t>(pair >> 16));
        v0_reg[t] = v0;
        v1_reg[t] = v1;
        sq = __fadd_rn(sq, __fmaf_rn(v0, v0, __fmul_rn(v1, v1)));
    }
    if (tid < 128) {
        uint32_t pair = row[tid + 1024];
        float v0 = bf16_to_f32(static_cast<uint16_t>(pair & 0xFFFFu));
        float v1 = bf16_to_f32(static_cast<uint16_t>(pair >> 16));
        v0_reg[4] = v0;
        v1_reg[4] = v1;
        sq = __fadd_rn(sq, __fmaf_rn(v0, v0, __fmul_rn(v1, v1)));
    } else {
        v0_reg[4] = 0.0f;
        v1_reg[4] = 0.0f;
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
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int idx = tid + t * 256;
        out_x[idx * 2] = __fmul_rn(v0_reg[t], __fmul_rn(scale, bf16_to_f32(static_cast<uint16_t>(nw_reg[t] & 0xFFFFu))));
        out_x[idx * 2 + 1] = __fmul_rn(v1_reg[t], __fmul_rn(scale, bf16_to_f32(static_cast<uint16_t>(nw_reg[t] >> 16))));
    }
    if (tid < 128) {
        const int idx = tid + 1024;
        out_x[idx * 2] = __fmul_rn(v0_reg[4], __fmul_rn(scale, bf16_to_f32(static_cast<uint16_t>(nw_reg[4] & 0xFFFFu))));
        out_x[idx * 2 + 1] = __fmul_rn(v1_reg[4], __fmul_rn(scale, bf16_to_f32(static_cast<uint16_t>(nw_reg[4] >> 16))));
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
    uint32_t gw_reg[5];
    if (has_gamma) {
        #pragma unroll
        for (int t = 0; t < 4; ++t) gw_reg[t] = gamma_bf16_pairs[tid + t * 256];
        gw_reg[4] = (tid < 128) ? gamma_bf16_pairs[tid + 1024] : 0u;
    }

    pdl_wait();
    pdl_trigger();
    inp = pdl_in(inp);

    float sq = 0.0f;
    for (int i = tid; i < kD; i += 256) {
        float v = inp[i];
        sq = __fmaf_rn(v, v, sq);
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
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int idx = tid + t * 256;
            uint32_t gw = gw_reg[t];
            float x0 = inp[idx * 2];
            float x1 = inp[idx * 2 + 1];
            out[idx * 2] = __fmul_rn(__fmul_rn(x0, k), bf16_to_f32(static_cast<uint16_t>(gw & 0xFFFFu)));
            out[idx * 2 + 1] = __fmul_rn(__fmul_rn(x1, k), bf16_to_f32(static_cast<uint16_t>(gw >> 16)));
        }
        if (tid < 128) {
            const int idx = tid + 1024;
            uint32_t gw = gw_reg[4];
            float x0 = inp[idx * 2];
            float x1 = inp[idx * 2 + 1];
            out[idx * 2] = __fmul_rn(__fmul_rn(x0, k), bf16_to_f32(static_cast<uint16_t>(gw & 0xFFFFu)));
            out[idx * 2 + 1] = __fmul_rn(__fmul_rn(x1, k), bf16_to_f32(static_cast<uint16_t>(gw >> 16)));
        }
    } else {
        for (int i = tid; i < kD; i += 256) {
            out[i] = __fmul_rn(inp[i], k);
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

    uint32_t gw_reg[5];
    #pragma unroll
    for (int it = 0; it < 4; ++it) {
        gw_reg[it] = gamma[(warp_id + it * 8) * 32 + lane];
    }
    gw_reg[4] = (warp_id < 4) ? gamma[(warp_id + 32) * 32 + lane] : 0u;

    pdl_wait();
    pdl_trigger();
    inp = pdl_in(inp);

    float sq = 0.0f;
    #pragma unroll
    for (int t = 0; t < 9; ++t) {
        float v = inp[tid + t * 256];
        sq = __fmaf_rn(v, v, sq);
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
    #pragma unroll
    for (int it = 0; it < 5; ++it) {
        const int blk = warp_id + it * 8;
        if (blk < (kD / 64)) {
            uint32_t gw = gw_reg[it];
            int base = blk * 64 + lane * 2;
            float y0 = __fmul_rn(__fmul_rn(inp[base], k), bf16_to_f32(static_cast<uint16_t>(gw & 0xFFFFu)));
            float y1 = __fmul_rn(__fmul_rn(inp[base + 1], k), bf16_to_f32(static_cast<uint16_t>(gw >> 16)));
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
}

// ----------------------------------------------------------------------------
// 3. Depthwise Attention-Residual `mix(states, queries)` (`n_states <= 6`, `n_q <= 4`)
//    Fused dual `(sq, dq)` reduction per state + optional `broadcast3`!
// ----------------------------------------------------------------------------
__device__ __forceinline__ void depth_mix_body(
    const float* __restrict__ blocks_buf,
    int32_t n_blocks,
    const float* __restrict__ cur_opt,
    int32_t append_cur,
    const float* __restrict__ queries,
    float* __restrict__ out_mixed,
    int32_t broadcast3,
    float* __restrict__ out_inv) {
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

    pdl_wait();
    pdl_trigger();
    blocks_buf = pdl_in(blocks_buf);
    cur_opt = pdl_in(cur_opt);

    float st_reg[6][9];
    #pragma unroll
    for (int s = 0; s < 6; ++s) {
        if (s < total_states) {
            const float* st = (s < n_blocks) ? (blocks_buf + static_cast<size_t>(s) * kD) : cur_opt;
            #pragma unroll
            for (int t = 0; t < 9; ++t) {
                st_reg[s][t] = st[tid + t * 256];
            }
        }
    }

    #pragma unroll
    for (int s = 0; s < 6; ++s) {
        if (s < total_states) {
            float sq = 0.0f;
            float dq = 0.0f;
            #pragma unroll
            for (int t = 0; t < 9; ++t) {
                float v = st_reg[s][t];
                sq = __fmaf_rn(v, v, sq);
                dq = __fmaf_rn(v, q_reg[t], dq);
            }
            sq = warp_sum(sq);
            dq = warp_sum(dq);
            if (lane == 0) {
                smem_sq[s][warp_id] = sq;
                smem_dq[s][warp_id] = dq;
            }
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
    float sq_out = 0.0f;
    #pragma unroll
    for (int t = 0; t < 9; ++t) {
        const int i = tid + t * 256;
        float acc = 0.0f;
        #pragma unroll
        for (int s = 0; s < 6; ++s) {
            if (s < total_states) {
                acc = __fmaf_rn(smem_probs[s], st_reg[s][t], acc);
            }
        }
        dst[i] = acc;
        if (broadcast3) {
            dst[kD + i] = acc;
            dst[2 * kD + i] = acc;
        }
        if (out_inv != nullptr) {
            sq_out = __fmaf_rn(acc, acc, sq_out);
        }
    }
    if (out_inv != nullptr) {
        sq_out = warp_sum(sq_out);
        if (lane == 0) smem_sq[0][warp_id] = sq_out;
        __syncthreads();
        if (warp_id == 0) {
            float sa = (lane < 8) ? smem_sq[0][lane] : 0.0f;
            sa = warp_sum(sa);
            if (lane == 0) {
                float inv = ptx_rsqrt(sa * (1.0f / static_cast<float>(kD)) + 1e-6f);
                out_inv[q_idx] = inv;
                if (broadcast3) {
                    out_inv[1] = inv;
                    out_inv[2] = inv;
                }
            }
        }
    }
}

extern "C" __global__ void ink2_depth_mix(
    const float* __restrict__ blocks_buf, // [max_blocks, D]
    int32_t n_blocks,
    const float* __restrict__ cur_opt,    // [D] if `append_cur == 1`
    int32_t append_cur,
    const float* __restrict__ queries,    // [n_q, D]
    float* __restrict__ out_mixed,        // [n_q, D] (or `[3, D]` if `broadcast3 == 1`)
    int32_t broadcast3) {
    pdl_wait();
    depth_mix_body(blocks_buf, n_blocks, cur_opt, append_cur, queries, out_mixed, broadcast3, nullptr);
}

extern "C" __global__ void ink2_depth_mix_inv(
    const float* __restrict__ blocks_buf, // [max_blocks, D]
    int32_t n_blocks,
    const float* __restrict__ cur_opt,    // [D] if `append_cur == 1`
    int32_t append_cur,
    const float* __restrict__ queries,    // [n_q, D]
    float* __restrict__ out_mixed,        // [n_q, D] (or `[3, D]` if `broadcast3 == 1`)
    int32_t broadcast3,
    float* __restrict__ out_inv) {        // [n_q] (or `[3]` if `broadcast3 == 1`)
    pdl_wait();
    depth_mix_body(blocks_buf, n_blocks, cur_opt, append_cur, queries, out_mixed, broadcast3, out_inv);
}

// Zero-Prologue-Barrier 4-Rows/CTA Controller Down (`Grid: (16, n_ctrls)`, `Block: 256`):
// Consumes precomputed `0-ULP` identical `(inv_cur, inv_msg)` from `ink2_depth_mix_inv`, hoists all
// `4` rows' BF16 weights before `pdl_wait()`, and shares `cur` + `msg` loads `4x` with 1 barrier!
extern "C" __global__ void ink2_ctrl_down_r4_fast(
    const float* __restrict__ cur_base,       // [n_ctrls, D]
    const float* __restrict__ msg_base,       // [n_ctrls, D]
    const float* __restrict__ inv_cur,        // [n_ctrls]
    const float* __restrict__ inv_msg,        // [n_ctrls]
    const uint4* __restrict__ down_bf16_u128, // [n_ctrls, 64, 576]
    float* __restrict__ g_out_base) {         // [n_ctrls, 64]
    constexpr int kRows = 4;
    __shared__ float partial[kRows][8];

    const int first_row = blockIdx.x * kRows; // 0..60
    const int c_idx = blockIdx.y;             // 0..n_ctrls-1
    const int tid = threadIdx.x;              // 0..255
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    uint4 w_main[kRows][2];
    uint4 w_tail[kRows];
    const bool has_tail = (tid < 64);
    #pragma unroll
    for (int r = 0; r < kRows; ++r) {
        const uint4* w_row = down_bf16_u128 + (static_cast<size_t>(c_idx) * 64u + first_row + r) * 576u;
        w_main[r][0] = w_row[tid];
        w_main[r][1] = w_row[tid + 256];
        w_tail[r] = has_tail ? w_row[tid + 512] : make_uint4(0u, 0u, 0u, 0u);
    }

    pdl_wait();
    pdl_trigger();
    cur_base = pdl_in(cur_base);
    msg_base = pdl_in(msg_base);
    inv_cur = pdl_in(inv_cur);
    inv_msg = pdl_in(inv_msg);

    const float* cur = cur_base + static_cast<size_t>(c_idx) * kD;
    const float* msg = msg_base + static_cast<size_t>(c_idx) * kD;
    const float ka = inv_cur[c_idx];
    const float kb = inv_msg[c_idx];

    float dot[kRows] = {0.0f, 0.0f, 0.0f, 0.0f};
    auto accum_c8 = [&](int it_idx, int c8, const uint4 w_step[kRows]) {
        (void)it_idx;
        const int base = c8 * 8;
        const float* src = (base < kD) ? (cur + base) : (msg + (base - kD));
        const float k_scale = (base < kD) ? ka : kb;
        float s_vals[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) s_vals[i] = src[i];
        #pragma unroll
        for (int r = 0; r < kRows; ++r) {
            uint32_t wu[4] = {w_step[r].x, w_step[r].y, w_step[r].z, w_step[r].w};
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                float w0 = bf16_to_f32(static_cast<uint16_t>(wu[p] & 0xFFFFu));
                float w1 = bf16_to_f32(static_cast<uint16_t>(wu[p] >> 16));
                float pair_dot = __fmaf_rn(w0, s_vals[p * 2], __fmul_rn(w1, s_vals[p * 2 + 1]));
                dot[r] = __fmaf_rn(pair_dot, k_scale, dot[r]);
            }
        }
    };

    {
        uint4 ws0[kRows] = {w_main[0][0], w_main[1][0], w_main[2][0], w_main[3][0]};
        accum_c8(0, tid, ws0);
    }
    {
        uint4 ws1[kRows] = {w_main[0][1], w_main[1][1], w_main[2][1], w_main[3][1]};
        accum_c8(1, tid + 256, ws1);
    }
    if (has_tail) {
        accum_c8(2, tid + 512, w_tail);
    }

    #pragma unroll
    for (int r = 0; r < kRows; ++r) {
        float s = warp_sum(dot[r]);
        if (lane == 0) partial[r][warp_id] = s;
    }
    __syncthreads();
    if (warp_id < kRows) {
        float s = (lane < 8) ? partial[warp_id][lane] : 0.0f;
        s = warp_sum(s);
        if (lane == 0) g_out_base[c_idx * 64 + first_row + warp_id] = s;
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
    pdl_wait();
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
    pdl_wait();
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
    // One warp per row with any multiple of 32 threads per CTA: the 128-row router launches one
    // warp per CTA so its rows spread over many SMs instead of 16.
    const int row = blockIdx.x * (blockDim.x >> 5) + warp_id;
    const int cols8 = cols >> 3;
    const uint4* w_row = w_bf16_u128 + static_cast<size_t>(row < rows ? row : 0) * cols8;

    if (cols8 == 288) {
        // D = 2304 (router): issue all 9 weight loads of this lane before `pdl_wait()` and the FMA
        // chain, keeping the same per-lane order (c8 = lane + 32 i) and therefore the same result.
        constexpr int kIters = 288 / 32;
        uint4 w[kIters];
        if (row < rows) {
            #pragma unroll
            for (int i = 0; i < kIters; ++i) w[i] = w_row[lane + 32 * i];
        }
        pdl_wait();
        const float4* x4 = pdl_in(reinterpret_cast<const float4*>(x));
        float acc = 0.0f;
        if (row < rows) {
            #pragma unroll
            for (int i = 0; i < kIters; ++i) {
                const int c8 = lane + 32 * i;
                acc = accum_bf16_8(acc, w[i], x4[c8 * 2], x4[c8 * 2 + 1]);
            }
            acc = warp_sum(acc);
        }
        pdl_trigger();
        if (row < rows && lane == 0) {
            if (softcap > 0.0f) {
                acc = softcap * ptx_tanh(acc * ptx_rcp(softcap));
            }
            y[row] = acc;
        }
        return;
    }

    pdl_wait();
    const float4* x4 = pdl_in(reinterpret_cast<const float4*>(x));
    float acc = 0.0f;
    if (row < rows) {
        for (int c8 = lane; c8 < cols8; c8 += 32) {
            acc = accum_bf16_8(acc, w_row[c8], x4[c8 * 2], x4[c8 * 2 + 1]);
        }
        acc = warp_sum(acc);
    }
    pdl_trigger();
    if (row < rows && lane == 0) {
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
    const float4 w = (c < channels) ? w_conv4[c] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    pdl_wait();
    pdl_trigger();
    if (c >= channels) return;
    cur = pdl_in(cur);
    dyn_tap = pdl_in(dyn_tap);
    hist3 = pdl_in_mut(hist3);
    res_add = pdl_in(res_add);

    const float x = cur[c];
    float* h = hist3 + c * 3;
    const float h0 = h[0];
    const float h1 = h[1];
    const float h2 = h[2];
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

    uint32_t cp0 = 0u, cp1 = 0u, rp = 0u;
    if (g_warp < 24) {
        const uint32_t* qc = q_clip_bf16 + g_warp * 64;
        cp0 = qc[lane * 2];
        cp1 = qc[lane * 2 + 1];
        if (lane < 8) {
            rp = (r_clip_bf16 + g_warp * 8)[lane];
        }
    }

    pdl_wait();
    pdl_trigger();
    q_buf = pdl_in_mut(q_buf);
    r_buf = pdl_in_mut(r_buf);
    k_in = pdl_in(k_in);
    v_in = pdl_in(v_in);
    pos_ptr = pdl_in(pos_ptr);

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
        float rv0 = 0.0f, rv1 = 0.0f;
        if (lane < 8) {
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
    pdl_wait();
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
    const float b0 = bias128[base + 0];
    const float b1 = bias128[base + 1];
    const float b2 = bias128[base + 2];
    const float b3 = bias128[base + 3];

    pdl_wait();
    pdl_trigger();
    logits128 = pdl_in(logits128);

    float p0 = ptx_sigmoid(logits128[base + 0]);
    float p1 = ptx_sigmoid(logits128[base + 1]);
    float p2 = ptx_sigmoid(logits128[base + 2]);
    float p3 = ptx_sigmoid(logits128[base + 3]);
    float s0 = p0 + b0;
    float s1 = p1 + b1;
    float s2 = p2 + b2;
    float s3 = p3 + b3;

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

// Fused 128-CTA BF16 Router GEMV (`[128, 2304]`) + Last-CTA Warp-Butterfly Top-8 (`Grid: 128`, `Block: 32`):
// Eliminates 48 kernel launches + 48 inter-kernel `ACQBULK` drains per token (`0-ULP` exact)!
// Each CTA (`1` warp) preloads all 9 BF16 `uint4` weight vectors AND `bias128` before `pdl_wait()`,
// writes `logits128[row]`, and synchronizes via GPU-scope `fence.acq_rel.gpu` + `atom.global.gpu.add.s32`
// on self-resetting `done_cnt` (`0 -> 128 -> 0`, CUDA-Graph replay safe!).
extern "C" __global__ void ink2_router_gemv_topk8(
    const uint4* __restrict__ w_bf16_u128, // [128, 288]
    const float* __restrict__ x,           // [2304]
    float* __restrict__ logits128,         // [128]
    const float* __restrict__ bias128,     // [128]
    int32_t* __restrict__ sel_ids,         // [8]
    float* __restrict__ sel_wts,           // [8]
    int32_t* __restrict__ done_cnt) {      // [1], zero-initialized & self-resetting
    __shared__ int smem_ids[8];
    __shared__ float smem_ps[8];

    const int row = blockIdx.x;            // 0..127
    const int lane = threadIdx.x & 31;     // 0..31
    const uint4* w_row = w_bf16_u128 + static_cast<size_t>(row) * 288u;

    constexpr int kIters = 288 / 32;       // 9
    uint4 w[kIters];
    #pragma unroll
    for (int i = 0; i < kIters; ++i) {
        w[i] = w_row[lane + 32 * i];
    }
    const int base = lane * 4;
    const float4 b4 = reinterpret_cast<const float4*>(bias128)[lane];

    pdl_wait();
    const float4* x4 = pdl_in(reinterpret_cast<const float4*>(x));
    float acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < kIters; ++i) {
        const int c8 = lane + 32 * i;
        acc = accum_bf16_8(acc, w[i], x4[c8 * 2], x4[c8 * 2 + 1]);
    }
    acc = warp_sum(acc);

    int is_last = 0;
    if (lane == 0) {
        logits128[row] = acc;
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        int prev = 0;
        asm volatile("atom.global.gpu.add.s32 %0, [%1], 1;"
                     : "=r"(prev) : "l"(done_cnt) : "memory");
        if (prev == 127) {
            is_last = 1;
        }
    }
    is_last = __shfl_sync(0xFFFFFFFFu, is_last, 0);
    if (!is_last) {
        return;
    }

    asm volatile("fence.acq_rel.gpu;" ::: "memory");
    if (lane == 0) {
        asm volatile("st.global.wb.s32 [%0], 0;" :: "l"(done_cnt) : "memory");
    }

    float l0, l1, l2, l3;
    asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(l0), "=f"(l1), "=f"(l2), "=f"(l3)
                 : "l"(logits128 + base)
                 : "memory");

    float p0 = ptx_sigmoid(l0);
    float p1 = ptx_sigmoid(l1);
    float p2 = ptx_sigmoid(l2);
    float p3 = ptx_sigmoid(l3);
    float s0 = p0 + b4.x;
    float s1 = p1 + b4.y;
    float s2 = p2 + b4.z;
    float s3 = p3 + b4.w;

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
    pdl_trigger();
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
    extern "C" __global__ void name params { pdl_wait(); name##_impl<false> args; } \
    extern "C" __global__ void name##_packed_scales params { pdl_wait(); name##_impl<true> args; } \
    template<bool PackedScales> __device__ __forceinline__ void name##_impl params

#define INK2_SCALE_KERNEL_LB(name, lb_threads, lb_blocks, params, args) \
    template<bool PackedScales> __device__ __forceinline__ void name##_impl params; \
    extern "C" __global__ __launch_bounds__(lb_threads, lb_blocks) void name params { pdl_wait(); name##_impl<false> args; } \
    extern "C" __global__ __launch_bounds__(lb_threads, lb_blocks) void name##_packed_scales params { pdl_wait(); name##_impl<true> args; } \
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

template<bool PackedScales, typename WeightRow>
__device__ __forceinline__ float warp_dot_mxfp8_u128_row(
    WeightRow row_fp8_u128,     // [n_u128]
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

// PRMT indexes eight bytes. Select between the two eight-entry table halves per byte.
__device__ __forceinline__ unsigned expand_four_indices(uint4 table, unsigned indices) {
    const unsigned lanes = indices & 0x7777u;
    const unsigned upper = __byte_perm(0u, ~0u, (indices >> 1) & 0x4444u);
    const unsigned lo = __byte_perm(table.x, table.y, lanes);
    const unsigned hi = __byte_perm(table.z, table.w, lanes);
    return (upper & hi) | (~upper & lo);
}

__device__ __forceinline__ uint4 load_indexed_mxfp8(const uint4* plane, uint4 table, size_t index) {
    const uint2 packed = reinterpret_cast<const uint2*>(plane + 1)[index];
    return make_uint4(expand_four_indices(table, packed.x), expand_four_indices(table, packed.x >> 16),
                      expand_four_indices(table, packed.y), expand_four_indices(table, packed.y >> 16));
}

// The remaining MXFP8 paths use the same logical 16-weight vector offsets for
// either storage format. In the indexed specialization only the payload is halved.
template<bool Indexed>
struct Mxfp8Row {
    const uint4* plane;
    size_t base;
    uint4 table;

    __device__ Mxfp8Row(const uint4* weights, size_t offset) : plane(weights), base(offset) {
        if constexpr (Indexed) table = weights[0];
    }
    __device__ __forceinline__ uint4 operator[](unsigned column) const {
        if constexpr (Indexed) return load_indexed_mxfp8(plane, table, base + column);
        return plane[base + column];
    }
};

#define INK2_INDEXED_KERNEL(name, bounds, params, args) \
    template<bool PackedScales, bool Indexed> __device__ __forceinline__ void name##_impl params; \
    extern "C" __global__ bounds void name params { pdl_wait(); name##_impl<false, false> args; } \
    extern "C" __global__ bounds void name##_packed_scales params { pdl_wait(); name##_impl<true, false> args; } \
    extern "C" __global__ bounds void name##_u4 params { pdl_wait(); name##_impl<false, true> args; } \
    extern "C" __global__ bounds void name##_u4_packed_scales params { pdl_wait(); name##_impl<true, true> args; } \
    template<bool PackedScales, bool Indexed> __device__ __forceinline__ void name##_impl params

template<bool PackedScales, bool Indexed>
__device__ __forceinline__ void mxfp8_up_situ_body(
    const uint4* __restrict__ w13_u128,    // [128, 2304, 144]
    const uint32_t* __restrict__ s13_u32,  // [128, 2304, 18]
    const float4* __restrict__ x4,         // [576]
    const int32_t* __restrict__ sel_ids,   // [8]
    float* __restrict__ act_out) {
    __shared__ float4 smem_x4[576];
    __shared__ float smem_gu[8];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int k = blockIdx.y;
    const int m = blockIdx.x * 4 + (warp_id >> 1);
    const int is_up = warp_id & 1;
    const int r = m * 2 + is_up;

    pdl_wait();
    x4 = pdl_in(x4);
    sel_ids = pdl_in(sel_ids);

    smem_x4[tid] = x4[tid];
    smem_x4[tid + 256] = x4[tid + 256];
    if (tid < 64) {
        smem_x4[tid + 512] = x4[tid + 512];
    }
    const int ex = sel_ids[k];
    const size_t row_idx = static_cast<size_t>(ex) * (2u * kFf) + r;
    uint4 table;
    if constexpr (Indexed) table = w13_u128[0];
    auto load_weight = [&](unsigned column) -> uint4 {
        const size_t idx = row_idx * 144u + column;
        if constexpr (Indexed) {
            return load_indexed_mxfp8(w13_u128, table, idx);
        } else {
            return w13_u128[idx];
        }
    };
    const unsigned sc_start = static_cast<unsigned>(row_idx * 18u);

    uint4 wp[4];
    float esc[4];
    #pragma unroll
    for (int it = 0; it < 4; ++it) {
        const int u16 = lane + 32 * it;
        wp[it] = load_weight(u16);
        const int blk32 = u16 >> 1;
        uint32_t s4 = expert_scale_word<PackedScales, 3, 112>(s13_u32, sc_start + (blk32 >> 2));
        esc[it] = __uint_as_float(((s4 >> ((blk32 & 3) * 8)) & 0xFFu) << 23);
    }
    uint4 wp4 = make_uint4(0u, 0u, 0u, 0u);
    float esc4 = 0.0f;
    if (lane < 16) {
        const int u16 = lane + 128;
        wp4 = load_weight(u16);
        const int blk32 = u16 >> 1;
        uint32_t s4 = expert_scale_word<PackedScales, 3, 112>(s13_u32, sc_start + (blk32 >> 2));
        esc4 = __uint_as_float(((s4 >> ((blk32 & 3) * 8)) & 0xFFu) << 23);
    }
    __syncthreads();

    float acc = 0.0f;
    #pragma unroll
    for (int it = 0; it < 4; ++it) {
        acc = __fmaf_rn(dot_mxfp8_u128_exact(wp[it], smem_x4 + (lane + 32 * it) * 4), esc[it], acc);
    }
    if (lane < 16) {
        acc = __fmaf_rn(dot_mxfp8_u128_exact(wp4, smem_x4 + (lane + 128) * 4), esc4, acc);
    }
    float dot = warp_sum(acc);
    if (lane == 0) {
        smem_gu[warp_id] = dot;
    }
    __syncthreads();
    pdl_trigger();
    if (lane == 0 && is_up == 0) {
        float g = smem_gu[warp_id];
        float u = smem_gu[warp_id + 1];
        float situ = 4.0f * ptx_tanh(g * 0.25f) * ptx_sigmoid(g) * 25.0f * ptx_tanh(u * 0.04f);
        act_out[k * kFf + m] = situ;
    }
}

INK2_SCALE_KERNEL(ink2_moe_mxfp8_up_situ, (    const uint4* __restrict__ w13_u128,    // [128, 2304, 144]
    const uint32_t* __restrict__ s13_u32,  // [128, 2304, 18]
    const float4* __restrict__ x4,         // [576]
    const int32_t* __restrict__ sel_ids,   // [8]
    float* __restrict__ act_out),
    (w13_u128, s13_u32, x4, sel_ids, act_out)) {
    mxfp8_up_situ_body<PackedScales, false>(w13_u128, s13_u32, x4, sel_ids, act_out);
}

INK2_SCALE_KERNEL(ink2_moe_mxfp8_up_situ_u4, (    const uint4* __restrict__ w13_u128,    // [128, 2304, 144]
    const uint32_t* __restrict__ s13_u32,  // [128, 2304, 18]
    const float4* __restrict__ x4,         // [576]
    const int32_t* __restrict__ sel_ids,   // [8]
    float* __restrict__ act_out),
    (w13_u128, s13_u32, x4, sel_ids, act_out)) {
    mxfp8_up_situ_body<PackedScales, true>(w13_u128, s13_u32, x4, sel_ids, act_out);
}

INK2_INDEXED_KERNEL(ink2_moe_mxfp8_down, , (
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
            Mxfp8Row<Indexed>(w2_u128, r_idx * 72u),
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

// Expert-parallel MXFP8 MoE Down (`Grid: 1152`, `Block: 256`, `2` output rows/CTA, `0-ULP` exact!):
// Warp `k` owns selected expert `k`; since `router_topk8` completed prior to `mxfp8_up_situ`,
// `sel_ids[k]` is already visible before `pdl_wait()`, allowing `w2` weights to stream during `up_situ`'s tail!
INK2_INDEXED_KERNEL(ink2_moe_mxfp8_down_xw, , (
    const uint4* __restrict__ w2_u128,     // [128, 2304, 72]
    const uint32_t* __restrict__ s2_u32,   // [128, 2304, 9]
    const float4* __restrict__ act4,       // [8, 288]
    const int32_t* __restrict__ sel_ids,   // [8]
    const float* __restrict__ sel_wts,     // [8]
    float* __restrict__ moe_out),
    (w2_u128, s2_u32, act4, sel_ids, sel_wts, moe_out)) {         // [2304]
    constexpr int kRows = 2;
    __shared__ float smem_dot[kRows][kTopK];

    const int tid = threadIdx.x;
    const int k = tid >> 5;
    const int lane = tid & 31;
    const int ex = ld_ca_i32(sel_ids + k);
    const int row0 = blockIdx.x * kRows;
    const bool tail = (lane < 8);

    uint4 wp[kRows][2];
    uint4 wp_tail[kRows];
    float esc[kRows][2];
    float esc_tail[kRows];
    #pragma unroll
    for (int r = 0; r < kRows; ++r) {
        const size_t r_idx = static_cast<size_t>(ex) * kD + (row0 + r);
        const Mxfp8Row<Indexed> row_fp8(w2_u128, r_idx * 72u);
        const unsigned sc_start = static_cast<unsigned>(r_idx * 9u);
        #pragma unroll
        for (int it = 0; it < 2; ++it) {
            const int u16 = lane + 32 * it;
            wp[r][it] = row_fp8[u16];
            const int blk32 = u16 >> 1;
            uint32_t s4 = expert_scale_word<PackedScales, 3, 112>(s2_u32, sc_start + (blk32 >> 2));
            esc[r][it] = __uint_as_float(((s4 >> ((blk32 & 3) * 8)) & 0xFFu) << 23);
        }
        if (tail) {
            const int u16 = lane + 64;
            wp_tail[r] = row_fp8[u16];
            const int blk32 = u16 >> 1;
            uint32_t s4 = expert_scale_word<PackedScales, 3, 112>(s2_u32, sc_start + (blk32 >> 2));
            esc_tail[r] = __uint_as_float(((s4 >> ((blk32 & 3) * 8)) & 0xFFu) << 23);
        }
    }

    pdl_wait();
    act4 = pdl_in(act4);
    sel_wts = pdl_in(sel_wts);

    const float4* ak = act4 + k * 288;
    #pragma unroll
    for (int r = 0; r < kRows; ++r) {
        float acc = 0.0f;
        acc = __fmaf_rn(dot_mxfp8_u128_exact(wp[r][0], ak + lane * 4), esc[r][0], acc);
        acc = __fmaf_rn(dot_mxfp8_u128_exact(wp[r][1], ak + (lane + 32) * 4), esc[r][1], acc);
        if (tail) {
            acc = __fmaf_rn(dot_mxfp8_u128_exact(wp_tail[r], ak + (lane + 64) * 4), esc_tail[r], acc);
        }
        const float dot = warp_sum(acc);
        if (lane == 0) smem_dot[r][k] = dot;
    }
    __syncthreads();
    pdl_trigger();
    if (tid < kRows) {
        float sum_y = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < kTopK; ++kk) {
            sum_y = __fmaf_rn(smem_dot[tid][kk], sel_wts[kk], sum_y);
        }
        moe_out[row0 + tid] = sum_y;
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
    pdl_wait();
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
    pdl_wait();
    pdl_trigger();
    x = pdl_in(x);
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
    pdl_wait();
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

// `ink2_q8_dp4a_gemv` specialised for `cols = 3072` (the attention output projection `wo`,
// 2,304 x 3,072): each lane loads all 6 of its weight `uint4`s for both rows before the
// `__dp4a` / FMA chain. Same lanes, same per-lane order (c16 = lane + 32 i) and the same
// `warp_sum`, so the output is bit-identical; only the loads are hoisted.
extern "C" __global__ void ink2_q8_dp4a_gemv_c3072(
    const uint4* __restrict__ w_q8_u128,   // [rows, 192]
    const float* __restrict__ w_sc64,      // [rows, 48]
    const uint4* __restrict__ x_q8_u128,   // [192]
    const float* __restrict__ x_sc64,      // [48]
    float* __restrict__ y,                 // [rows]
    int32_t rows,
    int32_t cols,
    float softcap) {
    constexpr int kCols16 = 3072 / 16;
    constexpr int kCols64 = 3072 / 64;
    constexpr int kIters = kCols16 / 32;
    __shared__ uint4 smem_x[kCols16];
    __shared__ float smem_xsc[kCols64];

    const int tid = threadIdx.x; // 256
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    (void)cols;

    const int row0 = (blockIdx.x * 8 + warp_id) * 2;
    const bool has_row0 = (row0 < rows);
    const bool has_row1 = (row0 + 1 < rows);

    uint4 w0[kIters], w1[kIters];
    float ws0[kIters], ws1[kIters];
    if (has_row0) {
        const uint4* w_r0 = w_q8_u128 + static_cast<size_t>(row0) * kCols16;
        const uint4* w_r1 = has_row1 ? (w_r0 + kCols16) : w_r0;
        const float* s_r0 = w_sc64 + static_cast<size_t>(row0) * kCols64;
        const float* s_r1 = has_row1 ? (s_r0 + kCols64) : s_r0;
        #pragma unroll
        for (int i = 0; i < kIters; ++i) {
            const int c16 = lane + 32 * i;
            w0[i] = w_r0[c16];
            w1[i] = w_r1[c16];
            ws0[i] = s_r0[c16 >> 2];
            ws1[i] = s_r1[c16 >> 2];
        }
    }

    pdl_wait();
    x_q8_u128 = pdl_in(x_q8_u128);
    x_sc64 = pdl_in(x_sc64);

    if (tid < kCols16) smem_x[tid] = x_q8_u128[tid];
    if (tid < kCols64) smem_xsc[tid] = x_sc64[tid];
    __syncthreads();

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    if (has_row0) {
        #pragma unroll
        for (int i = 0; i < kIters; ++i) {
            const int c16 = lane + 32 * i;
            uint4 xq = smem_x[c16];
            float xsc = smem_xsc[c16 >> 2];

            int d0 = __dp4a(static_cast<int>(w0[i].x), static_cast<int>(xq.x), 0);
            d0 = __dp4a(static_cast<int>(w0[i].y), static_cast<int>(xq.y), d0);
            d0 = __dp4a(static_cast<int>(w0[i].z), static_cast<int>(xq.z), d0);
            d0 = __dp4a(static_cast<int>(w0[i].w), static_cast<int>(xq.w), d0);
            acc0 = __fmaf_rn(static_cast<float>(d0), __fmul_rn(ws0[i], xsc), acc0);

            if (has_row1) {
                int d1 = __dp4a(static_cast<int>(w1[i].x), static_cast<int>(xq.x), 0);
                d1 = __dp4a(static_cast<int>(w1[i].y), static_cast<int>(xq.y), d1);
                d1 = __dp4a(static_cast<int>(w1[i].z), static_cast<int>(xq.z), d1);
                d1 = __dp4a(static_cast<int>(w1[i].w), static_cast<int>(xq.w), d1);
                acc1 = __fmaf_rn(static_cast<float>(d1), __fmul_rn(ws1[i], xsc), acc1);
            }
        }
        acc0 = warp_sum(acc0);
        acc1 = warp_sum(acc1);
    }
    pdl_trigger();
    if (has_row0 && lane == 0) {
        if (softcap > 0.0f) {
            float inv_sc = ptx_rcp(softcap);
            acc0 = softcap * ptx_tanh(acc0 * inv_sc);
            acc1 = softcap * ptx_tanh(acc1 * inv_sc);
        }
        y[row0] = acc0;
        if (has_row1) y[row0 + 1] = acc1;
    }
}

// 1-Row/Warp `c3072` (`Grid: 288`, `Block: 256`, `<= 40` regs -> `6` CTAs/SM = `48` warps/SM):
// Doubles grid CTAs from `144` to `288` on `wo` (`2304 x 3072`), eliminating `76`-SM Wave-2 starvation
// on 110-SM RTX PRO 5000 while preserving `0-ULP` bitwise identity!
extern "C" __global__ __launch_bounds__(256, 6) void ink2_q8_dp4a_gemv_c3072_r1(
    const uint4* __restrict__ w_q8_u128,   // [rows, 192]
    const float* __restrict__ w_sc64,      // [rows, 48]
    const uint4* __restrict__ x_q8_u128,   // [192]
    const float* __restrict__ x_sc64,      // [48]
    float* __restrict__ y,                 // [rows]
    int32_t rows,
    int32_t cols,
    float softcap) {
    constexpr int kCols16 = 3072 / 16;
    constexpr int kCols64 = 3072 / 64;
    constexpr int kIters = kCols16 / 32; // 6
    __shared__ uint4 smem_x[kCols16];
    __shared__ float smem_xsc[kCols64];

    const int tid = threadIdx.x; // 256
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    (void)cols;

    const int row = blockIdx.x * 8 + warp_id;
    const bool has_row = (row < rows);

    uint4 w0[kIters];
    float ws0[kIters];
    if (has_row) {
        const uint4* w_r0 = w_q8_u128 + static_cast<size_t>(row) * kCols16;
        const float* s_r0 = w_sc64 + static_cast<size_t>(row) * kCols64;
        #pragma unroll
        for (int i = 0; i < kIters; ++i) {
            const int c16 = lane + 32 * i;
            w0[i] = w_r0[c16];
            ws0[i] = s_r0[c16 >> 2];
        }
    }

    pdl_wait();
    x_q8_u128 = pdl_in(x_q8_u128);
    x_sc64 = pdl_in(x_sc64);

    if (tid < kCols16) smem_x[tid] = x_q8_u128[tid];
    if (tid < kCols64) smem_xsc[tid] = x_sc64[tid];
    __syncthreads();

    float acc0 = 0.0f;
    if (has_row) {
        #pragma unroll
        for (int i = 0; i < kIters; ++i) {
            const int c16 = lane + 32 * i;
            uint4 xq = smem_x[c16];
            float xsc = smem_xsc[c16 >> 2];

            int d0 = __dp4a(static_cast<int>(w0[i].x), static_cast<int>(xq.x), 0);
            d0 = __dp4a(static_cast<int>(w0[i].y), static_cast<int>(xq.y), d0);
            d0 = __dp4a(static_cast<int>(w0[i].z), static_cast<int>(xq.z), d0);
            d0 = __dp4a(static_cast<int>(w0[i].w), static_cast<int>(xq.w), d0);
            acc0 = __fmaf_rn(static_cast<float>(d0), __fmul_rn(ws0[i], xsc), acc0);
        }
        acc0 = warp_sum(acc0);
    }
    pdl_trigger();
    if (has_row && lane == 0) {
        if (softcap > 0.0f) {
            acc0 = softcap * ptx_tanh(acc0 * ptx_rcp(softcap));
        }
        y[row] = acc0;
    }
}

// `ink2_q8_dp4a_gemv` specialised for `cols = 2304` (`wq_all` 8,600 x 2,304 and `unembed` 200,058 x 2,304):
// Hoists all 4 full + 1 half-warp (`lane < 16`) weight `uint4`s and scales BEFORE `pdl_wait()`,
// preserving exact per-lane accumulation order (`c16 = lane + 32*i`) and `0-ULP` bitwise identity!
extern "C" __global__ void ink2_q8_dp4a_gemv_c2304(
    const uint4* __restrict__ w_q8_u128,   // [rows, 144]
    const float* __restrict__ w_sc64,      // [rows, 36]
    const uint4* __restrict__ x_q8_u128,   // [144]
    const float* __restrict__ x_sc64,      // [36]
    float* __restrict__ y,                 // [rows]
    int32_t rows,
    int32_t cols,
    float softcap) {
    constexpr int kCols16 = 2304 / 16; // 144 = 4 * 32 + 16
    constexpr int kCols64 = 2304 / 64; // 36
    __shared__ uint4 smem_x[kCols16];
    __shared__ float smem_xsc[kCols64];

    const int tid = threadIdx.x; // 256
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    (void)cols;

    const int row0 = (blockIdx.x * 8 + warp_id) * 2;
    const bool has_row0 = (row0 < rows);
    const bool has_row1 = (row0 + 1 < rows);
    const bool tail16 = (lane < 16);

    uint4 w0[4], w1[4];
    float ws0[4], ws1[4];
    uint4 w0_4 = make_uint4(0u, 0u, 0u, 0u), w1_4 = w0_4;
    float ws0_4 = 0.0f, ws1_4 = 0.0f;
    if (has_row0) {
        const uint4* w_r0 = w_q8_u128 + static_cast<size_t>(row0) * kCols16;
        const uint4* w_r1 = has_row1 ? (w_r0 + kCols16) : w_r0;
        const float* s_r0 = w_sc64 + static_cast<size_t>(row0) * kCols64;
        const float* s_r1 = has_row1 ? (s_r0 + kCols64) : s_r0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c16 = lane + 32 * i;
            w0[i] = w_r0[c16];
            w1[i] = w_r1[c16];
            ws0[i] = s_r0[c16 >> 2];
            ws1[i] = s_r1[c16 >> 2];
        }
        if (tail16) {
            const int c16 = lane + 128;
            w0_4 = w_r0[c16];
            w1_4 = w_r1[c16];
            ws0_4 = s_r0[c16 >> 2];
            ws1_4 = s_r1[c16 >> 2];
        }
    }

    pdl_wait();
    x_q8_u128 = pdl_in(x_q8_u128);
    x_sc64 = pdl_in(x_sc64);

    if (tid < kCols16) smem_x[tid] = x_q8_u128[tid];
    if (tid < kCols64) smem_xsc[tid] = x_sc64[tid];
    __syncthreads();

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    if (has_row0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c16 = lane + 32 * i;
            uint4 xq = smem_x[c16];
            float xsc = smem_xsc[c16 >> 2];

            int d0 = __dp4a(static_cast<int>(w0[i].x), static_cast<int>(xq.x), 0);
            d0 = __dp4a(static_cast<int>(w0[i].y), static_cast<int>(xq.y), d0);
            d0 = __dp4a(static_cast<int>(w0[i].z), static_cast<int>(xq.z), d0);
            d0 = __dp4a(static_cast<int>(w0[i].w), static_cast<int>(xq.w), d0);
            acc0 = __fmaf_rn(static_cast<float>(d0), __fmul_rn(ws0[i], xsc), acc0);

            if (has_row1) {
                int d1 = __dp4a(static_cast<int>(w1[i].x), static_cast<int>(xq.x), 0);
                d1 = __dp4a(static_cast<int>(w1[i].y), static_cast<int>(xq.y), d1);
                d1 = __dp4a(static_cast<int>(w1[i].z), static_cast<int>(xq.z), d1);
                d1 = __dp4a(static_cast<int>(w1[i].w), static_cast<int>(xq.w), d1);
                acc1 = __fmaf_rn(static_cast<float>(d1), __fmul_rn(ws1[i], xsc), acc1);
            }
        }
        if (tail16) {
            const int c16 = lane + 128;
            uint4 xq = smem_x[c16];
            float xsc = smem_xsc[c16 >> 2];

            int d0 = __dp4a(static_cast<int>(w0_4.x), static_cast<int>(xq.x), 0);
            d0 = __dp4a(static_cast<int>(w0_4.y), static_cast<int>(xq.y), d0);
            d0 = __dp4a(static_cast<int>(w0_4.z), static_cast<int>(xq.z), d0);
            d0 = __dp4a(static_cast<int>(w0_4.w), static_cast<int>(xq.w), d0);
            acc0 = __fmaf_rn(static_cast<float>(d0), __fmul_rn(ws0_4, xsc), acc0);

            if (has_row1) {
                int d1 = __dp4a(static_cast<int>(w1_4.x), static_cast<int>(xq.x), 0);
                d1 = __dp4a(static_cast<int>(w1_4.y), static_cast<int>(xq.y), d1);
                d1 = __dp4a(static_cast<int>(w1_4.z), static_cast<int>(xq.z), d1);
                d1 = __dp4a(static_cast<int>(w1_4.w), static_cast<int>(xq.w), d1);
                acc1 = __fmaf_rn(static_cast<float>(d1), __fmul_rn(ws1_4, xsc), acc1);
            }
        }
        acc0 = warp_sum(acc0);
        acc1 = warp_sum(acc1);
    }
    pdl_trigger();
    if (has_row0 && lane == 0) {
        if (softcap > 0.0f) {
            float inv_sc = ptx_rcp(softcap);
            acc0 = softcap * ptx_tanh(acc0 * inv_sc);
            acc1 = softcap * ptx_tanh(acc1 * inv_sc);
        }
        y[row0] = acc0;
        if (has_row1) y[row0 + 1] = acc1;
    }
}

// 1-Row/Warp `c2304` (`Grid: ceil(rows / 8)`, `Block: 256`, `<= 40` regs -> `6` CTAs/SM = `48` warps/SM):
// Turns `538`-CTA `wq_all` (`440 + 98` Wave-2 tail starvation at `63` regs) into `1,075` CTAs
// (`660 + 415` at `6` CTAs/SM on PRO 5000) while preserving exact `0-ULP` identity!
extern "C" __global__ __launch_bounds__(256, 5) void ink2_q8_dp4a_gemv_c2304_r1(
    const uint4* __restrict__ w_q8_u128,   // [rows, 144]
    const float* __restrict__ w_sc64,      // [rows, 36]
    const uint4* __restrict__ x_q8_u128,   // [144]
    const float* __restrict__ x_sc64,      // [36]
    float* __restrict__ y,                 // [rows]
    int32_t rows,
    int32_t cols,
    float softcap) {
    constexpr int kCols16 = 2304 / 16; // 144 = 4 * 32 + 16
    constexpr int kCols64 = 2304 / 64; // 36
    __shared__ uint4 smem_x[kCols16];
    __shared__ float smem_xsc[kCols64];

    const int tid = threadIdx.x; // 256
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    (void)cols;

    const int row = blockIdx.x * 8 + warp_id;
    const bool has_row = (row < rows);
    const bool tail16 = (lane < 16);

    uint4 w0[4];
    float ws0[4];
    uint4 w0_4 = make_uint4(0u, 0u, 0u, 0u);
    float ws0_4 = 0.0f;
    if (has_row) {
        const uint4* w_r0 = w_q8_u128 + static_cast<size_t>(row) * kCols16;
        const float* s_r0 = w_sc64 + static_cast<size_t>(row) * kCols64;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c16 = lane + 32 * i;
            w0[i] = w_r0[c16];
            ws0[i] = s_r0[c16 >> 2];
        }
        if (tail16) {
            const int c16 = lane + 128;
            w0_4 = w_r0[c16];
            ws0_4 = s_r0[c16 >> 2];
        }
    }

    pdl_wait();
    x_q8_u128 = pdl_in(x_q8_u128);
    x_sc64 = pdl_in(x_sc64);

    if (tid < kCols16) smem_x[tid] = x_q8_u128[tid];
    if (tid < kCols64) smem_xsc[tid] = x_sc64[tid];
    __syncthreads();

    float acc0 = 0.0f;
    if (has_row) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c16 = lane + 32 * i;
            uint4 xq = smem_x[c16];
            float xsc = smem_xsc[c16 >> 2];
            int d0 = __dp4a(static_cast<int>(w0[i].x), static_cast<int>(xq.x), 0);
            d0 = __dp4a(static_cast<int>(w0[i].y), static_cast<int>(xq.y), d0);
            d0 = __dp4a(static_cast<int>(w0[i].z), static_cast<int>(xq.z), d0);
            d0 = __dp4a(static_cast<int>(w0[i].w), static_cast<int>(xq.w), d0);
            acc0 = __fmaf_rn(static_cast<float>(d0), __fmul_rn(ws0[i], xsc), acc0);
        }
        if (tail16) {
            const int c16 = lane + 128;
            uint4 xq = smem_x[c16];
            float xsc = smem_xsc[c16 >> 2];
            int d0 = __dp4a(static_cast<int>(w0_4.x), static_cast<int>(xq.x), 0);
            d0 = __dp4a(static_cast<int>(w0_4.y), static_cast<int>(xq.y), d0);
            d0 = __dp4a(static_cast<int>(w0_4.z), static_cast<int>(xq.z), d0);
            d0 = __dp4a(static_cast<int>(w0_4.w), static_cast<int>(xq.w), d0);
            acc0 = __fmaf_rn(static_cast<float>(d0), __fmul_rn(ws0_4, xsc), acc0);
        }
        acc0 = warp_sum(acc0);
    }
    pdl_trigger();
    if (has_row && lane == 0) {
        if (softcap > 0.0f) {
            acc0 = softcap * ptx_tanh(acc0 * ptx_rcp(softcap));
        }
        y[row] = acc0;
    }
}

template<bool PackedScales>
__device__ __forceinline__ float lane_dot_nvfp4_dp4a_u128_row(
    const uint4* __restrict__ row_u128,         // [n_u128] (`16B` = `32` nibbles = `2` blocks of 16!)
    const uint32_t* __restrict__ row_scale_u32, unsigned scale_start, // [n_u128 / 2]
    const uint4* __restrict__ xq16,        // [n_u128 * 2]
    const float* __restrict__ xsc16,       // [n_u128 * 2]
    int n_u128,
    int lane) {
    float acc = 0.0f;
    for (int u16 = lane; u16 < n_u128; u16 += 32) {
        uint4 pk16 = row_u128[u16];
        uint32_t sc4 = expert_scale_word<PackedScales, 6, 64>(row_scale_u32, scale_start + (u16 >> 1));
        int sh = (u16 & 1) * 16;
        float s0 = __fmul_rn(e4m3_byte_to_f32((sc4 >> sh) & 0xFFu) * 0.5f, xsc16[u16 * 2]);
        float s1 = __fmul_rn(e4m3_byte_to_f32((sc4 >> (sh + 8)) & 0xFFu) * 0.5f, xsc16[u16 * 2 + 1]);

        uint4 xq0 = xq16[u16 * 2];
        uint4 xq1 = xq16[u16 * 2 + 1];
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
    return acc;
}

template<bool PackedScales>
__device__ __forceinline__ float warp_dot_nvfp4_dp4a_u128_row(
    const uint4* __restrict__ row_u128,         // [n_u128] (`16B` = `32` nibbles = `2` blocks of 16!)
    const uint32_t* __restrict__ row_scale_u32, unsigned scale_start, // [n_u128 / 2]
    const uint4* __restrict__ smem_xq16,        // [n_u128 * 2]
    const float* __restrict__ smem_xsc16,       // [n_u128 * 2]
    int n_u128,
    int lane) {
    return warp_sum(lane_dot_nvfp4_dp4a_u128_row<PackedScales>(
        row_u128, row_scale_u32, scale_start, smem_xq16, smem_xsc16, n_u128, lane));
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

__device__ __forceinline__ float accum_nvfp4_dp4a_u128_step(
    float acc,
    uint4 pk16,
    uint32_t sc4,
    int u16,
    const uint4* xq16,
    const float* xsc16) {
    int sh = (u16 & 1) * 16;
    float s0 = __fmul_rn(e4m3_byte_to_f32((sc4 >> sh) & 0xFFu) * 0.5f, xsc16[u16 * 2]);
    float s1 = __fmul_rn(e4m3_byte_to_f32((sc4 >> (sh + 8)) & 0xFFu) * 0.5f, xsc16[u16 * 2 + 1]);
    uint4 xq0 = xq16[u16 * 2];
    uint4 xq1 = xq16[u16 * 2 + 1];
    int d0 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.x & 0xFFFFu)), static_cast<int>(xq0.x), 0);
    d0 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.x >> 16)), static_cast<int>(xq0.y), d0);
    d0 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.y & 0xFFFFu)), static_cast<int>(xq0.z), d0);
    d0 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.y >> 16)), static_cast<int>(xq0.w), d0);
    int d1 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.z & 0xFFFFu)), static_cast<int>(xq1.x), 0);
    d1 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.z >> 16)), static_cast<int>(xq1.y), d1);
    d1 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.w & 0xFFFFu)), static_cast<int>(xq1.z), d1);
    d1 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.w >> 16)), static_cast<int>(xq1.w), d1);
    return __fmaf_rn(static_cast<float>(d1), s1, __fmaf_rn(static_cast<float>(d0), s0, acc));
}

template<bool PackedScales, bool TriggerPdl = false>
__device__ __forceinline__ void nvfp4_half_warp_situ(
    const uint4* weights, const uint32_t* scales, unsigned scale_offset, float alpha,
    const uint4* x_q8_u128, const float* x_sc16, int lane, float* output) {
    const int index = lane & 15;
    const bool has4 = (index < 8);
    const int u0 = index;
    const int u1 = index + 16;
    const int u2 = index + 32;
    const int u3 = index + 48;
    const int u4 = index + 64;

    // Issue all 5 weight uint4 loads + scale words up front before the first __dp4a,
    // preserving exact per-half (lo: u0->u2->[u4], hi: u1->u3) 0-ULP accumulation order!
    const uint4 pk0 = weights[u0];
    const uint4 pk1 = weights[u1];
    const uint4 pk2 = weights[u2];
    const uint4 pk3 = weights[u3];
    uint4 pk4 = make_uint4(0u, 0u, 0u, 0u);
    if (has4) {
        pk4 = weights[u4];
    }
    const uint32_t sc0 = expert_scale_word<PackedScales, 6, 64>(scales, scale_offset + (u0 >> 1));
    const uint32_t sc1 = expert_scale_word<PackedScales, 6, 64>(scales, scale_offset + (u1 >> 1));
    const uint32_t sc2 = expert_scale_word<PackedScales, 6, 64>(scales, scale_offset + (u2 >> 1));
    const uint32_t sc3 = expert_scale_word<PackedScales, 6, 64>(scales, scale_offset + (u3 >> 1));
    uint32_t sc4 = 0u;
    if (has4) {
        sc4 = expert_scale_word<PackedScales, 6, 64>(scales, scale_offset + (u4 >> 1));
    }

    float lo = accum_nvfp4_dp4a_u128_step(0.0f, pk0, sc0, u0, x_q8_u128, x_sc16);
    lo = accum_nvfp4_dp4a_u128_step(lo, pk2, sc2, u2, x_q8_u128, x_sc16);
    if (has4) {
        lo = accum_nvfp4_dp4a_u128_step(lo, pk4, sc4, u4, x_q8_u128, x_sc16);
    }
    float hi = accum_nvfp4_dp4a_u128_step(0.0f, pk1, sc1, u1, x_q8_u128, x_sc16);
    hi = accum_nvfp4_dp4a_u128_step(hi, pk3, sc3, u3, x_q8_u128, x_sc16);

    float dot = lo + hi;
    #pragma unroll
    for (int mask = 8; mask; mask >>= 1) {
        dot += __shfl_xor_sync(0xffffffffu, dot, mask);
    }
    dot *= alpha * (1.0f / 2688.0f);
    const float u = __shfl_sync(0xffffffffu, dot, 16);
    if constexpr (TriggerPdl) {
        pdl_trigger();
    }
    if (lane == 0) {
        const float g = dot;
        *output =
            4.0f * ptx_tanh(g * 0.25f) * ptx_sigmoid(g) * 25.0f * ptx_tanh(u * 0.04f);
    }
}

INK2_SCALE_KERNEL(ink2_moe_nvfp4_dp4a_up_half, (
    const uint4* __restrict__ w13_u128,    // [128, 2304, 72]
    const uint32_t* __restrict__ s13_u32,  // [128, 2304, 36]
    const float* __restrict__ a13,         // [128]
    const uint4* __restrict__ x_q8_u128,   // [144]
    const float* __restrict__ x_sc16,      // [144]
    const int32_t* __restrict__ sel_ids,   // [8]
    float* __restrict__ act_out),
    (w13_u128, s13_u32, a13, x_q8_u128, x_sc16, sel_ids, act_out)) {         // [8, 1152]
    // Four warps produce four channels. Each half warp handles one of the
    // two interleaved rows (gate/up), so their SiTU needs no CTA barrier.
    const int lane = threadIdx.x & 31;
    const int channel = blockIdx.x * 4 + threadIdx.x / 32;
    const int k = blockIdx.y;
    pdl_wait();
    x_q8_u128 = pdl_in(x_q8_u128);
    x_sc16 = pdl_in(x_sc16);
    sel_ids = pdl_in(sel_ids);
    const int ex = sel_ids[k];
    const size_t row = (static_cast<size_t>(ex) * kFf + channel) * 2 + (lane >> 4);
    nvfp4_half_warp_situ<PackedScales, true>(w13_u128 + row * 72u, s13_u32, row * 36u,
        a13[ex], x_q8_u128, x_sc16, lane, act_out + k * kFf + channel);
}

// Fused 16-Warp Half-Warp NVFP4 Gate/Up + In-Block `Q8_16` Epilogue (`Grid: (72, 8)`, `Block: 512`):
// Stages `x_q8_u128`/`x_sc16` (`2,880 B` SMEM, `2 * (2,944 + 1,024) = 7,936 B <= 8 KiB` -> `120 KiB` L1!)
// concurrently with all 5 `uint4` GDDR7 weight fetches (`pk0..pk4`) + 5 scale words BEFORE `__syncthreads()`,
// eliminating sequential `__dp4a` load stalls while preserving exact `0-ULP` arithmetic!
INK2_SCALE_KERNEL_LB(ink2_moe_nvfp4_dp4a_up_half_q8, 512, 2, (
    const uint4* __restrict__ w13_u128,    // [128, 2304, 72]
    const uint32_t* __restrict__ s13_u32,  // [128, 2304, 36]
    const float* __restrict__ a13,         // [128]
    const uint4* __restrict__ x_q8_u128,   // [144]
    const float* __restrict__ x_sc16,      // [144]
    const int32_t* __restrict__ sel_ids,   // [8]
    uint32_t* __restrict__ out_q8_u32,     // [8, 288]
    float* __restrict__ out_sc16),         // [8, 72]
    (w13_u128, s13_u32, a13, x_q8_u128, x_sc16, sel_ids, out_q8_u32, out_sc16)) {
    __shared__ uint4 smem_xq[144];         // 2,304 B
    __shared__ float smem_xsc[144];        //   576 B
    __shared__ float smem_ch[16];          //    64 B -> 2,944 B user SMEM
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;          // 0..15
    const int lane = tid & 31;             // 0..31
    const int index = lane & 15;           // 0..15
    const int blk16 = blockIdx.x;          // 0..71
    const int channel = blk16 * 16 + warp_id;
    const int k = blockIdx.y;              // 0..7
    const bool has4 = (index < 8);

    pdl_wait();
    x_q8_u128 = pdl_in(x_q8_u128);
    x_sc16 = pdl_in(x_sc16);
    sel_ids = pdl_in(sel_ids);

    if (tid < 144) {
        smem_xq[tid] = x_q8_u128[tid];
        smem_xsc[tid] = x_sc16[tid];
    }

    const int ex = sel_ids[k];
    const float alpha = a13[ex];
    const size_t row = (static_cast<size_t>(ex) * kFf + channel) * 2 + (lane >> 4);
    const uint4* weights = w13_u128 + row * 72u;
    const unsigned sc_off = static_cast<unsigned>(row * 36u);

    const int u0 = index;
    const int u1 = index + 16;
    const int u2 = index + 32;
    const int u3 = index + 48;
    const int u4 = index + 64;

    const uint4 pk0 = weights[u0];
    const uint4 pk1 = weights[u1];
    const uint4 pk2 = weights[u2];
    const uint4 pk3 = weights[u3];
    uint4 pk4 = make_uint4(0u, 0u, 0u, 0u);
    if (has4) {
        pk4 = weights[u4];
    }
    const uint32_t sc0 = expert_scale_word<PackedScales, 6, 64>(s13_u32, sc_off + (u0 >> 1));
    const uint32_t sc1 = expert_scale_word<PackedScales, 6, 64>(s13_u32, sc_off + (u1 >> 1));
    const uint32_t sc2 = expert_scale_word<PackedScales, 6, 64>(s13_u32, sc_off + (u2 >> 1));
    const uint32_t sc3 = expert_scale_word<PackedScales, 6, 64>(s13_u32, sc_off + (u3 >> 1));
    uint32_t sc4 = 0u;
    if (has4) {
        sc4 = expert_scale_word<PackedScales, 6, 64>(s13_u32, sc_off + (u4 >> 1));
    }

    __syncthreads();

    float lo = accum_nvfp4_dp4a_u128_step(0.0f, pk0, sc0, u0, smem_xq, smem_xsc);
    lo = accum_nvfp4_dp4a_u128_step(lo, pk2, sc2, u2, smem_xq, smem_xsc);
    if (has4) {
        lo = accum_nvfp4_dp4a_u128_step(lo, pk4, sc4, u4, smem_xq, smem_xsc);
    }
    float hi = accum_nvfp4_dp4a_u128_step(0.0f, pk1, sc1, u1, smem_xq, smem_xsc);
    hi = accum_nvfp4_dp4a_u128_step(hi, pk3, sc3, u3, smem_xq, smem_xsc);

    float dot = lo + hi;
    #pragma unroll
    for (int mask = 8; mask; mask >>= 1) {
        dot += __shfl_xor_sync(0xffffffffu, dot, mask);
    }
    dot *= alpha * (1.0f / 2688.0f);
    const float u = __shfl_sync(0xffffffffu, dot, 16);
    if (lane == 0) {
        const float g = dot;
        smem_ch[warp_id] =
            4.0f * ptx_tanh(g * 0.25f) * ptx_sigmoid(g) * 25.0f * ptx_tanh(u * 0.04f);
    }
    __syncthreads();
    pdl_trigger();
    if (threadIdx.x < 16) {
        const int lane16 = threadIdx.x;
        float v = smem_ch[lane16];
        float amax = fabsf(v);
        #pragma unroll
        for (int mask = 8; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0x0000FFFFu, amax, mask, 16));
        }
        float sc = amax * (1.0f / 127.0f);
        float inv = (amax > 0.0f) ? (127.0f * ptx_rcp(amax)) : 0.0f;
        int q = __float2int_rn(v * inv);
        if (q < -127) q = -127;
        if (q > 127) q = 127;
        uint32_t qb = static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(q)));
        uint32_t p4 = qb
            | (__shfl_sync(0x0000FFFFu, qb, lane16 + 1, 16) << 8)
            | (__shfl_sync(0x0000FFFFu, qb, lane16 + 2, 16) << 16)
            | (__shfl_sync(0x0000FFFFu, qb, lane16 + 3, 16) << 24);
        if ((lane16 & 3) == 0) {
            out_q8_u32[(k * 288) + blk16 * 4 + (lane16 >> 2)] = p4;
        }
        if (lane16 == 0) {
            out_sc16[(k * 72) + blk16] = sc;
        }
    }
}

// Copy contiguous spans of the selected host-backed rows before their dot products.
extern "C" __global__ void ink2_gather_nvfp4_up_192(
    const uint4* __restrict__ mapped, const int32_t* __restrict__ selected,
    uint4* __restrict__ staged, int first_channel) {
    pdl_wait();
    constexpr int vectors_per_channel = 2 * 72;
    constexpr int tile_vectors = 192 * vectors_per_channel;
    const int slot = blockIdx.y;
    const int first = (selected[slot] * kFf + first_channel) * vectors_per_channel;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < tile_vectors;
         i += gridDim.x * blockDim.x) {
        staged[slot * tile_vectors + i] = mapped[first + i];
    }
}

INK2_SCALE_KERNEL(ink2_moe_nvfp4_dp4a_up_half_staged, (
    const uint4* __restrict__ w13_u128,    // [128, 2304, 72]
    const uint32_t* __restrict__ s13_u32,  // [128, 2304, 36]
    const float* __restrict__ a13,         // [128]
    const uint4* __restrict__ x_q8_u128,   // [144]
    const float* __restrict__ x_sc16,      // [144]
    const int32_t* __restrict__ sel_ids,   // [8]
    float* __restrict__ act_out, int first_channel),
    (w13_u128, s13_u32, a13, x_q8_u128, x_sc16, sel_ids, act_out, first_channel)) {         // [8, 1152]
    // Four warps produce four channels. Each half warp handles one of the
    // two interleaved rows (gate/up), so their SiTU needs no CTA barrier.
    const int lane = threadIdx.x & 31;
    constexpr int tile_channels = 192;
    const int local_channel = blockIdx.x * 4 + threadIdx.x / 32;
    const int channel = first_channel + local_channel;
    const int k = blockIdx.y;
    const int ex = sel_ids[k];
    const size_t row = (static_cast<size_t>(ex) * kFf + channel) * 2 + (lane >> 4);
    const size_t staged_row = (k * tile_channels + local_channel) * 2 + (lane >> 4);
    nvfp4_half_warp_situ<PackedScales>(w13_u128 + staged_row * 72u, s13_u32, row * 36u,
        a13[ex], x_q8_u128, x_sc16, lane, act_out + k * kFf + channel);
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

// Expert-parallel NVFP4 `__dp4a` MoE Down for the single decode row. Warp `k` of a CTA owns
// selected expert `k` and computes its dot for `kRows = 2` consecutive output rows; one
// thread per row then folds the 8 dots in expert order. Every dot uses the lane -> `u16` mapping,
// scale products, `__dp4a` chains and `warp_sum` tree of `warp_dot_nvfp4_dp4a_u128_row`, and the
// fold is the same `__fmaf_rn` chain over `k = 0..7`, so `moe_out` is bit-identical to
// `ink2_moe_nvfp4_dp4a_down`. The 8 independent expert rows per CTA (vs. 8 sequential rows per
// warp) give 8x the warps and loads in flight; the activations stay in registers (no smem copy).
INK2_SCALE_KERNEL(ink2_moe_nvfp4_dp4a_down_xw, (
    const uint4* __restrict__ w2_u128,     // [128, 2304, 36]
    const uint32_t* __restrict__ s2_u32,   // [128, 2304, 18]
    const float* __restrict__ a2,          // [128]
    const uint4* __restrict__ act_q8_u128, // [8, 72]
    const float* __restrict__ act_sc16,    // [8, 72]
    const int32_t* __restrict__ sel_ids,   // [8]
    const float* __restrict__ sel_wts,     // [8]
    float* __restrict__ moe_out),
    (w2_u128, s2_u32, a2, act_q8_u128, act_sc16, sel_ids, sel_wts, moe_out)) {         // [2304]
    constexpr int kRows = 2;           // grid = kD / 2 = 1,152 CTAs
    __shared__ float smem_dot[kRows][kTopK];

    const int tid = threadIdx.x;
    const int k = tid >> 5;                // 8 warps = 8 selected experts
    const int lane = tid & 31;
    // `router_topk8` completed prior to `up_situ` + `quant_q8_16_packed`, so `sel_ids[k]` is already
    // visible before `pdl_wait()`, allowing `w2` weights to prefetch while `quant_q8_16_packed` finishes!
    const int ex = ld_ca_i32(sel_ids + k);
    const int row0 = blockIdx.x * kRows;
    const bool tail = lane < 4;

    uint4 pk_a[kRows], pk_b[kRows];
    uint32_t sc_a[kRows], sc_b[kRows];
    #pragma unroll
    for (int r = 0; r < kRows; ++r) {
        const size_t r_idx = static_cast<size_t>(ex) * kD + (row0 + r);
        const uint4* row_u128 = w2_u128 + r_idx * 36u;
        const size_t sc_start = r_idx * 18u;
        pk_a[r] = row_u128[lane];
        sc_a[r] = expert_scale_word<PackedScales, 6, 64>(s2_u32, sc_start + (lane >> 1));
        if (tail) {
            const int u16_b = lane + 32;
            pk_b[r] = row_u128[u16_b];
            sc_b[r] = expert_scale_word<PackedScales, 6, 64>(s2_u32, sc_start + (u16_b >> 1));
        } else {
            pk_b[r] = make_uint4(0u, 0u, 0u, 0u);
            sc_b[r] = 0u;
        }
    }

    pdl_wait();
    act_q8_u128 = pdl_in(act_q8_u128);
    act_sc16 = pdl_in(act_sc16);
    sel_ids = pdl_in(sel_ids);
    sel_wts = pdl_in(sel_wts);

    const uint4* aq = act_q8_u128 + k * 72;
    const float* asc = act_sc16 + k * 72;
    const uint4 xa0 = aq[lane * 2];
    const uint4 xa1 = aq[lane * 2 + 1];
    const float sa0 = asc[lane * 2];
    const float sa1 = asc[lane * 2 + 1];
    uint4 xb0 = make_uint4(0u, 0u, 0u, 0u), xb1 = xb0;
    float sb0 = 0.0f, sb1 = 0.0f;
    if (tail) {
        xb0 = aq[(lane + 32) * 2];
        xb1 = aq[(lane + 32) * 2 + 1];
        sb0 = asc[(lane + 32) * 2];
        sb1 = asc[(lane + 32) * 2 + 1];
    }

    auto step_hoisted = [&](float acc, int u16, const uint4& pk16, uint32_t sc4,
                            const uint4& xq0, const uint4& xq1, float xs0, float xs1) {
        int sh = (u16 & 1) * 16;
        float s0 = __fmul_rn(e4m3_byte_to_f32((sc4 >> sh) & 0xFFu) * 0.5f, xs0);
        float s1 = __fmul_rn(e4m3_byte_to_f32((sc4 >> (sh + 8)) & 0xFFu) * 0.5f, xs1);
        int d0 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.x & 0xFFFFu)), static_cast<int>(xq0.x), 0);
        d0 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.x >> 16)), static_cast<int>(xq0.y), d0);
        d0 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.y & 0xFFFFu)), static_cast<int>(xq0.z), d0);
        d0 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.y >> 16)), static_cast<int>(xq0.w), d0);
        int d1 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.z & 0xFFFFu)), static_cast<int>(xq1.x), 0);
        d1 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.z >> 16)), static_cast<int>(xq1.y), d1);
        d1 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.w & 0xFFFFu)), static_cast<int>(xq1.z), d1);
        d1 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.w >> 16)), static_cast<int>(xq1.w), d1);
        return __fmaf_rn(static_cast<float>(d1), s1, __fmaf_rn(static_cast<float>(d0), s0, acc));
    };

    #pragma unroll
    for (int r = 0; r < kRows; ++r) {
        float acc = step_hoisted(0.0f, lane, pk_a[r], sc_a[r], xa0, xa1, sa0, sa1);
        if (tail) acc = step_hoisted(acc, lane + 32, pk_b[r], sc_b[r], xb0, xb1, sb0, sb1);
        const float dot = warp_sum(acc);
        if (lane == 0) smem_dot[r][k] = dot;
    }
    __syncthreads();
    pdl_trigger();
    if (tid < kRows) {
        float sum_y = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < kTopK; ++kk) {
            const int e = sel_ids[kk];
            sum_y = __fmaf_rn(__fmul_rn(smem_dot[tid][kk], a2[e] * (1.0f / 2688.0f)), sel_wts[kk], sum_y);
        }
        moe_out[row0 + tid] = sum_y;
    }
}

extern "C" __global__ void ink2_inc_pos(int32_t* __restrict__ pos_ptr) {
    pdl_wait();
    pdl_trigger();
    pos_ptr = pdl_in_mut(pos_ptr);
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
    pdl_wait();
    pdl_trigger();
    src = pdl_in(src);
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

    pdl_wait();
    logits = pdl_in(logits);
    pos_ptr = pdl_in_mut(pos_ptr);

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
    pdl_wait();
    pdl_trigger();
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
        sq = __fadd_rn(sq, __fmaf_rn(v0, v0, __fmul_rn(v1, v1)));
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
        out_x[idx * 2] = __fmul_rn(out_x[idx * 2], __fmul_rn(scale, bf16_to_f32(static_cast<uint16_t>(nw & 0xFFFFu))));
        out_x[idx * 2 + 1] = __fmul_rn(out_x[idx * 2 + 1], __fmul_rn(scale, bf16_to_f32(static_cast<uint16_t>(nw >> 16))));
    }
}

extern "C" __global__ void ink2_rmsnorm_d_b(
    const float* __restrict__ inp_base,             // [B, D]
    const uint32_t* __restrict__ gamma_bf16_pairs,  // [D / 2]
    float* __restrict__ out_base,                   // [B, D]
    int32_t has_gamma) {
    pdl_wait();
    pdl_trigger();
    __shared__ float smem_red[8];
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const float* inp = inp_base + static_cast<size_t>(b) * kD;
    float* out = out_base + static_cast<size_t>(b) * kD;

    float sq = 0.0f;
    for (int i = tid; i < kD; i += 256) {
        float v = inp[i];
        sq = __fmaf_rn(v, v, sq);
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
            out[idx * 2] = __fmul_rn(__fmul_rn(x0, k), bf16_to_f32(static_cast<uint16_t>(gw & 0xFFFFu)));
            out[idx * 2 + 1] = __fmul_rn(__fmul_rn(x1, k), bf16_to_f32(static_cast<uint16_t>(gw >> 16)));
        }
    } else {
        for (int i = tid; i < kD; i += 256) {
            out[i] = __fmul_rn(inp[i], k);
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
    pdl_wait();
    pdl_trigger();
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
        sq = __fmaf_rn(v, v, sq);
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
        float y0 = __fmul_rn(__fmul_rn(inp[base], k), bf16_to_f32(static_cast<uint16_t>(gw & 0xFFFFu)));
        float y1 = __fmul_rn(__fmul_rn(inp[base + 1], k), bf16_to_f32(static_cast<uint16_t>(gw >> 16)));
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
    pdl_wait();
    pdl_trigger();
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
    pdl_wait();
    pdl_trigger();
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

    // Reductions use the element -> thread mapping of `ink2_depth_mix` (`tid + t * 256`) so B >= 1
    // verify rows stay bit-identical to the B = 1 decode graph (`--spec 0 == --spec K`, #88).
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
            sq = __fmaf_rn(v, v, sq);
            dq = __fmaf_rn(v, q_reg[t], dq);
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

    float4* dst4 = reinterpret_cast<float4*>(out_mixed + (static_cast<size_t>(q_idx) * B_out + b_out) * kD);
    const size_t plane_stride4 = static_cast<size_t>(B_out) * 576u;
    #pragma unroll
    for (int p = 0; p < 3; ++p) {
        const int i4 = p * 256 + tid;
        if (i4 < 576) {
            float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int s = 0; s < total_states; ++s) {
                const float4* st4 = reinterpret_cast<const float4*>((s < n_blocks)
                    ? (blocks_buf + (static_cast<size_t>(s) * stride_B + b_src) * kD)
                    : (cur_opt + static_cast<size_t>(b_src) * kD));
                const float pr = smem_probs[s];
                const float4 sv = st4[i4];
                acc.x = __fmaf_rn(pr, sv.x, acc.x);
                acc.y = __fmaf_rn(pr, sv.y, acc.y);
                acc.z = __fmaf_rn(pr, sv.z, acc.z);
                acc.w = __fmaf_rn(pr, sv.w, acc.w);
            }
            dst4[i4] = acc;
            if (broadcast3) {
                dst4[plane_stride4 + i4] = acc;
                dst4[2u * plane_stride4 + i4] = acc;
            }
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

    // The first `c8` pass of the (constant) weights is loaded before `pdl_wait()` (#152 P0-B).
    uint4 pre[Rows];
    #pragma unroll
    for (int row = 0; row < Rows; ++row) {
        const size_t rank = static_cast<size_t>(c_idx) * 64 + first_row + row;
        pre[row] = down_bf16_u128[rank * 576 + tid];
    }
    pdl_wait();
    cur_base = pdl_in(cur_base);
    msg_base = pdl_in(msg_base);

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
        sq_a = __fmaf_rn(va, va, sq_a);
        sq_b = __fmaf_rn(vb, vb, sq_b);
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
    #pragma unroll
    for (int pass = 0; pass < 3; ++pass) {
        const int c8 = tid + pass * 256;
        if (c8 >= 576) break;
        const int offset = c8 * 8;
        const float* src = offset < kD
            ? (Rows > 1 ? cur_tile : cur) + offset
            : (Rows > 1 ? msg_tile : msg) + (offset - kD);
        const float scale = inv_ab[offset < kD ? 0 : 1];
        #pragma unroll
        for (int row = 0; row < Rows; ++row) {
            size_t rank = static_cast<size_t>(c_idx) * 64 + first_row + row;
            const uint4 pack = pass == 0 ? pre[row] : down_bf16_u128[rank * 576 + c8];
            uint32_t words[4] = {pack.x, pack.y, pack.z, pack.w};
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                float w0 = bf16_to_f32(static_cast<uint16_t>(words[p]));
                float w1 = bf16_to_f32(static_cast<uint16_t>(words[p] >> 16));
                float pair_dot = __fmaf_rn(w0, src[2 * p], __fmul_rn(w1, src[2 * p + 1]));
                dot[row] = __fmaf_rn(pair_dot, scale, dot[row]);
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
    // pdl_wait() runs inside ctrl_down_rows, after the weight prefetch.
    ctrl_down_rows<1>(cur_base, msg_base, down_bf16_u128, g_out_base, B);
}

extern "C" __global__ void ink2_ctrl_down_r8_b(
    const float* __restrict__ cur_base,       // [n_ctrls, B, D]
    const float* __restrict__ msg_base,       // [n_ctrls, B, D]
    const uint4* __restrict__ down_bf16_u128, // [n_ctrls, 64, 576]
    float* __restrict__ g_out_base,           // [n_ctrls, B, 64]
    int32_t B) {
    // pdl_wait() runs inside ctrl_down_rows, after the weight prefetch.
    ctrl_down_rows<8>(cur_base, msg_base, down_bf16_u128, g_out_base, B);
}

extern "C" __global__ void ink2_ctrl_down_r8_t2_b(
    const float* __restrict__ cur_base,       // [n_ctrls, B, D]
    const float* __restrict__ msg_base,       // [n_ctrls, B, D]
    const uint4* __restrict__ down_bf16_u128, // [n_ctrls, 64, 576]
    float* __restrict__ g_out_base,           // [n_ctrls, B, 64]
    int32_t B) {
    pdl_wait();
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
    pdl_wait();
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
    pdl_wait();
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
    pdl_wait();
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

// Multi-sequence rounds (#107): `ink2_q8_dp4a_gemm_b` with `kTiles` 8-token tiles per block.
// A warp loads its weight fragments once per K=16 step and reuses them for every tile, instead
// of one block (and one weight read) per 8 tokens. Each (row, token) keeps the same MMA K=16
// integer dot, FP32 `__fmaf_rn` sequence, virtual-lane combine and cross-warp sum as
// `ink2_q8_dp4a_gemm_b` (`0-ULP` to `ink2_q8_dp4a_gemv`). `Grid (ceil(rows / 16),
// ceil(B / (8 * kTiles)))`, `Block 256`.
template <int kTiles>
__device__ __forceinline__ void q8_gemm_ms_body(
    const uint4* __restrict__ w_q8_u128,
    const float* __restrict__ w_sc64,
    const uint4* __restrict__ x_q8_u128,
    const float* __restrict__ x_sc64,
    float* __restrict__ y_base,
    int32_t rows,
    int32_t cols,
    int32_t B,
    int32_t y_stride,
    float softcap) {
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int fragment = lane % 4;
    const int first_row = blockIdx.x * 16;
    const int first_token = blockIdx.y * 8 * kTiles;
    const int row0 = min(first_row + lane / 4, rows - 1);
    const int row1 = min(first_row + lane / 4 + 8, rows - 1);
    const int cols16 = cols / 16;
    const int cols64 = cols / 64;
    const auto* top = reinterpret_cast<const uint32_t*>(w_q8_u128 + static_cast<size_t>(row0) * cols16);
    const auto* bottom = reinterpret_cast<const uint32_t*>(w_q8_u128 + static_cast<size_t>(row1) * cols16);
    const float* top_scales = w_sc64 + static_cast<size_t>(row0) * cols64;
    const float* bottom_scales = w_sc64 + static_cast<size_t>(row1) * cols64;
    const uint32_t* inputs[kTiles];
    const float* input_scales[kTiles];
    #pragma unroll
    for (int t = 0; t < kTiles; ++t) {
        const int token = min(first_token + 8 * t + lane / 4, B - 1);
        inputs[t] = reinterpret_cast<const uint32_t*>(x_q8_u128 + static_cast<size_t>(token) * cols16);
        input_scales[t] = x_sc64 + static_cast<size_t>(token) * cols64;
    }
    __shared__ float partials[kTiles][8][128];
    float sum[kTiles][4] = {};
    float left[kTiles][4] = {};

    #pragma unroll 1
    for (int step = 0; step < 4; ++step) {
        float acc[kTiles][4] = {};
        const int k_lane = warp + (step % 2) * 16 + (step / 2) * 8;
        for (int k = k_lane; k < cols16; k += 32) {
            const unsigned a0 = top[k * 4 + fragment];
            const unsigned a1 = bottom[k * 4 + fragment];
            const float w0 = top_scales[k / 4];
            const float w1 = bottom_scales[k / 4];
            #pragma unroll
            for (int t = 0; t < kTiles; ++t) {
                const unsigned b = inputs[t][k * 4 + fragment];
                int d0, d1, d2, d3;
                mma_s8_k16(a0, a1, b, d0, d1, d2, d3);
                const float xs = input_scales[t][k / 4];
                const float x0 = __shfl_sync(0xffffffffu, xs, 8 * fragment);
                const float x1 = __shfl_sync(0xffffffffu, xs, 8 * fragment + 4);
                acc[t][0] = __fmaf_rn(float(d0), __fmul_rn(w0, x0), acc[t][0]);
                acc[t][1] = __fmaf_rn(float(d1), __fmul_rn(w0, x1), acc[t][1]);
                acc[t][2] = __fmaf_rn(float(d2), __fmul_rn(w1, x0), acc[t][2]);
                acc[t][3] = __fmaf_rn(float(d3), __fmul_rn(w1, x1), acc[t][3]);
            }
        }
        #pragma unroll
        for (int t = 0; t < kTiles; ++t) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (step % 2 == 0) {
                    left[t][i] = acc[t][i];
                } else if (step == 1) {
                    sum[t][i] = left[t][i] + acc[t][i];
                } else {
                    sum[t][i] += left[t][i] + acc[t][i];
                }
            }
        }
    }
    const int dst = (lane / 4) * 8 + 2 * fragment;
    #pragma unroll
    for (int t = 0; t < kTiles; ++t) {
        partials[t][warp][dst] = sum[t][0];
        partials[t][warp][dst + 1] = sum[t][1];
        partials[t][warp][dst + 64] = sum[t][2];
        partials[t][warp][dst + 65] = sum[t][3];
    }
    __syncthreads();

    #pragma unroll
    for (int t0 = 0; t0 < kTiles; t0 += 2) {
        const int t = t0 + tid / 128;
        const int e = tid % 128;
        const int out_row = first_row + e / 8;
        const int out_token = first_token + 8 * t + e % 8;
        if (t < kTiles && out_row < rows && out_token < B) {
            const float s0 = partials[t][0][e] + partials[t][4][e];
            const float s1 = partials[t][1][e] + partials[t][5][e];
            const float s2 = partials[t][2][e] + partials[t][6][e];
            const float s3 = partials[t][3][e] + partials[t][7][e];
            float value = (s0 + s2) + (s1 + s3);
            if (softcap > 0.0f) value = softcap * ptx_tanh(value * ptx_rcp(softcap));
            y_base[static_cast<size_t>(out_token) * y_stride + out_row] = value;
        }
    }
}

extern "C" __global__ void ink2_q8_dp4a_gemm_b_ms4(
    const uint4* __restrict__ w_q8_u128, const float* __restrict__ w_sc64,
    const uint4* __restrict__ x_q8_u128, const float* __restrict__ x_sc64,
    float* __restrict__ y_base, int32_t rows, int32_t cols, int32_t B, int32_t y_stride,
    float softcap) {
    pdl_wait();
    q8_gemm_ms_body<4>(w_q8_u128, w_sc64, x_q8_u128, x_sc64, y_base, rows, cols, B, y_stride,
                       softcap);
}

extern "C" __global__ void ink2_q8_dp4a_gemm_b_ms8(
    const uint4* __restrict__ w_q8_u128, const float* __restrict__ w_sc64,
    const uint4* __restrict__ x_q8_u128, const float* __restrict__ x_sc64,
    float* __restrict__ y_base, int32_t rows, int32_t cols, int32_t B, int32_t y_stride,
    float softcap) {
    pdl_wait();
    q8_gemm_ms_body<8>(w_q8_u128, w_sc64, x_q8_u128, x_sc64, y_base, rows, cols, B, y_stride,
                       softcap);
}

// Register-Persistent Causal `short_conv4` over `b = 0..B-1` (`0-ULP` bit-identical!)
// Optional `seq_out` (`[3 + B, channels]`) records `[hist3; x_0..x_{B-1}]` for `O(1)` speculative rewind!
// One channel of `ink2_short_conv4_b` (shared with `ink2_short_conv4_pair_b`).
__device__ __forceinline__ void short_conv4_b_channel(
    int c,
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
    float* __restrict__ seq_out,
    int32_t channels,
    int32_t B,
    int32_t has_dyn,
    int32_t do_res_add) {

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
    pdl_wait();
    pdl_trigger();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    short_conv4_b_channel(c, cur_base, in_stride, w_conv4, dyn_base, dyn_f4_stride, hist3, out_base,
                          out_stride, res_base, res_stride, seq_out, channels, B, has_dyn,
                          do_res_add);
}

// Arguments of one `ink2_short_conv4_b` launch, passed by value (pointers first: 88 bytes).
struct Conv4BArgs {
    const float* cur_base;
    const float4* w_conv4;
    const float4* dyn_base;
    float* hist3;
    float* out_base;
    const float* res_base;
    float* seq_out;
    int32_t in_stride;
    int32_t dyn_f4_stride;
    int32_t out_stride;
    int32_t res_stride;
    int32_t channels;
    int32_t B;
    int32_t has_dyn;
    int32_t do_res_add;
};

// Two independent `ink2_short_conv4_b` launches in one grid (`blockIdx.y` picks the K or V
// arguments): the per-channel arithmetic is unchanged. `Grid (ceil(max channels / 256), 2)`.
extern "C" __global__ void ink2_short_conv4_pair_b(Conv4BArgs a0, Conv4BArgs a1) {
    pdl_wait();
    pdl_trigger();
    const bool second = blockIdx.y != 0;
#define INK2_PICK(field) (second ? a1.field : a0.field)
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t channels = INK2_PICK(channels);
    if (c >= channels) return;
    short_conv4_b_channel(c, INK2_PICK(cur_base), INK2_PICK(in_stride), INK2_PICK(w_conv4),
                          INK2_PICK(dyn_base), INK2_PICK(dyn_f4_stride), INK2_PICK(hist3),
                          INK2_PICK(out_base), INK2_PICK(out_stride), INK2_PICK(res_base),
                          INK2_PICK(res_stride), INK2_PICK(seq_out), channels, INK2_PICK(B),
                          INK2_PICK(has_dyn), INK2_PICK(do_res_add));
#undef INK2_PICK
}

// Row `b` of a multi-sequence batch (#107): `ms_rows[b] = {pos, seq}`; sequence `seq`'s KV
// cache / scale plane starts `seq * seq_kv_bytes` / `seq * seq_scale_floats` after sequence 0.
template <bool kMs>
__device__ __forceinline__ void prep_qkv_and_cache_b_body(
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
    int32_t is_global,
    const int2* __restrict__ ms_rows = nullptr,
    long long seq_kv_bytes = 0,
    int32_t seq_scale_floats = 0) {
    const int g_warp = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int b = blockIdx.y;
    const int lane = threadIdx.x & 31;
    int pos;
    if constexpr (kMs) {
        const int2 row = ms_rows[b];
        pos = row.x;
        k_cache_fp16 = reinterpret_cast<uint16_t*>(reinterpret_cast<char*>(k_cache_fp16) + row.y * seq_kv_bytes);
        v_cache_fp16 = reinterpret_cast<uint16_t*>(reinterpret_cast<char*>(v_cache_fp16) + row.y * seq_kv_bytes);
        if (k_scale != nullptr) {
            k_scale += static_cast<size_t>(row.y) * seq_scale_floats;
            v_scale += static_cast<size_t>(row.y) * seq_scale_floats;
        }
    } else {
        pos = (*pos_ptr) + b;
    }
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

// Multi-sequence rows (#107): grid `(channels / 256, B)`. Row `b` belongs to sequence
// `ms_rows[b].y` and steps that sequence's own history exactly like the B = 1 decode kernel.
template <bool kSave>
__device__ __forceinline__ void short_conv4_ms_body(
    const float* __restrict__ cur_base, int32_t in_stride, const float4* __restrict__ w_conv4,
    const float4* __restrict__ dyn_base, int32_t dyn_f4_stride, float* __restrict__ hist3,
    float* out_base, int32_t out_stride, const float* res_base, int32_t res_stride,
    float* __restrict__ save_prev_base, const int2* __restrict__ ms_rows, int32_t channels,
    int32_t has_dyn, int32_t do_res_add, int32_t hist_seq_floats) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (c >= channels) return;
    const size_t ob = static_cast<size_t>(b) * out_stride + c;
    if (kSave && save_prev_base != nullptr) {
        save_prev_base[ob] = out_base[ob];
    }
    float* h = hist3 + static_cast<size_t>(ms_rows[b].y) * hist_seq_floats + c * 3;
    const float h0 = h[0];
    const float h1 = h[1];
    const float h2 = h[2];
    const float4 w = w_conv4[c];
    const int head = c >> 7;
    const float x = cur_base[static_cast<size_t>(b) * in_stride + c];
    const float4 d = has_dyn ? dyn_base[static_cast<size_t>(b) * dyn_f4_stride + head] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float res = do_res_add ? res_base[static_cast<size_t>(b) * res_stride + c] : 0.0f;
    const float y = eval_conv4_step(w, h0, h1, h2, x, has_dyn, d, do_res_add, res);
    h[0] = h1;
    h[1] = h2;
    h[2] = x;
    out_base[ob] = y;
}
extern "C" __global__ void ink2_short_conv4_ms(
    const float* __restrict__ cur_base, int32_t in_stride, const float4* __restrict__ w_conv4,
    const float4* __restrict__ dyn_base, int32_t dyn_f4_stride, float* __restrict__ hist3,
    float* out_base, int32_t out_stride, const float* res_base, int32_t res_stride,
    const int2* __restrict__ ms_rows, int32_t channels, int32_t has_dyn, int32_t do_res_add,
    int32_t hist_seq_floats) {
    pdl_wait();
    short_conv4_ms_body<false>(cur_base, in_stride, w_conv4, dyn_base, dyn_f4_stride, hist3, out_base,
        out_stride, res_base, res_stride, nullptr, ms_rows, channels, has_dyn, do_res_add, hist_seq_floats);
}
extern "C" __global__ void ink2_short_conv4_save_ms(
    const float* __restrict__ cur_base, int32_t in_stride, const float4* __restrict__ w_conv4,
    const float4* __restrict__ dyn_base, int32_t dyn_f4_stride, float* __restrict__ hist3,
    float* out_base, int32_t out_stride, const float* res_base, int32_t res_stride,
    float* __restrict__ save_prev_base, const int2* __restrict__ ms_rows, int32_t channels,
    int32_t has_dyn, int32_t do_res_add, int32_t hist_seq_floats) {
    pdl_wait();
    short_conv4_ms_body<true>(cur_base, in_stride, w_conv4, dyn_base, dyn_f4_stride, hist3, out_base,
        out_stride, res_base, res_stride, save_prev_base, ms_rows, channels, has_dyn, do_res_add, hist_seq_floats);
}

// Multi-sequence verify rows (#107 stage 3): rows of one sequence are contiguous and
// `ms_j[b]` is row `b`'s index inside its sequence. Row `b` sees the window
// `L[j], L[j + 1], L[j + 2], x_b` of `L = [hist0, hist1, hist2, x_{b - j}, ...]`, i.e. the
// same inputs a B = 1 decode of that sequence would see after the earlier rows, with the
// same `eval_conv4_step` arithmetic. The history is not written here: each input is recorded
// in `xrec[b]` and `ink2_conv_commit_ms` keeps the accepted prefix afterwards.
template <bool kSave>
__device__ __forceinline__ void short_conv4_vms_body(
    const float* __restrict__ cur_base, int32_t in_stride, const float4* __restrict__ w_conv4,
    const float4* __restrict__ dyn_base, int32_t dyn_f4_stride, const float* __restrict__ hist3,
    float* out_base, int32_t out_stride, const float* res_base, int32_t res_stride,
    float* __restrict__ save_prev_base, const int2* __restrict__ ms_rows,
    const int32_t* __restrict__ ms_j, float* __restrict__ xrec, int32_t channels,
    int32_t has_dyn, int32_t do_res_add, int32_t hist_seq_floats) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (c >= channels) return;
    const size_t ob = static_cast<size_t>(b) * out_stride + c;
    if (kSave && save_prev_base != nullptr) {
        save_prev_base[ob] = out_base[ob];
    }
    const float* h = hist3 + static_cast<size_t>(ms_rows[b].y) * hist_seq_floats + c * 3;
    const int j = ms_j[b];
    float win[3];
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int idx = j + i;  // index into L
        win[i] = idx < 3 ? h[idx] : cur_base[static_cast<size_t>(b - j + idx - 3) * in_stride + c];
    }
    const float4 w = w_conv4[c];
    const int head = c >> 7;
    const float x = cur_base[static_cast<size_t>(b) * in_stride + c];
    xrec[static_cast<size_t>(b) * channels + c] = x;
    const float4 d = has_dyn ? dyn_base[static_cast<size_t>(b) * dyn_f4_stride + head] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float res = do_res_add ? res_base[static_cast<size_t>(b) * res_stride + c] : 0.0f;
    out_base[ob] = eval_conv4_step(w, win[0], win[1], win[2], x, has_dyn, d, do_res_add, res);
}
extern "C" __global__ void ink2_short_conv4_vms(
    const float* __restrict__ cur_base, int32_t in_stride, const float4* __restrict__ w_conv4,
    const float4* __restrict__ dyn_base, int32_t dyn_f4_stride, const float* __restrict__ hist3,
    float* out_base, int32_t out_stride, const float* res_base, int32_t res_stride,
    const int2* __restrict__ ms_rows, const int32_t* __restrict__ ms_j, float* __restrict__ xrec,
    int32_t channels, int32_t has_dyn, int32_t do_res_add, int32_t hist_seq_floats) {
    pdl_wait();
    short_conv4_vms_body<false>(cur_base, in_stride, w_conv4, dyn_base, dyn_f4_stride, hist3, out_base,
        out_stride, res_base, res_stride, nullptr, ms_rows, ms_j, xrec, channels, has_dyn, do_res_add,
        hist_seq_floats);
}
extern "C" __global__ void ink2_short_conv4_save_vms(
    const float* __restrict__ cur_base, int32_t in_stride, const float4* __restrict__ w_conv4,
    const float4* __restrict__ dyn_base, int32_t dyn_f4_stride, const float* __restrict__ hist3,
    float* out_base, int32_t out_stride, const float* res_base, int32_t res_stride,
    float* __restrict__ save_prev_base, const int2* __restrict__ ms_rows,
    const int32_t* __restrict__ ms_j, float* __restrict__ xrec, int32_t channels,
    int32_t has_dyn, int32_t do_res_add, int32_t hist_seq_floats) {
    pdl_wait();
    short_conv4_vms_body<true>(cur_base, in_stride, w_conv4, dyn_base, dyn_f4_stride, hist3, out_base,
        out_stride, res_base, res_stride, save_prev_base, ms_rows, ms_j, xrec, channels, has_dyn,
        do_res_add, hist_seq_floats);
}

// Keep each verified sequence's accepted prefix in its `short_conv4` histories (#107 stage 3).
// `runs[r] = {first row, rows kept, sequence, 0}`; the new history is `L[keep .. keep + 2]` of
// `L = [hist0, hist1, hist2, x_first, ...]` with the inputs recorded by `ink2_short_conv4_vms`.
// Grid `(9, 48 * 4, runs)`, Block `256`; `xrec_base` is `[48][6656 * rows_stride]` floats with
// the four kinds at `0, 1024, 2048, 4352` x `rows_stride`.
extern "C" __global__ void ink2_conv_commit_ms(
    const float* __restrict__ xrec_base,
    const uint64_t* __restrict__ hist_ptrs,   // [seq][48 * 4]
    const int4* __restrict__ runs,
    int32_t rows_stride) {
    pdl_wait();
    const int slot = blockIdx.y >> 2;
    const int kind = blockIdx.y & 3;
    const int c = blockIdx.x * 256 + threadIdx.x;
    const int ch = kind < 2 ? 1024 : 2304;
    if (c >= ch) return;
    const int kind_off = kind == 0 ? 0 : (kind == 1 ? 1024 : (kind == 2 ? 2048 : 4352));
    const int4 run = runs[blockIdx.z];
    const float* xr = xrec_base + static_cast<size_t>(slot) * 6656u * rows_stride
                      + static_cast<size_t>(kind_off) * rows_stride;
    float* h = reinterpret_cast<float*>(hist_ptrs[static_cast<size_t>(run.z) * 192 + blockIdx.y]) + c * 3;
    float L[3];
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int idx = run.y + i;
        L[i] = idx < 3 ? h[idx] : xr[static_cast<size_t>(run.x + idx - 3) * ch + c];
    }
    h[0] = L[0];
    h[1] = L[1];
    h[2] = L[2];
}

// Grid: `(5, B)`, Block: `256`
extern "C" __global__ void ink2_prep_qkv_and_cache_b(
    float* __restrict__ qrgs_base, int32_t qrgs_stride,
    const float* __restrict__ k_in_base, const float* __restrict__ v_in_base,
    const uint32_t* __restrict__ q_clip_bf16, const uint32_t* __restrict__ r_clip_bf16,
    uint16_t* __restrict__ k_cache_fp16, uint16_t* __restrict__ v_cache_fp16,
    float* __restrict__ k_scale, float* __restrict__ v_scale,
    const int32_t* __restrict__ pos_ptr, int32_t kv_cap, int32_t is_global) {
    pdl_wait();
    pdl_trigger();
    prep_qkv_and_cache_b_body<false>(qrgs_base, qrgs_stride, k_in_base, v_in_base, q_clip_bf16,
        r_clip_bf16, k_cache_fp16, v_cache_fp16, k_scale, v_scale, pos_ptr, kv_cap, is_global);
}

// Multi-sequence rows (#107): each row writes its own sequence's KV at its own position.
extern "C" __global__ void ink2_prep_qkv_and_cache_ms(
    float* __restrict__ qrgs_base, int32_t qrgs_stride,
    const float* __restrict__ k_in_base, const float* __restrict__ v_in_base,
    const uint32_t* __restrict__ q_clip_bf16, const uint32_t* __restrict__ r_clip_bf16,
    uint16_t* __restrict__ k_cache_fp16, uint16_t* __restrict__ v_cache_fp16,
    float* __restrict__ k_scale, float* __restrict__ v_scale,
    const int2* __restrict__ ms_rows, int32_t kv_cap, int32_t is_global,
    long long seq_kv_bytes, int32_t seq_scale_floats) {
    pdl_wait();
    prep_qkv_and_cache_b_body<true>(qrgs_base, qrgs_stride, k_in_base, v_in_base, q_clip_bf16,
        r_clip_bf16, k_cache_fp16, v_cache_fp16, k_scale, v_scale, nullptr, kv_cap, is_global,
        ms_rows, seq_kv_bytes, seq_scale_floats);
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

    pdl_wait();
    pos_ptr = pdl_in(pos_ptr);
    qrgs_base = pdl_in(qrgs_base);
    k_cache_fp16 = pdl_in(k_cache_fp16);
    v_cache_fp16 = pdl_in(v_cache_fp16);
    k_scale = pdl_in(k_scale);
    v_scale = pdl_in(v_scale);

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
    pdl_wait();
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

// Grid: `(128, ceil(B / 4))`, Block: `32`. One warp per expert row; the row's 9 weight
// vectors are loaded before `pdl_wait()` so they overlap the preceding kernel. Each lane
// accumulates `c8 = lane, lane + 32, ...` and `warp_sum` reduces, exactly as
// `ink2_bf16_gemm_router_b` and `ink2_router_gemv_topk8`, so the logits are bit-identical.
extern "C" __global__ void ink2_bf16_gemv_router_b(
    const uint4* __restrict__ w_bf16_u128, // [128, 288]
    const float4* __restrict__ x4_base,    // [B, 576]
    float* __restrict__ y_base,            // [B, 128]
    int32_t B) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const uint4* w_row = w_bf16_u128 + static_cast<size_t>(row) * 288u;
    constexpr int kIters = 288 / 32;
    uint4 w[kIters];
    #pragma unroll
    for (int i = 0; i < kIters; ++i) w[i] = w_row[lane + 32 * i];

    pdl_wait();
    pdl_trigger();
    x4_base = pdl_in(x4_base);
    const int b_base = blockIdx.y * 4;
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        if (b_base + t >= B) break;
        const float4* x4 = x4_base + static_cast<size_t>(b_base + t) * 576u;
        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < kIters; ++i) {
            const int c8 = lane + 32 * i;
            acc = accum_bf16_8(acc, w[i], x4[c8 * 2], x4[c8 * 2 + 1]);
        }
        acc = warp_sum(acc);
        if (lane == 0) y_base[(b_base + t) * kExp + row] = acc;
    }
}

// Grid: `(B, 1)`, Block: `32`
// Top-8 of one token's 128 router logits (one warp); shared by `ink2_router_topk8_b` and
// `ink2_router_topk8_groups_b`.
__device__ __forceinline__ void router_topk8_row(
    const float* __restrict__ logits128_base,
    const float* __restrict__ bias128,
    int32_t* __restrict__ sel_ids_base,
    float* __restrict__ sel_wts_base,
    int* smem_ids,
    float* smem_ps,
    int b,
    int lane) {
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

extern "C" __global__ void ink2_router_topk8_b(
    const float* __restrict__ logits128_base, // [B, 128]
    const float* __restrict__ bias128,        // [128]
    int32_t* __restrict__ sel_ids_base,       // [B, 8]
    float* __restrict__ sel_wts_base) {       // [B, 8]
    pdl_wait();
    pdl_trigger();
    __shared__ int smem_ids[8];
    __shared__ float smem_ps[8];

    const int b = blockIdx.x;
    const int lane = threadIdx.x & 31;
    router_topk8_row(logits128_base, bias128, sel_ids_base, sel_wts_base, smem_ids, smem_ps, b, lane);
}

// Deterministic Zero-Atomic Expert Grouping (`1` CTA, `128` threads = `1` thread/expert!)
extern "C" __global__ void ink2_build_expert_groups_b(
    const int32_t* __restrict__ sel_ids, // [B * 8]
    int32_t* __restrict__ exp_counts,    // [128]
    int32_t* __restrict__ exp_toks,      // [128, 64]
    int32_t* __restrict__ exp_pairs,     // [128, 64]
    int32_t B) {
    pdl_wait();
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
    pdl_wait();
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

// Grid: 128 CTAs of 256 threads (8 warps per expert). Partitions `[0, B * 8)` across
// 8 warps monotonically with an 8-element SMEM prefix sum: 32x higher warp parallelism
// (`1,024` warps vs `32`) while preserving 100% bit-for-bit identical `exp_toks`/`exp_pairs`!
extern "C" __global__ void ink2_build_expert_groups_par_b(
    const int32_t* __restrict__ sel_ids,
    int32_t* __restrict__ exp_counts,
    int32_t* __restrict__ exp_toks,
    int32_t* __restrict__ exp_pairs,
    int32_t B) {
    pdl_wait();
    __shared__ int32_t sm_wcnt[8];
    __shared__ int32_t sm_woff[8];

    const int expert = blockIdx.x;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int total_pairs = B * 8;
    const int chunk = (((total_pairs + 7) >> 3) + 31) & ~31;
    const int w_begin = min(total_pairs, warp_id * chunk);
    const int w_end = min(total_pairs, w_begin + chunk);

    int my_cnt = 0;
    for (int first = w_begin; first < w_end; first += 32) {
        const int pair = first + lane;
        const bool match = pair < w_end && sel_ids[pair] == expert;
        my_cnt += __popc(__ballot_sync(0xffffffffu, match));
    }
    if (lane == 0) {
        sm_wcnt[warp_id] = my_cnt;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int acc = 0;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            sm_woff[w] = acc;
            acc += sm_wcnt[w];
        }
        exp_counts[expert] = acc;
    }
    __syncthreads();

    int count = sm_woff[warp_id];
    for (int first = w_begin; first < w_end; first += 32) {
        const int pair = first + lane;
        const bool match = pair < w_end && sel_ids[pair] == expert;
        const unsigned hits = __ballot_sync(0xffffffffu, match);
        if (match) {
            const int rank = count + __popc(hits & ((1u << lane) - 1u));
            const int dst = expert_map_offset(expert, rank);
            exp_toks[dst] = pair >> 3;
            exp_pairs[dst] = pair;
        }
        count += __popc(hits);
    }
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
INK2_INDEXED_KERNEL(ink2_moe_mxfp8_grouped_up_situ, , (
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
            const Mxfp8Row<Indexed> r_u128(w13_u128, row_idx * 144u);
            const unsigned scale_start = row_idx * 18u;

            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            // Issue the row's 4-5 weight loads before the first FMA (same accumulation order).
            uint4 pw[5];
            float pes[5];
            #pragma unroll
            for (int i = 0; i < 5; ++i) {
                const int u16 = lane + 32 * i;
                if (u16 < 144) {
                    pw[i] = r_u128[u16];
                    pes[i] = load_e8m0_scale<PackedScales>(s13_u32, scale_start, u16);
                }
            }
            #pragma unroll
            for (int i = 0; i < 5; ++i) {
                const int u16 = lane + 32 * i;
                if (u16 >= 144) break;
                const uint4 p = pw[i];
                const float es = pes[i];
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
INK2_INDEXED_KERNEL(ink2_moe_mxfp8_grouped_down, , (
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
            const Mxfp8Row<Indexed> r_u128(w2_u128, r_idx * 72u);
            const unsigned scale_start = r_idx * 9u;

            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
            // Issue the row's 2-3 weight loads before the first FMA (same accumulation order).
            uint4 pw[3];
            float pes[3];
            #pragma unroll
            for (int i = 0; i < 3; ++i) {
                const int u16 = lane + 32 * i;
                if (u16 < 72) {
                    pw[i] = r_u128[u16];
                    pes[i] = load_e8m0_scale<PackedScales>(s2_u32, scale_start, u16);
                }
            }
            #pragma unroll
            for (int i = 0; i < 3; ++i) {
                const int u16 = lane + 32 * i;
                if (u16 >= 72) break;
                const uint4 p = pw[i];
                const float es = pes[i];
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
    pdl_wait();
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
    pdl_wait();
    if (threadIdx.x == 0) {
        *pos_ptr += delta;
    }
}

// Multi-Token Greedy ArgMax (`Grid (B_out, 1)`, `Block 256`) for Speculative Verification!
extern "C" __global__ void ink2_argmax_b(
    const float* __restrict__ logits_base, // [B_out, VOCAB]
    int32_t* __restrict__ out_toks) {      // [B_out]
    pdl_wait();
    pdl_trigger();
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
    pdl_wait();
    pdl_trigger();
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
    pdl_wait();
    pdl_trigger();
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
    pdl_wait();
    pdl_trigger();
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
    pdl_wait();
    pdl_trigger();
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
    pdl_wait();
    pdl_trigger();
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
    pdl_wait();
    pdl_trigger();
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
    pdl_wait();
    pdl_trigger();
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
    pdl_wait();
    pdl_trigger();
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

// Long-context DSpark attention (#107): `mtp_attn` with the keys `[lo..hi) + 8 block rows`
// split into `gridDim.y` slices of `kps` keys. Grid `(8 KV heads, S)`, Block 256: warp `t` is
// draft row `t` and keeps the online softmax of its 3 query heads over its slice (one K/V load
// per key per warp, shared through L1). `mtp_attn_merge` combines the slices.
extern "C" __global__ void mtp_attn_split(
    const float* __restrict__ qkv,            // [8, 5120]
    const uint16_t* __restrict__ kc_fp16,     // [cap, 1024]
    const uint16_t* __restrict__ vc_fp16,     // [cap, 1024]
    int32_t ring,
    int32_t lo,
    int32_t hi,
    int32_t kps,
    float* __restrict__ part_ml,              // [S, 8, 24, 2]
    float* __restrict__ part_o) {             // [S, 8, 24, 128]
    pdl_wait();
    pdl_trigger();
    const int kvh = blockIdx.x;
    const int s = blockIdx.y;
    const int t = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nc = hi - lo;
    const int total = nc + 8;
    const int j0 = s * kps;
    const int j1 = min(total, j0 + kps);
    const float scale = 0.08838834764831845f; // 1 / sqrt(128)

    float q[3][4];
    #pragma unroll
    for (int hh = 0; hh < 3; ++hh) {
        const float4 q4 = reinterpret_cast<const float4*>(
            qkv + static_cast<size_t>(t) * 5120u + (kvh * 3 + hh) * 128)[lane];
        q[hh][0] = q4.x; q[hh][1] = q4.y; q[hh][2] = q4.z; q[hh][3] = q4.w;
    }
    float m_w[3] = {-1e30f, -1e30f, -1e30f};
    float l_w[3] = {0.0f, 0.0f, 0.0f};
    float o[3][4] = {};

    auto load_kv = [&](int j, float (&k)[4], float (&v)[4]) {
        if (j < nc) {
            const int cpos = lo + j;
            const int row = (ring > 0) ? (cpos % ring) : cpos;
            const uint64_t kw = reinterpret_cast<const uint64_t*>(
                kc_fp16 + static_cast<size_t>(row) * 1024u + kvh * 128)[lane];
            const uint64_t vw = reinterpret_cast<const uint64_t*>(
                vc_fp16 + static_cast<size_t>(row) * 1024u + kvh * 128)[lane];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                k[i] = fp16_bits_to_f32(static_cast<uint16_t>((kw >> (16 * i)) & 0xFFFFu));
                v[i] = fp16_bits_to_f32(static_cast<uint16_t>((vw >> (16 * i)) & 0xFFFFu));
            }
        } else {
            const float* b_row = qkv + static_cast<size_t>(j - nc) * 5120u;
            const float4 kv4 = reinterpret_cast<const float4*>(b_row + 3072 + kvh * 128)[lane];
            const float4 vv4 = reinterpret_cast<const float4*>(b_row + 4096 + kvh * 128)[lane];
            k[0] = kv4.x; k[1] = kv4.y; k[2] = kv4.z; k[3] = kv4.w;
            v[0] = vv4.x; v[1] = vv4.y; v[2] = vv4.z; v[3] = vv4.w;
        }
    };

    float kc[4], vc[4];
    if (j0 < j1) load_kv(j0, kc, vc);
    for (int j = j0; j < j1; ++j) {
        float kn[4], vn[4];
        if (j + 1 < j1) load_kv(j + 1, kn, vn);
        float dots[3];
        #pragma unroll
        for (int hh = 0; hh < 3; ++hh) {
            dots[hh] = q[hh][0] * kc[0] + q[hh][1] * kc[1] + q[hh][2] * kc[2] + q[hh][3] * kc[3];
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
            const float dot = dots[hh] * scale;
            const float m_new = (dot > m_w[hh]) ? dot : m_w[hh];
            const float alpha = (l_w[hh] > 0.0f) ? ptx_exp(m_w[hh] - m_new) : 0.0f;
            const float p = ptx_exp(dot - m_new);
            l_w[hh] = l_w[hh] * alpha + p;
            m_w[hh] = m_new;
            #pragma unroll
            for (int i = 0; i < 4; ++i) o[hh][i] = o[hh][i] * alpha + p * vc[i];
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) { kc[i] = kn[i]; vc[i] = vn[i]; }
    }
    #pragma unroll
    for (int hh = 0; hh < 3; ++hh) {
        const size_t idx = (static_cast<size_t>(s) * 8 + t) * 24 + kvh * 3 + hh;
        reinterpret_cast<float4*>(part_o + idx * 128)[lane] = make_float4(o[hh][0], o[hh][1], o[hh][2], o[hh][3]);
        if (lane == 0) {
            part_ml[idx * 2] = m_w[hh];
            part_ml[idx * 2 + 1] = l_w[hh];
        }
    }
}

// Grid `(24, 8)`, Block 128: merges the `S` slices of `mtp_attn_split` into `att_out`.
extern "C" __global__ void mtp_attn_merge(
    const float* __restrict__ part_ml,
    const float* __restrict__ part_o,
    int32_t S,
    float* __restrict__ att_out) {            // [8, 3072]
    pdl_wait();
    pdl_trigger();
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int d = threadIdx.x;
    float m_max = -1e30f;
    for (int s = 0; s < S; ++s) {
        const size_t idx = (static_cast<size_t>(s) * 8 + t) * 24 + h;
        if (part_ml[idx * 2 + 1] > 0.0f) m_max = fmaxf(m_max, part_ml[idx * 2]);
    }
    float denom = 0.0f, sum_d = 0.0f;
    for (int s = 0; s < S; ++s) {
        const size_t idx = (static_cast<size_t>(s) * 8 + t) * 24 + h;
        const float l = part_ml[idx * 2 + 1];
        if (l > 0.0f) {
            const float sc = ptx_exp(part_ml[idx * 2] - m_max);
            denom += l * sc;
            sum_d += part_o[idx * 128 + d] * sc;
        }
    }
    att_out[static_cast<size_t>(t) * 3072u + h * 128 + d] = sum_d * ptx_rcp(fmaxf(denom, 1e-20f));
}

extern "C" __global__ void mtp_add_inplace(
    float* __restrict__ x,
    const float* __restrict__ o,
    int32_t n) {
    pdl_wait();
    pdl_trigger();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += o[i];
}

extern "C" __global__ void mtp_silu_mul(
    const float* __restrict__ gu, // [8, 18432]
    float* __restrict__ act) {    // [8, 9216]
    pdl_wait();
    pdl_trigger();
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
    pdl_wait();
    pdl_trigger();
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
    pdl_wait();
    pdl_trigger();
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
    pdl_wait();
    pdl_trigger();
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
    pdl_wait();
    pdl_trigger();
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

// Unit-Stride `512B`-Coalesced Load-Hoisted Planar `Q8_64` GEMM for `1 <= B <= 8`, `cols <= 3072`
// (`DSpark` Verify `wq_all`/`qkv`/`wo`/`unembed` + MTP `unembed_q8`, `Grid: (ceil(rows / 16), 1)`, `Block: 256`):
// Eliminates `ink2_q8_dp4a_gemm_b`'s 8-row strided `50%` L2 sector waste while preserving exact per-lane
// `c16 = lane + 32*i` `__dp4a` + `__fmaf_rn` + `warp_sum` (`0-ULP` bit-identical to `ink2_q8_dp4a_gemv`!).
extern "C" __global__ void ink2_q8_dp4a_gemm_b8_fast(
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
    __shared__ uint4 smem_x[8][192];   // 24,576 B (`max cols = 3072`)
    __shared__ float smem_xsc[8][48];  // 1,536 B

    const int tid = threadIdx.x; // 256
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int cols16 = cols >> 4;
    const int cols64 = cols >> 6;
    const int row0 = (blockIdx.x * 8 + warp_id) * 2;
    const bool has_row0 = (row0 < rows);
    const bool has_row1 = (row0 + 1 < rows);

    // The weights are constant: for the unrolled widths (`cols16 = 144 / 192`) they are loaded
    // before `pdl_wait()`, overlapping the kernel that produces `x` (#152 P0-B).
    const uint4* w_r0 = w_q8_u128 + static_cast<size_t>(has_row0 ? row0 : 0) * cols16;
    const uint4* w_r1 = has_row1 ? (w_r0 + cols16) : w_r0;
    const float* s_r0 = w_sc64 + static_cast<size_t>(has_row0 ? row0 : 0) * cols64;
    const float* s_r1 = has_row1 ? (s_r0 + cols64) : s_r0;
    uint4 w0[6], w1[6];
    float ws0[6], ws1[6];
    uint4 w0_4 = make_uint4(0u, 0u, 0u, 0u), w1_4 = w0_4;
    float ws0_4 = 0.0f, ws1_4 = 0.0f;
    if (cols16 == 144) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c16 = lane + 32 * i;
            w0[i] = w_r0[c16];
            w1[i] = w_r1[c16];
            ws0[i] = s_r0[c16 >> 2];
            ws1[i] = s_r1[c16 >> 2];
        }
        if (lane < 16) {
            const int c16 = lane + 128;
            w0_4 = w_r0[c16];
            w1_4 = w_r1[c16];
            ws0_4 = s_r0[c16 >> 2];
            ws1_4 = s_r1[c16 >> 2];
        }
    } else if (cols16 == 192) {
        #pragma unroll
        for (int i = 0; i < 6; ++i) {
            const int c16 = lane + 32 * i;
            w0[i] = w_r0[c16];
            w1[i] = w_r1[c16];
            ws0[i] = s_r0[c16 >> 2];
            ws1[i] = s_r1[c16 >> 2];
        }
    }

    pdl_wait();
    pdl_trigger();
    x_q8_u128 = pdl_in(x_q8_u128);
    x_sc64 = pdl_in(x_sc64);
    for (int t = 0; t < B; ++t) {
        if (tid < cols16) smem_x[t][tid] = x_q8_u128[static_cast<size_t>(t) * cols16 + tid];
        if (tid < cols64) smem_xsc[t][tid] = x_sc64[static_cast<size_t>(t) * cols64 + tid];
    }
    __syncthreads();
    if (!has_row0) return;

    float a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float b[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    auto step_col = [&](int c16, const uint4& w0, const uint4& w1, float ws0, float ws1) {
        const int c64 = c16 >> 2;
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < B) {
                uint4 xq = smem_x[t][c16];
                float xs = smem_xsc[t][c64];
                int d0 = dp4a_u128(w0, xq);
                a[t] = __fmaf_rn(static_cast<float>(d0), __fmul_rn(ws0, xs), a[t]);
                if (has_row1) {
                    int d1 = dp4a_u128(w1, xq);
                    b[t] = __fmaf_rn(static_cast<float>(d1), __fmul_rn(ws1, xs), b[t]);
                }
            }
        }
    };

    if (cols16 == 144) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            step_col(lane + 32 * i, w0[i], w1[i], ws0[i], ws1[i]);
        }
        if (lane < 16) {
            step_col(lane + 128, w0_4, w1_4, ws0_4, ws1_4);
        }
    } else if (cols16 == 192) {
        #pragma unroll
        for (int i = 0; i < 6; ++i) {
            step_col(lane + 32 * i, w0[i], w1[i], ws0[i], ws1[i]);
        }
    } else {
        for (int c16 = lane; c16 < cols16; c16 += 32) {
            step_col(c16, w_r0[c16], w_r1[c16], s_r0[c16 >> 2], s_r1[c16 >> 2]);
        }
    }

    const float inv_sc = (softcap > 0.0f) ? ptx_rcp(softcap) : 0.0f;
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        if (t < B) {
            float v0 = warp_sum(a[t]);
            float v1 = warp_sum(b[t]);
            if (lane == 0) {
                if (softcap > 0.0f) {
                    v0 = softcap * ptx_tanh(v0 * inv_sc);
                    v1 = softcap * ptx_tanh(v1 * inv_sc);
                }
                float* dst = y_base + static_cast<size_t>(t) * y_stride;
                dst[row0] = v0;
                if (has_row1) dst[row0 + 1] = v1;
            }
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
    pdl_wait();
    pdl_trigger();
    if (*cum_ptr < 0.0f) return;
    __shared__ uint4 smem_lq[16];
    __shared__ float smem_lsc[4];
    __shared__ float smem_bv[8];
    __shared__ float smem_bz[8];
    __shared__ int smem_bi[8];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int q = lane & 3;
    const int r = (blockIdx.x * 8 + warp_id) * 8 + (lane >> 2);

    uint4 wr0 = make_uint4(0u, 0u, 0u, 0u), wr1 = wr0, wr2 = wr0, wr3 = wr0;
    float fsc = 0.0f;
    float fb_val = 0.0f;
    float bl_val = 0.0f;
    if (r < kVocab) {
        const uint4* wr = fold_q8_u128 + static_cast<size_t>(r) * 16u + q * 4;
        wr0 = wr[0];
        wr1 = wr[1];
        wr2 = wr[2];
        wr3 = wr[3];
        fsc = fold_sc64[static_cast<size_t>(r) * 4u + q];
        if (q == 0) {
            fb_val = fold_b[r];
            bl_val = base_logits[r];
        }
    }
    if (tid < 16) smem_lq[tid] = lat_q8_u128[tid];
    if (tid < 4) smem_lsc[tid] = lat_sc64[tid];
    __syncthreads();

    float s = 0.0f;
    if (r < kVocab) {
        int d = dp4a_u128(wr0, smem_lq[q * 4]);
        d += dp4a_u128(wr1, smem_lq[q * 4 + 1]);
        d += dp4a_u128(wr2, smem_lq[q * 4 + 2]);
        d += dp4a_u128(wr3, smem_lq[q * 4 + 3]);
        s = static_cast<float>(d) * (fsc * smem_lsc[q]);
    }
    s += __shfl_xor_sync(0xFFFFFFFFu, s, 1);
    s += __shfl_xor_sync(0xFFFFFFFFu, s, 2);

    float v = (r < kVocab && q == 0)
        ? (30.0f * ptx_tanh(((s + fb_val) + bl_val) * (1.0f / 30.0f)))
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
    pdl_wait();
    pdl_trigger();
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
// kPipe: depth-1 pipelined FP16 row loop, used for decode / verify rows only (prefill keeps the
// lower-register generic loop). Same arithmetic either way.
template <bool kInt8, bool kPipe = false, bool kMs = false>
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
    float* __restrict__ part_o,                 // [B, 24, G, 128]
    const int2* __restrict__ ms_rows = nullptr, // kMs: per-row {pos, seq} (#107)
    long long seq_kv_bytes = 0,
    int32_t seq_scale_floats = 0) {
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

    pdl_wait();
    pos_ptr = pdl_in(pos_ptr);
    qrgs_base = pdl_in(qrgs_base);
    k_cache_fp16 = pdl_in(k_cache_fp16);
    v_cache_fp16 = pdl_in(v_cache_fp16);
    k_scale = pdl_in(k_scale);
    v_scale = pdl_in(v_scale);

    int pos;
    if constexpr (kMs) {
        const int2 row = ms_rows[b];
        pos = row.x;
        k_cache_fp16 = reinterpret_cast<const uint16_t*>(reinterpret_cast<const char*>(k_cache_fp16) + row.y * seq_kv_bytes);
        v_cache_fp16 = reinterpret_cast<const uint16_t*>(reinterpret_cast<const char*>(v_cache_fp16) + row.y * seq_kv_bytes);
        if constexpr (kInt8) {
            k_scale += static_cast<size_t>(row.y) * seq_scale_floats;
            v_scale += static_cast<size_t>(row.y) * seq_scale_floats;
        }
    } else {
        pos = (*pos_ptr) + b;
    }
    const int n = pos + 1;
    const int first = is_global ? 0 : ((n > kExtent) ? (n - kExtent) : 0);
    const int total_early = n - first;
    const int n_chunks_early = (total_early + kChunk - 1) / kChunk;
    const int active_g = (n_chunks_early > 1) ? ((n_chunks_early < G) ? n_chunks_early : G) : 1;
    if (g >= active_g) {
        pdl_trigger();
        return;
    }

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
        } else if constexpr (kPipe) {
            // FP16: the same depth-1 pipeline on the raw FP16 words (row jj+8's K/V in flight while
            // row jj is computed). Per-row arithmetic and row order are unchanged (0-ULP).
            int jj = warp_id;
            int j = j0 + jj;
            uint64_t wk = 0u, wv = 0u;
            if (jj < kChunk && j < n) {
                wk = kv_raw_f16(k_cache_fp16, ring_row(j), kvh, lane);
                wv = kv_raw_f16(v_cache_fp16, ring_row(j), kvh, lane);
            }
            while (jj < kChunk && j < n) {
                const int jn = jj + 8;
                const int jnx = j0 + jn;
                uint64_t nwk = 0u, nwv = 0u;
                if (jn < kChunk && jnx < n) {
                    nwk = kv_raw_f16(k_cache_fp16, ring_row(jnx), kvh, lane);
                    nwv = kv_raw_f16(v_cache_fp16, ring_row(jnx), kvh, lane);
                }
                float k0, k1, k2, k3, v0, v1, v2, v3;
                kv_unpack_f16(wk, k0, k1, k2, k3);
                kv_unpack_f16(wv, v0, v1, v2, v3);
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
                wk = nwk; wv = nwv;
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
    pdl_trigger();
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
    pdl_wait();
    gqa_attn_split_body<false>(qrgs_base, qrgs_stride, rel_proj_bf16, k_cache_fp16, v_cache_fp16, k_scale, v_scale, pos_ptr, kv_cap, is_global, part_ml, part_o);
}

// FP16 decode / verify instantiation with the pipelined row loop (bit-identical to the above).
extern "C" __global__ void ink2_gqa_attn_split_dec(
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
    pdl_wait();
    gqa_attn_split_body<false, true>(qrgs_base, qrgs_stride, rel_proj_bf16, k_cache_fp16, v_cache_fp16, k_scale, v_scale, pos_ptr, kv_cap, is_global, part_ml, part_o);
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
    pdl_wait();
    gqa_attn_split_body<true>(qrgs_base, qrgs_stride, rel_proj_bf16, k_cache_fp16, v_cache_fp16, k_scale, v_scale, pos_ptr, kv_cap, is_global, part_ml, part_o);
}

// Multi-sequence decode rows (#107): row `b` attends its own sequence's KV up to its own
// position (`ms_rows[b] = {pos, seq}`). Same per-row arithmetic as the instantiations above.
#define INK2_SPLIT_MS_PARAMS \
    const float* __restrict__ qrgs_base, int32_t qrgs_stride, const uint32_t* __restrict__ rel_proj_bf16, \
    const uint16_t* __restrict__ k_cache_fp16, const uint16_t* __restrict__ v_cache_fp16, \
    const float* __restrict__ k_scale, const float* __restrict__ v_scale, \
    const int2* __restrict__ ms_rows, int32_t kv_cap, int32_t is_global, \
    float* __restrict__ part_ml, float* __restrict__ part_o, long long seq_kv_bytes, int32_t seq_scale_floats
extern "C" __global__ void ink2_gqa_attn_split_dec_ms(INK2_SPLIT_MS_PARAMS) {
    pdl_wait();
    gqa_attn_split_body<false, true, true>(qrgs_base, qrgs_stride, rel_proj_bf16, k_cache_fp16, v_cache_fp16,
        k_scale, v_scale, nullptr, kv_cap, is_global, part_ml, part_o, ms_rows, seq_kv_bytes, seq_scale_floats);
}
extern "C" __global__ void __launch_bounds__(256, 4) ink2_gqa_attn_split_i8_ms(INK2_SPLIT_MS_PARAMS) {
    pdl_wait();
    gqa_attn_split_body<true, false, true>(qrgs_base, qrgs_stride, rel_proj_bf16, k_cache_fp16, v_cache_fp16,
        k_scale, v_scale, nullptr, kv_cap, is_global, part_ml, part_o, ms_rows, seq_kv_bytes, seq_scale_floats);
}
#undef INK2_SPLIT_MS_PARAMS

// Merge `G` split partials per head + sigmoid gate. Grid: `(24, B)`, Block: `128`.
extern "C" __global__ void ink2_gqa_attn_combine(
    const float* __restrict__ part_ml,   // [B, 24, G, 2]
    const float* __restrict__ part_o,    // [B, 24, G, 128]
    const float* __restrict__ qrgs_base, // [B, qrgs_stride] (`gate @ 3456`)
    int32_t qrgs_stride,
    int32_t G,
    float* __restrict__ attn_out_base) { // [B, 24, 128]
    pdl_wait();
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
    const uint32_t* gamma = gamma_bf16_pairs + static_cast<size_t>(q_idx) * (kD / 2);
    const float inv_sqrt_d = 0.020833333333333332f;

    float q_reg[9];
    #pragma unroll
    for (int t = 0; t < 9; ++t) {
        q_reg[t] = q[tid + t * 256];
    }
    uint32_t gw_reg[5];
    #pragma unroll
    for (int it = 0; it < 4; ++it) {
        gw_reg[it] = gamma[(warp_id + it * 8) * 32 + lane];
    }
    gw_reg[4] = (warp_id < 4) ? gamma[(warp_id + 32) * 32 + lane] : 0u;

    pdl_wait();
    pdl_trigger();
    blocks_buf = pdl_in(blocks_buf);
    cur_opt = pdl_in(cur_opt);

    float st_reg[6][9];
    #pragma unroll
    for (int s = 0; s < 6; ++s) {
        if (s < total_states) {
            const float* st = (s < n_blocks)
                ? (blocks_buf + (static_cast<size_t>(s) * stride_B + b_src) * kD)
                : (cur_opt + static_cast<size_t>(b_src) * kD);
            #pragma unroll
            for (int t = 0; t < 9; ++t) {
                st_reg[s][t] = st[tid + t * 256];
            }
        }
    }

    #pragma unroll
    for (int s = 0; s < 6; ++s) {
        if (s < total_states) {
            float sq = 0.0f;
            float dq = 0.0f;
            #pragma unroll
            for (int t = 0; t < 9; ++t) {
                float v = st_reg[s][t];
                sq = __fmaf_rn(v, v, sq);
                dq = __fmaf_rn(v, q_reg[t], dq);
            }
            sq = warp_sum(sq);
            dq = warp_sum(dq);
            if (lane == 0) {
                smem_sq[s][warp_id] = sq;
                smem_dq[s][warp_id] = dq;
            }
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
        #pragma unroll
        for (int s = 0; s < 6; ++s) {
            if (s < total_states) {
                acc = __fmaf_rn(smem_probs[s], st_reg[s][t], acc);
            }
        }
        smem_mix[i] = acc;
        sq_norm = __fmaf_rn(acc, acc, sq_norm);
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
    uint32_t* dst_q8 = out_q8_u32 + vec_idx * (kD / 4);
    float* dst_sc = out_sc64 + vec_idx * (kD / 64);

    #pragma unroll
    for (int it = 0; it < 5; ++it) {
        const int blk = warp_id + it * 8;
        if (blk < (kD / 64)) {
            uint32_t gw = gw_reg[it];
            int base = blk * 64 + lane * 2;
            float y0 = __fmul_rn(__fmul_rn(smem_mix[base], k), bf16_to_f32(static_cast<uint16_t>(gw & 0xFFFFu)));
            float y1 = __fmul_rn(__fmul_rn(smem_mix[base + 1], k), bf16_to_f32(static_cast<uint16_t>(gw >> 16)));
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
    uint32_t gw_reg[5];
    #pragma unroll
    for (int t = 0; t < 4; ++t) gw_reg[t] = gamma_bf16_pairs[tid + t * 256];
    gw_reg[4] = (tid < 128) ? gamma_bf16_pairs[tid + 1024] : 0u;

    pdl_wait();
    pdl_trigger();
    blocks_buf = pdl_in(blocks_buf);
    cur_opt = pdl_in(cur_opt);

    float st_reg[6][9];
    #pragma unroll
    for (int s = 0; s < 6; ++s) {
        if (s < total_states) {
            const float* st = (s < n_blocks)
                ? (blocks_buf + (static_cast<size_t>(s) * stride_B + b_src) * kD)
                : (cur_opt + static_cast<size_t>(b_src) * kD);
            #pragma unroll
            for (int t = 0; t < 9; ++t) {
                st_reg[s][t] = st[tid + t * 256];
            }
        }
    }

    #pragma unroll
    for (int s = 0; s < 6; ++s) {
        if (s < total_states) {
            float sq = 0.0f;
            float dq = 0.0f;
            #pragma unroll
            for (int t = 0; t < 9; ++t) {
                float v = st_reg[s][t];
                sq = __fmaf_rn(v, v, sq);
                dq = __fmaf_rn(v, q_reg[t], dq);
            }
            sq = warp_sum(sq);
            dq = warp_sum(dq);
            if (lane == 0) {
                smem_sq[s][warp_id] = sq;
                smem_dq[s][warp_id] = dq;
            }
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

    float mix_reg[9];
    float sq_norm = 0.0f;
    #pragma unroll
    for (int t = 0; t < 9; ++t) {
        float acc = 0.0f;
        #pragma unroll
        for (int s = 0; s < 6; ++s) {
            if (s < total_states) {
                acc = __fmaf_rn(smem_probs[s], st_reg[s][t], acc);
            }
        }
        mix_reg[t] = acc;
        sq_norm = __fmaf_rn(acc, acc, sq_norm);
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

    // Write `mix_reg` to `smem_x` once (`tid + t*256`), then normalize with `idx * 2` (`1` fewer barrier)!
    #pragma unroll
    for (int t = 0; t < 9; ++t) {
        smem_x[tid + t * 256] = mix_reg[t];
    }
    __syncthreads();

    float* dst_f32 = out_normed + static_cast<size_t>(b_out) * kD;
    #pragma unroll
    for (int t = 0; t < 5; ++t) {
        const int idx = tid + t * 256;
        if (idx < kD / 2) {
            uint32_t gw = gw_reg[t];
            float y0 = __fmul_rn(__fmul_rn(smem_x[idx * 2], k), bf16_to_f32(static_cast<uint16_t>(gw & 0xFFFFu)));
            float y1 = __fmul_rn(__fmul_rn(smem_x[idx * 2 + 1], k), bf16_to_f32(static_cast<uint16_t>(gw >> 16)));
            dst_f32[idx * 2] = y0;
            dst_f32[idx * 2 + 1] = y1;
            if (do_q8_16) {
                smem_x[idx * 2] = y0;
                smem_x[idx * 2 + 1] = y1;
            }
        }
    }
    if (!do_q8_16) return;
    __syncthreads();

    const int hw = tid >> 4;     // 0..15 (16 half-warps)
    const int lane16 = tid & 15; // 0..15
    uint32_t* dst_q8 = out_q8_u32 + static_cast<size_t>(b_out) * (kD >> 2);
    float* dst_sc = out_sc16 + static_cast<size_t>(b_out) * (kD >> 4);
    #pragma unroll
    for (int it = 0; it < 9; ++it) {
        const int blk16 = hw + it * 16;
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

    uint32_t gw_reg[5];
    #pragma unroll
    for (int t = 0; t < 4; ++t) gw_reg[t] = gamma_bf16_pairs[tid + t * 256];
    gw_reg[4] = (tid < 128) ? gamma_bf16_pairs[tid + 1024] : 0u;

    pdl_wait();
    pdl_trigger();
    inp = pdl_in(inp);

    float sq = 0.0f;
    #pragma unroll
    for (int t = 0; t < 9; ++t) {
        const int i = tid + t * 256;
        float v = inp[i];
        smem_x[i] = v;
        sq = __fmaf_rn(v, v, sq);
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

    #pragma unroll
    for (int t = 0; t < 5; ++t) {
        const int idx = tid + t * 256;
        if (idx < kD / 2) {
            uint32_t gw = gw_reg[t];
            float y0 = __fmul_rn(__fmul_rn(smem_x[idx * 2], k), bf16_to_f32(static_cast<uint16_t>(gw & 0xFFFFu)));
            float y1 = __fmul_rn(__fmul_rn(smem_x[idx * 2 + 1], k), bf16_to_f32(static_cast<uint16_t>(gw >> 16)));
            dst_f32[idx * 2] = y0;
            dst_f32[idx * 2 + 1] = y1;
            smem_x[idx * 2] = y0;
            smem_x[idx * 2 + 1] = y1;
        }
    }
    __syncthreads();

    const int hw = tid >> 4;
    const int lane16 = tid & 15;
    uint32_t* dst_q8 = out_q8_u32 + static_cast<size_t>(b) * (kD >> 2);
    float* dst_sc = out_sc16 + static_cast<size_t>(b) * (kD >> 4);
    #pragma unroll
    for (int it = 0; it < 9; ++it) {
        const int blk16 = hw + it * 16;
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

    const size_t tile_row_offset = (static_cast<size_t>(c_idx) * kD + row_base) * 8u;
    const uint4* g_tile = gain_bf16_u128 + tile_row_offset;
    const uint4* f_tile = forget_bf16_u128 + tile_row_offset;

    uint4 fp_reg[8];
    #pragma unroll
    for (int it = 0; it < 8; ++it) {
        const int idx = it * 32 + lane;
        smem_buf[warp_id][idx >> 3][idx & 7] = g_tile[idx];
        fp_reg[it] = f_tile[idx];
    }

    pdl_wait();
    g_vec_base = pdl_in(g_vec_base);
    cur_inout_base = pdl_in_mut(cur_inout_base);
    msg_base = pdl_in(msg_base);

    smem_gv[tid] = g_vec_base[cb * 64 + tid];
    __syncthreads();

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
        smem_buf[warp_id][idx >> 3][idx & 7] = fp_reg[it];
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
    float m = msg[row];
    pdl_trigger();
    cur_inout[row] = c - ptx_tanh(forget_acc) * c + ptx_tanh(gain_acc) * m;
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

    pdl_wait();
    pos_ptr = pdl_in(pos_ptr);
    qrgs_base = pdl_in(qrgs_base);
    k_cache_fp16 = pdl_in(k_cache_fp16);
    v_cache_fp16 = pdl_in(v_cache_fp16);
    k_scale = pdl_in(k_scale);
    v_scale = pdl_in(v_scale);

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
    pdl_trigger();
    quant_head128_epilogue_q8_64(tid, h, b, out_val, smem_qmax, smem_qinv, out_q8_u32, out_sc64);
}

extern "C" __global__ void ink2_gqa_rel_sink_attn_q8_b(
    const float* __restrict__ qrgs_base, int32_t qrgs_stride, const uint32_t* __restrict__ rel_proj_bf16,
    const uint16_t* __restrict__ k_cache_fp16, const uint16_t* __restrict__ v_cache_fp16,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    const int32_t* __restrict__ pos_ptr, int32_t kv_cap, int32_t is_global,
    uint32_t* __restrict__ out_q8_u32, float* __restrict__ out_sc64) {
    pdl_wait();
    gqa_rel_sink_attn_q8<false, false>(qrgs_base, qrgs_stride, rel_proj_bf16,
        k_cache_fp16, v_cache_fp16, k_scale, v_scale, pos_ptr, kv_cap, is_global, out_q8_u32, out_sc64);
}

extern "C" __global__ void ink2_gqa_rel_sink_attn_q8_decode(
    const float* __restrict__ qrgs_base, int32_t qrgs_stride, const uint32_t* __restrict__ rel_proj_bf16,
    const uint16_t* __restrict__ k_cache_fp16, const uint16_t* __restrict__ v_cache_fp16,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    const int32_t* __restrict__ pos_ptr, int32_t kv_cap, int32_t is_global,
    uint32_t* __restrict__ out_q8_u32, float* __restrict__ out_sc64) {
    pdl_wait();
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
    pdl_wait();
    gqa_rel_sink_attn_q8<true, false>(qrgs_base, qrgs_stride, rel_proj_bf16,
        k_cache_fp16, v_cache_fp16, k_scale, v_scale, pos_ptr, kv_cap, is_global, out_q8_u32, out_sc64);
}

extern "C" __global__ void ink2_gqa_rel_sink_attn_q8_decode_i8(
    const float* __restrict__ qrgs_base, int32_t qrgs_stride, const uint32_t* __restrict__ rel_proj_bf16,
    const uint16_t* __restrict__ k_cache_fp16, const uint16_t* __restrict__ v_cache_fp16,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    const int32_t* __restrict__ pos_ptr, int32_t kv_cap, int32_t is_global,
    uint32_t* __restrict__ out_q8_u32, float* __restrict__ out_sc64) {
    pdl_wait();
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
    pdl_wait();
    pdl_trigger();
    part_ml = pdl_in(part_ml);
    part_o = pdl_in(part_o);
    qrgs_base = pdl_in(qrgs_base);
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

// Dynamic Active-Split Combine + In-Register `Q8_64` Epilogue (`Grid: (24, B)`, `Block: 128`):
// Reads ONLY the `active_g = clamp(ceil((n - first) / 64), 1, G)` non-empty splits produced by
// `gqa_attn_split_body`, eliminating up to `31/32` (`97%`) empty DRAM partial reads at short context
// while remaining `0-ULP` bit-for-bit identical across Solo (`B=1`) and Speculative Verify (`B<=8`)!
template <bool kMs>
__device__ __forceinline__ void gqa_attn_combine_q8_dyn_body(
    const float* __restrict__ part_ml,   // [B, 24, G, 2]
    const float* __restrict__ part_o,    // [B, 24, G, 128]
    const float* __restrict__ qrgs_base, // [B, qrgs_stride] (`gate @ 3456`)
    int32_t qrgs_stride,
    int32_t G,
    const int32_t* __restrict__ pos_ptr, // kMs: `int2 {pos, seq}` per row (#107)
    int32_t is_global,
    uint32_t* __restrict__ out_q8_u32,   // [B, 768]
    float* __restrict__ out_sc64) {      // [B, 48]
    __shared__ float smem_qmax[2][2];
    __shared__ float smem_qinv[2];

    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int dd = threadIdx.x; // 0..127
    const size_t bh = static_cast<size_t>(b) * 24u + h;
    pdl_wait();
    pdl_trigger();
    part_ml = pdl_in(part_ml);
    part_o = pdl_in(part_o);
    qrgs_base = pdl_in(qrgs_base);
    pos_ptr = pdl_in(pos_ptr);

    const int pos = kMs ? reinterpret_cast<const int2*>(pos_ptr)[b].x : (*pos_ptr) + b;
    const int n = pos + 1;
    const int first = is_global ? 0 : ((n > kExtent) ? (n - kExtent) : 0);
    const int n_chunks = ((n - first) + 63) >> 6;
    const int active_g = (n_chunks > 1) ? ((n_chunks < G) ? n_chunks : G) : 1;
    const float* gate_buf = qrgs_base + static_cast<size_t>(b) * qrgs_stride + 3456;

    float m_max = -1e30f;
    for (int g = 0; g < active_g; ++g) {
        m_max = fmaxf(m_max, part_ml[(bh * G + g) * 2]);
    }
    float denom = 0.0f;
    float sum_d = 0.0f;
    for (int g = 0; g < active_g; ++g) {
        const float s = ptx_exp(part_ml[(bh * G + g) * 2] - m_max);
        denom += part_ml[(bh * G + g) * 2 + 1] * s;
        sum_d += part_o[(bh * G + g) * kHd + dd] * s;
    }
    const float gt = ptx_sigmoid(gate_buf[h * kHd + dd]);
    const float out_val = (sum_d * ptx_rcp(fmaxf(denom, 1e-20f))) * gt;
    quant_head128_epilogue_q8_64(dd, h, b, out_val, smem_qmax, smem_qinv, out_q8_u32, out_sc64);
}

#define INK2_COMBINE_DYN_PARAMS \
    const float* __restrict__ part_ml, const float* __restrict__ part_o, \
    const float* __restrict__ qrgs_base, int32_t qrgs_stride, int32_t G, \
    const int32_t* __restrict__ pos_ptr, int32_t is_global, \
    uint32_t* __restrict__ out_q8_u32, float* __restrict__ out_sc64
extern "C" __global__ void ink2_gqa_attn_combine_q8_dyn(INK2_COMBINE_DYN_PARAMS) {
    pdl_wait();
    gqa_attn_combine_q8_dyn_body<false>(part_ml, part_o, qrgs_base, qrgs_stride, G, pos_ptr, is_global, out_q8_u32, out_sc64);
}
// Multi-sequence rows (#107): `pos_ptr` points at the `int2 {pos, seq}` row table.
extern "C" __global__ void ink2_gqa_attn_combine_q8_dyn_ms(INK2_COMBINE_DYN_PARAMS) {
    pdl_wait();
    gqa_attn_combine_q8_dyn_body<true>(part_ml, part_o, qrgs_base, qrgs_stride, G, pos_ptr, is_global, out_q8_u32, out_sc64);
}
#undef INK2_COMBINE_DYN_PARAMS

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
    const float4 w = (c < channels) ? w_conv4[c] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    pdl_wait();
    pdl_trigger();
    if (c >= channels) return;
    cur = pdl_in(cur);
    dyn_tap = pdl_in(dyn_tap);
    hist3 = pdl_in_mut(hist3);
    out = pdl_in_mut(out);
    res_add = pdl_in(res_add);

    if (save_prev_out != nullptr) {
        save_prev_out[c] = out[c];
    }
    const float x = cur[c];
    float* h = hist3 + c * 3;
    const float h0 = h[0];
    const float h1 = h[1];
    const float h2 = h[2];
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
    pdl_wait();
    pdl_trigger();
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
    pdl_wait();
    short_conv4_tiled(cur, in_stride, weights, dyn, dyn_stride, hist3, out, out_stride, residual, res_stride,
                      nullptr, seq, channels, B, has_dyn, do_res_add);
}

extern "C" __global__ void ink2_short_conv4_tiled_save_b(
    const float* cur, int in_stride, const float4* weights, const float4* dyn, int dyn_stride,
    float* hist3, float* out, int out_stride, const float* residual, int res_stride,
    float* save_prev, float* seq, int channels, int B, int has_dyn, int do_res_add) {
    pdl_wait();
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
    pdl_wait();
    pdl_trigger();
    cur_buf = pdl_in_mut(cur_buf);
    blocks_buf = pdl_in_mut(blocks_buf);
    prelude_x_buf = pdl_in_mut(prelude_x_buf);
    prelude_block0_buf = pdl_in_mut(prelude_block0_buf);
    bank_buf = pdl_in_mut(bank_buf);
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
    pdl_wait();
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
// Capped at `<= 64` regs (`4` CTAs/SM = `32` warps/SM, matching `c2304` instead of falling to `3` CTAs/SM at `70` regs).
extern "C" __global__ __launch_bounds__(256, 4) void ink2_q8_dp4a_qkv3_gemv(
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
    const int row0 = (cta_local * 8 + warp_id) * 2;
    const bool has_row0 = (row0 < rows);
    const bool has_row1 = (row0 + 1 < rows);
    const bool tail16 = (lane < 16);

    uint4 w0[4], w1[4];
    float ws0[4], ws1[4];
    uint4 w0_4 = make_uint4(0u, 0u, 0u, 0u), w1_4 = w0_4;
    float ws0_4 = 0.0f, ws1_4 = 0.0f;
    if (has_row0) {
        const uint4* w_r0 = w_u128 + static_cast<size_t>(row0) * 144u;
        const uint4* w_r1 = has_row1 ? (w_r0 + 144u) : w_r0;
        const float* s_r0 = w_sc + static_cast<size_t>(row0) * 36u;
        const float* s_r1 = has_row1 ? (s_r0 + 36u) : s_r0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c16 = lane + 32 * i;
            w0[i] = w_r0[c16];
            w1[i] = w_r1[c16];
            ws0[i] = s_r0[c16 >> 2];
            ws1[i] = s_r1[c16 >> 2];
        }
        if (tail16) {
            const int c16 = lane + 128;
            w0_4 = w_r0[c16];
            w1_4 = w_r1[c16];
            ws0_4 = s_r0[c16 >> 2];
            ws1_4 = s_r1[c16 >> 2];
        }
    }

    pdl_wait();
    x_q8_3_u128 = pdl_in(x_q8_3_u128);
    x_sc64_3 = pdl_in(x_sc64_3);

    const uint4* x_q8 = x_q8_3_u128 + slot * 144;
    const float* x_sc = x_sc64_3 + slot * 36;
    if (tid < 144) {
        smem_x[tid] = x_q8[tid];
    }
    if (tid < 36) {
        smem_xsc[tid] = x_sc[tid];
    }
    __syncthreads();

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    if (has_row0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c16 = lane + 32 * i;
            uint4 xq = smem_x[c16];
            float xsc = smem_xsc[c16 >> 2];

            int d0 = __dp4a(static_cast<int>(w0[i].x), static_cast<int>(xq.x), 0);
            d0 = __dp4a(static_cast<int>(w0[i].y), static_cast<int>(xq.y), d0);
            d0 = __dp4a(static_cast<int>(w0[i].z), static_cast<int>(xq.z), d0);
            d0 = __dp4a(static_cast<int>(w0[i].w), static_cast<int>(xq.w), d0);
            acc0 = __fmaf_rn(static_cast<float>(d0), __fmul_rn(ws0[i], xsc), acc0);

            if (has_row1) {
                int d1 = __dp4a(static_cast<int>(w1[i].x), static_cast<int>(xq.x), 0);
                d1 = __dp4a(static_cast<int>(w1[i].y), static_cast<int>(xq.y), d1);
                d1 = __dp4a(static_cast<int>(w1[i].z), static_cast<int>(xq.z), d1);
                d1 = __dp4a(static_cast<int>(w1[i].w), static_cast<int>(xq.w), d1);
                acc1 = __fmaf_rn(static_cast<float>(d1), __fmul_rn(ws1[i], xsc), acc1);
            }
        }
        if (tail16) {
            const int c16 = lane + 128;
            uint4 xq = smem_x[c16];
            float xsc = smem_xsc[c16 >> 2];

            int d0 = __dp4a(static_cast<int>(w0_4.x), static_cast<int>(xq.x), 0);
            d0 = __dp4a(static_cast<int>(w0_4.y), static_cast<int>(xq.y), d0);
            d0 = __dp4a(static_cast<int>(w0_4.z), static_cast<int>(xq.z), d0);
            d0 = __dp4a(static_cast<int>(w0_4.w), static_cast<int>(xq.w), d0);
            acc0 = __fmaf_rn(static_cast<float>(d0), __fmul_rn(ws0_4, xsc), acc0);

            if (has_row1) {
                int d1 = __dp4a(static_cast<int>(w1_4.x), static_cast<int>(xq.x), 0);
                d1 = __dp4a(static_cast<int>(w1_4.y), static_cast<int>(xq.y), d1);
                d1 = __dp4a(static_cast<int>(w1_4.z), static_cast<int>(xq.z), d1);
                d1 = __dp4a(static_cast<int>(w1_4.w), static_cast<int>(xq.w), d1);
                acc1 = __fmaf_rn(static_cast<float>(d1), __fmul_rn(ws1_4, xsc), acc1);
            }
        }
        acc0 = warp_sum(acc0);
        acc1 = warp_sum(acc1);
    }
    pdl_trigger();
    if (has_row0 && lane == 0) {
        y[row0] = acc0;
        if (has_row1) y[row0 + 1] = acc1;
    }
}

// 1-Row/Warp Fused 3-Projection `(Q, K, V)` Planar `Q8_64` `__dp4a` GEMV (`Block: 256` = `8` rows/CTA,
// `<= 40` regs -> `6` CTAs/SM = `48` warps/SM = `660` CTAs/wave on PRO 5000, `0-ULP` exact!):
extern "C" __global__ __launch_bounds__(256, 5) void ink2_q8_dp4a_qkv3_gemv_r1(
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

    const int cta_q = (rows_q + 7) >> 3;
    const int cta_k = (rows_k + 7) >> 3;
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
    const int row = cta_local * 8 + warp_id;
    const bool has_row = (row < rows);
    const bool tail16 = (lane < 16);

    uint4 w0[4];
    float ws0[4];
    uint4 w0_4 = make_uint4(0u, 0u, 0u, 0u);
    float ws0_4 = 0.0f;
    if (has_row) {
        const uint4* w_r0 = w_u128 + static_cast<size_t>(row) * 144u;
        const float* s_r0 = w_sc + static_cast<size_t>(row) * 36u;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c16 = lane + 32 * i;
            w0[i] = w_r0[c16];
            ws0[i] = s_r0[c16 >> 2];
        }
        if (tail16) {
            const int c16 = lane + 128;
            w0_4 = w_r0[c16];
            ws0_4 = s_r0[c16 >> 2];
        }
    }

    pdl_wait();
    x_q8_3_u128 = pdl_in(x_q8_3_u128);
    x_sc64_3 = pdl_in(x_sc64_3);

    const uint4* x_q8 = x_q8_3_u128 + slot * 144;
    const float* x_sc = x_sc64_3 + slot * 36;
    if (tid < 144) smem_x[tid] = x_q8[tid];
    if (tid < 36) smem_xsc[tid] = x_sc[tid];
    __syncthreads();

    float acc0 = 0.0f;
    if (has_row) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c16 = lane + 32 * i;
            uint4 xq = smem_x[c16];
            float xsc = smem_xsc[c16 >> 2];
            int d0 = __dp4a(static_cast<int>(w0[i].x), static_cast<int>(xq.x), 0);
            d0 = __dp4a(static_cast<int>(w0[i].y), static_cast<int>(xq.y), d0);
            d0 = __dp4a(static_cast<int>(w0[i].z), static_cast<int>(xq.z), d0);
            d0 = __dp4a(static_cast<int>(w0[i].w), static_cast<int>(xq.w), d0);
            acc0 = __fmaf_rn(static_cast<float>(d0), __fmul_rn(ws0[i], xsc), acc0);
        }
        if (tail16) {
            const int c16 = lane + 128;
            uint4 xq = smem_x[c16];
            float xsc = smem_xsc[c16 >> 2];
            int d0 = __dp4a(static_cast<int>(w0_4.x), static_cast<int>(xq.x), 0);
            d0 = __dp4a(static_cast<int>(w0_4.y), static_cast<int>(xq.y), d0);
            d0 = __dp4a(static_cast<int>(w0_4.z), static_cast<int>(xq.z), d0);
            d0 = __dp4a(static_cast<int>(w0_4.w), static_cast<int>(xq.w), d0);
            acc0 = __fmaf_rn(static_cast<float>(d0), __fmul_rn(ws0_4, xsc), acc0);
        }
        acc0 = warp_sum(acc0);
    }
    pdl_trigger();
    if (has_row && lane == 0) {
        y[row] = acc0;
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
    pdl_wait();
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
    pdl_wait();
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
    const bool is_v = (blockIdx.y != 0);
    const float4* w_conv4 = is_v ? w_v : w_k;
    const float4 w = (c < 1024) ? w_conv4[c] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    pdl_wait();
    pdl_trigger();
    if (c >= 1024) return;

    const float* cur = pdl_in(is_v ? cur_v : cur_k);
    const float4* dyn_tap = pdl_in(is_v ? dyn_v : dyn_k);
    float* hist3 = pdl_in_mut(is_v ? hist_v : hist_k);
    float* out = is_v ? out_v : out_k;

    const float x = cur[c];
    float* h = hist3 + c * 3;
    const float h0 = h[0];
    const float h1 = h[1];
    const float h2 = h[2];
    const float4 d = has_dyn ? dyn_tap[c >> 7] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    h[0] = h1;
    h[1] = h2;
    h[2] = x;

    out[c] = eval_conv4_step(w, h0, h1, h2, x, has_dyn, d, 0, 0.0f);
}

// ============================================================================
// 17b. `v0.8.0` Solo Decode `short_conv4` Producer-Consumer Fusions (`#106`)
//      & Speculative Verify (`1 <= B <= 8`) Active-Expert `0-ULP` MoE (`#104`)
// ============================================================================

// Fused K+V `short_conv4` + Q/R/K Clip + V RMSNorm + KV Cache Write (`Grid: (5, 1)`, `Block: 256`, `0 B` SMEM):
// Eliminates `ink2_short_conv4_kv2` (`-48` launches/tok) and `d_k_conv`/`d_v_conv` DRAM round-trips (`0-ULP` exact).
extern "C" __global__ void ink2_kv_conv4_prep_cache(
    float* __restrict__ q_buf,                 // [24 * 128]
    float* __restrict__ r_buf,                 // [24 * 16]
    const float* __restrict__ k_raw,           // [1024 (+ 32 if global)]
    const float4* __restrict__ w_k,            // [1024]
    float* __restrict__ hist_k,                // [1024 * 3]
    const float* __restrict__ v_raw,           // [1024 (+ 32 if global)]
    const float4* __restrict__ w_v,            // [1024]
    float* __restrict__ hist_v,                // [1024 * 3]
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

    uint32_t cp0 = 0u, cp1 = 0u, rp = 0u;
    float4 w_conv[4];
    if (g_warp < 24) {
        const uint32_t* qc = q_clip_bf16 + g_warp * 64;
        cp0 = qc[lane * 2];
        cp1 = qc[lane * 2 + 1];
        if (lane < 8) {
            rp = (r_clip_bf16 + g_warp * 8)[lane];
        }
    } else if (g_warp < 40) {
        const bool is_v = (g_warp >= 32);
        const int h = is_v ? (g_warp - 32) : (g_warp - 24);
        const float4* wc_base = (is_v ? w_v : w_k) + h * 128 + lane * 4;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            w_conv[j] = wc_base[j];
        }
    }

    pdl_wait();
    pdl_trigger();
    q_buf = pdl_in_mut(q_buf);
    r_buf = pdl_in_mut(r_buf);
    k_raw = pdl_in(k_raw);
    hist_k = pdl_in_mut(hist_k);
    v_raw = pdl_in(v_raw);
    hist_v = pdl_in_mut(hist_v);
    pos_ptr = pdl_in(pos_ptr);

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
        const int h = g_warp;
        float* qh = q_buf + h * 128;
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

        float* rh = r_buf + h * 16;
        float rv0 = 0.0f, rv1 = 0.0f;
        if (lane < 8) {
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
        const int c_base = kh * 128 + lane * 4;
        const float4 dk = is_global
            ? reinterpret_cast<const float4*>(k_raw + 1024)[kh]
            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float kv[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = c_base + j;
            const float x = k_raw[c];
            float* h = hist_k + c * 3;
            const float h0 = h[0], h1 = h[1], h2 = h[2];
            h[0] = h1;
            h[1] = h2;
            h[2] = x;
            kv[j] = eval_conv4_step(w_conv[j], h0, h1, h2, x, is_global, dk, 0, 0.0f);
        }
        float k_sq = warp_sum(__fadd_rn(
            __fmaf_rn(kv[1], kv[1], __fmul_rn(kv[0], kv[0])),
            __fmaf_rn(kv[3], kv[3], __fmul_rn(kv[2], kv[2]))));
        float k_f = 22.627416997969522f * ptx_rcp(ptx_sqrt(k_sq) + 1e-6f);
        if (k_f > 1.0f) k_f = 1.0f;
        kv_store4(k_cache_fp16, k_scale, cache_row, kh, lane,
                  kv[0] * k_f, kv[1] * k_f, kv[2] * k_f, kv[3] * k_f);
    } else if (g_warp < 40) {
        const int vh = g_warp - 32;
        const int c_base = vh * 128 + lane * 4;
        const float4 dv = is_global
            ? reinterpret_cast<const float4*>(v_raw + 1024)[vh]
            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float vv[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = c_base + j;
            const float x = v_raw[c];
            float* h = hist_v + c * 3;
            const float h0 = h[0], h1 = h[1], h2 = h[2];
            h[0] = h1;
            h[1] = h2;
            h[2] = x;
            vv[j] = eval_conv4_step(w_conv[j], h0, h1, h2, x, is_global, dv, 0, 0.0f);
        }
        float v_sq = warp_sum(__fadd_rn(
            __fmaf_rn(vv[1], vv[1], __fmul_rn(vv[0], vv[0])),
            __fmaf_rn(vv[3], vv[3], __fmul_rn(vv[2], vv[2]))));
        float v_inv = ptx_rsqrt(v_sq * (1.0f / 128.0f) + 1e-6f);
        kv_store4(v_cache_fp16, v_scale, cache_row, vh, lane,
                  vv[0] * v_inv, vv[1] * v_inv, vv[2] * v_inv, vv[3] * v_inv);
    }
}

// Fused `wo` `c3072` GEMV + `attn_sconv` + Pre-Overwrite Block Save (`Grid: (144, 1)`, `Block: 256`, `3,264 B` SMEM):
// Eliminates `ink2_short_conv4_save` (`-48` launches/tok) and `d_wo_out` DRAM round-trips (`0-ULP` exact).
extern "C" __global__ void ink2_wo_conv4_save_c3072(
    const uint4* __restrict__ w_q8_u128,   // [2304, 192]
    const float* __restrict__ w_sc64,      // [2304, 48]
    const uint4* __restrict__ x_q8_u128,   // [192]
    const float* __restrict__ x_sc64,      // [48]
    const float4* __restrict__ w_conv4,    // [2304]
    float* __restrict__ hist3,             // [2304 * 3]
    float* __restrict__ cur_inout,         // [2304]
    float* __restrict__ save_prev_out,     // nullable (`d_blocks[blk]` when `is_block_start`)
    int32_t do_res_add) {
    constexpr int kCols16 = 3072 / 16;
    constexpr int kCols64 = 3072 / 64;
    constexpr int kIters = kCols16 / 32;
    __shared__ uint4 smem_x[kCols16];
    __shared__ float smem_xsc[kCols64];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row0 = (blockIdx.x * 8 + warp_id) * 2; // 144 * 16 = 2304 exact

    uint4 w0[kIters], w1[kIters];
    float ws0[kIters], ws1[kIters];
    const uint4* w_r0 = w_q8_u128 + static_cast<size_t>(row0) * kCols16;
    const uint4* w_r1 = w_r0 + kCols16;
    const float* s_r0 = w_sc64 + static_cast<size_t>(row0) * kCols64;
    const float* s_r1 = s_r0 + kCols64;
    #pragma unroll
    for (int i = 0; i < kIters; ++i) {
        const int c16 = lane + 32 * i;
        w0[i] = w_r0[c16];
        w1[i] = w_r1[c16];
        ws0[i] = s_r0[c16 >> 2];
        ws1[i] = s_r1[c16 >> 2];
    }
    const float4 wc = (lane < 2) ? w_conv4[row0 + lane] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    pdl_wait();
    x_q8_u128 = pdl_in(x_q8_u128);
    x_sc64 = pdl_in(x_sc64);
    hist3 = pdl_in_mut(hist3);
    cur_inout = pdl_in_mut(cur_inout);

    if (tid < kCols16) smem_x[tid] = x_q8_u128[tid];
    if (tid < kCols64) smem_xsc[tid] = x_sc64[tid];
    __syncthreads();

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    #pragma unroll
    for (int i = 0; i < kIters; ++i) {
        const int c16 = lane + 32 * i;
        uint4 xq = smem_x[c16];
        float xsc = smem_xsc[c16 >> 2];

        int d0 = __dp4a(static_cast<int>(w0[i].x), static_cast<int>(xq.x), 0);
        d0 = __dp4a(static_cast<int>(w0[i].y), static_cast<int>(xq.y), d0);
        d0 = __dp4a(static_cast<int>(w0[i].z), static_cast<int>(xq.z), d0);
        d0 = __dp4a(static_cast<int>(w0[i].w), static_cast<int>(xq.w), d0);
        acc0 = __fmaf_rn(static_cast<float>(d0), __fmul_rn(ws0[i], xsc), acc0);

        int d1 = __dp4a(static_cast<int>(w1[i].x), static_cast<int>(xq.x), 0);
        d1 = __dp4a(static_cast<int>(w1[i].y), static_cast<int>(xq.y), d1);
        d1 = __dp4a(static_cast<int>(w1[i].z), static_cast<int>(xq.z), d1);
        d1 = __dp4a(static_cast<int>(w1[i].w), static_cast<int>(xq.w), d1);
        acc1 = __fmaf_rn(static_cast<float>(d1), __fmul_rn(ws1[i], xsc), acc1);
    }
    acc0 = warp_sum(acc0);
    acc1 = warp_sum(acc1);
    const float x1 = __shfl_sync(0xffffffffu, acc1, 0);
    pdl_trigger();

    if (lane < 2) {
        const int c = row0 + lane;
        const float x = (lane == 0) ? acc0 : x1;
        if (save_prev_out != nullptr) {
            save_prev_out[c] = cur_inout[c];
        }
        const float res = do_res_add ? cur_inout[c] : 0.0f;
        float* h = hist3 + c * 3;
        const float h0 = h[0], h1 = h[1], h2 = h[2];
        h[0] = h1;
        h[1] = h2;
        h[2] = x;
        cur_inout[c] = eval_conv4_step(wc, h0, h1, h2, x, 0, make_float4(0.0f, 0.0f, 0.0f, 0.0f), do_res_add, res);
    }
}

// Fused Expert-Parallel NVFP4 MoE Down + `mlp_sconv` + Residual Add (`Grid: 1152`, `Block: 256`, `64 B` SMEM):
// Eliminates `mlp_sconv` (`-44` launches/tok) and `d_moe_y` DRAM round-trips (`0-ULP` exact).
INK2_SCALE_KERNEL(ink2_moe_nvfp4_dp4a_down_xw_conv, (
    const uint4* __restrict__ w2_u128,     // [128, 2304, 36]
    const uint32_t* __restrict__ s2_u32,   // [128, 2304, 18]
    const float* __restrict__ a2,          // [128]
    const uint4* __restrict__ act_q8_u128, // [8, 72]
    const float* __restrict__ act_sc16,    // [8, 72]
    const int32_t* __restrict__ sel_ids,   // [8]
    const float* __restrict__ sel_wts,     // [8]
    const float4* __restrict__ w_conv4,    // [2304]
    float* __restrict__ hist3,             // [2304 * 3]
    float* __restrict__ cur_inout),        // [2304]
    (w2_u128, s2_u32, a2, act_q8_u128, act_sc16, sel_ids, sel_wts, w_conv4, hist3, cur_inout)) {
    constexpr int kRows = 2;
    __shared__ float smem_dot[kRows][kTopK];

    const int tid = threadIdx.x;
    const int k = tid >> 5;
    const int lane = tid & 31;
    const int ex = ld_ca_i32(sel_ids + k);
    const int row0 = blockIdx.x * kRows;
    const bool tail = lane < 4;

    uint4 pk_a[kRows], pk_b[kRows];
    uint32_t sc_a[kRows], sc_b[kRows];
    #pragma unroll
    for (int r = 0; r < kRows; ++r) {
        const size_t r_idx = static_cast<size_t>(ex) * kD + (row0 + r);
        const uint4* row_u128 = w2_u128 + r_idx * 36u;
        const size_t sc_start = r_idx * 18u;
        pk_a[r] = row_u128[lane];
        sc_a[r] = expert_scale_word<PackedScales, 6, 64>(s2_u32, sc_start + (lane >> 1));
        if (tail) {
            const int u16_b = lane + 32;
            pk_b[r] = row_u128[u16_b];
            sc_b[r] = expert_scale_word<PackedScales, 6, 64>(s2_u32, sc_start + (u16_b >> 1));
        } else {
            pk_b[r] = make_uint4(0u, 0u, 0u, 0u);
            sc_b[r] = 0u;
        }
    }
    const float4 wc = (tid < kRows) ? w_conv4[row0 + tid] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    pdl_wait();
    act_q8_u128 = pdl_in(act_q8_u128);
    act_sc16 = pdl_in(act_sc16);
    sel_ids = pdl_in(sel_ids);
    sel_wts = pdl_in(sel_wts);
    hist3 = pdl_in_mut(hist3);
    cur_inout = pdl_in_mut(cur_inout);

    const uint4* aq = act_q8_u128 + k * 72;
    const float* asc = act_sc16 + k * 72;
    const uint4 xa0 = aq[lane * 2];
    const uint4 xa1 = aq[lane * 2 + 1];
    const float sa0 = asc[lane * 2];
    const float sa1 = asc[lane * 2 + 1];
    uint4 xb0 = make_uint4(0u, 0u, 0u, 0u), xb1 = xb0;
    float sb0 = 0.0f, sb1 = 0.0f;
    if (tail) {
        xb0 = aq[(lane + 32) * 2];
        xb1 = aq[(lane + 32) * 2 + 1];
        sb0 = asc[(lane + 32) * 2];
        sb1 = asc[(lane + 32) * 2 + 1];
    }

    auto step_hoisted = [&](float acc, int u16, const uint4& pk16, uint32_t sc4,
                            const uint4& xq0, const uint4& xq1, float xs0, float xs1) {
        int sh = (u16 & 1) * 16;
        float s0 = __fmul_rn(e4m3_byte_to_f32((sc4 >> sh) & 0xFFu) * 0.5f, xs0);
        float s1 = __fmul_rn(e4m3_byte_to_f32((sc4 >> (sh + 8)) & 0xFFu) * 0.5f, xs1);
        int d0 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.x & 0xFFFFu)), static_cast<int>(xq0.x), 0);
        d0 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.x >> 16)), static_cast<int>(xq0.y), d0);
        d0 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.y & 0xFFFFu)), static_cast<int>(xq0.z), d0);
        d0 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.y >> 16)), static_cast<int>(xq0.w), d0);
        int d1 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.z & 0xFFFFu)), static_cast<int>(xq1.x), 0);
        d1 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.z >> 16)), static_cast<int>(xq1.y), d1);
        d1 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.w & 0xFFFFu)), static_cast<int>(xq1.z), d1);
        d1 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.w >> 16)), static_cast<int>(xq1.w), d1);
        return __fmaf_rn(static_cast<float>(d1), s1, __fmaf_rn(static_cast<float>(d0), s0, acc));
    };

    #pragma unroll
    for (int r = 0; r < kRows; ++r) {
        float acc = step_hoisted(0.0f, lane, pk_a[r], sc_a[r], xa0, xa1, sa0, sa1);
        if (tail) acc = step_hoisted(acc, lane + 32, pk_b[r], sc_b[r], xb0, xb1, sb0, sb1);
        const float dot = warp_sum(acc);
        if (lane == 0) smem_dot[r][k] = dot;
    }
    __syncthreads();
    pdl_trigger();
    if (tid < kRows) {
        float sum_y = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < kTopK; ++kk) {
            const int e = sel_ids[kk];
            sum_y = __fmaf_rn(__fmul_rn(smem_dot[tid][kk], a2[e] * (1.0f / 2688.0f)), sel_wts[kk], sum_y);
        }
        const int c = row0 + tid;
        const float res = cur_inout[c];
        float* h = hist3 + c * 3;
        const float h0 = h[0], h1 = h[1], h2 = h[2];
        h[0] = h1;
        h[1] = h2;
        h[2] = sum_y;
        cur_inout[c] = eval_conv4_step(wc, h0, h1, h2, sum_y, 0, make_float4(0.0f, 0.0f, 0.0f, 0.0f), 1, res);
    }
}

// Fused Expert-Parallel MXFP8 MoE Down + `mlp_sconv` + Residual Add (`Grid: 1152`, `Block: 256`, `64 B` SMEM):
// Eliminates `mlp_sconv` (`-4` launches/tok) and `d_moe_y` DRAM round-trips (`0-ULP` exact).
INK2_INDEXED_KERNEL(ink2_moe_mxfp8_down_xw_conv, , (
    const uint4* __restrict__ w2_u128,     // [128, 2304, 72]
    const uint32_t* __restrict__ s2_u32,   // [128, 2304, 9]
    const float4* __restrict__ act4,       // [8, 288]
    const int32_t* __restrict__ sel_ids,   // [8]
    const float* __restrict__ sel_wts,     // [8]
    const float4* __restrict__ w_conv4,    // [2304]
    float* __restrict__ hist3,             // [2304 * 3]
    float* __restrict__ cur_inout),        // [2304]
    (w2_u128, s2_u32, act4, sel_ids, sel_wts, w_conv4, hist3, cur_inout)) {
    constexpr int kRows = 2;
    __shared__ float smem_dot[kRows][kTopK];

    const int tid = threadIdx.x;
    const int k = tid >> 5;
    const int lane = tid & 31;
    const int ex = ld_ca_i32(sel_ids + k);
    const int row0 = blockIdx.x * kRows;
    const bool tail = (lane < 8);

    uint4 wp[kRows][2];
    uint4 wp_tail[kRows];
    float esc[kRows][2];
    float esc_tail[kRows];
    #pragma unroll
    for (int r = 0; r < kRows; ++r) {
        const size_t r_idx = static_cast<size_t>(ex) * kD + (row0 + r);
        const Mxfp8Row<Indexed> row_fp8(w2_u128, r_idx * 72u);
        const unsigned sc_start = static_cast<unsigned>(r_idx * 9u);
        #pragma unroll
        for (int it = 0; it < 2; ++it) {
            const int u16 = lane + 32 * it;
            wp[r][it] = row_fp8[u16];
            const int blk32 = u16 >> 1;
            uint32_t s4 = expert_scale_word<PackedScales, 3, 112>(s2_u32, sc_start + (blk32 >> 2));
            esc[r][it] = __uint_as_float(((s4 >> ((blk32 & 3) * 8)) & 0xFFu) << 23);
        }
        if (tail) {
            const int u16 = lane + 64;
            wp_tail[r] = row_fp8[u16];
            const int blk32 = u16 >> 1;
            uint32_t s4 = expert_scale_word<PackedScales, 3, 112>(s2_u32, sc_start + (blk32 >> 2));
            esc_tail[r] = __uint_as_float(((s4 >> ((blk32 & 3) * 8)) & 0xFFu) << 23);
        }
    }
    const float4 wc = (tid < kRows) ? w_conv4[row0 + tid] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    pdl_wait();
    act4 = pdl_in(act4);
    sel_wts = pdl_in(sel_wts);
    hist3 = pdl_in_mut(hist3);
    cur_inout = pdl_in_mut(cur_inout);

    const float4* ak = act4 + k * 288;
    #pragma unroll
    for (int r = 0; r < kRows; ++r) {
        float acc = 0.0f;
        acc = __fmaf_rn(dot_mxfp8_u128_exact(wp[r][0], ak + lane * 4), esc[r][0], acc);
        acc = __fmaf_rn(dot_mxfp8_u128_exact(wp[r][1], ak + (lane + 32) * 4), esc[r][1], acc);
        if (tail) {
            acc = __fmaf_rn(dot_mxfp8_u128_exact(wp_tail[r], ak + (lane + 64) * 4), esc_tail[r], acc);
        }
        const float dot = warp_sum(acc);
        if (lane == 0) smem_dot[r][k] = dot;
    }
    __syncthreads();
    pdl_trigger();
    if (tid < kRows) {
        float sum_y = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < kTopK; ++kk) {
            sum_y = __fmaf_rn(smem_dot[tid][kk], sel_wts[kk], sum_y);
        }
        const int c = row0 + tid;
        const float res = cur_inout[c];
        float* h = hist3 + c * 3;
        const float h0 = h[0], h1 = h[1], h2 = h[2];
        h[0] = h1;
        h[1] = h2;
        h[2] = sum_y;
        cur_inout[c] = eval_conv4_step(wc, h0, h1, h2, sum_y, 0, make_float4(0.0f, 0.0f, 0.0f, 0.0f), 1, res);
    }
}

// Active-Expert Grouping + Compaction (`Grid: 1`, `Block: 128`, `1 <= B <= 8`):
// Builds `exp_counts`/`exp_toks`/`exp_pairs` AND writes compacted active expert IDs into
// `active_exps[0 .. n_active)` (`-1` for `[n_active .. 127]`) in deterministic ascending order.
extern "C" __global__ void ink2_build_expert_groups_active_b(
    const int32_t* __restrict__ sel_ids, // [B * 8]
    int32_t* __restrict__ exp_counts,    // [128]
    int32_t* __restrict__ exp_toks,      // [128, 64]
    int32_t* __restrict__ exp_pairs,     // [128, 64]
    int32_t* __restrict__ active_exps,   // [128]
    int32_t B) {
    pdl_wait();
    pdl_trigger();
    __shared__ int32_t smem_sel[128];
    __shared__ int32_t sm_wcnt[4];

    const int ex = threadIdx.x; // 0..127
    const int warp = ex >> 5;
    const int lane = ex & 31;
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

    const bool active = (cnt > 0);
    const unsigned mask = __ballot_sync(0xffffffffu, active);
    if (lane == 0) {
        sm_wcnt[warp] = __popc(mask);
    }
    __syncthreads();

    int base = 0;
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        if (w < warp) base += sm_wcnt[w];
    }
    const int total_active = sm_wcnt[0] + sm_wcnt[1] + sm_wcnt[2] + sm_wcnt[3];
    if (active) {
        const int dst = base + __popc(mask & ((1u << lane) - 1u));
        active_exps[dst] = ex;
    }
    if (total_active + ex < 128) {
        active_exps[total_active + ex] = -1;
    }
}

// Speculative verify (`1 <= B <= 8`): `ink2_router_topk8_b` (warp `w` = token `w`) followed by
// `ink2_build_expert_groups_active_b` (threads `0..127` = experts) in one CTA, reading the
// selected ids from shared memory. Same per-token top-8 and the same grouping order.
// `Grid (1, 1)`, `Block 256`.
extern "C" __global__ void ink2_router_topk8_groups_b(
    const float* __restrict__ logits128_base, // [B, 128]
    const float* __restrict__ bias128,        // [128]
    int32_t* __restrict__ sel_ids_base,       // [B, 8]
    float* __restrict__ sel_wts_base,         // [B, 8]
    int32_t* __restrict__ exp_counts,         // [128]
    int32_t* __restrict__ exp_toks,           // [128, 64]
    int32_t* __restrict__ exp_pairs,          // [128, 64]
    int32_t* __restrict__ active_exps,        // [128]
    int32_t B) {
    pdl_wait();
    pdl_trigger();
    __shared__ int smem_ids[8][8];
    __shared__ float smem_ps[8][8];
    __shared__ int32_t smem_sel[64];
    __shared__ int32_t sm_wcnt[4];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (warp < B) {
        router_topk8_row(logits128_base, bias128, sel_ids_base, sel_wts_base, smem_ids[warp],
                         smem_ps[warp], warp, lane);
        if (lane < 8) smem_sel[warp * 8 + lane] = smem_ids[warp][lane];
    }
    __syncthreads();
    if (tid >= 128) return;

    const int ex = tid;
    const int total_pairs = B * 8;
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

    const bool active = (cnt > 0);
    const unsigned mask = __ballot_sync(0xffffffffu, active);
    if (lane == 0) {
        sm_wcnt[warp] = __popc(mask);
    }
    // Only the 128 expert threads remain: named barrier 1 instead of `__syncthreads()`.
    asm volatile("bar.sync 1, 128;" ::: "memory");

    int base = 0;
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        if (w < warp) base += sm_wcnt[w];
    }
    const int total_active = sm_wcnt[0] + sm_wcnt[1] + sm_wcnt[2] + sm_wcnt[3];
    if (active) {
        const int dst = base + __popc(mask & ((1u << lane) - 1u));
        active_exps[dst] = ex;
    }
    if (total_active + ex < 128) {
        active_exps[total_active + ex] = -1;
    }
}

// Active-Expert Register-Cached Half-Warp NVFP4 Gate/Up + Fused `Q8_16` Epilogue (`1 <= B <= 8`):
// `Grid: (72, min(128, B * 8))`, `Block: 512` (`2` CTAs/SM = `32` warps/SM, `544 B` SMEM, `120 KiB` L1!).
// Loads `w13` weights ONCE into registers per active expert and reuses across all `t = 0 .. cnt - 1` tokens,
// matching `nvfp4_half_warp_situ` + `ink2_quant_q8_16_packed` to `0.0000-ULP`!
INK2_SCALE_KERNEL(ink2_moe_nvfp4_dp4a_verify_up_half_q8, (
    const uint4* __restrict__ w13_u128,        // [128, 2304, 72]
    const uint32_t* __restrict__ s13_u32,      // [128, 2304, 36]
    const float* __restrict__ a13,             // [128]
    const uint4* __restrict__ x_q8_u128,       // [B, 144]
    const float* __restrict__ x_sc16,          // [B, 144]
    const int32_t* __restrict__ active_exps,   // [128]
    const int32_t* __restrict__ exp_counts,    // [128]
    const int32_t* __restrict__ exp_toks,      // [128, 64]
    const int32_t* __restrict__ exp_pairs,     // [128, 64]
    uint32_t* __restrict__ out_q8_u32,         // [B * 8, 288]
    float* __restrict__ out_sc16),             // [B * 8, 72]
    (w13_u128, s13_u32, a13, x_q8_u128, x_sc16, active_exps, exp_counts, exp_toks, exp_pairs, out_q8_u32, out_sc16)) {
    pdl_trigger();
    const int ex = active_exps[blockIdx.y];
    if (ex < 0) return;
    const int cnt = exp_counts[ex];
    if (cnt <= 0) return;

    __shared__ float smem_ch[8][16];
    __shared__ int32_t smem_pairs[8];

    const int warp_id = threadIdx.x >> 5;   // 0..15
    const int lane = threadIdx.x & 31;      // 0..31
    const int blk16 = blockIdx.x;           // 0..71
    const int channel = blk16 * 16 + warp_id;
    const int index = lane & 15;            // 0..15
    const bool tail8 = (index < 8);
    const size_t row = (static_cast<size_t>(ex) * kFf + channel) * 2u + (lane >> 4);
    const uint4* row_u128 = w13_u128 + row * 72u;
    const unsigned sc_start = static_cast<unsigned>(row * 36u);
    const float alpha_scaled = a13[ex] * (1.0f / 2688.0f);

    const uint4 pk0 = row_u128[index];
    const uint4 pk1 = row_u128[index + 16];
    const uint4 pk2 = row_u128[index + 32];
    const uint4 pk3 = row_u128[index + 48];
    const uint4 pk4 = tail8 ? row_u128[index + 64] : make_uint4(0u, 0u, 0u, 0u);
    const uint32_t sc0 = expert_scale_word<PackedScales, 6, 64>(s13_u32, sc_start + (index >> 1));
    const uint32_t sc1 = expert_scale_word<PackedScales, 6, 64>(s13_u32, sc_start + ((index + 16) >> 1));
    const uint32_t sc2 = expert_scale_word<PackedScales, 6, 64>(s13_u32, sc_start + ((index + 32) >> 1));
    const uint32_t sc3 = expert_scale_word<PackedScales, 6, 64>(s13_u32, sc_start + ((index + 48) >> 1));
    const uint32_t sc4 = tail8 ? expert_scale_word<PackedScales, 6, 64>(s13_u32, sc_start + ((index + 64) >> 1)) : 0u;

    auto step_u16 = [](float acc, int u16, const uint4& pk16, uint32_t sc_word,
                       const uint4* xq16, const float* xsc16) {
        const int sh = (u16 & 1) * 16;
        const float s0 = __fmul_rn(e4m3_byte_to_f32((sc_word >> sh) & 0xFFu) * 0.5f, xsc16[u16 * 2]);
        const float s1 = __fmul_rn(e4m3_byte_to_f32((sc_word >> (sh + 8)) & 0xFFu) * 0.5f, xsc16[u16 * 2 + 1]);
        const uint4 xq0 = xq16[u16 * 2];
        const uint4 xq1 = xq16[u16 * 2 + 1];
        int d0 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.x & 0xFFFFu)), static_cast<int>(xq0.x), 0);
        d0 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.x >> 16)), static_cast<int>(xq0.y), d0);
        d0 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.y & 0xFFFFu)), static_cast<int>(xq0.z), d0);
        d0 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.y >> 16)), static_cast<int>(xq0.w), d0);
        int d1 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.z & 0xFFFFu)), static_cast<int>(xq1.x), 0);
        d1 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.z >> 16)), static_cast<int>(xq1.y), d1);
        d1 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.w & 0xFFFFu)), static_cast<int>(xq1.z), d1);
        d1 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.w >> 16)), static_cast<int>(xq1.w), d1);
        return __fmaf_rn(static_cast<float>(d1), s1, __fmaf_rn(static_cast<float>(d0), s0, acc));
    };

    for (int t_base = 0; t_base < cnt; t_base += 8) {
        if (t_base > 0) __syncthreads();
        const int tile_t = min(8, cnt - t_base);
        if (threadIdx.x < tile_t) {
            smem_pairs[threadIdx.x] = exp_pairs[ex * 64 + t_base + threadIdx.x];
        }
        for (int t = 0; t < tile_t; ++t) {
            const int tok = exp_toks[ex * 64 + t_base + t];
            const uint4* xq_tok = x_q8_u128 + static_cast<size_t>(tok) * 144u;
            const float* xs_tok = x_sc16 + static_cast<size_t>(tok) * 144u;

            float lo = step_u16(0.0f, index, pk0, sc0, xq_tok, xs_tok);
            lo = step_u16(lo, index + 32, pk2, sc2, xq_tok, xs_tok);
            if (tail8) {
                lo = step_u16(lo, index + 64, pk4, sc4, xq_tok, xs_tok);
            }
            float hi = step_u16(0.0f, index + 16, pk1, sc1, xq_tok, xs_tok);
            hi = step_u16(hi, index + 48, pk3, sc3, xq_tok, xs_tok);

            float dot = __fadd_rn(lo, hi);
            #pragma unroll
            for (int mask = 8; mask > 0; mask >>= 1) {
                dot = __fadd_rn(dot, __shfl_xor_sync(0xffffffffu, dot, mask));
            }
            dot = __fmul_rn(dot, alpha_scaled);
            const float u = __shfl_sync(0xffffffffu, dot, 16);
            if (lane == 0) {
                const float g = dot;
                smem_ch[t][warp_id] =
                    4.0f * ptx_tanh(g * 0.25f) * ptx_sigmoid(g) * 25.0f * ptx_tanh(u * 0.04f);
            }
        }
        __syncthreads();

        if (warp_id < tile_t && lane < 16) {
            const int pair = smem_pairs[warp_id];
            const float v = smem_ch[warp_id][lane];
            float amax = fabsf(v);
            #pragma unroll
            for (int mask = 8; mask > 0; mask >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0x0000FFFFu, amax, mask, 16));
            }
            const float sc = amax * (1.0f / 127.0f);
            const float inv = (amax > 0.0f) ? (127.0f * ptx_rcp(amax)) : 0.0f;
            int q = __float2int_rn(v * inv);
            if (q < -127) q = -127;
            if (q > 127) q = 127;
            const uint32_t qb = static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(q)));
            const uint32_t p4 = qb
                | (__shfl_sync(0x0000FFFFu, qb, lane + 1, 16) << 8)
                | (__shfl_sync(0x0000FFFFu, qb, lane + 2, 16) << 16)
                | (__shfl_sync(0x0000FFFFu, qb, lane + 3, 16) << 24);
            if ((lane & 3) == 0) {
                out_q8_u32[static_cast<size_t>(pair) * 288u + blk16 * 4 + (lane >> 2)] = p4;
            }
            if (lane == 0) {
                out_sc16[static_cast<size_t>(pair) * 72u + blk16] = sc;
            }
        }
    }
}

// Active-Expert Register-Cached NVFP4 MoE Down (`Grid: (288, min(128, B * 8))`, `Block: 256`, `0 B` SMEM!):
// `6` CTAs/SM = `48` warps/SM (`100%` occupancy, `120 KiB` L1), `0` barriers, `0.0000-ULP` exact!
INK2_SCALE_KERNEL(ink2_moe_nvfp4_dp4a_verify_down, (
    const uint4* __restrict__ w2_u128,         // [128, 2304, 36]
    const uint32_t* __restrict__ s2_u32,       // [128, 2304, 18]
    const float* __restrict__ a2,              // [128]
    const uint4* __restrict__ act_q8_u128,     // [B * 8, 72]
    const float* __restrict__ act_sc16,        // [B * 8, 72]
    const int32_t* __restrict__ active_exps,   // [128]
    const int32_t* __restrict__ exp_counts,    // [128]
    const int32_t* __restrict__ exp_pairs,     // [128, 64]
    float* __restrict__ pair_y),               // [B * 8, 2304]
    (w2_u128, s2_u32, a2, act_q8_u128, act_sc16, active_exps, exp_counts, exp_pairs, pair_y)) {
    pdl_trigger();
    const int ex = active_exps[blockIdx.y];
    if (ex < 0) return;
    const int cnt = exp_counts[ex];
    if (cnt <= 0) return;

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp_id; // 288 * 8 = 2304 exact
    const bool tail = (lane < 4);
    const size_t r_idx = static_cast<size_t>(ex) * kD + row;
    const uint4* row_u128 = w2_u128 + r_idx * 36u;
    const size_t sc_start = r_idx * 18u;
    const float expert_scale = a2[ex] * (1.0f / 2688.0f);

    const uint4 pk_a = row_u128[lane];
    const uint32_t sc_a = expert_scale_word<PackedScales, 6, 64>(s2_u32, sc_start + (lane >> 1));
    const uint4 pk_b = tail ? row_u128[lane + 32] : make_uint4(0u, 0u, 0u, 0u);
    const uint32_t sc_b = tail ? expert_scale_word<PackedScales, 6, 64>(s2_u32, sc_start + ((lane + 32) >> 1)) : 0u;

    auto step_hoisted = [](float acc, int u16, const uint4& pk16, uint32_t sc4,
                           const uint4* aq, const float* asc) {
        const int sh = (u16 & 1) * 16;
        const float s0 = __fmul_rn(e4m3_byte_to_f32((sc4 >> sh) & 0xFFu) * 0.5f, asc[u16 * 2]);
        const float s1 = __fmul_rn(e4m3_byte_to_f32((sc4 >> (sh + 8)) & 0xFFu) * 0.5f, asc[u16 * 2 + 1]);
        const uint4 xq0 = aq[u16 * 2];
        const uint4 xq1 = aq[u16 * 2 + 1];
        int d0 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.x & 0xFFFFu)), static_cast<int>(xq0.x), 0);
        d0 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.x >> 16)), static_cast<int>(xq0.y), d0);
        d0 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.y & 0xFFFFu)), static_cast<int>(xq0.z), d0);
        d0 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.y >> 16)), static_cast<int>(xq0.w), d0);
        int d1 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.z & 0xFFFFu)), static_cast<int>(xq1.x), 0);
        d1 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.z >> 16)), static_cast<int>(xq1.y), d1);
        d1 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.w & 0xFFFFu)), static_cast<int>(xq1.z), d1);
        d1 = __dp4a(static_cast<int>(decode_4x_e2m1_nibbles16_to_2x_s8(pk16.w >> 16)), static_cast<int>(xq1.w), d1);
        return __fmaf_rn(static_cast<float>(d1), s1, __fmaf_rn(static_cast<float>(d0), s0, acc));
    };

    for (int t = 0; t < cnt; ++t) {
        const int pair = exp_pairs[ex * 64 + t];
        const uint4* aq = act_q8_u128 + static_cast<size_t>(pair) * 72u;
        const float* asc = act_sc16 + static_cast<size_t>(pair) * 72u;
        float acc = step_hoisted(0.0f, lane, pk_a, sc_a, aq, asc);
        if (tail) {
            acc = step_hoisted(acc, lane + 32, pk_b, sc_b, aq, asc);
        }
        const float dot = warp_sum(acc);
        if (lane == 0) {
            pair_y[static_cast<size_t>(pair) * kD + row] = __fmul_rn(dot, expert_scale);
        }
    }
}

// Active-Expert Register-Cached MXFP8 Gate/Up + SiTU (`Grid: (288, min(128, B * 8))`, `Block: 256`, `288 B` SMEM!):
// Replaces `36 KiB`-SMEM (`16.7%` occ) `mxfp8_grouped_up_situ` at `1 <= B <= 8` (`0.0000-ULP` exact!).
INK2_INDEXED_KERNEL(ink2_moe_mxfp8_verify_up_situ, __launch_bounds__(256, 2), (
    const uint4* __restrict__ w13_u128,        // [128, 2304, 144]
    const uint32_t* __restrict__ s13_u32,      // [128, 2304, 18]
    const float4* __restrict__ x4_base,        // [B, 576]
    const int32_t* __restrict__ active_exps,   // [128]
    const int32_t* __restrict__ exp_counts,    // [128]
    const int32_t* __restrict__ exp_toks,      // [128, 64]
    const int32_t* __restrict__ exp_pairs,     // [128, 64]
    float* __restrict__ act_out),              // [B * 8, 1152]
    (w13_u128, s13_u32, x4_base, active_exps, exp_counts, exp_toks, exp_pairs, act_out)) {
    pdl_trigger();
    const int ex = active_exps[blockIdx.y];
    if (ex < 0) return;
    const int cnt = exp_counts[ex];
    if (cnt <= 0) return;

    __shared__ float smem_gu[8][8];
    __shared__ int32_t smem_pairs[8];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int m = blockIdx.x * 4 + (warp_id >> 1);
    const int is_up = warp_id & 1;
    const int r = m * 2 + is_up;
    const size_t row_idx = static_cast<size_t>(ex) * (2u * kFf) + r;
    const Mxfp8Row<Indexed> row_fp8(w13_u128, row_idx * 144u);
    const unsigned sc_start = static_cast<unsigned>(row_idx * 18u);

    uint4 wp[4];
    float esc[4];
    #pragma unroll
    for (int it = 0; it < 4; ++it) {
        const int u16 = lane + 32 * it;
        wp[it] = row_fp8[u16];
        esc[it] = load_e8m0_scale<PackedScales>(s13_u32, sc_start, u16);
    }
    const bool tail16 = (lane < 16);
    const uint4 wp4 = tail16 ? row_fp8[lane + 128] : make_uint4(0u, 0u, 0u, 0u);
    const float esc4 = tail16 ? load_e8m0_scale<PackedScales>(s13_u32, sc_start, lane + 128) : 0.0f;

    for (int t_base = 0; t_base < cnt; t_base += 8) {
        if (t_base > 0) __syncthreads();
        const int tile_t = min(8, cnt - t_base);
        if (tid < tile_t) {
            smem_pairs[tid] = exp_pairs[ex * 64 + t_base + tid];
        }
        for (int t = 0; t < tile_t; ++t) {
            const int tok = exp_toks[ex * 64 + t_base + t];
            const float4* x4 = x4_base + static_cast<size_t>(tok) * 576u;
            float acc = 0.0f;
            #pragma unroll
            for (int it = 0; it < 4; ++it) {
                acc = __fmaf_rn(dot_mxfp8_u128_exact(wp[it], x4 + (lane + 32 * it) * 4), esc[it], acc);
            }
            if (tail16) {
                acc = __fmaf_rn(dot_mxfp8_u128_exact(wp4, x4 + (lane + 128) * 4), esc4, acc);
            }
            const float dot = warp_sum(acc);
            if (lane == 0) {
                smem_gu[t][warp_id] = dot;
            }
        }
        __syncthreads();
        if (is_up == 0 && lane < tile_t) {
            const float g = smem_gu[lane][warp_id];
            const float u = smem_gu[lane][warp_id + 1];
            const float situ = 4.0f * ptx_tanh(g * 0.25f) * ptx_sigmoid(g) * 25.0f * ptx_tanh(u * 0.04f);
            act_out[static_cast<size_t>(smem_pairs[lane]) * kFf + m] = situ;
        }
    }
}

// Active-Expert Register-Cached MXFP8 Down (`Grid: (288, min(128, B * 8))`, `Block: 256`, `0 B` SMEM!):
// `4` CTAs/SM = `32` warps/SM, `0` barriers, `0.0000-ULP` exact!
INK2_INDEXED_KERNEL(ink2_moe_mxfp8_verify_down, , (
    const uint4* __restrict__ w2_u128,         // [128, 2304, 72]
    const uint32_t* __restrict__ s2_u32,       // [128, 2304, 9]
    const float4* __restrict__ act4_base,      // [B * 8, 288]
    const int32_t* __restrict__ active_exps,   // [128]
    const int32_t* __restrict__ exp_counts,    // [128]
    const int32_t* __restrict__ exp_pairs,     // [128, 64]
    float* __restrict__ pair_y),               // [B * 8, 2304]
    (w2_u128, s2_u32, act4_base, active_exps, exp_counts, exp_pairs, pair_y)) {
    pdl_trigger();
    const int ex = active_exps[blockIdx.y];
    if (ex < 0) return;
    const int cnt = exp_counts[ex];
    if (cnt <= 0) return;

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp_id; // 288 * 8 = 2304 exact
    const bool tail8 = (lane < 8);
    const size_t r_idx = static_cast<size_t>(ex) * kD + row;
    const Mxfp8Row<Indexed> row_fp8(w2_u128, r_idx * 72u);
    const unsigned sc_start = static_cast<unsigned>(r_idx * 9u);

    uint4 wp[2];
    float esc[2];
    #pragma unroll
    for (int it = 0; it < 2; ++it) {
        const int u16 = lane + 32 * it;
        wp[it] = row_fp8[u16];
        esc[it] = load_e8m0_scale<PackedScales>(s2_u32, sc_start, u16);
    }
    const uint4 wp_tail = tail8 ? row_fp8[lane + 64] : make_uint4(0u, 0u, 0u, 0u);
    const float esc_tail = tail8 ? load_e8m0_scale<PackedScales>(s2_u32, sc_start, lane + 64) : 0.0f;

    for (int t = 0; t < cnt; ++t) {
        const int pair = exp_pairs[ex * 64 + t];
        const float4* ak = act4_base + static_cast<size_t>(pair) * 288u;
        float acc = 0.0f;
        acc = __fmaf_rn(dot_mxfp8_u128_exact(wp[0], ak + lane * 4), esc[0], acc);
        acc = __fmaf_rn(dot_mxfp8_u128_exact(wp[1], ak + (lane + 32) * 4), esc[1], acc);
        if (tail8) {
            acc = __fmaf_rn(dot_mxfp8_u128_exact(wp_tail, ak + (lane + 64) * 4), esc_tail, acc);
        }
        const float dot = warp_sum(acc);
        if (lane == 0) {
            pair_y[static_cast<size_t>(pair) * kD + row] = dot;
        }
    }
}

#undef INK2_INDEXED_KERNEL
#undef INK2_SCALE_KERNEL_LB
#undef INK2_SCALE_KERNEL

// ============================================================================
// 18. `v0.7.0` SM120a Wide Tensor Core Prefill Suite (`Issue #10` -> `14K+ tok/s`)
//     - Parallel 2D Causal 4-Tap FIR `short_conv4` (`0.0000-ULP` bitwise exact!)
//     - Double-Buffered `cp.async.cg` `64x64` Multi-Warp `Q8_64` `IMMA.16832` GEMM
//     - Resident-Weight `32x32` Multi-Warp `NVFP4` `IMMA.16832` Grouped MoE (`0` `partials` SMEM!)
//     - Native SM120a `mxf8f6f4` (`m16n8k32.e4m3.ue8m0`) `64x64` `MXFP8` Grouped MoE
//     - `22-Bit` (`q_hi + q_lo/2048`) FP16 Tensor Core FlashAttention + In-Register `Q8_64` Epilogue
// ============================================================================

#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL) || \
    (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200 || __CUDA_ARCH_SPECIFIC__ == 1210)) || \
    (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200 || __CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
#define INK2_HAS_SM120_BLOCKSCALE 1
#else
#define INK2_HAS_SM120_BLOCKSCALE 0
#endif

namespace leansonic::inkling2 {

__device__ __forceinline__ uint32_t smem_u32_ptr(const void* ptr) {
    uint64_t s64;
    asm("cvta.to.shared.u64 %0, %1;\n" : "=l"(s64) : "l"(ptr));
    return static_cast<uint32_t>(s64);
}

__device__ __forceinline__ void cp_async16_cg_pred(void* smem_dst, const void* gmem_src, bool valid) {
    const uint32_t sa = smem_u32_ptr(smem_dst);
    const int sz = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sa), "l"(gmem_src), "r"(sz) : "memory");
}

__device__ __forceinline__ void cp_async_commit_grp() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int Keep>
__device__ __forceinline__ void cp_async_wait_grp() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(Keep) : "memory");
}

__device__ __forceinline__ void mma_s8_m16n8k32(
    int& d0, int& d1, int& d2, int& d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1,
    int c0, int c1, int c2, int c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3));
}

__device__ __forceinline__ void mma_s8_m16n8k16_via_k32(
    int& d0, int& d1, int& d2, int& d3,
    unsigned a0, unsigned a1, unsigned b0) {
    mma_s8_m16n8k32(d0, d1, d2, d3, a0, a1, 0u, 0u, b0, 0u, 0, 0, 0, 0);
}

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, mask));
    }
    return v;
}

__device__ __forceinline__ void cp_async16_cg(void* smem_dst, const void* gmem_src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_u32_ptr(smem_dst)), "l"(gmem_src) : "memory");
}

__device__ __forceinline__ void cp_async4_ca(void* smem_dst, const void* gmem_src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(smem_u32_ptr(smem_dst)), "l"(gmem_src) : "memory");
}

template <int TileTokens>
__device__ __forceinline__ bool resolve_flat_expert_tile(
    const int32_t* __restrict__ exp_counts,
    int32_t (&sm_blk_off)[129],
    int32_t (&sm_row_off)[129],
    int& out_expert,
    int& out_first,
    int& out_count,
    size_t& out_sorted_base) {
    __shared__ int32_t sm_warp_b[4];
    __shared__ int32_t sm_warp_r[4];
    const int tid = threadIdx.x;
    if (tid < 128) {
        const int lane = tid & 31;
        const int wid = tid >> 5;
        const int c = exp_counts[tid];
        int b_scan = (c + TileTokens - 1) / TileTokens;
        int r_scan = c;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int nb = __shfl_up_sync(0xFFFFFFFFu, b_scan, o);
            const int nr = __shfl_up_sync(0xFFFFFFFFu, r_scan, o);
            if (lane >= o) {
                b_scan += nb;
                r_scan += nr;
            }
        }
        sm_blk_off[tid + 1] = b_scan;
        sm_row_off[tid + 1] = r_scan;
        if (lane == 31) {
            sm_warp_b[wid] = b_scan;
            sm_warp_r[wid] = r_scan;
        }
    } else if (tid == 128) {
        sm_blk_off[0] = 0;
        sm_row_off[0] = 0;
    }
    __syncthreads();
    if (tid >= 32 && tid < 128) {
        const int wid = tid >> 5;
        int add_b = 0;
        int add_r = 0;
        #pragma unroll
        for (int w = 0; w < 3; ++w) {
            if (w < wid) {
                add_b += sm_warp_b[w];
                add_r += sm_warp_r[w];
            }
        }
        sm_blk_off[tid + 1] += add_b;
        sm_row_off[tid + 1] += add_r;
    }
    __syncthreads();
    const int tile_y = static_cast<int>(blockIdx.y);
    if (tile_y >= sm_blk_off[128]) {
        return false;
    }
    int lo = 0;
    int hi = 128;
    #pragma unroll
    for (int step = 0; step < 7; ++step) {
        const int mid = (lo + hi) >> 1;
        if (sm_blk_off[mid] <= tile_y) lo = mid;
        else hi = mid;
    }
    out_expert = lo;
    const int tp = tile_y - sm_blk_off[lo];
    out_first = tp * TileTokens;
    out_count = sm_row_off[lo + 1] - sm_row_off[lo];
    out_sorted_base = static_cast<size_t>(sm_row_off[lo] + out_first);
    return true;
}

__device__ __forceinline__ void cp_async4_ca_pred(void* smem_dst, const void* gmem_src, bool valid) {
    const uint32_t sa = smem_u32_ptr(smem_dst);
    const int sz = valid ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(sa), "l"(gmem_src), "r"(sz) : "memory");
}

__device__ __forceinline__ void cp_async8_ca_pred(void* smem_dst, const void* gmem_src, bool valid) {
    const uint32_t sa = smem_u32_ptr(smem_dst);
    const int sz = valid ? 8 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" :: "r"(sa), "l"(gmem_src), "r"(sz) : "memory");
}

__device__ __forceinline__ void ldsm_x2_b16(unsigned& r0, unsigned& r1, const void* p) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(smem_u32_ptr(p)));
}

__device__ __forceinline__ void ldsm_x4_b16(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_u32_ptr(p)));
}

__device__ __forceinline__ void ldsm_x4_trans_b16(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_u32_ptr(p)));
}

__device__ __forceinline__ void mma_f16_m16n8k16(
    float& c0, float& c1, float& c2, float& c3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_bf16_m16n8k16(
    float& c0, float& c1, float& c2, float& c3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_f16x2_rn(float lo, float hi) {
    uint32_t packed;
    asm("cvt.rn.f16x2.f32 %0, %2, %1;\n" : "=r"(packed) : "f"(lo), "f"(hi));
    return packed;
}

__device__ __forceinline__ uint32_t pack_bf16x2_rn(float lo, float hi) {
    uint32_t packed;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;\n" : "=r"(packed) : "f"(lo), "f"(hi));
    return packed;
}

__device__ __forceinline__ float ue4m3_to_f32_exact(uint32_t b) {
    b &= 0x7Fu;
    if (b == 0u || b == 0x7Fu) return 0.0f;
    if (b < 8u) return static_cast<float>(b) * (1.0f / 512.0f);
    return __uint_as_float((b + (120u << 3)) << 20);
}

__device__ __forceinline__ uint8_t cvt_f32_to_ue4m3_byte(float x) {
    if (!(x > 0.0f)) return 0u;
    uint16_t pair;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\n" : "=h"(pair) : "f"(x), "f"(0.0f));
    return static_cast<uint8_t>(pair & 0x7Fu);
}

__device__ __forceinline__ float e2m1_code_to_f32(uint32_t c) {
    const uint32_t m = c & 7u;
    const float mag = (m < 2u)
        ? (0.5f * static_cast<float>(m))
        : (((m & 1u) ? 1.5f : 1.0f) * static_cast<float>(1u << ((m >> 1) - 1u)));
    return (c & 8u) ? -mag : mag;
}

__device__ __forceinline__ uint32_t f32_to_e2m1_nibble_sw(float v) {
    const uint32_t sgn = (v < 0.0f) ? 8u : 0u;
    const float a = fabsf(v);
    uint32_t m;
    if (a <= 0.25f) m = 0u;
    else if (a < 0.75f) m = 1u;
    else if (a <= 1.25f) m = 2u;
    else if (a < 1.75f) m = 3u;
    else if (a <= 2.5f) m = 4u;
    else if (a < 3.5f) m = 5u;
    else if (a <= 5.0f) m = 6u;
    else m = 7u;
    return sgn | m;
}

__device__ __forceinline__ uint2 pack_e2m1_subblock16(const float (&x)[16], float inv) {
    uint32_t w0 = 0u;
    uint32_t w1 = 0u;
#if INK2_HAS_SM120_BLOCKSCALE
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint16_t c0, c1;
        asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; cvt.u16.u8 %0, t; }\n"
            : "=h"(c0) : "f"(x[j] * inv), "f"(x[j + 8] * inv));
        asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; cvt.u16.u8 %0, t; }\n"
            : "=h"(c1) : "f"(x[j + 4] * inv), "f"(x[j + 12] * inv));
        w0 |= static_cast<uint32_t>(c0 & 0xFFu) << (8 * j);
        w1 |= static_cast<uint32_t>(c1 & 0xFFu) << (8 * j);
    }
#else
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t b0 = f32_to_e2m1_nibble_sw(x[j] * inv) | (f32_to_e2m1_nibble_sw(x[j + 8] * inv) << 4);
        const uint32_t b1 = f32_to_e2m1_nibble_sw(x[j + 4] * inv) | (f32_to_e2m1_nibble_sw(x[j + 12] * inv) << 4);
        w0 |= b0 << (8 * j);
        w1 |= b1 << (8 * j);
    }
#endif
    return make_uint2(w0, w1);
}

// Spreads 4 nibbles `[n3 n2 n1 n0]` in `x16` into `[0 n3 0 n2 0 n1 0 n0]` in 4 integer ops!
__device__ __forceinline__ uint32_t spread4_nibbles(uint32_t x16) {
    const uint32_t u = (x16 | (x16 << 8)) & 0x00FF00FFu;
    return (u | (u << 4)) & 0x0F0F0F0Fu;
}

// Converts linear safetensors NVFP4 `(lo32 = v[0..7], hi32 = v[8..15])` into `mxf4nvf4` sub-block layout `(v[j+8]<<4)|v[j]` in 5 PTX ops!
__device__ __forceinline__ uint32_t interleave_nvfp4_sub_sel(uint32_t lo32, uint32_t hi32, uint32_t prmt_sel) {
    const uint32_t p = __byte_perm(lo32, hi32, prmt_sel);
    return (p & 0xF00FF00Fu) | ((p << 4) & 0x0F000F00u) | ((p >> 4) & 0x00F000F0u);
}

__device__ __forceinline__ uint32_t sanitize_ue4m3x4(uint32_t raw) {
    const uint32_t v = raw & 0x7F7F7F7Fu;
    return v & ~__vcmpeq4(v, 0x7F7F7F7Fu);
}

__device__ __forceinline__ void mma_nvfp4_m16n8k64(
    float (&c)[4],
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1,
    unsigned sfa, unsigned sfb,
    const uint32_t* sm_a_r0_lo, const uint32_t* sm_a_r0_hi,
    const uint32_t* sm_a_r1_lo, const uint32_t* sm_a_r1_hi,
    const uint32_t* sm_b_c0_lo, const uint32_t* sm_b_c0_hi,
    const uint32_t* sm_b_c1_lo, const uint32_t* sm_b_c1_hi,
    uint32_t sfa_r0, uint32_t sfa_r1, uint32_t sfb_c0, uint32_t sfb_c1) {
#if INK2_HAS_SM120_BLOCKSCALE
    (void)sm_a_r0_lo; (void)sm_a_r0_hi; (void)sm_a_r1_lo; (void)sm_a_r1_hi;
    (void)sm_b_c0_lo; (void)sm_b_c0_hi; (void)sm_b_c1_lo; (void)sm_b_c1_hi;
    (void)sfa_r0; (void)sfa_r1; (void)sfb_c0; (void)sfb_c1;
    constexpr unsigned short kId = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(sfa), "h"(kId), "h"(kId),
          "r"(sfb), "h"(kId), "h"(kId));
#else
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1; (void)sfa; (void)sfb;
    auto dot64_sw = [](const uint32_t* w_lo, const uint32_t* w_hi, const uint32_t* x_lo, const uint32_t* x_hi, uint32_t ws4, uint32_t xs4) -> float {
        float acc = 0.0f;
        #pragma unroll
        for (int sb = 0; sb < 4; ++sb) {
            const float ws = ue4m3_to_f32_exact((ws4 >> (sb * 8)) & 0x7Fu);
            const float xs = ue4m3_to_f32_exact((xs4 >> (sb * 8)) & 0x7Fu);
            const uint32_t* w_p = (sb < 2) ? (w_lo + 2 * sb) : (w_hi + 2 * (sb - 2));
            const uint32_t* x_p = (sb < 2) ? (x_lo + 2 * sb) : (x_hi + 2 * (sb - 2));
            const uint32_t wl0 = w_p[0], wl1 = w_p[1];
            const uint32_t xl0 = x_p[0], xl1 = x_p[1];
            float sub_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t xb0 = (xl0 >> (8 * j)) & 0xFFu;
                const uint32_t xb1 = (xl1 >> (8 * j)) & 0xFFu;
                sub_sum += e2m1_code_to_f32((wl0 >> (4 * j)) & 0xFu) * e2m1_code_to_f32(xb0 & 0xFu);
                sub_sum += e2m1_code_to_f32((wl1 >> (4 * j)) & 0xFu) * e2m1_code_to_f32((xb0 >> 4) & 0xFu);
                sub_sum += e2m1_code_to_f32((wl0 >> (16 + 4 * j)) & 0xFu) * e2m1_code_to_f32(xb1 & 0xFu);
                sub_sum += e2m1_code_to_f32((wl1 >> (16 + 4 * j)) & 0xFu) * e2m1_code_to_f32((xb1 >> 4) & 0xFu);
            }
            acc += sub_sum * (ws * xs);
        }
        return acc;
    };
    c[0] += dot64_sw(sm_a_r0_lo, sm_a_r0_hi, sm_b_c0_lo, sm_b_c0_hi, sfa_r0, sfb_c0);
    c[1] += dot64_sw(sm_a_r0_lo, sm_a_r0_hi, sm_b_c1_lo, sm_b_c1_hi, sfa_r0, sfb_c1);
    c[2] += dot64_sw(sm_a_r1_lo, sm_a_r1_hi, sm_b_c0_lo, sm_b_c0_hi, sfa_r1, sfb_c0);
    c[3] += dot64_sw(sm_a_r1_lo, sm_a_r1_hi, sm_b_c1_lo, sm_b_c1_hi, sfa_r1, sfb_c1);
#endif
}

template <int V>
struct KsConst {
    static constexpr int k = V;
};

template <int KS>
__device__ __forceinline__ void mma_mxfp8_m16n8k32(
    float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1, unsigned sfa, unsigned sfb,
    const uint32_t* smem_a_row0, const uint32_t* smem_a_row1,
    const uint32_t* smem_b_col0, const uint32_t* smem_b_col1,
    float sfa0_f, float sfa1_f, float sfb0_f, float sfb1_f) {
#if INK2_HAS_SM120_BLOCKSCALE
    (void)smem_a_row0; (void)smem_a_row1; (void)smem_b_col0; (void)smem_b_col1;
    (void)sfa0_f; (void)sfa1_f; (void)sfb0_f; (void)sfb1_f;
    constexpr unsigned short kBid = static_cast<unsigned short>(KS);
    constexpr unsigned short kTid = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b0), "r"(b1),
          "r"(sfa), "h"(kBid), "h"(kTid),
          "r"(sfb), "h"(kBid), "h"(kTid));
#else
    (void)a; (void)b0; (void)b1; (void)sfa; (void)sfb;
    auto dot32_e4m3 = [](const uint32_t* pa, const uint32_t* pb) -> float {
        float s = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            uint32_t wa = pa[KS * 8 + w];
            uint32_t wb = pb[KS * 8 + w];
            s += e4m3_byte_to_f32(wa & 0xFFu) * e4m3_byte_to_f32(wb & 0xFFu);
            s += e4m3_byte_to_f32((wa >> 8) & 0xFFu) * e4m3_byte_to_f32((wb >> 8) & 0xFFu);
            s += e4m3_byte_to_f32((wa >> 16) & 0xFFu) * e4m3_byte_to_f32((wb >> 16) & 0xFFu);
            s += e4m3_byte_to_f32((wa >> 24) & 0xFFu) * e4m3_byte_to_f32((wb >> 24) & 0xFFu);
        }
        return s;
    };
    c[0] += dot32_e4m3(smem_a_row0, smem_b_col0) * (sfa0_f * sfb0_f);
    c[1] += dot32_e4m3(smem_a_row0, smem_b_col1) * (sfa0_f * sfb1_f);
    c[2] += dot32_e4m3(smem_a_row1, smem_b_col0) * (sfa1_f * sfb0_f);
    c[3] += dot32_e4m3(smem_a_row1, smem_b_col1) * (sfa1_f * sfb1_f);
#endif
}

} // namespace leansonic::inkling2

// 18.1 `0.0000-ULP` 2D Token-Parallel Causal 4-Tap FIR `short_conv4` (`Grid: (ceil(ch/256), ceil(B/32))`, `Block: 256`):
// Eliminates the `40.3 ms` 9-CTA serial `b = 0..B-1` bottleneck in wide prefill while keeping `hist3` read-only (`0` WAR race)!
extern "C" __global__ void ink2_short_conv4_save_par_b(
    const float* __restrict__ cur_base,      // [B, in_stride]
    int32_t in_stride,
    const float4* __restrict__ w_conv4,      // [channels]
    const float4* __restrict__ dyn_base,     // nullable (`[B, dyn_f4_stride]`)
    int32_t dyn_f4_stride,
    const float* __restrict__ hist3,         // [channels * 3] (strictly read-only during `par_b`!)
    float* out_base,                         // [B, out_stride]
    int32_t out_stride,
    const float* res_base,                   // nullable (`[B, res_stride]`)
    int32_t res_stride,
    float* __restrict__ save_blk_base,       // nullable (`[B, channels]`)
    int32_t channels,
    int32_t B,
    int32_t has_dyn,
    int32_t do_res_add) {
    pdl_wait();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    const int b_start = blockIdx.y * 32;
    if (b_start >= B) return;
    const int b_end = min(b_start + 32, B);

    float h0, h1, h2;
    if (b_start == 0) {
        const float* h = hist3 + c * 3;
        h0 = h[0];
        h1 = h[1];
        h2 = h[2];
    } else {
        h0 = cur_base[static_cast<size_t>(b_start - 3) * in_stride + c];
        h1 = cur_base[static_cast<size_t>(b_start - 2) * in_stride + c];
        h2 = cur_base[static_cast<size_t>(b_start - 1) * in_stride + c];
    }

    const float4 w = w_conv4[c];
    const int head = c >> 7;

    for (int b = b_start; b < b_end; ++b) {
        const float x = cur_base[static_cast<size_t>(b) * in_stride + c];
        if (save_blk_base != nullptr) {
            save_blk_base[static_cast<size_t>(b) * channels + c] =
                out_base[static_cast<size_t>(b) * out_stride + c];
        }
        const float4 d = has_dyn
            ? dyn_base[static_cast<size_t>(b) * dyn_f4_stride + head]
            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float res = do_res_add ? res_base[static_cast<size_t>(b) * res_stride + c] : 0.0f;
        const float y = eval_conv4_step(w, h0, h1, h2, x, has_dyn, d, do_res_add, res);
        h0 = h1;
        h1 = h2;
        h2 = x;
        out_base[static_cast<size_t>(b) * out_stride + c] = y;
    }
}

// Commits the trailing `3` samples `cur_base[B-3..B-1]` into `hist3` after `ink2_short_conv4_save_par_b` (`B >= 3`).
extern "C" __global__ void ink2_short_conv4_commit_hist_b(
    const float* __restrict__ cur_base,
    int32_t in_stride,
    float* __restrict__ hist3,
    int32_t channels,
    int32_t B) {
    pdl_wait();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float* h = hist3 + c * 3;
    h[0] = cur_base[static_cast<size_t>(B - 3) * in_stride + c];
    h[1] = cur_base[static_cast<size_t>(B - 2) * in_stride + c];
    h[2] = cur_base[static_cast<size_t>(B - 1) * in_stride + c];
}

// 18.2 Triple-Buffered (`3`-Stage, `64,512` B = exact `64.0 KiB` w/ sys -> `64 KiB` L1!) `ldmatrix.x4` `128x128` 16-Warp Planar `Q8_64` `IMMA.16832` GEMM
struct Q8TcSmem3Stage {
    uint4 smem_w[3][128][5]; // 30,720 B (stride = 20 dwords -> 32-bank bijection!)
    uint4 smem_x[3][128][5]; // 30,720 B (stride = 20 dwords -> 32-bank bijection!)
    float smem_wsc[3][128];  //  1,536 B
    float smem_xsc[3][128];  //  1,536 B
}; // Total = 64,512 B + 1,024 B sys = 65,536 B (64.000 KiB -> full 64 KiB L1 preserved!)

__device__ __forceinline__ void q8_tc_gemm_tile128(
    const uint4* __restrict__ w_q8_u128,
    const float* __restrict__ w_sc64,
    const uint4* __restrict__ x_q8_u128,
    const float* __restrict__ x_sc64,
    float* __restrict__ y_base,
    int32_t r_cta,
    int32_t rows,
    int32_t cols,
    int32_t B,
    int32_t y_stride,
    float softcap) {
    extern __shared__ __align__(16) uint8_t dyn_q8_smem[];
    Q8TcSmem3Stage* s = reinterpret_cast<Q8TcSmem3Stage*>(dyn_q8_smem);

    const int t_cta = blockIdx.y * 128;
    const int tid = threadIdx.x; // 0..511
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int wr = warp_id >> 1; // 0..7 (16 rows each -> 128 rows!)
    const int wt = warp_id & 1;  // 0..1 (64 tokens each -> 128 tokens!)
    const int g = lane >> 2;     // 0..7
    const int t = lane & 3;      // 0..3

    const int cols16 = cols >> 4;
    const int cols64 = cols >> 6;
    const int lr = tid >> 2; // 0..127
    const int lq = tid & 3;  // 0..3
    const int gr = r_cta + lr;
    const int gt = t_cta + lr;

    auto issue_stage = [&](int st, int b64) {
        cp_async16_cg_pred(
            &s->smem_w[st][lr][lq],
            w_q8_u128 + static_cast<size_t>(gr) * cols16 + b64 * 4 + lq,
            gr < rows);
        cp_async16_cg_pred(
            &s->smem_x[st][lr][lq],
            x_q8_u128 + static_cast<size_t>(gt) * cols16 + b64 * 4 + lq,
            gt < B);
        if (tid < 128) {
            const int sr = r_cta + tid;
            cp_async4_ca_pred(&s->smem_wsc[st][tid], w_sc64 + static_cast<size_t>(sr) * cols64 + b64, sr < rows);
        } else if (tid < 256) {
            const int st_tok = t_cta + (tid - 128);
            cp_async4_ca_pred(&s->smem_xsc[st][tid - 128], x_sc64 + static_cast<size_t>(st_tok) * cols64 + b64, st_tok < B);
        }
        cp_async_commit_grp();
    };

    float acc[8][4] = {};
    issue_stage(0, 0);
    if (cols64 > 1) {
        issue_stage(1, 1);
    }

    int st = 0;
    int st_ld = 2;
    for (int b64 = 0; b64 < cols64; ++b64) {
        if (b64 + 2 < cols64) {
            issue_stage(st_ld, b64 + 2);
            st_ld = (st_ld == 2) ? 0 : (st_ld + 1);
            cp_async_wait_grp<2>();
        } else if (b64 + 1 < cols64) {
            cp_async_wait_grp<1>();
        } else {
            cp_async_wait_grp<0>();
        }
        __syncthreads();

        if (r_cta + wr * 16 < rows && t_cta + wt * 64 < B) {
            const int r_base = wr * 16;
            const uint8_t* w_ldm = reinterpret_cast<const uint8_t*>(s->smem_w[st][r_base + (lane & 15)]) + (lane >> 4) * 16;
            unsigned a0, a1, a2, a3, a0p, a1p, a2p, a3p;
            ldsm_x4_b16(a0, a1, a2, a3, w_ldm);
            ldsm_x4_b16(a0p, a1p, a2p, a3p, w_ldm + 32);
            const float ws0 = s->smem_wsc[st][r_base + g];
            const float ws1 = s->smem_wsc[st][r_base + 8 + g];

            #pragma unroll
            for (int nt = 0; nt < 8; nt += 2) {
                if (t_cta + wt * 64 + nt * 8 < B) {
                    const uint8_t* x_ldm = reinterpret_cast<const uint8_t*>(
                        s->smem_x[st][wt * 64 + nt * 8 + (lane & 7) + ((lane >> 4) << 3)]) + ((lane >> 3) & 1) * 16;
                    unsigned b00, b01, b10, b11;
                    unsigned b02, b03, b12, b13;
                    ldsm_x4_b16(b00, b01, b10, b11, x_ldm);
                    ldsm_x4_b16(b02, b03, b12, b13, x_ldm + 32);

                    const int tc0 = wt * 64 + nt * 8 + 2 * t;
                    const float xs00 = s->smem_xsc[st][tc0];
                    const float xs01 = s->smem_xsc[st][tc0 + 1];
                    int d0, d1, d2, d3;
                    mma_s8_m16n8k32(d0, d1, d2, d3, a0, a1, a2, a3, b00, b01, 0, 0, 0, 0);
                    mma_s8_m16n8k32(d0, d1, d2, d3, a0p, a1p, a2p, a3p, b02, b03, d0, d1, d2, d3);
                    acc[nt][0] = __fmaf_rn(static_cast<float>(d0), __fmul_rn(ws0, xs00), acc[nt][0]);
                    acc[nt][1] = __fmaf_rn(static_cast<float>(d1), __fmul_rn(ws0, xs01), acc[nt][1]);
                    acc[nt][2] = __fmaf_rn(static_cast<float>(d2), __fmul_rn(ws1, xs00), acc[nt][2]);
                    acc[nt][3] = __fmaf_rn(static_cast<float>(d3), __fmul_rn(ws1, xs01), acc[nt][3]);

                    if (t_cta + wt * 64 + (nt + 1) * 8 < B) {
                        const int tc1 = wt * 64 + (nt + 1) * 8 + 2 * t;
                        const float xs10 = s->smem_xsc[st][tc1];
                        const float xs11 = s->smem_xsc[st][tc1 + 1];
                        int e0, e1, e2, e3;
                        mma_s8_m16n8k32(e0, e1, e2, e3, a0, a1, a2, a3, b10, b11, 0, 0, 0, 0);
                        mma_s8_m16n8k32(e0, e1, e2, e3, a0p, a1p, a2p, a3p, b12, b13, e0, e1, e2, e3);
                        acc[nt + 1][0] = __fmaf_rn(static_cast<float>(e0), __fmul_rn(ws0, xs10), acc[nt + 1][0]);
                        acc[nt + 1][1] = __fmaf_rn(static_cast<float>(e1), __fmul_rn(ws0, xs11), acc[nt + 1][1]);
                        acc[nt + 1][2] = __fmaf_rn(static_cast<float>(e2), __fmul_rn(ws1, xs10), acc[nt + 1][2]);
                        acc[nt + 1][3] = __fmaf_rn(static_cast<float>(e3), __fmul_rn(ws1, xs11), acc[nt + 1][3]);
                    }
                }
            }
        }
        __syncthreads();
        st = (st == 2) ? 0 : (st + 1);
    }

    const float inv_sc = (softcap > 0.0f) ? ptx_rcp(softcap) : 0.0f;
    const int out_r0 = r_cta + wr * 16 + g;
    const int out_r1 = out_r0 + 8;
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int out_t = t_cta + wt * 64 + nt * 8 + 2 * t + c;
            if (out_t < B) {
                float v0 = acc[nt][c];
                float v1 = acc[nt][2 + c];
                if (softcap > 0.0f) {
                    v0 = softcap * ptx_tanh(v0 * inv_sc);
                    v1 = softcap * ptx_tanh(v1 * inv_sc);
                }
                float* dst = y_base + static_cast<size_t>(out_t) * y_stride;
                if (out_r0 < rows) dst[out_r0] = v0;
                if (out_r1 < rows) dst[out_r1] = v1;
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(512, 1) void ink2_q8_tc_gemm_b(
    const uint4* __restrict__ w_q8_u128,
    const float* __restrict__ w_sc64,
    const uint4* __restrict__ x_q8_u128,
    const float* __restrict__ x_sc64,
    float* __restrict__ y_base,
    int32_t rows,
    int32_t cols,
    int32_t B,
    int32_t y_stride,
    float softcap) {
    pdl_wait();
    q8_tc_gemm_tile128(
        w_q8_u128, w_sc64, x_q8_u128, x_sc64, y_base,
        static_cast<int32_t>(blockIdx.x) * 128, rows, cols, B, y_stride, softcap);
}

// Fused 3-Projection (`qrgs` + `wk` + `wv`) Wave-Packed `Q8_64` Tensor Core GEMM (`Grid: (nb_q + 2*nb_kv, ceil(B/128))`, `Block: 512`)!
extern "C" __global__ __launch_bounds__(512, 1) void ink2_q8_tc_qkv3_gemm_b(
    const uint4* __restrict__ wq_u128,
    const float* __restrict__ wq_sc64,
    const uint4* __restrict__ wk_u128,
    const float* __restrict__ wk_sc64,
    const uint4* __restrict__ wv_u128,
    const float* __restrict__ wv_sc64,
    const uint4* __restrict__ x_q8_base,
    const float* __restrict__ x_sc_base,
    float* __restrict__ y_qrgs,
    float* __restrict__ y_k,
    float* __restrict__ y_v,
    int32_t rows_q,
    int32_t rows_kv,
    int32_t B) {
    pdl_wait();
    const int nb_q = (rows_q + 127) >> 7;
    const int nb_kv = (rows_kv + 127) >> 7;
    const int bx = static_cast<int>(blockIdx.x);
    const size_t x_slot_u128 = static_cast<size_t>(B) * (kD >> 4);
    const size_t x_slot_sc = static_cast<size_t>(B) * (kD >> 6);
    if (bx < nb_q) {
        q8_tc_gemm_tile128(wq_u128, wq_sc64, x_q8_base, x_sc_base, y_qrgs, bx * 128, rows_q, kD, B, 8600, 0.0f);
    } else if (bx < nb_q + nb_kv) {
        q8_tc_gemm_tile128(
            wk_u128, wk_sc64, x_q8_base + x_slot_u128, x_sc_base + x_slot_sc,
            y_k, (bx - nb_q) * 128, rows_kv, kD, B, rows_kv, 0.0f);
    } else {
        q8_tc_gemm_tile128(
            wv_u128, wv_sc64, x_q8_base + 2u * x_slot_u128, x_sc_base + 2u * x_slot_sc,
            y_v, (bx - nb_q - nb_kv) * 128, rows_kv, kD, B, rows_kv, 0.0f);
    }
}

// 18.3 Two-Term (`x_hi + x_lo`, 8-Bit Fidelity) Native SM120a `mxf4nvf4` (`m16n8k64`) `128x128` Grouped MoE
// `ink2_quant_nvfp4_2term_b` Grid: `(ceil(n_rows / 4), 1)`, Block: `128` (`4` warps, `1` warp per row)
extern "C" __global__ void ink2_quant_nvfp4_2term_b(
    const float4* __restrict__ x4_base,       // [n_rows, cols / 4]
    uint2* __restrict__ out_codes_u64,        // [n_rows, 2 * (cols / 16)] (`4` `uint2`s Term 0 + `4` `uint2`s Term 1 per `ks64`)
    uint2* __restrict__ out_sc_u64,           // [n_rows, cols / 64] (`.x` = Term 0 `u32` scale, `.y` = Term 1 `u32` scale)
    float* __restrict__ out_gs,               // [n_rows]
    int32_t n_rows,
    int32_t cols) {
    pdl_wait();
    const int r = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (r >= n_rows) return;
    const int lane = threadIdx.x & 31;
    const int nb = cols >> 4;   // 144 (D=2304) or 72 (FF=1152)
    const int n64 = cols >> 6;  // 36 or 18
    const float4* xr4 = x4_base + static_cast<size_t>(r) * (cols >> 2);

    float amax = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const float4* p = xr4 + b * 4;
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float4 v = p[q];
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        }
    }
    amax = warp_max(amax);
    const float g_tok = amax * (1.0f / 2688.0f);
    if (lane == 0) {
        out_gs[r] = g_tok;
    }
    const float inv_g = (amax > 0.0f) ? ptx_rcp(g_tok) : 0.0f;

    uint2* row_codes = out_codes_u64 + static_cast<size_t>(r) * (2 * nb);
    uint2* row_sc = out_sc_u64 + static_cast<size_t>(r) * n64;

    for (int base_b = 0; base_b < nb; base_b += 32) {
        const int b = base_b + lane;
        uint32_t sb0_u = 0u;
        uint32_t sb1_u = 0u;
        if (b < nb) {
            const float4* p = xr4 + b * 4;
            float x[16];
            float bm = 0.0f;
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float4 v = p[q];
                x[4 * q + 0] = v.x;
                x[4 * q + 1] = v.y;
                x[4 * q + 2] = v.z;
                x[4 * q + 3] = v.w;
                bm = fmaxf(bm, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
            }

            const uint8_t sb0 = cvt_f32_to_ue4m3_byte(bm * inv_g * (1.0f / 6.0f));
            const float sd0 = ue4m3_to_f32_exact(sb0) * g_tok;
            const float inv0 = (sd0 > 0.0f) ? ptx_rcp(sd0) : 0.0f;
            const uint2 w0 = pack_e2m1_subblock16(x, inv0);

            float m2 = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t b0 = (w0.x >> (8 * j)) & 0xFFu;
                const uint32_t b1 = (w0.y >> (8 * j)) & 0xFFu;
                x[j]      -= e2m1_code_to_f32(b0 & 0xFu) * sd0;
                x[j + 8]  -= e2m1_code_to_f32(b0 >> 4)   * sd0;
                x[j + 4]  -= e2m1_code_to_f32(b1 & 0xFu) * sd0;
                x[j + 12] -= e2m1_code_to_f32(b1 >> 4)   * sd0;
                m2 = fmaxf(m2, fmaxf(fmaxf(fabsf(x[j]), fabsf(x[j + 8])), fmaxf(fabsf(x[j + 4]), fabsf(x[j + 12]))));
            }

            const uint8_t sb1 = cvt_f32_to_ue4m3_byte(m2 * inv_g * (1.0f / 6.0f));
            const float sd1 = ue4m3_to_f32_exact(sb1) * g_tok;
            const float inv1 = (sd1 > 0.0f) ? ptx_rcp(sd1) : 0.0f;
            const uint2 w1 = pack_e2m1_subblock16(x, inv1);

            const int ks64 = b >> 2;
            const int sub4 = b & 3;
            row_codes[(2 * ks64) * 4 + sub4] = w0;
            row_codes[(2 * ks64 + 1) * 4 + sub4] = w1;
            sb0_u = sb0;
            sb1_u = sb1;
        }
        uint32_t p0 = sb0_u | (__shfl_xor_sync(0xFFFFFFFFu, sb0_u, 1) << 8);
        p0 |= __shfl_xor_sync(0xFFFFFFFFu, p0, 2) << 16;
        uint32_t p1 = sb1_u | (__shfl_xor_sync(0xFFFFFFFFu, sb1_u, 1) << 8);
        p1 |= __shfl_xor_sync(0xFFFFFFFFu, p1, 2) << 16;
        if (b < nb && (lane & 3) == 0) {
            row_sc[b >> 2] = make_uint2(p0, p1);
        }
    }
}

// 1-Pass Register-Cached 2-Term NVFP4 Activation Quantizer (`Grid: (n_rows, 1)`, `Block: 160` = `5` warps)
// Holds `1` sub-block (`16` floats) per thread (`tid < 144` for `D=2304`, `tid < 72` for `FF=1152`) in registers,
// cutting global memory reads by `50%` (`-94.4 MiB/visit`) with `0.0000-ULP` bitwise parity to `ink2_quant_nvfp4_2term_b`!
extern "C" __global__ void ink2_quant_nvfp4_2term_1p_b(
    const float4* __restrict__ x4_base,       // [n_rows, cols / 4]
    uint2* __restrict__ out_codes_u64,        // [n_rows, 2 * (cols / 16)]
    uint2* __restrict__ out_sc_u64,           // [n_rows, cols / 64]
    float* __restrict__ out_gs,               // [n_rows]
    int32_t n_rows,
    int32_t cols) {
    pdl_wait();
    __shared__ float sm_wmax[5];
    __shared__ float sm_gtok[2];

    const int r = blockIdx.x;
    if (r >= n_rows) return;
    const int tid = threadIdx.x; // 0..159
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int nb = cols >> 4;    // 144 or 72
    const int n64 = cols >> 6;   // 36 or 18
    const float4* xr4 = x4_base + static_cast<size_t>(r) * (cols >> 2);

    float x[16];
    float bm = 0.0f;
    if (tid < nb) {
        const float4* p = xr4 + tid * 4;
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float4 v = p[q];
            x[4 * q + 0] = v.x;
            x[4 * q + 1] = v.y;
            x[4 * q + 2] = v.z;
            x[4 * q + 3] = v.w;
            bm = fmaxf(bm, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        }
    }

    const float wmax = warp_max(bm);
    if (lane == 0) {
        sm_wmax[warp_id] = wmax;
    }
    __syncthreads();
    if (warp_id == 0) {
        float amax = (lane < 5) ? sm_wmax[lane] : 0.0f;
        amax = warp_max(amax);
        if (lane == 0) {
            const float g_tok = amax * (1.0f / 2688.0f);
            out_gs[r] = g_tok;
            sm_gtok[0] = g_tok;
            sm_gtok[1] = (amax > 0.0f) ? ptx_rcp(g_tok) : 0.0f;
        }
    }
    __syncthreads();

    const float g_tok = sm_gtok[0];
    const float inv_g = sm_gtok[1];
    uint32_t sb0_u = 0u;
    uint32_t sb1_u = 0u;
    if (tid < nb) {
        const uint8_t sb0 = cvt_f32_to_ue4m3_byte(bm * inv_g * (1.0f / 6.0f));
        const float sd0 = ue4m3_to_f32_exact(sb0) * g_tok;
        const float inv0 = (sd0 > 0.0f) ? ptx_rcp(sd0) : 0.0f;
        const uint2 w0 = pack_e2m1_subblock16(x, inv0);

        float m2 = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t b0 = (w0.x >> (8 * j)) & 0xFFu;
            const uint32_t b1 = (w0.y >> (8 * j)) & 0xFFu;
            x[j]      -= e2m1_code_to_f32(b0 & 0xFu) * sd0;
            x[j + 8]  -= e2m1_code_to_f32(b0 >> 4)   * sd0;
            x[j + 4]  -= e2m1_code_to_f32(b1 & 0xFu) * sd0;
            x[j + 12] -= e2m1_code_to_f32(b1 >> 4)   * sd0;
            m2 = fmaxf(m2, fmaxf(fmaxf(fabsf(x[j]), fabsf(x[j + 8])), fmaxf(fabsf(x[j + 4]), fabsf(x[j + 12]))));
        }

        const uint8_t sb1 = cvt_f32_to_ue4m3_byte(m2 * inv_g * (1.0f / 6.0f));
        const float sd1 = ue4m3_to_f32_exact(sb1) * g_tok;
        const float inv1 = (sd1 > 0.0f) ? ptx_rcp(sd1) : 0.0f;
        const uint2 w1 = pack_e2m1_subblock16(x, inv1);

        uint2* row_codes = out_codes_u64 + static_cast<size_t>(r) * (2 * nb);
        const int ks64 = tid >> 2;
        const int sub4 = tid & 3;
        row_codes[(2 * ks64) * 4 + sub4] = w0;
        row_codes[(2 * ks64 + 1) * 4 + sub4] = w1;
        sb0_u = sb0;
        sb1_u = sb1;
    }
    uint32_t p0 = sb0_u | (__shfl_xor_sync(0xFFFFFFFFu, sb0_u, 1) << 8);
    p0 |= __shfl_xor_sync(0xFFFFFFFFu, p0, 2) << 16;
    uint32_t p1 = sb1_u | (__shfl_xor_sync(0xFFFFFFFFu, sb1_u, 1) << 8);
    p1 |= __shfl_xor_sync(0xFFFFFFFFu, p1, 2) << 16;
    if (tid < nb && (lane & 3) == 0) {
        out_sc_u64[static_cast<size_t>(r) * n64 + (tid >> 2)] = make_uint2(p0, p1);
    }
}

// Pre-gather `[B, D]` 2-term NVFP4 activations ONCE into contiguous `sorted_base` order inside `pair_output`!
// Grid: `(1, max_tiles_128)`, Block: `256` (`TileTokens = 128` matches `128x128` `2` CTAs/SM MoE tiles!)
extern "C" __global__ void ink2_gather_nvfp4_sorted_b(
    const uint4* __restrict__ src_codes_u128, // [B, 144]
    const uint4* __restrict__ src_sc_u128,    // [B, 18] (`36` `uint2`s)
    const float* __restrict__ src_gs,         // [B]
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_toks,
    uint4* __restrict__ dst_codes_u128,       // [B * 8, 144]
    uint4* __restrict__ dst_sc_u128,          // [B * 8, 18]
    float* __restrict__ dst_gs) {             // [B * 8]
    pdl_wait();
    __shared__ int32_t sm_blk_off[129];
    __shared__ int32_t sm_row_off[129];
    __shared__ int32_t sm_toks[128];

    int expert = 0;
    int first = 0;
    int count = 0;
    size_t sorted_base = 0;
    if (!resolve_flat_expert_tile<128>(exp_counts, sm_blk_off, sm_row_off, expert, first, count, sorted_base)) {
        return;
    }
    const int tid = threadIdx.x;
    const int live = min(128, count - first);
    if (tid < live) {
        const int tok = exp_toks[expert_map_offset(expert, first + tid)];
        sm_toks[tid] = tok;
        dst_gs[sorted_base + static_cast<size_t>(tid)] = src_gs[tok];
    }
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    for (int t = warp_id; t < live; t += 8) {
        const size_t src = static_cast<size_t>(sm_toks[t]);
        const size_t dst = sorted_base + static_cast<size_t>(t);
        const uint4* sc_p = src_codes_u128 + src * 144u;
        uint4* dc_p = dst_codes_u128 + dst * 144u;
        #pragma unroll
        for (int i = lane; i < 144; i += 32) {
            dc_p[i] = sc_p[i];
        }
        if (lane < 18) {
            dst_sc_u128[dst * 18u + lane] = src_sc_u128[src * 18u + lane];
        }
    }
}

// Grid: `(9, max_tiles_128)`, Block: `512` (`256 weight rows x 128 tokens`, 3-Stage `K_STAGE=64`, XOR-Swizzled `57,352` B SMEM!)
// Halves `sorted_x` activation reads (`18x -> 9x`) while running `16` warps/SM AND keeping SMEM `<= 64 KiB` (`64 KiB` full L1 preserved)!
struct Nvfp4TcSmem256x128_3Stage {
    uint4 sm_w[3][256][2];     // 24,576 B (XOR swizzled `wq2 ^ ((r >> 2) & 1)` -> bank-conflict-free!)
    uint4 sm_x[3][128][4];     // 24,576 B (XOR swizzled `xq4 ^ ((xr >> 1) & 3)` -> 32-bank bijection!)
    uint32_t sm_wsc[8][256];   //  8,192 B (ping-pong 2x4-stage banks -> 512-thread 4-stage batch unpack, 0 race!)
    uint2 sm_xsc[3][128];      //  3,072 B
    float sm_xgs[128];         //    512 B
    int32_t sm_out_pairs[128]; //    512 B
    int32_t sm_blk_off[129];   //    516 B
    int32_t sm_row_off[129];   //    516 B
}; // Total = 62,472 B (`+ 32 + 1,024 = 63,528 B <= 64 KiB` -> `16` warps/SM + `64 KiB` full L1!)

template <bool GateUp, bool PackedScales>
__device__ __forceinline__ void grouped_nvfp4_tc_impl(
    const uint4* __restrict__ weight_data,
    const uint32_t* __restrict__ weight_scales_u32,
    const float* __restrict__ amax,
    const uint4* __restrict__ x_codes_u128,
    const uint2* __restrict__ x_sc_u64,
    const float* __restrict__ x_gs,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_pairs,
    int32_t exp_begin,
    int32_t exp_end,
    void* __restrict__ output_raw) {
    constexpr int kCols = GateUp ? kD : kFf;        // 2304 or 1152
    constexpr int kStages64 = kCols / 64;           // 36 or 18
    constexpr int kWRowU128 = kCols / 32;           // 72 or 36
    constexpr int kWScRowU32 = kCols / 64;          // 36 or 18 (`uint32_t`s)
    constexpr int kXRowU128 = 2 * (kCols / 32);     // 144 or 72
    constexpr int kXScRowU64 = kCols / 64;          // 36 or 18 (`uint2`s)

    extern __shared__ __align__(16) uint8_t dyn_nvfp4_smem[];
    Nvfp4TcSmem256x128_3Stage* s = reinterpret_cast<Nvfp4TcSmem256x128_3Stage*>(dyn_nvfp4_smem);

    int expert = 0;
    int first = 0;
    int count = 0;
    size_t sorted_base = 0;
    if (!resolve_flat_expert_tile<128>(exp_counts, s->sm_blk_off, s->sm_row_off, expert, first, count, sorted_base)) {
        return;
    }
    if (expert < exp_begin || expert >= exp_end) {
        return;
    }

    const int tid = threadIdx.x; // 0..511 (`16` warps/CTA!)
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int wr = warp_id >> 1; // 0..7 (`8` row-warps x `32` rows = `256` weight rows!)
    const int wt = warp_id & 1;  // 0..1 (`2` token-warps x `64` tokens = `128` tokens!)
    const int g = lane >> 2;     // 0..7
    const int t = lane & 3;      // 0..3
    const int sfa_row = ((lane & 1) << 3) | (lane >> 2);
    const int sub01 = (t >> 1) * 2;
    const uint32_t prmt_sel = (t & 1) ? 0x7362u : 0x5140u;

    auto map_w_row = [&](int sr) -> int {
        if constexpr (GateUp) {
            const int m_local = ((sr >> 4) << 3) | (sr & 7); // 0..127
            const int is_up = (sr >> 3) & 1;
            return blockIdx.x * 256 + 2 * m_local + is_up;
        } else {
            return blockIdx.x * 256 + sr;
        }
    };

    const int local_exp = expert - exp_begin;
    const size_t exp_base_row = static_cast<size_t>(local_exp) * (GateUp ? 2 * kFf : kD);
    const size_t full_exp_base_row = static_cast<size_t>(expert) * (GateUp ? 2 * kFf : kD);
    const int wr_r = tid >> 1;  // 0..255 (`2` lanes x `16B` = `32B` across all `512` threads!)
    const int wq2 = tid & 1;    // 0..1
    const size_t grow_wr = exp_base_row + map_w_row(wr_r);
    const size_t grow_sc_wr = full_exp_base_row + map_w_row(wr_r);
    const size_t grow_sc_tid256 = full_exp_base_row + map_w_row(tid & 255);
    const float expert_scale = amax[expert] * (1.0f / 2688.0f);

    const int live = min(128, count - first);
    if (tid < 128) {
        const int clamped = min(tid, live - 1);
        const size_t seq_idx = sorted_base + static_cast<size_t>(clamped);
        s->sm_xgs[tid] = (tid < live) ? x_gs[seq_idx] : 0.0f;
        if constexpr (!GateUp) {
            s->sm_out_pairs[tid] = exp_pairs[expert_map_offset(expert, first + clamped)];
        }
    }
    __syncthreads();

    const int xr = tid >> 2;  // 0..127 (`512` threads load ALL `128 x 64B` in `1` single pass!)
    const int xq4 = tid & 3;  // 0..3
    const size_t src_tok_xr = sorted_base + static_cast<size_t>(min(xr, live - 1));
    const int xid = tid - 256;
    const size_t seq_tok_xid = sorted_base + static_cast<size_t>(min(max(0, xid), live - 1));

    auto issue_stage = [&](int st, int ks64) {
        cp_async16_cg(
            &s->sm_w[st][wr_r][wq2 ^ ((wr_r >> 2) & 1)],
            weight_data + grow_wr * kWRowU128 + ks64 * 2 + wq2);
        cp_async16_cg_pred(
            &s->sm_x[st][xr][xq4 ^ ((xr >> 1) & 3)],
            x_codes_u128 + src_tok_xr * kXRowU128 + ks64 * 4 + xq4,
            xr < live);
        if (tid < 256) {
            if constexpr (!PackedScales) {
                cp_async4_ca_pred(&s->sm_wsc[st][tid], weight_scales_u32 + grow_sc_tid256 * kWScRowU32 + ks64, true);
            }
        } else if (tid < 384) {
            cp_async8_ca_pred(&s->sm_xsc[st][xid], x_sc_u64 + seq_tok_xid * kXScRowU64 + ks64, xid < live);
        }
        cp_async_commit_grp();
    };

    float acc[2][8][4] = {};
    // Packed-scale producers already read through registers. Sanitize once before
    // sharing each word. Raw-scale producers copy directly to shared memory.
    auto finite_packed_scale = [&](unsigned index) {
        return sanitize_ue4m3x4(expert_scale_word<true, 6, 64>(weight_scales_u32, index));
    };
    auto shared_scale = [&](int stage, int row) {
        const uint32_t word = s->sm_wsc[stage][row];
        if constexpr (PackedScales) return word;
        return sanitize_ue4m3x4(word);
    };

    issue_stage(0, 0);
    issue_stage(1, 1);
    if constexpr (PackedScales) {
        const int k_init = wq2 * 2;
        s->sm_wsc[k_init][wr_r] = finite_packed_scale(static_cast<unsigned>(grow_sc_wr * kWScRowU32 + k_init));
        s->sm_wsc[k_init + 1][wr_r] = finite_packed_scale(static_cast<unsigned>(grow_sc_wr * kWScRowU32 + k_init + 1));
    }

    int st = 0;
    int st_ld = 2;
    for (int ks64 = 0; ks64 < kStages64; ++ks64) {
        if (ks64 + 2 < kStages64) {
            issue_stage(st_ld, ks64 + 2);
            st_ld = (st_ld == 2) ? 0 : (st_ld + 1);
            cp_async_wait_grp<2>();
        } else if (ks64 + 1 < kStages64) {
            cp_async_wait_grp<1>();
        } else {
            cp_async_wait_grp<0>();
        }
        __syncthreads();

        uint32_t next_wsc0 = 0u;
        uint32_t next_wsc1 = 0u;
        if constexpr (PackedScales) {
            if (((ks64 & 3) == 3) && (ks64 + 1 < kStages64)) {
                const int k_next = ks64 + 1 + wq2 * 2;
                if (k_next < kStages64) {
                    next_wsc0 = finite_packed_scale(static_cast<unsigned>(grow_sc_wr * kWScRowU32 + k_next));
                }
                if (k_next + 1 < kStages64) {
                    next_wsc1 = finite_packed_scale(static_cast<unsigned>(grow_sc_wr * kWScRowU32 + k_next + 1));
                }
            }
        }

        const int wsc_st = PackedScales ? (ks64 & 7) : st;
        if (wt * 64 < live) {
            unsigned wa[2][4];
            unsigned wsfa[2];
            const uint32_t* w0_lo_sw[2];
            const uint32_t* w0_hi_sw[2];
            const uint32_t* w1_lo_sw[2];
            const uint32_t* w1_hi_sw[2];
            uint32_t sfa_r0_sw[2];
            uint32_t sfa_r1_sw[2];

            #pragma unroll
            for (int rm = 0; rm < 2; ++rm) {
                const int r0 = wr * 32 + rm * 16 + g;
                const int r1 = r0 + 8;
                const uint32_t* w0_lo = reinterpret_cast<const uint32_t*>(&s->sm_w[st][r0][0 ^ ((r0 >> 2) & 1)]);
                const uint32_t* w0_hi = reinterpret_cast<const uint32_t*>(&s->sm_w[st][r0][1 ^ ((r0 >> 2) & 1)]);
                const uint32_t* w1_lo = reinterpret_cast<const uint32_t*>(&s->sm_w[st][r1][0 ^ ((r1 >> 2) & 1)]);
                const uint32_t* w1_hi = reinterpret_cast<const uint32_t*>(&s->sm_w[st][r1][1 ^ ((r1 >> 2) & 1)]);
                wa[rm][0] = interleave_nvfp4_sub_sel(w0_lo[sub01], w0_lo[sub01 + 1], prmt_sel);
                wa[rm][1] = interleave_nvfp4_sub_sel(w1_lo[sub01], w1_lo[sub01 + 1], prmt_sel);
                wa[rm][2] = interleave_nvfp4_sub_sel(w0_hi[sub01], w0_hi[sub01 + 1], prmt_sel);
                wa[rm][3] = interleave_nvfp4_sub_sel(w1_hi[sub01], w1_hi[sub01 + 1], prmt_sel);
                wsfa[rm] = shared_scale(wsc_st, wr * 32 + rm * 16 + sfa_row);
                w0_lo_sw[rm] = w0_lo;
                w0_hi_sw[rm] = w0_hi;
                w1_lo_sw[rm] = w1_lo;
                w1_hi_sw[rm] = w1_hi;
                sfa_r0_sw[rm] = shared_scale(wsc_st, r0);
                sfa_r1_sw[rm] = shared_scale(wsc_st, r1);
            }

            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                if (wt * 64 + nt * 8 < live) {
                    const int tok_row = wt * 64 + nt * 8 + g;
                    const int tc0 = wt * 64 + nt * 8 + 2 * t;
                    const int tc1 = tc0 + 1;
                    const uint2 xsc2 = s->sm_xsc[st][tok_row];
                    const uint2 xsc_c0 = s->sm_xsc[st][tc0];
                    const uint2 xsc_c1 = s->sm_xsc[st][tc1];

                    #pragma unroll
                    for (int u = 0; u < 2; ++u) {
                        const uint32_t* x_lo = reinterpret_cast<const uint32_t*>(&s->sm_x[st][tok_row][(2 * u) ^ ((tok_row >> 1) & 3)]);
                        const uint32_t* x_hi = reinterpret_cast<const uint32_t*>(&s->sm_x[st][tok_row][(2 * u + 1) ^ ((tok_row >> 1) & 3)]);
                        const unsigned b0 = x_lo[t];
                        const unsigned b1 = x_hi[t];
                        const unsigned sfb = (u == 0) ? xsc2.x : xsc2.y;
                        const uint32_t sfb_c0 = (u == 0) ? xsc_c0.x : xsc_c0.y;
                        const uint32_t sfb_c1 = (u == 0) ? xsc_c1.x : xsc_c1.y;
                        const uint32_t* xb_c0_lo = reinterpret_cast<const uint32_t*>(&s->sm_x[st][tc0][(2 * u) ^ ((tc0 >> 1) & 3)]);
                        const uint32_t* xb_c0_hi = reinterpret_cast<const uint32_t*>(&s->sm_x[st][tc0][(2 * u + 1) ^ ((tc0 >> 1) & 3)]);
                        const uint32_t* xb_c1_lo = reinterpret_cast<const uint32_t*>(&s->sm_x[st][tc1][(2 * u) ^ ((tc1 >> 1) & 3)]);
                        const uint32_t* xb_c1_hi = reinterpret_cast<const uint32_t*>(&s->sm_x[st][tc1][(2 * u + 1) ^ ((tc1 >> 1) & 3)]);

                        #pragma unroll
                        for (int rm = 0; rm < 2; ++rm) {
                            mma_nvfp4_m16n8k64(
                                acc[rm][nt], wa[rm][0], wa[rm][1], wa[rm][2], wa[rm][3],
                                b0, b1, wsfa[rm], sfb,
                                w0_lo_sw[rm], w0_hi_sw[rm], w1_lo_sw[rm], w1_hi_sw[rm],
                                xb_c0_lo, xb_c0_hi, xb_c1_lo, xb_c1_hi,
                                sfa_r0_sw[rm], sfa_r1_sw[rm], sfb_c0, sfb_c1);
                        }
                    }
                }
            }
        }
        const int st_next = (st == 2) ? 0 : (st + 1);
        if constexpr (PackedScales) {
            if (((ks64 & 3) == 3) && (ks64 + 1 < kStages64)) {
                const int next_bank = ((ks64 + 1) & 4) + wq2 * 2;
                s->sm_wsc[next_bank][wr_r] = next_wsc0;
                s->sm_wsc[next_bank + 1][wr_r] = next_wsc1;
            }
        }
        __syncthreads();
        st = st_next;
    }

    if (wt * 64 < live) {
        #pragma unroll
        for (int rm = 0; rm < 2; ++rm) {
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                #pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int tok_local = wt * 64 + nt * 8 + 2 * t + c;
                    const float tot_scale = (tok_local < live) ? (expert_scale * s->sm_xgs[tok_local]) : 0.0f;
                    if constexpr (GateUp) {
                        if (tok_local < live) {
                            float* output = static_cast<float*>(output_raw);
                            const size_t dst_seq = sorted_base + static_cast<size_t>(tok_local);
                            const float gv = acc[rm][nt][c] * tot_scale;
                            const float uv = acc[rm][nt][2 + c] * tot_scale;
                            const float situ =
                                4.0f * ptx_tanh(gv * 0.25f) * ptx_sigmoid(gv) * 25.0f * ptx_tanh(uv * 0.04f);
                            const int m = blockIdx.x * 128 + wr * 16 + rm * 8 + g;
                            output[dst_seq * kFf + m] = situ;
                        }
                    } else {
                        const float v0 = acc[rm][nt][c] * tot_scale;
                        const float v1 = acc[rm][nt][2 + c] * tot_scale;
                        const float v0_odd = __shfl_xor_sync(0xFFFFFFFFu, v0, 4);
                        const float v1_odd = __shfl_xor_sync(0xFFFFFFFFu, v1, 4);
                        if (tok_local < live && (g & 1) == 0) {
                            uint32_t* out_f16x2 = static_cast<uint32_t*>(output_raw);
                            const size_t dst_pair = static_cast<size_t>(s->sm_out_pairs[tok_local]);
                            const int r0 = blockIdx.x * 256 + wr * 32 + rm * 16 + g;
                            out_f16x2[(dst_pair * kD + r0) >> 1] = pack_f16x2_rn(v0, v0_odd);
                            out_f16x2[(dst_pair * kD + r0 + 8) >> 1] = pack_f16x2_rn(v1, v1_odd);
                        }
                    }
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(512, 1) void ink2_moe_nvfp4_tc_up_situ(
    const uint4* __restrict__ w13_u128,
    const uint32_t* __restrict__ s13_u32,
    const float* __restrict__ a13,
    const uint4* __restrict__ sorted_x_codes_u128,
    const uint2* __restrict__ sorted_x_sc_u64,
    const float* __restrict__ sorted_x_gs,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_toks,
    const int32_t* __restrict__ exp_pairs,
    int32_t exp_begin,
    int32_t exp_end,
    float* __restrict__ act_out) {
    pdl_wait();
    (void)exp_toks;
    grouped_nvfp4_tc_impl<true, false>(
        w13_u128, s13_u32, a13, sorted_x_codes_u128, sorted_x_sc_u64, sorted_x_gs,
        exp_counts, exp_pairs, exp_begin, exp_end, act_out);
}

extern "C" __global__ __launch_bounds__(512, 1) void ink2_moe_nvfp4_tc_up_situ_packed_scales(
    const uint4* __restrict__ w13_u128,
    const uint32_t* __restrict__ s13_u32,
    const float* __restrict__ a13,
    const uint4* __restrict__ sorted_x_codes_u128,
    const uint2* __restrict__ sorted_x_sc_u64,
    const float* __restrict__ sorted_x_gs,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_toks,
    const int32_t* __restrict__ exp_pairs,
    int32_t exp_begin,
    int32_t exp_end,
    float* __restrict__ act_out) {
    pdl_wait();
    (void)exp_toks;
    grouped_nvfp4_tc_impl<true, true>(
        w13_u128, s13_u32, a13, sorted_x_codes_u128, sorted_x_sc_u64, sorted_x_gs,
        exp_counts, exp_pairs, exp_begin, exp_end, act_out);
}

extern "C" __global__ __launch_bounds__(512, 1) void ink2_moe_nvfp4_tc_down(
    const uint4* __restrict__ w2_u128,
    const uint32_t* __restrict__ s2_u32,
    const float* __restrict__ a2,
    const uint4* __restrict__ act_codes_u128,
    const uint2* __restrict__ act_sc_u64,
    const float* __restrict__ act_gs,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_pairs,
    int32_t exp_begin,
    int32_t exp_end,
    void* __restrict__ pair_y_f16) {
    pdl_wait();
    grouped_nvfp4_tc_impl<false, false>(
        w2_u128, s2_u32, a2, act_codes_u128, act_sc_u64, act_gs,
        exp_counts, exp_pairs, exp_begin, exp_end, pair_y_f16);
}

extern "C" __global__ __launch_bounds__(512, 1) void ink2_moe_nvfp4_tc_down_packed_scales(
    const uint4* __restrict__ w2_u128,
    const uint32_t* __restrict__ s2_u32,
    const float* __restrict__ a2,
    const uint4* __restrict__ act_codes_u128,
    const uint2* __restrict__ act_sc_u64,
    const float* __restrict__ act_gs,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_pairs,
    int32_t exp_begin,
    int32_t exp_end,
    void* __restrict__ pair_y_f16) {
    pdl_wait();
    grouped_nvfp4_tc_impl<false, true>(
        w2_u128, s2_u32, a2, act_codes_u128, act_sc_u64, act_gs,
        exp_counts, exp_pairs, exp_begin, exp_end, pair_y_f16);
}

// V13 (`#79`): `TileTokens = 160` (`192 weight rows x 160 tokens`, 3-Stage `57,608` B `<= 64 KiB` SMEM, `Block = 384`!)
// Eliminates `TileTokens = 128` boundary splitting at `B = 2048` (`mean = 128, std = 11` -> `99.6%` of experts fit in `1` tile!)
// while preserving full 3-Stage `cp.async`, `64 KiB` L1 cache, and active `wt = 0, 1` warps!
struct Nvfp4TcSmem192x160_3Stage {
    uint4 sm_w[3][192][2];     // 18,432 B
    uint4 sm_x[3][160][4];     // 30,720 B
    uint2 sm_xsc[3][160];      //  3,840 B
    uint32_t sm_wsc[3][192];   //  2,304 B
    float sm_xgs[160];         //    640 B
    int32_t sm_out_pairs[160]; //    640 B
    int32_t sm_blk_off[129];   //    516 B
    int32_t sm_row_off[129];   //    516 B
}; // Total = 57,608 B (`+ 32 B + 1,024 B sys <= 64 KiB` carveout -> `64 KiB` full L1 preserved!)

template <bool GateUp, bool PackedScales>
__device__ __forceinline__ void grouped_nvfp4_tc_t160_impl(
    const uint4* __restrict__ weight_data,
    const uint32_t* __restrict__ weight_scales_u32,
    const float* __restrict__ amax,
    const uint4* __restrict__ x_codes_u128,
    const uint2* __restrict__ x_sc_u64,
    const float* __restrict__ x_gs,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_pairs,
    int32_t exp_begin,
    int32_t exp_end,
    void* __restrict__ output_raw) {
    constexpr int kCols = GateUp ? kD : kFf;        // 2304 or 1152
    constexpr int kStages64 = kCols / 64;           // 36 or 18
    constexpr int kWRowU128 = kCols / 32;           // 72 or 36
    constexpr int kWScRowU32 = kCols / 64;          // 36 or 18
    constexpr int kXRowU128 = 2 * (kCols / 32);     // 144 or 72
    constexpr int kXScRowU64 = kCols / 64;          // 36 or 18

    extern __shared__ __align__(16) uint8_t dyn_nvfp4_t160_smem[];
    Nvfp4TcSmem192x160_3Stage* s = reinterpret_cast<Nvfp4TcSmem192x160_3Stage*>(dyn_nvfp4_t160_smem);

    int expert = 0;
    int first = 0;
    int count = 0;
    size_t sorted_base = 0;
    if (!resolve_flat_expert_tile<160>(exp_counts, s->sm_blk_off, s->sm_row_off, expert, first, count, sorted_base)) {
        return;
    }
    if (expert < exp_begin || expert >= exp_end) {
        return;
    }

    const int tid = threadIdx.x; // 0..383 (`12` warps/CTA!)
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int wr = warp_id >> 1; // 0..5 (`6` row-warps x `32` rows = `192` weight rows!)
    const int wt = warp_id & 1;  // 0..1 (`2` token-warps x `80` tokens = `160` tokens!)
    const int g = lane >> 2;     // 0..7
    const int t = lane & 3;      // 0..3
    const int sfa_row = ((lane & 1) << 3) | (lane >> 2);
    const int sub01 = (t >> 1) * 2;
    const uint32_t prmt_sel = (t & 1) ? 0x7362u : 0x5140u;

    auto map_w_row = [&](int sr) -> int {
        if constexpr (GateUp) {
            const int m_local = ((sr >> 4) << 3) | (sr & 7); // 0..95
            const int is_up = (sr >> 3) & 1;
            return blockIdx.x * 192 + 2 * m_local + is_up;
        } else {
            return blockIdx.x * 192 + sr;
        }
    };

    const int local_exp = expert - exp_begin;
    const size_t exp_base_row = static_cast<size_t>(local_exp) * (GateUp ? 2 * kFf : kD);
    const size_t full_exp_base_row = static_cast<size_t>(expert) * (GateUp ? 2 * kFf : kD);
    const int wr_r = tid >> 1; // 0..191 (`384` threads load `192 x 32B` in `1` single pass!)
    const int wq2 = tid & 1;   // 0..1
    const size_t grow_wr = exp_base_row + map_w_row(wr_r);
    const size_t grow_sc_tid192 = full_exp_base_row + map_w_row(tid < 192 ? tid : 0);
    const float expert_scale = amax[expert] * (1.0f / 2688.0f);

    const int live = min(160, count - first);
    if (tid < 160) {
        const int clamped = min(tid, live - 1);
        const size_t seq_idx = sorted_base + static_cast<size_t>(clamped);
        s->sm_xgs[tid] = (tid < live) ? x_gs[seq_idx] : 0.0f;
        if constexpr (!GateUp) {
            s->sm_out_pairs[tid] = exp_pairs[expert_map_offset(expert, first + clamped)];
        }
    }
    __syncthreads();

    const int xid = tid - 192; // 0..159 for `192 <= tid < 352`
    const size_t seq_tok_xid = sorted_base + static_cast<size_t>(min(max(0, xid), live - 1));

    auto issue_stage = [&](int st, int ks64) {
        cp_async16_cg(
            &s->sm_w[st][wr_r][wq2 ^ ((wr_r >> 2) & 1)],
            weight_data + grow_wr * kWRowU128 + ks64 * 2 + wq2);
        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int lin = p * 384 + tid; // 0..639 (`160` tokens x `4` uint4s)
            if (lin < 640) {
                const int xr = lin >> 2;
                const int xq4 = lin & 3;
                const size_t src_tok_xr = sorted_base + static_cast<size_t>(min(xr, live - 1));
                cp_async16_cg_pred(
                    &s->sm_x[st][xr][xq4 ^ ((xr >> 1) & 3)],
                    x_codes_u128 + src_tok_xr * kXRowU128 + ks64 * 4 + xq4,
                    xr < live);
            }
        }
        if (tid < 192) {
            if constexpr (!PackedScales) {
                cp_async4_ca_pred(&s->sm_wsc[st][tid], weight_scales_u32 + grow_sc_tid192 * kWScRowU32 + ks64, true);
            }
        } else if (tid < 352) {
            cp_async8_ca_pred(&s->sm_xsc[st][xid], x_sc_u64 + seq_tok_xid * kXScRowU64 + ks64, xid < live);
        }
        cp_async_commit_grp();
    };

    float acc[2][10][4] = {};
    issue_stage(0, 0);
    issue_stage(1, 1);
    if constexpr (PackedScales) {
        if (tid < 192) {
            s->sm_wsc[0][tid] = expert_scale_word<true, 6, 64>(
                weight_scales_u32,
                static_cast<unsigned>(grow_sc_tid192 * kWScRowU32 + 0));
        }
    }

    int st = 0;
    int st_ld = 2;
    for (int ks64 = 0; ks64 < kStages64; ++ks64) {
        if (ks64 + 2 < kStages64) {
            issue_stage(st_ld, ks64 + 2);
            st_ld = (st_ld == 2) ? 0 : (st_ld + 1);
            cp_async_wait_grp<2>();
        } else if (ks64 + 1 < kStages64) {
            cp_async_wait_grp<1>();
        } else {
            cp_async_wait_grp<0>();
        }
        __syncthreads();

        uint32_t next_wsc = 0u;
        if constexpr (PackedScales) {
            if ((ks64 + 1 < kStages64) && tid < 192) {
                next_wsc = expert_scale_word<true, 6, 64>(
                    weight_scales_u32,
                    static_cast<unsigned>(grow_sc_tid192 * kWScRowU32 + (ks64 + 1)));
            }
        }

        if (wt * 80 < live) {
            unsigned wa[2][4];
            unsigned wsfa[2];
            const uint32_t* w0_lo_sw[2];
            const uint32_t* w0_hi_sw[2];
            const uint32_t* w1_lo_sw[2];
            const uint32_t* w1_hi_sw[2];
            uint32_t sfa_r0_sw[2];
            uint32_t sfa_r1_sw[2];

            #pragma unroll
            for (int rm = 0; rm < 2; ++rm) {
                const int r0 = wr * 32 + rm * 16 + g;
                const int r1 = r0 + 8;
                const uint32_t* w0_lo = reinterpret_cast<const uint32_t*>(&s->sm_w[st][r0][0 ^ ((r0 >> 2) & 1)]);
                const uint32_t* w0_hi = reinterpret_cast<const uint32_t*>(&s->sm_w[st][r0][1 ^ ((r0 >> 2) & 1)]);
                const uint32_t* w1_lo = reinterpret_cast<const uint32_t*>(&s->sm_w[st][r1][0 ^ ((r1 >> 2) & 1)]);
                const uint32_t* w1_hi = reinterpret_cast<const uint32_t*>(&s->sm_w[st][r1][1 ^ ((r1 >> 2) & 1)]);
                wa[rm][0] = interleave_nvfp4_sub_sel(w0_lo[sub01], w0_lo[sub01 + 1], prmt_sel);
                wa[rm][1] = interleave_nvfp4_sub_sel(w1_lo[sub01], w1_lo[sub01 + 1], prmt_sel);
                wa[rm][2] = interleave_nvfp4_sub_sel(w0_hi[sub01], w0_hi[sub01 + 1], prmt_sel);
                wa[rm][3] = interleave_nvfp4_sub_sel(w1_hi[sub01], w1_hi[sub01 + 1], prmt_sel);
                wsfa[rm] = sanitize_ue4m3x4(s->sm_wsc[st][wr * 32 + rm * 16 + sfa_row]);
                w0_lo_sw[rm] = w0_lo;
                w0_hi_sw[rm] = w0_hi;
                w1_lo_sw[rm] = w1_lo;
                w1_hi_sw[rm] = w1_hi;
                sfa_r0_sw[rm] = sanitize_ue4m3x4(s->sm_wsc[st][r0]);
                sfa_r1_sw[rm] = sanitize_ue4m3x4(s->sm_wsc[st][r1]);
            }

            #pragma unroll
            for (int nt = 0; nt < 10; ++nt) {
                if (wt * 80 + nt * 8 < live) {
                    const int tok_row = wt * 80 + nt * 8 + g;
                    const int tc0 = wt * 80 + nt * 8 + 2 * t;
                    const int tc1 = tc0 + 1;
                    const uint2 xsc2 = s->sm_xsc[st][tok_row];
                    const uint2 xsc_c0 = s->sm_xsc[st][tc0];
                    const uint2 xsc_c1 = s->sm_xsc[st][tc1];

                    #pragma unroll
                    for (int u = 0; u < 2; ++u) {
                        const uint32_t* x_lo = reinterpret_cast<const uint32_t*>(&s->sm_x[st][tok_row][(2 * u) ^ ((tok_row >> 1) & 3)]);
                        const uint32_t* x_hi = reinterpret_cast<const uint32_t*>(&s->sm_x[st][tok_row][(2 * u + 1) ^ ((tok_row >> 1) & 3)]);
                        const unsigned b0 = x_lo[t];
                        const unsigned b1 = x_hi[t];
                        const unsigned sfb = (u == 0) ? xsc2.x : xsc2.y;
                        const uint32_t sfb_c0 = (u == 0) ? xsc_c0.x : xsc_c0.y;
                        const uint32_t sfb_c1 = (u == 0) ? xsc_c1.x : xsc_c1.y;
                        const uint32_t* xb_c0_lo = reinterpret_cast<const uint32_t*>(&s->sm_x[st][tc0][(2 * u) ^ ((tc0 >> 1) & 3)]);
                        const uint32_t* xb_c0_hi = reinterpret_cast<const uint32_t*>(&s->sm_x[st][tc0][(2 * u + 1) ^ ((tc0 >> 1) & 3)]);
                        const uint32_t* xb_c1_lo = reinterpret_cast<const uint32_t*>(&s->sm_x[st][tc1][(2 * u) ^ ((tc1 >> 1) & 3)]);
                        const uint32_t* xb_c1_hi = reinterpret_cast<const uint32_t*>(&s->sm_x[st][tc1][(2 * u + 1) ^ ((tc1 >> 1) & 3)]);

                        #pragma unroll
                        for (int rm = 0; rm < 2; ++rm) {
                            mma_nvfp4_m16n8k64(
                                acc[rm][nt], wa[rm][0], wa[rm][1], wa[rm][2], wa[rm][3],
                                b0, b1, wsfa[rm], sfb,
                                w0_lo_sw[rm], w0_hi_sw[rm], w1_lo_sw[rm], w1_hi_sw[rm],
                                xb_c0_lo, xb_c0_hi, xb_c1_lo, xb_c1_hi,
                                sfa_r0_sw[rm], sfa_r1_sw[rm], sfb_c0, sfb_c1);
                        }
                    }
                }
            }
        }

        const int st_next = (st == 2) ? 0 : (st + 1);
        if constexpr (PackedScales) {
            if ((ks64 + 1 < kStages64) && tid < 192) {
                s->sm_wsc[st_next][tid] = next_wsc;
            }
        }
        __syncthreads();
        st = st_next;
    }

    if (wt * 80 < live) {
        #pragma unroll
        for (int rm = 0; rm < 2; ++rm) {
            #pragma unroll
            for (int nt = 0; nt < 10; ++nt) {
                #pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int tok_local = wt * 80 + nt * 8 + 2 * t + c;
                    const float tot_scale = (tok_local < live) ? (expert_scale * s->sm_xgs[tok_local]) : 0.0f;
                    if constexpr (GateUp) {
                        if (tok_local < live) {
                            float* output = static_cast<float*>(output_raw);
                            const size_t dst_seq = sorted_base + static_cast<size_t>(tok_local);
                            const float gv = acc[rm][nt][c] * tot_scale;
                            const float uv = acc[rm][nt][2 + c] * tot_scale;
                            const float situ =
                                4.0f * ptx_tanh(gv * 0.25f) * ptx_sigmoid(gv) * 25.0f * ptx_tanh(uv * 0.04f);
                            const int m = blockIdx.x * 96 + wr * 16 + rm * 8 + g;
                            output[dst_seq * kFf + m] = situ;
                        }
                    } else {
                        const float v0 = acc[rm][nt][c] * tot_scale;
                        const float v1 = acc[rm][nt][2 + c] * tot_scale;
                        const float v0_odd = __shfl_xor_sync(0xFFFFFFFFu, v0, 4);
                        const float v1_odd = __shfl_xor_sync(0xFFFFFFFFu, v1, 4);
                        if (tok_local < live && (g & 1) == 0) {
                            uint32_t* out_f16x2 = static_cast<uint32_t*>(output_raw);
                            const size_t dst_pair = static_cast<size_t>(s->sm_out_pairs[tok_local]);
                            const int r0 = blockIdx.x * 192 + wr * 32 + rm * 16 + g;
                            out_f16x2[(dst_pair * kD + r0) >> 1] = pack_f16x2_rn(v0, v0_odd);
                            out_f16x2[(dst_pair * kD + r0 + 8) >> 1] = pack_f16x2_rn(v1, v1_odd);
                        }
                    }
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(384, 1) void ink2_moe_nvfp4_tc_up_situ_t160(
    const uint4* __restrict__ w13_u128,
    const uint32_t* __restrict__ s13_u32,
    const float* __restrict__ a13,
    const uint4* __restrict__ sorted_x_codes_u128,
    const uint2* __restrict__ sorted_x_sc_u64,
    const float* __restrict__ sorted_x_gs,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_toks,
    const int32_t* __restrict__ exp_pairs,
    int32_t exp_begin,
    int32_t exp_end,
    float* __restrict__ act_out) {
    pdl_wait();
    (void)exp_toks;
    grouped_nvfp4_tc_t160_impl<true, false>(
        w13_u128, s13_u32, a13, sorted_x_codes_u128, sorted_x_sc_u64, sorted_x_gs,
        exp_counts, exp_pairs, exp_begin, exp_end, act_out);
}

extern "C" __global__ __launch_bounds__(384, 1) void ink2_moe_nvfp4_tc_up_situ_t160_packed_scales(
    const uint4* __restrict__ w13_u128,
    const uint32_t* __restrict__ s13_u32,
    const float* __restrict__ a13,
    const uint4* __restrict__ sorted_x_codes_u128,
    const uint2* __restrict__ sorted_x_sc_u64,
    const float* __restrict__ sorted_x_gs,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_toks,
    const int32_t* __restrict__ exp_pairs,
    int32_t exp_begin,
    int32_t exp_end,
    float* __restrict__ act_out) {
    pdl_wait();
    (void)exp_toks;
    grouped_nvfp4_tc_t160_impl<true, true>(
        w13_u128, s13_u32, a13, sorted_x_codes_u128, sorted_x_sc_u64, sorted_x_gs,
        exp_counts, exp_pairs, exp_begin, exp_end, act_out);
}

extern "C" __global__ __launch_bounds__(384, 1) void ink2_moe_nvfp4_tc_down_t160(
    const uint4* __restrict__ w2_u128,
    const uint32_t* __restrict__ s2_u32,
    const float* __restrict__ a2,
    const uint4* __restrict__ act_codes_u128,
    const uint2* __restrict__ act_sc_u64,
    const float* __restrict__ act_gs,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_pairs,
    int32_t exp_begin,
    int32_t exp_end,
    void* __restrict__ pair_y_f16) {
    pdl_wait();
    grouped_nvfp4_tc_t160_impl<false, false>(
        w2_u128, s2_u32, a2, act_codes_u128, act_sc_u64, act_gs,
        exp_counts, exp_pairs, exp_begin, exp_end, pair_y_f16);
}

extern "C" __global__ __launch_bounds__(384, 1) void ink2_moe_nvfp4_tc_down_t160_packed_scales(
    const uint4* __restrict__ w2_u128,
    const uint32_t* __restrict__ s2_u32,
    const float* __restrict__ a2,
    const uint4* __restrict__ act_codes_u128,
    const uint2* __restrict__ act_sc_u64,
    const float* __restrict__ act_gs,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_pairs,
    int32_t exp_begin,
    int32_t exp_end,
    void* __restrict__ pair_y_f16) {
    pdl_wait();
    grouped_nvfp4_tc_t160_impl<false, true>(
        w2_u128, s2_u32, a2, act_codes_u128, act_sc_u64, act_gs,
        exp_counts, exp_pairs, exp_begin, exp_end, pair_y_f16);
}

// 18.4 Native Block-32 `MXFP8` (`UE8M0 + E4M3`) Activation Quantizer + `64x64` `mxf8f6f4` Grouped MoE
// Grid: `(cols / 128, n_rows)`, Block: `32` (`1` warp = `128` elements = `4` `MXFP8` blocks of 32!)
extern "C" __global__ void ink2_quant_mxfp8_b(
    const float4* __restrict__ x4_base,       // [n_rows, cols / 4]
    uint4* __restrict__ out_fp8_u128,         // [n_rows, cols / 16]
    uint32_t* __restrict__ out_ue8m0_u32,     // [n_rows, cols / 128]
    int32_t cols) {
    pdl_wait();
    const int blk128 = blockIdx.x;
    const int row = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int sub32 = lane >> 3;     // 0..3
    const int sub_lane = lane & 7;   // 0..7

    const float4 v = x4_base[static_cast<size_t>(row) * (cols >> 2) + blk128 * 32 + lane];
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 4));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1));

    uint32_t ue8m0 = 0u;
    float inv_scale = 0.0f;
    if (amax > 0.0f) {
        const uint32_t raw_bits = __float_as_uint(amax);
        if ((raw_bits & 0x7F800000u) == 0x7F800000u) {
            ue8m0 = 0xFEu;
            inv_scale = 0.0f;
        } else {
            const float sc_min = amax * (1.0f / 448.0f);
            const uint32_t sbits = __float_as_uint(sc_min);
            int biased_exp = static_cast<int>((sbits + 0x007FFFFFu) >> 23);
            if (biased_exp < 1) biased_exp = 1;
            if (biased_exp > 247) biased_exp = 247;
            ue8m0 = static_cast<uint32_t>(biased_exp);
            inv_scale = __uint_as_float((254u - ue8m0) << 23);
        }
    }

    uint16_t q_lo, q_hi;
    const float sx0 = v.x * inv_scale;
    const float sx1 = v.y * inv_scale;
    const float sx2 = v.z * inv_scale;
    const float sx3 = v.w * inv_scale;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\n" : "=h"(q_lo) : "f"(sx0), "f"(sx1));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\n" : "=h"(q_hi) : "f"(sx2), "f"(sx3));
    const uint32_t q4 = static_cast<uint32_t>(q_lo) | (static_cast<uint32_t>(q_hi) << 16);

    const uint32_t q4_n1 = __shfl_sync(0xFFFFFFFFu, q4, lane + 1);
    const uint32_t q4_n2 = __shfl_sync(0xFFFFFFFFu, q4, lane + 2);
    const uint32_t q4_n3 = __shfl_sync(0xFFFFFFFFu, q4, lane + 3);
    if ((lane & 3) == 0) {
        out_fp8_u128[static_cast<size_t>(row) * (cols >> 4) + blk128 * 8 + (lane >> 2)] =
            make_uint4(q4, q4_n1, q4_n2, q4_n3);
    }

    const uint32_t sc_b0 = __shfl_sync(0xFFFFFFFFu, ue8m0, 0);
    const uint32_t sc_b1 = __shfl_sync(0xFFFFFFFFu, ue8m0, 8);
    const uint32_t sc_b2 = __shfl_sync(0xFFFFFFFFu, ue8m0, 16);
    const uint32_t sc_b3 = __shfl_sync(0xFFFFFFFFu, ue8m0, 24);
    if (lane == 0) {
        (void)sub32; (void)sub_lane;
        out_ue8m0_u32[static_cast<size_t>(row) * (cols >> 7) + blk128] =
            sc_b0 | (sc_b1 << 8) | (sc_b2 << 16) | (sc_b3 << 24);
    }
}

// Double-Buffered `128 tokens x 128 weight rows` `mxf8f6f4` Grouped MXFP8 MoE (`Grid: (18, max_tiles_128)`, `Block: 512`)
// Doubles warp occupancy (`8 -> 16` warps/SM) and halves scattered `input_ids` activation gathers (`36x -> 18x`)!
struct Mxfp8TcSmem128x128 {
    uint4 smem_a[2][128][9];    // 36,864 B (stride = 36 dwords -> 32-bank bijection!)
    uint4 smem_b[2][128][9];    // 36,864 B (stride = 36 dwords -> 32-bank bijection!)
    uint32_t smem_asc[2][128];  //  1,024 B
    uint32_t smem_bsc[2][128];  //  1,024 B
    int32_t smem_in_ids[128];   //    512 B
    int32_t smem_out_pairs[128];//    512 B
    int32_t sm_blk_off[129];    //    516 B
    int32_t sm_row_off[129];    //    516 B
}; // Total = 77,832 B (`<= 99 KiB`, `1` CTA x `16` warps/SM!)

template <bool GateUp, bool PackedScales, bool Indexed = false>
__device__ __forceinline__ void grouped_mxfp8_tc_impl(
    const uint4* __restrict__ w_u128,       // [128, out_rows, cols / 16]
    const uint32_t* __restrict__ w_sc_u32,  // [128, out_rows, cols / 128]
    const uint4* __restrict__ x_u128,       // [n_in, cols / 16]
    const uint32_t* __restrict__ x_sc_u32,  // [n_in, cols / 16]
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ input_ids,
    const int32_t* __restrict__ exp_pairs,
    void* __restrict__ output_raw) {
    constexpr int kCols = GateUp ? kD : kFf;  // 2304 or 1152
    constexpr int kU128 = kCols / 16;         // 144 or 72
    constexpr int kKTiles = kCols / 128;      // 18 or 9

    extern __shared__ __align__(16) uint8_t dyn_mxfp8_smem[];
    Mxfp8TcSmem128x128* s = reinterpret_cast<Mxfp8TcSmem128x128*>(dyn_mxfp8_smem);

    int expert = 0;
    int first = 0;
    int count = 0;
    size_t sorted_base = 0;
    if (!resolve_flat_expert_tile<128>(exp_counts, s->sm_blk_off, s->sm_row_off, expert, first, count, sorted_base)) {
        return;
    }

    const int tid = threadIdx.x; // 0..511 (`16` warps/CTA!)
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int wm = warp_id >> 2;  // 0..3 (32 tokens each = 2 m16 tiles -> 128 tokens!)
    const int wn = warp_id & 3;   // 0..3 (32 weight rows each = 4 n8 tiles -> 128 weight rows!)
    const int g = lane >> 2;      // 0..7
    const int tig = lane & 3;     // 0..3
    const int sfa_row = ((lane & 1) << 3) | g;

    auto map_w_row = [&](int sr) -> int {
        if constexpr (GateUp) {
            const int wn_s = sr >> 5;                  // 0..3
            const int is_up = (sr >> 4) & 1;           // 0..1 (`r=0,1` gate, `r=2,3` up)
            const int m_local = wn_s * 16 + (sr & 15); // 0..63
            return blockIdx.x * 128 + 2 * m_local + is_up;
        } else {
            return blockIdx.x * 128 + sr;
        }
    };

    const size_t exp_base_row = static_cast<size_t>(expert) * (GateUp ? 2 * kFf : kD);
    const int live = min(128, count - first);
    if (tid < 128) {
        const int clamped = min(tid, live - 1);
        const int map_idx = expert_map_offset(expert, first + clamped);
        if constexpr (GateUp) {
            s->smem_in_ids[tid] = input_ids[map_idx];
        } else {
            s->smem_out_pairs[tid] = exp_pairs[map_idx];
            s->smem_in_ids[tid] = static_cast<int32_t>(sorted_base + static_cast<size_t>(clamped));
        }
    }
    __syncthreads();

    uint4 weight_table;
    if constexpr (Indexed) weight_table = w_u128[0];

    auto issue_stage = [&](int st, int kt) {
        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int i = p * 512 + tid;
            const int r = i >> 3;
            const int q = i & 7;
            const size_t src_tok = static_cast<size_t>(s->smem_in_ids[r]);
            cp_async16_cg_pred(
                &s->smem_a[st][r][q],
                x_u128 + src_tok * kU128 + kt * 8 + q,
                r < live);
            const size_t grow = exp_base_row + map_w_row(r);
            const size_t weight_index = grow * kU128 + kt * 8 + q;
            if constexpr (Indexed) {
                s->smem_b[st][r][q] = load_indexed_mxfp8(w_u128, weight_table, weight_index);
            } else {
                cp_async16_cg(&s->smem_b[st][r][q], w_u128 + weight_index);
            }
        }
        if (tid < 128) {
            const size_t src_tok = static_cast<size_t>(s->smem_in_ids[tid]);
            cp_async4_ca_pred(&s->smem_asc[st][tid], x_sc_u32 + src_tok * kKTiles + kt, tid < live);
        } else if (tid < 256) {
            if constexpr (!PackedScales) {
                const int wr_i = tid - 128;
                const size_t grow = exp_base_row + map_w_row(wr_i);
                cp_async4_ca(&s->smem_bsc[st][wr_i], w_sc_u32 + grow * kKTiles + kt);
            }
        }
        cp_async_commit_grp();
    };

    auto san_ue8m0x4 = [](uint32_t v) -> uint32_t {
        return v & ~__vcmpeq4(v, 0xFFFFFFFFu);
    };

    float acc[2][4][4] = {};
    issue_stage(0, 0);
    if constexpr (PackedScales) {
        if (tid >= 128 && tid < 256) {
            const int wr_i = tid - 128;
            const size_t grow = exp_base_row + map_w_row(wr_i);
            s->smem_bsc[0][wr_i] = expert_scale_word<true, 3, 112>(
                w_sc_u32,
                static_cast<unsigned>(grow * kKTiles + 0));
        }
    }

    for (int kt = 0; kt < kKTiles; ++kt) {
        if (kt + 1 < kKTiles) {
            issue_stage((kt + 1) & 1, kt + 1);
            cp_async_wait_grp<1>();
        } else {
            cp_async_wait_grp<0>();
        }
        __syncthreads();

        uint32_t next_bsc = 0u;
        if constexpr (PackedScales) {
            if ((kt + 1 < kKTiles) && tid >= 128 && tid < 256) {
                const int wr_i = tid - 128;
                const size_t grow = exp_base_row + map_w_row(wr_i);
                next_bsc = expert_scale_word<true, 3, 112>(
                    w_sc_u32,
                    static_cast<unsigned>(grow * kKTiles + (kt + 1)));
            }
        }

        if (wm * 32 < live) {
            const int st = kt & 1;
            unsigned sfa[2];
            sfa[0] = san_ue8m0x4(s->smem_asc[st][wm * 32 + sfa_row]);
            sfa[1] = san_ue8m0x4(s->smem_asc[st][wm * 32 + 16 + sfa_row]);
            unsigned sfb[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                sfb[r] = san_ue8m0x4(s->smem_bsc[st][wn * 32 + r * 8 + g]);
            }

            auto run_ks = [&](auto ks_tag) {
                constexpr int KS = decltype(ks_tag)::k;
                #pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    if (wm * 32 + mt * 16 < live) {
                        const uint32_t* a_r0 = reinterpret_cast<const uint32_t*>(s->smem_a[st][wm * 32 + mt * 16 + g]);
                        const uint32_t* a_r1 = reinterpret_cast<const uint32_t*>(s->smem_a[st][wm * 32 + mt * 16 + 8 + g]);
                        const unsigned af[4] = {
                            a_r0[KS * 8 + tig],
                            a_r1[KS * 8 + tig],
                            a_r0[KS * 8 + 4 + tig],
                            a_r1[KS * 8 + 4 + tig]
                        };
                        const uint32_t asc0 = san_ue8m0x4(s->smem_asc[st][wm * 32 + mt * 16 + g]);
                        const uint32_t asc1 = san_ue8m0x4(s->smem_asc[st][wm * 32 + mt * 16 + 8 + g]);
                        const float sfa0_f = __uint_as_float(((asc0 >> (KS * 8)) & 0xFFu) << 23);
                        const float sfa1_f = __uint_as_float(((asc1 >> (KS * 8)) & 0xFFu) << 23);
                        #pragma unroll
                        for (int r = 0; r < 4; ++r) {
                            const uint32_t* b_r = reinterpret_cast<const uint32_t*>(s->smem_b[st][wn * 32 + r * 8 + g]);
                            const unsigned b0 = b_r[KS * 8 + tig];
                            const unsigned b1 = b_r[KS * 8 + 4 + tig];
                            const uint32_t* b_c0 = reinterpret_cast<const uint32_t*>(s->smem_b[st][wn * 32 + r * 8 + 2 * tig]);
                            const uint32_t* b_c1 = reinterpret_cast<const uint32_t*>(s->smem_b[st][wn * 32 + r * 8 + 2 * tig + 1]);
                            const uint32_t bsc0 = san_ue8m0x4(s->smem_bsc[st][wn * 32 + r * 8 + 2 * tig]);
                            const uint32_t bsc1 = san_ue8m0x4(s->smem_bsc[st][wn * 32 + r * 8 + 2 * tig + 1]);
                            const float sfb0_f = __uint_as_float(((bsc0 >> (KS * 8)) & 0xFFu) << 23);
                            const float sfb1_f = __uint_as_float(((bsc1 >> (KS * 8)) & 0xFFu) << 23);
                            mma_mxfp8_m16n8k32<KS>(
                                acc[mt][r], af, b0, b1, sfa[mt], sfb[r],
                                a_r0, a_r1, b_c0, b_c1, sfa0_f, sfa1_f, sfb0_f, sfb1_f);
                        }
                    }
                }
            };
            run_ks(KsConst<0>{});
            run_ks(KsConst<1>{});
            run_ks(KsConst<2>{});
            run_ks(KsConst<3>{});
        }
        if constexpr (PackedScales) {
            if ((kt + 1 < kKTiles) && tid >= 128 && tid < 256) {
                s->smem_bsc[(kt + 1) & 1][tid - 128] = next_bsc;
            }
        }
        __syncthreads();
    }

    if (wm * 32 < live) {
        #pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int tok_local = wm * 32 + mt * 16 + h * 8 + g;
                if (tok_local < live) {
                    if constexpr (GateUp) {
                        float* output = static_cast<float*>(output_raw);
                        const size_t dst_seq = sorted_base + static_cast<size_t>(tok_local);
                        #pragma unroll
                        for (int rh = 0; rh < 2; ++rh) {
                            const int m0 = blockIdx.x * 64 + wn * 16 + rh * 8 + 2 * tig;
                            const float g0 = acc[mt][rh][2 * h];
                            const float g1 = acc[mt][rh][2 * h + 1];
                            const float u0 = acc[mt][2 + rh][2 * h];
                            const float u1 = acc[mt][2 + rh][2 * h + 1];
                            const float s0 = 4.0f * ptx_tanh(g0 * 0.25f) * ptx_sigmoid(g0) * 25.0f * ptx_tanh(u0 * 0.04f);
                            const float s1 = 4.0f * ptx_tanh(g1 * 0.25f) * ptx_sigmoid(g1) * 25.0f * ptx_tanh(u1 * 0.04f);
                            output[dst_seq * kFf + m0] = s0;
                            output[dst_seq * kFf + m0 + 1] = s1;
                        }
                    } else {
                        uint32_t* out_f16x2 = static_cast<uint32_t*>(output_raw);
                        const size_t dst_pair = static_cast<size_t>(s->smem_out_pairs[tok_local]);
                        #pragma unroll
                        for (int r = 0; r < 4; ++r) {
                            const int col0 = blockIdx.x * 128 + wn * 32 + r * 8 + 2 * tig;
                            out_f16x2[(dst_pair * kD + col0) >> 1] = pack_f16x2_rn(acc[mt][r][2 * h], acc[mt][r][2 * h + 1]);
                        }
                    }
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(512, 1) void ink2_moe_mxfp8_tc_up_situ(
    const uint4* __restrict__ w13_u128,
    const uint32_t* __restrict__ s13_u32,
    const uint4* __restrict__ x_u128,
    const uint32_t* __restrict__ x_sc_u32,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_toks,
    const int32_t* __restrict__ exp_pairs,
    float* __restrict__ act_out) {
    pdl_wait();
    grouped_mxfp8_tc_impl<true, false>(w13_u128, s13_u32, x_u128, x_sc_u32, exp_counts, exp_toks, exp_pairs, act_out);
}

extern "C" __global__ __launch_bounds__(512, 1) void ink2_moe_mxfp8_tc_up_situ_u4(
    const uint4* __restrict__ w13_u128,
    const uint32_t* __restrict__ s13_u32,
    const uint4* __restrict__ x_u128,
    const uint32_t* __restrict__ x_sc_u32,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_toks,
    const int32_t* __restrict__ exp_pairs,
    float* __restrict__ act_out) {
    pdl_wait();
    grouped_mxfp8_tc_impl<true, false, true>(w13_u128, s13_u32, x_u128, x_sc_u32, exp_counts, exp_toks, exp_pairs, act_out);
}

extern "C" __global__ __launch_bounds__(512, 1) void ink2_moe_mxfp8_tc_up_situ_packed_scales(
    const uint4* __restrict__ w13_u128,
    const uint32_t* __restrict__ s13_u32,
    const uint4* __restrict__ x_u128,
    const uint32_t* __restrict__ x_sc_u32,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_toks,
    const int32_t* __restrict__ exp_pairs,
    float* __restrict__ act_out) {
    pdl_wait();
    grouped_mxfp8_tc_impl<true, true>(w13_u128, s13_u32, x_u128, x_sc_u32, exp_counts, exp_toks, exp_pairs, act_out);
}

extern "C" __global__ __launch_bounds__(512, 1) void ink2_moe_mxfp8_tc_up_situ_u4_packed_scales(
    const uint4* __restrict__ w13_u128,
    const uint32_t* __restrict__ s13_u32,
    const uint4* __restrict__ x_u128,
    const uint32_t* __restrict__ x_sc_u32,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_toks,
    const int32_t* __restrict__ exp_pairs,
    float* __restrict__ act_out) {
    pdl_wait();
    grouped_mxfp8_tc_impl<true, true, true>(w13_u128, s13_u32, x_u128, x_sc_u32, exp_counts, exp_toks, exp_pairs, act_out);
}

extern "C" __global__ __launch_bounds__(512, 1) void ink2_moe_mxfp8_tc_down(
    const uint4* __restrict__ w2_u128,
    const uint32_t* __restrict__ s2_u32,
    const uint4* __restrict__ act_u128,
    const uint32_t* __restrict__ act_sc_u32,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_pairs,
    void* __restrict__ pair_y_f16) {
    pdl_wait();
    grouped_mxfp8_tc_impl<false, false>(w2_u128, s2_u32, act_u128, act_sc_u32, exp_counts, exp_pairs, exp_pairs, pair_y_f16);
}

extern "C" __global__ __launch_bounds__(512, 1) void ink2_moe_mxfp8_tc_down_u4(
    const uint4* __restrict__ w2_u128,
    const uint32_t* __restrict__ s2_u32,
    const uint4* __restrict__ act_u128,
    const uint32_t* __restrict__ act_sc_u32,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_pairs,
    void* __restrict__ pair_y_f16) {
    pdl_wait();
    grouped_mxfp8_tc_impl<false, false, true>(w2_u128, s2_u32, act_u128, act_sc_u32, exp_counts, exp_pairs, exp_pairs, pair_y_f16);
}

extern "C" __global__ __launch_bounds__(512, 1) void ink2_moe_mxfp8_tc_down_packed_scales(
    const uint4* __restrict__ w2_u128,
    const uint32_t* __restrict__ s2_u32,
    const uint4* __restrict__ act_u128,
    const uint32_t* __restrict__ act_sc_u32,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_pairs,
    void* __restrict__ pair_y_f16) {
    pdl_wait();
    grouped_mxfp8_tc_impl<false, true>(w2_u128, s2_u32, act_u128, act_sc_u32, exp_counts, exp_pairs, exp_pairs, pair_y_f16);
}

extern "C" __global__ __launch_bounds__(512, 1) void ink2_moe_mxfp8_tc_down_u4_packed_scales(
    const uint4* __restrict__ w2_u128,
    const uint32_t* __restrict__ s2_u32,
    const uint4* __restrict__ act_u128,
    const uint32_t* __restrict__ act_sc_u32,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_pairs,
    void* __restrict__ pair_y_f16) {
    pdl_wait();
    grouped_mxfp8_tc_impl<false, true, true>(w2_u128, s2_u32, act_u128, act_sc_u32, exp_counts, exp_pairs, exp_pairs, pair_y_f16);
}

// V13 (`#79`): `TileTokens = 160 x 128` Grouped MXFP8 MoE TC (`Block = 256`, `86,664 B <= 99 KiB` SMEM!)
struct Mxfp8TcSmem160x128 {
    uint4 smem_a[2][160][9];     // 46,080 B
    uint4 smem_b[2][128][9];     // 36,864 B
    uint32_t smem_asc[2][160];   //  1,280 B
    uint32_t smem_bsc[2][128];   //  1,024 B
    int32_t smem_in_ids[160];    //    640 B
    int32_t smem_out_pairs[160]; //    640 B
    int32_t sm_blk_off[129];     //    516 B
    int32_t sm_row_off[129];     //    516 B
}; // Total = 87,560 B (`<= 99 KiB`, `1` CTA/SM!)

template <bool GateUp, bool PackedScales, bool Indexed = false>
__device__ __forceinline__ void grouped_mxfp8_tc_t160_impl(
    const uint4* __restrict__ w_u128,
    const uint32_t* __restrict__ w_sc_u32,
    const uint4* __restrict__ x_u128,
    const uint32_t* __restrict__ x_sc_u32,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ input_ids,
    const int32_t* __restrict__ exp_pairs,
    void* __restrict__ output_raw) {
    constexpr int kCols = GateUp ? kD : kFf;  // 2304 or 1152
    constexpr int kU128 = kCols / 16;         // 144 or 72
    constexpr int kKTiles = kCols / 128;      // 18 or 9

    extern __shared__ __align__(16) uint8_t dyn_mxfp8_t160_smem[];
    Mxfp8TcSmem160x128* s = reinterpret_cast<Mxfp8TcSmem160x128*>(dyn_mxfp8_t160_smem);

    int expert = 0;
    int first = 0;
    int count = 0;
    size_t sorted_base = 0;
    if (!resolve_flat_expert_tile<160>(exp_counts, s->sm_blk_off, s->sm_row_off, expert, first, count, sorted_base)) {
        return;
    }

    const int tid = threadIdx.x; // 0..255 (`8` warps/CTA!)
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int wm = warp_id >> 2;  // 0..1 (`2` warps x `80` tokens = `160` tokens!)
    const int wn = warp_id & 3;   // 0..3 (`4` warps x `32` weight rows = `128` weight rows!)
    const int g = lane >> 2;      // 0..7
    const int tig = lane & 3;     // 0..3
    const int sfa_row = ((lane & 1) << 3) | g;

    auto map_w_row = [&](int sr) -> int {
        if constexpr (GateUp) {
            const int wn_s = sr >> 5;                  // 0..3
            const int is_up = (sr >> 4) & 1;           // 0..1
            const int m_local = wn_s * 16 + (sr & 15); // 0..63
            return blockIdx.x * 128 + 2 * m_local + is_up;
        } else {
            return blockIdx.x * 128 + sr;
        }
    };

    const size_t exp_base_row = static_cast<size_t>(expert) * (GateUp ? 2 * kFf : kD);
    const int live = min(160, count - first);
    if (tid < 160) {
        const int clamped = min(tid, live - 1);
        const int map_idx = expert_map_offset(expert, first + clamped);
        if constexpr (GateUp) {
            s->smem_in_ids[tid] = input_ids[map_idx];
        } else {
            s->smem_out_pairs[tid] = exp_pairs[map_idx];
            s->smem_in_ids[tid] = static_cast<int32_t>(sorted_base + static_cast<size_t>(clamped));
        }
    }
    __syncthreads();

    uint4 weight_table;
    if constexpr (Indexed) weight_table = w_u128[0];

    auto issue_stage = [&](int st, int kt) {
        #pragma unroll
        for (int p = 0; p < 5; ++p) {
            const int i = p * 256 + tid; // 0..1279 -> `160 x 8` uint4s
            const int r = i >> 3;
            const int q = i & 7;
            const size_t src_tok = static_cast<size_t>(s->smem_in_ids[r]);
            cp_async16_cg_pred(
                &s->smem_a[st][r][q],
                x_u128 + src_tok * kU128 + kt * 8 + q,
                r < live);
            if (p < 4) {
                const size_t grow = exp_base_row + map_w_row(r);
                const size_t weight_index = grow * kU128 + kt * 8 + q;
                if constexpr (Indexed) {
                    s->smem_b[st][r][q] = load_indexed_mxfp8(w_u128, weight_table, weight_index);
                } else {
                    cp_async16_cg(&s->smem_b[st][r][q], w_u128 + weight_index);
                }
            }
        }
        if (tid < 160) {
            const size_t src_tok = static_cast<size_t>(s->smem_in_ids[tid]);
            cp_async4_ca_pred(&s->smem_asc[st][tid], x_sc_u32 + src_tok * kKTiles + kt, tid < live);
        }
        if constexpr (!PackedScales) {
            if (tid >= 128) {
                const int wr_i = tid - 128;
                const size_t grow = exp_base_row + map_w_row(wr_i);
                cp_async4_ca(&s->smem_bsc[st][wr_i], w_sc_u32 + grow * kKTiles + kt);
            }
        }
        cp_async_commit_grp();
    };

    auto san_ue8m0x4 = [](uint32_t v) -> uint32_t {
        return v & ~__vcmpeq4(v, 0xFFFFFFFFu);
    };

    float acc[5][4][4] = {};
    issue_stage(0, 0);
    if constexpr (PackedScales) {
        if (tid >= 128) {
            const int wr_i = tid - 128;
            const size_t grow = exp_base_row + map_w_row(wr_i);
            s->smem_bsc[0][wr_i] = expert_scale_word<true, 3, 112>(
                w_sc_u32,
                static_cast<unsigned>(grow * kKTiles + 0));
        }
    }

    for (int kt = 0; kt < kKTiles; ++kt) {
        if (kt + 1 < kKTiles) {
            issue_stage((kt + 1) & 1, kt + 1);
            cp_async_wait_grp<1>();
        } else {
            cp_async_wait_grp<0>();
        }
        __syncthreads();

        uint32_t next_bsc = 0u;
        if constexpr (PackedScales) {
            if ((kt + 1 < kKTiles) && tid >= 128) {
                const int wr_i = tid - 128;
                const size_t grow = exp_base_row + map_w_row(wr_i);
                next_bsc = expert_scale_word<true, 3, 112>(
                    w_sc_u32,
                    static_cast<unsigned>(grow * kKTiles + (kt + 1)));
            }
        }

        if (wm * 80 < live) {
            const int st = kt & 1;
            unsigned sfa[5];
            #pragma unroll
            for (int mt = 0; mt < 5; ++mt) {
                sfa[mt] = san_ue8m0x4(s->smem_asc[st][wm * 80 + mt * 16 + sfa_row]);
            }
            unsigned sfb[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                sfb[r] = san_ue8m0x4(s->smem_bsc[st][wn * 32 + r * 8 + g]);
            }

            auto run_ks = [&](auto ks_tag) {
                constexpr int KS = decltype(ks_tag)::k;
                #pragma unroll
                for (int mt = 0; mt < 5; ++mt) {
                    if (wm * 80 + mt * 16 < live) {
                        const uint32_t* a_r0 = reinterpret_cast<const uint32_t*>(s->smem_a[st][wm * 80 + mt * 16 + g]);
                        const uint32_t* a_r1 = reinterpret_cast<const uint32_t*>(s->smem_a[st][wm * 80 + mt * 16 + 8 + g]);
                        const unsigned af[4] = {
                            a_r0[KS * 8 + tig],
                            a_r1[KS * 8 + tig],
                            a_r0[KS * 8 + 4 + tig],
                            a_r1[KS * 8 + 4 + tig]
                        };
                        const uint32_t asc0 = san_ue8m0x4(s->smem_asc[st][wm * 80 + mt * 16 + g]);
                        const uint32_t asc1 = san_ue8m0x4(s->smem_asc[st][wm * 80 + mt * 16 + 8 + g]);
                        const float sfa0_f = __uint_as_float(((asc0 >> (KS * 8)) & 0xFFu) << 23);
                        const float sfa1_f = __uint_as_float(((asc1 >> (KS * 8)) & 0xFFu) << 23);
                        #pragma unroll
                        for (int r = 0; r < 4; ++r) {
                            const uint32_t* b_r = reinterpret_cast<const uint32_t*>(s->smem_b[st][wn * 32 + r * 8 + g]);
                            const unsigned b0 = b_r[KS * 8 + tig];
                            const unsigned b1 = b_r[KS * 8 + 4 + tig];
                            const uint32_t* b_c0 = reinterpret_cast<const uint32_t*>(s->smem_b[st][wn * 32 + r * 8 + 2 * tig]);
                            const uint32_t* b_c1 = reinterpret_cast<const uint32_t*>(s->smem_b[st][wn * 32 + r * 8 + 2 * tig + 1]);
                            const uint32_t bsc0 = san_ue8m0x4(s->smem_bsc[st][wn * 32 + r * 8 + 2 * tig]);
                            const uint32_t bsc1 = san_ue8m0x4(s->smem_bsc[st][wn * 32 + r * 8 + 2 * tig + 1]);
                            const float sfb0_f = __uint_as_float(((bsc0 >> (KS * 8)) & 0xFFu) << 23);
                            const float sfb1_f = __uint_as_float(((bsc1 >> (KS * 8)) & 0xFFu) << 23);
                            mma_mxfp8_m16n8k32<KS>(
                                acc[mt][r], af, b0, b1, sfa[mt], sfb[r],
                                a_r0, a_r1, b_c0, b_c1, sfa0_f, sfa1_f, sfb0_f, sfb1_f);
                        }
                    }
                }
            };
            run_ks(KsConst<0>{});
            run_ks(KsConst<1>{});
            run_ks(KsConst<2>{});
            run_ks(KsConst<3>{});
        }
        if constexpr (PackedScales) {
            if ((kt + 1 < kKTiles) && tid >= 128) {
                s->smem_bsc[(kt + 1) & 1][tid - 128] = next_bsc;
            }
        }
        __syncthreads();
    }

    if (wm * 80 < live) {
        #pragma unroll
        for (int mt = 0; mt < 5; ++mt) {
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int tok_local = wm * 80 + mt * 16 + h * 8 + g;
                if (tok_local < live) {
                    if constexpr (GateUp) {
                        float* output = static_cast<float*>(output_raw);
                        const size_t dst_seq = sorted_base + static_cast<size_t>(tok_local);
                        #pragma unroll
                        for (int rh = 0; rh < 2; ++rh) {
                            const int m0 = blockIdx.x * 64 + wn * 16 + rh * 8 + 2 * tig;
                            const float g0 = acc[mt][rh][2 * h];
                            const float g1 = acc[mt][rh][2 * h + 1];
                            const float u0 = acc[mt][2 + rh][2 * h];
                            const float u1 = acc[mt][2 + rh][2 * h + 1];
                            const float s0 = 4.0f * ptx_tanh(g0 * 0.25f) * ptx_sigmoid(g0) * 25.0f * ptx_tanh(u0 * 0.04f);
                            const float s1 = 4.0f * ptx_tanh(g1 * 0.25f) * ptx_sigmoid(g1) * 25.0f * ptx_tanh(u1 * 0.04f);
                            output[dst_seq * kFf + m0] = s0;
                            output[dst_seq * kFf + m0 + 1] = s1;
                        }
                    } else {
                        uint32_t* out_f16x2 = static_cast<uint32_t*>(output_raw);
                        const size_t dst_pair = static_cast<size_t>(s->smem_out_pairs[tok_local]);
                        #pragma unroll
                        for (int r = 0; r < 4; ++r) {
                            const int col0 = blockIdx.x * 128 + wn * 32 + r * 8 + 2 * tig;
                            out_f16x2[(dst_pair * kD + col0) >> 1] = pack_f16x2_rn(acc[mt][r][2 * h], acc[mt][r][2 * h + 1]);
                        }
                    }
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 1) void ink2_moe_mxfp8_tc_up_situ_t160(
    const uint4* __restrict__ w13_u128,
    const uint32_t* __restrict__ s13_u32,
    const uint4* __restrict__ x_u128,
    const uint32_t* __restrict__ x_sc_u32,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_toks,
    const int32_t* __restrict__ exp_pairs,
    float* __restrict__ act_out) {
    pdl_wait();
    grouped_mxfp8_tc_t160_impl<true, false>(w13_u128, s13_u32, x_u128, x_sc_u32, exp_counts, exp_toks, exp_pairs, act_out);
}

extern "C" __global__ __launch_bounds__(256, 1) void ink2_moe_mxfp8_tc_up_situ_t160_u4(
    const uint4* __restrict__ w13_u128,
    const uint32_t* __restrict__ s13_u32,
    const uint4* __restrict__ x_u128,
    const uint32_t* __restrict__ x_sc_u32,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_toks,
    const int32_t* __restrict__ exp_pairs,
    float* __restrict__ act_out) {
    pdl_wait();
    grouped_mxfp8_tc_t160_impl<true, false, true>(w13_u128, s13_u32, x_u128, x_sc_u32, exp_counts, exp_toks, exp_pairs, act_out);
}

extern "C" __global__ __launch_bounds__(256, 1) void ink2_moe_mxfp8_tc_up_situ_t160_packed_scales(
    const uint4* __restrict__ w13_u128,
    const uint32_t* __restrict__ s13_u32,
    const uint4* __restrict__ x_u128,
    const uint32_t* __restrict__ x_sc_u32,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_toks,
    const int32_t* __restrict__ exp_pairs,
    float* __restrict__ act_out) {
    pdl_wait();
    grouped_mxfp8_tc_t160_impl<true, true>(w13_u128, s13_u32, x_u128, x_sc_u32, exp_counts, exp_toks, exp_pairs, act_out);
}

extern "C" __global__ __launch_bounds__(256, 1) void ink2_moe_mxfp8_tc_up_situ_t160_u4_packed_scales(
    const uint4* __restrict__ w13_u128,
    const uint32_t* __restrict__ s13_u32,
    const uint4* __restrict__ x_u128,
    const uint32_t* __restrict__ x_sc_u32,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_toks,
    const int32_t* __restrict__ exp_pairs,
    float* __restrict__ act_out) {
    pdl_wait();
    grouped_mxfp8_tc_t160_impl<true, true, true>(w13_u128, s13_u32, x_u128, x_sc_u32, exp_counts, exp_toks, exp_pairs, act_out);
}

extern "C" __global__ __launch_bounds__(256, 1) void ink2_moe_mxfp8_tc_down_t160(
    const uint4* __restrict__ w2_u128,
    const uint32_t* __restrict__ s2_u32,
    const uint4* __restrict__ act_u128,
    const uint32_t* __restrict__ act_sc_u32,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_pairs,
    void* __restrict__ pair_y_f16) {
    pdl_wait();
    grouped_mxfp8_tc_t160_impl<false, false>(w2_u128, s2_u32, act_u128, act_sc_u32, exp_counts, exp_pairs, exp_pairs, pair_y_f16);
}

extern "C" __global__ __launch_bounds__(256, 1) void ink2_moe_mxfp8_tc_down_t160_u4(
    const uint4* __restrict__ w2_u128,
    const uint32_t* __restrict__ s2_u32,
    const uint4* __restrict__ act_u128,
    const uint32_t* __restrict__ act_sc_u32,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_pairs,
    void* __restrict__ pair_y_f16) {
    pdl_wait();
    grouped_mxfp8_tc_t160_impl<false, false, true>(w2_u128, s2_u32, act_u128, act_sc_u32, exp_counts, exp_pairs, exp_pairs, pair_y_f16);
}

extern "C" __global__ __launch_bounds__(256, 1) void ink2_moe_mxfp8_tc_down_t160_packed_scales(
    const uint4* __restrict__ w2_u128,
    const uint32_t* __restrict__ s2_u32,
    const uint4* __restrict__ act_u128,
    const uint32_t* __restrict__ act_sc_u32,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_pairs,
    void* __restrict__ pair_y_f16) {
    pdl_wait();
    grouped_mxfp8_tc_t160_impl<false, true>(w2_u128, s2_u32, act_u128, act_sc_u32, exp_counts, exp_pairs, exp_pairs, pair_y_f16);
}

extern "C" __global__ __launch_bounds__(256, 1) void ink2_moe_mxfp8_tc_down_t160_u4_packed_scales(
    const uint4* __restrict__ w2_u128,
    const uint32_t* __restrict__ s2_u32,
    const uint4* __restrict__ act_u128,
    const uint32_t* __restrict__ act_sc_u32,
    const int32_t* __restrict__ exp_counts,
    const int32_t* __restrict__ exp_pairs,
    void* __restrict__ pair_y_f16) {
    pdl_wait();
    grouped_mxfp8_tc_t160_impl<false, true, true>(w2_u128, s2_u32, act_u128, act_sc_u32, exp_counts, exp_pairs, exp_pairs, pair_y_f16);
}

// 18.5 GQA `3:1` + `64`-Query Shared-KV (`192` Query-Heads/KV Load!) FP16/INT8 Tensor Core FlashAttention + `Q8_64` Epilogue
// Grid: `(8, ceil(B / 64))`, Block: `384` (`1` CTA x `12` warps/SM via `__launch_bounds__(384, 1)`, `58,240` B SMEM -> `64 KiB` L1 + `2x` KV reuse!)
struct alignas(16) FlashKvStageF16 {
    uint16_t k[32][136]; // 8,704 B (272 B/row -> 16B aligned & bank-conflict-free!)
    uint16_t v[32][136]; // 8,704 B
};

template <bool kInt8>
struct FlashKvBuffer;
template <>
struct alignas(16) FlashKvBuffer<false> {
    FlashKvStageF16 st[2]; // 34,816 B
};
template <>
struct alignas(16) FlashKvBuffer<true> {
    uint32_t i8[2][64][32]; // 16,384 B (`[0..31]` = K `int8x4`, `[32..63]` = V `int8x4`)
    float sc[2][64];        //    512 B (`[0..31]` = K scales, `[32..63]` = V scales)
    FlashKvStageF16 f16;    // 17,408 B (single unpacked stage -> ZERO aliasing & 1 fewer CTA barrier/tile!)
}; // Total = 34,304 B (< 34,816 B!)

template <bool kInt8>
struct alignas(16) FlashAttnSmem64 {
    FlashKvBuffer<kInt8> kv;           // 34,816 B (`FP16`) or 34,304 B (`INT8`)
    alignas(16) uint16_t band[96][24]; //  4,608 B (`64` queries + `32` keys = `95` relative offsets!)
    uint4 rf[384];                     //  6,144 B (`R` staged in SMEM -> frees 4 regs across KV loop, 0 spills!)
    float gx[12][8][49];               // 18,816 B (`__syncwarp()` only across all 12 warps!)
}; // Total = 64,384 B (`FP16`) / 63,872 B (`INT8`) + 1,024 B sys <= 65,536 B (`64.0 KiB` -> full `64 KiB` L1!)

template <bool kInt8, bool kSplit = false>
__device__ __forceinline__ void flash_attn_tc_q8_b_impl(
    const float* __restrict__ qrgs_base,        // [B, qrgs_stride]
    int32_t qrgs_stride,
    const uint32_t* __restrict__ rel_proj_bf16, // [16, 512]
    const uint16_t* __restrict__ k_cache_fp16,  // [kv_cap, 1024] (or `int8_t[kv_cap, 1024]` if `kInt8`)
    const uint16_t* __restrict__ v_cache_fp16,  // [kv_cap, 1024] (or `int8_t[kv_cap, 1024]` if `kInt8`)
    const float* __restrict__ k_scale,          // [kv_cap, 8] if `kInt8`
    const float* __restrict__ v_scale,          // [kv_cap, 8] if `kInt8`
    const int32_t* __restrict__ pos_ptr,
    int32_t kv_cap,
    int32_t is_global,
    int32_t B,
    uint32_t* __restrict__ out_q8_u32,          // [B, 768] (when `!kSplit`)
    float* __restrict__ out_sc64,               // [B, 48]  (when `!kSplit`)
    uint32_t* __restrict__ out_part_o_f16x2 = nullptr, // [G_b, B, 24, 64] (when `kSplit`)
    float2* __restrict__ out_part_ml = nullptr) { // [G_b, B, 24]     (when `kSplit`)
    extern __shared__ __align__(16) uint8_t dyn_flash_smem[];
    FlashAttnSmem64<kInt8>* f_sm = reinterpret_cast<FlashAttnSmem64<kInt8>*>(dyn_flash_smem);
    auto& sm_kv = f_sm->kv;
    auto& sm_band = f_sm->band;
    auto& sm_gx = f_sm->gx;

    const int kvh = blockIdx.x; // 0..7
    const int q_cta = static_cast<int>(gridDim.y - 1u - blockIdx.y) * 64;
    if (q_cta >= B) return;
    const int q_live = min(64, B - q_cta);

    const int tid = threadIdx.x; // 0..383
    const int w = tid >> 5;      // 0..11
    const int lane = tid & 31;
    const int gqa = w >> 2;      // 0..2
    const int wq = w & 3;        // 0..3 (`4 x 16 = 64` queries sharing 1 KV stage!)
    const int h = kvh * 3 + gqa; // 0..23
    const int g = lane >> 2;     // 0..7
    const int t = lane & 3;      // 0..3

    for (int idx = tid; idx < 96 * 24; idx += 384) {
        sm_band[idx / 24][idx % 24] = 0u;
    }

    const int base_pos = *pos_ptr;
    const int qr0 = wq * 16 + g;
    const int qr1 = qr0 + 8;
    const bool vq0 = qr0 < q_live;
    const bool vq1 = qr1 < q_live;
    const int pos_q0 = base_pos + q_cta + qr0;
    const int pos_q1 = base_pos + q_cta + qr1;

    // Load `Q` (`16 x 128`) and `R` (`16 x 16`) directly into warp registers once!
    uint32_t qf[8][4];
    {
        const float* qa_ptr = qrgs_base + static_cast<size_t>(q_cta + (vq0 ? qr0 : 0)) * qrgs_stride + h * kHd;
        const float* qb_ptr = qrgs_base + static_cast<size_t>(q_cta + (vq1 ? qr1 : 0)) * qrgs_stride + h * kHd;
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            const int c0 = kk * 16 + 2 * t;
            const int c1 = c0 + 8;
            qf[kk][0] = vq0 ? pack_f16x2_rn(qa_ptr[c0], qa_ptr[c0 + 1]) : 0u;
            qf[kk][1] = vq1 ? pack_f16x2_rn(qb_ptr[c0], qb_ptr[c0 + 1]) : 0u;
            qf[kk][2] = vq0 ? pack_f16x2_rn(qa_ptr[c1], qa_ptr[c1 + 1]) : 0u;
            qf[kk][3] = vq1 ? pack_f16x2_rn(qb_ptr[c1], qb_ptr[c1 + 1]) : 0u;
        }
    }
    {
        const float* ra_ptr = qrgs_base + static_cast<size_t>(q_cta + (vq0 ? qr0 : 0)) * qrgs_stride + 3072 + h * kRel + 2 * t;
        const float* rb_ptr = qrgs_base + static_cast<size_t>(q_cta + (vq1 ? qr1 : 0)) * qrgs_stride + 3072 + h * kRel + 2 * t;
        f_sm->rf[tid] = make_uint4(
            vq0 ? pack_f16x2_rn(ra_ptr[0], ra_ptr[1]) : 0u,
            vq1 ? pack_f16x2_rn(rb_ptr[0], rb_ptr[1]) : 0u,
            vq0 ? pack_f16x2_rn(ra_ptr[8], ra_ptr[9]) : 0u,
            vq1 ? pack_f16x2_rn(rb_ptr[8], rb_ptr[9]) : 0u);
    }

    const int pos_q_min = base_pos + q_cta;
    const int pos_q_max = base_pos + q_cta + q_live - 1;
    int k_first = is_global ? 0 : max(0, pos_q_min + 1 - kExtent);
    int k_end = pos_q_max + 1;
    if constexpr (kSplit) {
        const int split_idx = static_cast<int>(blockIdx.z);
        const int n_splits = static_cast<int>(gridDim.z);
        const int n_tiles32 = (pos_q_max + 1 + 31) >> 5;
        const int t0 = (n_tiles32 * split_idx) / n_splits;
        const int t1 = (n_tiles32 * (split_idx + 1)) / n_splits;
        k_first = t0 * 32;
        k_end = min(pos_q_max + 1, t1 * 32);
    }
    const bool include_sink = !kSplit || (blockIdx.z == 0u);
    float m0 = (include_sink && vq0) ? qrgs_base[static_cast<size_t>(q_cta + qr0) * qrgs_stride + 6528 + h] : (kSplit ? -1e30f : 0.0f);
    float m1 = (include_sink && vq1) ? qrgs_base[static_cast<size_t>(q_cta + qr1) * qrgs_stride + 6528 + h] : (kSplit ? -1e30f : 0.0f);
    float l0 = (include_sink && (!kSplit || vq0) && t == 0) ? 1.0f : 0.0f;
    float l1 = (include_sink && (!kSplit || vq1) && t == 0) ? 1.0f : 0.0f;
    float o_acc[16][4] = {};
    const float attn_scale = 0.08838834764831845f;

    auto issue_kv_tile = [&](int st, int k0) {
        const int ring_base = is_global ? k0 : ((k0 >= 0) ? (k0 % kv_cap) : 0);
        auto ring_row = [&](int kc) -> int {
            int sr = ring_base + kc;
            if (!is_global && sr >= kv_cap) sr -= kv_cap;
            return sr;
        };
        if constexpr (!kInt8) {
            #pragma unroll
            for (int p = 0; p < 3; ++p) {
                const int idx = p * 384 + tid;
                if (idx < 1024) {
                    const bool is_v = idx >= 512;
                    const int rem = is_v ? (idx - 512) : idx;
                    const int kc = rem >> 4;        // 0..31
                    const int c8 = (rem & 15) * 8;  // 0, 8, ..., 120 halves
                    const int pos_k = k0 + kc;
                    const bool ok = (pos_k >= k_first) && (pos_k < k_end);
                    const int slot_row = ok ? ring_row(kc) : 0;
                    const uint16_t* src = (is_v ? v_cache_fp16 : k_cache_fp16)
                        + static_cast<size_t>(slot_row) * 1024u + kvh * 128 + c8;
                    uint16_t* dst = is_v ? &sm_kv.st[st].v[kc][c8] : &sm_kv.st[st].k[kc][c8];
                    cp_async16_cg_pred(dst, src, ok);
                }
            }
        } else {
            const int8_t* k_i8 = reinterpret_cast<const int8_t*>(k_cache_fp16);
            const int8_t* v_i8 = reinterpret_cast<const int8_t*>(v_cache_fp16);
            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int idx = p * 384 + tid;
                if (idx < 512) {
                    const bool is_v = idx >= 256;
                    const int rem = is_v ? (idx - 256) : idx;
                    const int kc = rem >> 3;         // 0..31
                    const int c16 = (rem & 7) * 16;  // 0, 16, ..., 112 int8 bytes
                    const int pos_k = k0 + kc;
                    const bool ok = (pos_k >= k_first) && (pos_k < k_end);
                    const int slot_row = ok ? ring_row(kc) : 0;
                    const int8_t* src = (is_v ? v_i8 : k_i8)
                        + static_cast<size_t>(slot_row) * 1024u + kvh * 128 + c16;
                    uint32_t* dst = &sm_kv.i8[st][(is_v ? 32 : 0) + kc][(rem & 7) * 4];
                    cp_async16_cg_pred(dst, src, ok);
                }
            }
            if (tid < 64) {
                const bool is_v = tid >= 32;
                const int kc = tid & 31;
                const int pos_k = k0 + kc;
                const bool ok = (pos_k >= k_first) && (pos_k < k_end);
                const int slot_row = ok ? ring_row(kc) : 0;
                const float* sc_src = (is_v ? v_scale : k_scale) + static_cast<size_t>(slot_row) * 8u + kvh;
                cp_async4_ca_pred(&sm_kv.sc[st][tid], sc_src, ok);
            }
        }
        cp_async_commit_grp();
    };

    __syncthreads();
    if (k_first < k_end) {
        issue_kv_tile(0, k_first);
    }

    int st_idx = 0;
    for (int k_base = k_first; k_base < k_end; k_base += 32, st_idx ^= 1) {
        const bool has_next = (k_base + 32) < k_end;
        if (has_next) {
            issue_kv_tile(st_idx ^ 1, k_base + 32);
        }

        const int d_min = pos_q_min - (k_base + 31);
        const int d_max = pos_q_max - k_base;
        const bool has_rel = (d_min < kExtent) && (d_max >= 0);
        if (has_rel) {
            const int d0 = pos_q_min - k_base - 31;
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int idx = p * 384 + tid;
                const int br = idx >> 4;
                const int ri = idx & 15;
                const int d = d0 + br;
                uint16_t hbits = 0u;
                if (d >= 0 && d < kExtent) {
                    const uint32_t pair32 = rel_proj_bf16[ri * 512 + (d >> 1)];
                    const uint16_t bf16_u16 = static_cast<uint16_t>((d & 1) ? (pair32 >> 16) : (pair32 & 0xFFFFu));
                    hbits = f32_to_fp16_bits(bf16_to_f32(bf16_u16));
                }
                sm_band[br][ri] = hbits;
            }
        }

        if (has_next) {
            cp_async_wait_grp<1>();
        } else {
            cp_async_wait_grp<0>();
        }
        __syncthreads();

        if constexpr (kInt8) {
            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int idx = p * 384 + tid;
                if (idx < 512) {
                    const bool is_v = idx >= 256;
                    const int rem = is_v ? (idx - 256) : idx;
                    const int kc = rem >> 3;  // 0..31
                    const int c16 = rem & 7;  // 0..7 (`16` int8s = `1` uint4 -> `2` f16x8 uint4s!)
                    const uint4 w16 = *reinterpret_cast<const uint4*>(
                        &sm_kv.i8[st_idx][(is_v ? 32 : 0) + kc][c16 * 4]);
                    const float sc = sm_kv.sc[st_idx][(is_v ? 32 : 0) + kc];
                    float x0, x1, x2, x3, x4, x5, x6, x7;
                    kv_dequant_i8(w16.x, sc, x0, x1, x2, x3);
                    kv_dequant_i8(w16.y, sc, x4, x5, x6, x7);
                    uint4* dst_u128 = reinterpret_cast<uint4*>(
                        is_v ? &sm_kv.f16.v[kc][c16 * 16] : &sm_kv.f16.k[kc][c16 * 16]);
                    dst_u128[0] = make_uint4(
                        pack_f16x2_rn(x0, x1), pack_f16x2_rn(x2, x3),
                        pack_f16x2_rn(x4, x5), pack_f16x2_rn(x6, x7));
                    kv_dequant_i8(w16.z, sc, x0, x1, x2, x3);
                    kv_dequant_i8(w16.w, sc, x4, x5, x6, x7);
                    dst_u128[1] = make_uint4(
                        pack_f16x2_rn(x0, x1), pack_f16x2_rn(x2, x3),
                        pack_f16x2_rn(x4, x5), pack_f16x2_rn(x6, x7));
                }
            }
            __syncthreads();
        }
        const FlashKvStageF16& cur_kv = [&]() -> const FlashKvStageF16& {
            if constexpr (kInt8) return sm_kv.f16;
            else return sm_kv.st[st_idx];
        }();

        const int p_first = base_pos + q_cta + wq * 16;
        const int p_last = p_first + 15;
        const bool active = !(k_base > p_last || (!is_global && (k_base + 31) <= (p_first - kExtent)));

        if (active) {
            float s[4][4] = {};
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                #pragma unroll
                for (int n = 0; n < 4; n += 2) {
                    unsigned b0, b1, b2, b3;
                    ldsm_x4_b16(
                        b0, b1, b2, b3,
                        &cur_kv.k[n * 8 + (lane & 7) + ((lane >> 4) << 3)][kk * 16 + ((lane >> 3) & 1) * 8]);
                    mma_f16_m16n8k16(s[n][0], s[n][1], s[n][2], s[n][3], qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], b0, b1);
                    mma_f16_m16n8k16(s[n + 1][0], s[n + 1][1], s[n + 1][2], s[n + 1][3], qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], b2, b3);
                }
            }

            const bool warp_has_rel = has_rel && ((p_first - (k_base + 31)) < kExtent);
            if (warp_has_rel) {
                const uint4 rfu = f_sm->rf[tid];
                float ga_hi[6][2];
                #pragma unroll
                for (int nt = 0; nt < 6; ++nt) {
                    unsigned b0, b1;
                    ldsm_x2_b16(b0, b1, &sm_band[wq * 16 + nt * 8 + (lane & 7)][((lane >> 3) & 1) * 8]);
                    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
                    mma_f16_m16n8k16(g0, g1, g2, g3, rfu.x, rfu.y, rfu.z, rfu.w, b0, b1);
                    sm_gx[w][g][nt * 8 + 2 * t] = g0;
                    sm_gx[w][g][nt * 8 + 2 * t + 1] = g1;
                    ga_hi[nt][0] = g2;
                    ga_hi[nt][1] = g3;
                }
                __syncwarp();
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    const int kk0 = n * 8 + 2 * t;
                    s[n][0] = __fmaf_rn(s[n][0], attn_scale, sm_gx[w][g][g + 31 - kk0]);
                    s[n][1] = __fmaf_rn(s[n][1], attn_scale, sm_gx[w][g][g + 31 - (kk0 + 1)]);
                }
                __syncwarp();
                // Pass B: row `g + 8` (`ga_hi[nt][0..1]`)
                #pragma unroll
                for (int nt = 0; nt < 6; ++nt) {
                    sm_gx[w][g][nt * 8 + 2 * t] = ga_hi[nt][0];
                    sm_gx[w][g][nt * 8 + 2 * t + 1] = ga_hi[nt][1];
                }
                __syncwarp();
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    const int kk0 = n * 8 + 2 * t;
                    s[n][2] = __fmaf_rn(s[n][2], attn_scale, sm_gx[w][g][(g + 8) + 31 - kk0]);
                    s[n][3] = __fmaf_rn(s[n][3], attn_scale, sm_gx[w][g][(g + 8) + 31 - (kk0 + 1)]);
                }
                __syncwarp();
            } else {
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        s[n][e] *= attn_scale;
                    }
                }
            }

            bool valid_mask[4][4];
            float tmax0 = m0;
            float tmax1 = m1;
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                #pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int pos_k = k_base + n * 8 + 2 * t + c;
                    const int d0 = pos_q0 - pos_k;
                    const bool v0 = vq0 && (pos_k < k_end) && (d0 >= 0) && (is_global || d0 < kExtent);
                    valid_mask[n][c] = v0;
                    if (v0) tmax0 = fmaxf(tmax0, s[n][c]);

                    const int d1 = pos_q1 - pos_k;
                    const bool v1 = vq1 && (pos_k < k_end) && (d1 >= 0) && (is_global || d1 < kExtent);
                    valid_mask[n][2 + c] = v1;
                    if (v1) tmax1 = fmaxf(tmax1, s[n][2 + c]);
                }
            }

            tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xFFFFFFFFu, tmax0, 1));
            tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xFFFFFFFFu, tmax0, 2));
            tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xFFFFFFFFu, tmax1, 1));
            tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xFFFFFFFFu, tmax1, 2));

            if (tmax0 > m0) {
                const float alpha0 = ptx_exp(m0 - tmax0);
                l0 *= alpha0;
                #pragma unroll
                for (int nd = 0; nd < 16; ++nd) {
                    o_acc[nd][0] *= alpha0;
                    o_acc[nd][1] *= alpha0;
                }
                m0 = tmax0;
            }
            if (tmax1 > m1) {
                const float alpha1 = ptx_exp(m1 - tmax1);
                l1 *= alpha1;
                #pragma unroll
                for (int nd = 0; nd < 16; ++nd) {
                    o_acc[nd][2] *= alpha1;
                    o_acc[nd][3] *= alpha1;
                }
                m1 = tmax1;
            }

            unsigned pf[2][4];
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int n0 = 2 * j;
                const int n1 = n0 + 1;
                const float p00 = valid_mask[n0][0] ? ptx_exp(s[n0][0] - m0) : 0.0f;
                const float p01 = valid_mask[n0][1] ? ptx_exp(s[n0][1] - m0) : 0.0f;
                const float p02 = valid_mask[n0][2] ? ptx_exp(s[n0][2] - m1) : 0.0f;
                const float p03 = valid_mask[n0][3] ? ptx_exp(s[n0][3] - m1) : 0.0f;
                l0 += p00 + p01;
                l1 += p02 + p03;
                pf[j][0] = pack_f16x2_rn(p00, p01);
                pf[j][1] = pack_f16x2_rn(p02, p03);

                const float p10 = valid_mask[n1][0] ? ptx_exp(s[n1][0] - m0) : 0.0f;
                const float p11 = valid_mask[n1][1] ? ptx_exp(s[n1][1] - m0) : 0.0f;
                const float p12 = valid_mask[n1][2] ? ptx_exp(s[n1][2] - m1) : 0.0f;
                const float p13 = valid_mask[n1][3] ? ptx_exp(s[n1][3] - m1) : 0.0f;
                l0 += p10 + p11;
                l1 += p12 + p13;
                pf[j][2] = pack_f16x2_rn(p10, p11);
                pf[j][3] = pack_f16x2_rn(p12, p13);
            }

            #pragma unroll
            for (int nd = 0; nd < 16; nd += 2) {
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    unsigned b0, b1, b2, b3;
                    ldsm_x4_trans_b16(
                        b0, b1, b2, b3,
                        &cur_kv.v[j * 16 + (lane & 15)][nd * 8 + (lane >> 4) * 8]);
                    mma_f16_m16n8k16(
                        o_acc[nd][0], o_acc[nd][1], o_acc[nd][2], o_acc[nd][3],
                        pf[j][0], pf[j][1], pf[j][2], pf[j][3], b0, b1);
                    mma_f16_m16n8k16(
                        o_acc[nd + 1][0], o_acc[nd + 1][1], o_acc[nd + 1][2], o_acc[nd + 1][3],
                        pf[j][0], pf[j][1], pf[j][2], pf[j][3], b2, b3);
                }
            }
        }
        __syncthreads();
    }

    // Quad reduction of `l0, l1`
    l0 += __shfl_xor_sync(0xFFFFFFFFu, l0, 1);
    l0 += __shfl_xor_sync(0xFFFFFFFFu, l0, 2);
    l1 += __shfl_xor_sync(0xFFFFFFFFu, l1, 1);
    l1 += __shfl_xor_sync(0xFFFFFFFFu, l1, 2);

    if constexpr (kSplit) {
        const size_t split_base = static_cast<size_t>(blockIdx.z) * static_cast<size_t>(B);
        if (vq0) {
            const float inv_split_l0 = (l0 > 0.0f) ? ptx_rcp(l0) : 0.0f;
            const size_t qh0 = (split_base + static_cast<size_t>(q_cta + qr0)) * 24u + h;
            uint32_t* dst0 = out_part_o_f16x2 + qh0 * 64u;
            #pragma unroll
            for (int nd = 0; nd < 16; ++nd) {
                dst0[nd * 4 + t] = pack_f16x2_rn(o_acc[nd][0] * inv_split_l0, o_acc[nd][1] * inv_split_l0);
            }
            if (t == 0) {
                out_part_ml[qh0] = make_float2(m0, l0);
            }
        }
        if (vq1) {
            const float inv_split_l1 = (l1 > 0.0f) ? ptx_rcp(l1) : 0.0f;
            const size_t qh1 = (split_base + static_cast<size_t>(q_cta + qr1)) * 24u + h;
            uint32_t* dst1 = out_part_o_f16x2 + qh1 * 64u;
            #pragma unroll
            for (int nd = 0; nd < 16; ++nd) {
                dst1[nd * 4 + t] = pack_f16x2_rn(o_acc[nd][2] * inv_split_l1, o_acc[nd][3] * inv_split_l1);
            }
            if (t == 0) {
                out_part_ml[qh1] = make_float2(m1, l1);
            }
        }
        return;
    }

    const float inv_l0 = ptx_rcp(fmaxf(l0, 1e-20f));
    const float inv_l1 = ptx_rcp(fmaxf(l1, 1e-20f));

    auto quantize_and_store_half64 = [&](int qr, bool vq, float inv_l, int hi_offset, int half64) {
        const int blk64_idx = h * 2 + half64;
        const float* gate_ptr = vq
            ? (qrgs_base + static_cast<size_t>(q_cta + qr) * qrgs_stride + 3456 + h * kHd + half64 * 64)
            : nullptr;
        float vals[8][2];
        float amax = 0.0f;
        #pragma unroll
        for (int nd = 0; nd < 8; ++nd) {
            const int d0 = nd * 8 + 2 * t;
            float v0 = 0.0f;
            float v1 = 0.0f;
            if (vq) {
                v0 = (o_acc[half64 * 8 + nd][hi_offset] * inv_l) * ptx_sigmoid(gate_ptr[d0]);
                v1 = (o_acc[half64 * 8 + nd][hi_offset + 1] * inv_l) * ptx_sigmoid(gate_ptr[d0 + 1]);
            }
            vals[nd][0] = v0;
            vals[nd][1] = v1;
            amax = fmaxf(amax, fmaxf(fabsf(v0), fabsf(v1)));
        }
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 2));

        if (vq) {
            const float inv = (amax > 0.0f) ? (127.0f * ptx_rcp(amax)) : 0.0f;
            uint32_t* dst_q8 = out_q8_u32 + static_cast<size_t>(q_cta + qr) * 768u + blk64_idx * 16;
            #pragma unroll
            for (int nd = 0; nd < 8; ++nd) {
                int q0 = __float2int_rn(vals[nd][0] * inv);
                int q1 = __float2int_rn(vals[nd][1] * inv);
                if (q0 < -127) q0 = -127;
                if (q0 > 127) q0 = 127;
                if (q1 < -127) q1 = -127;
                if (q1 > 127) q1 = 127;
                const uint32_t pair16 =
                    static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(q0))) |
                    (static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(q1))) << 8);
                const uint32_t pack32 = pair16 | (__shfl_xor_sync(0xFFFFFFFFu, pair16, 1) << 16);
                if ((t & 1) == 0) {
                    dst_q8[nd * 2 + (t >> 1)] = pack32;
                }
            }
            if (t == 0) {
                out_sc64[static_cast<size_t>(q_cta + qr) * 48u + blk64_idx] = amax * (1.0f / 127.0f);
            }
        }
    };

    quantize_and_store_half64(qr0, vq0, inv_l0, 0, 0);
    quantize_and_store_half64(qr0, vq0, inv_l0, 0, 1);
    quantize_and_store_half64(qr1, vq1, inv_l1, 2, 0);
    quantize_and_store_half64(qr1, vq1, inv_l1, 2, 1);
}

extern "C" __global__ __launch_bounds__(384, 1) void ink2_flash_attn_tc_q8_b(
    const float* __restrict__ qrgs_base,
    int32_t qrgs_stride,
    const uint32_t* __restrict__ rel_proj_bf16,
    const uint16_t* __restrict__ k_cache_fp16,
    const uint16_t* __restrict__ v_cache_fp16,
    const int32_t* __restrict__ pos_ptr,
    int32_t kv_cap,
    int32_t is_global,
    int32_t B,
    uint32_t* __restrict__ out_q8_u32,
    float* __restrict__ out_sc64) {
    pdl_wait();
    flash_attn_tc_q8_b_impl<false>(
        qrgs_base, qrgs_stride, rel_proj_bf16, k_cache_fp16, v_cache_fp16,
        nullptr, nullptr, pos_ptr, kv_cap, is_global, B, out_q8_u32, out_sc64);
}

extern "C" __global__ __launch_bounds__(384, 1) void ink2_flash_attn_tc_q8_b_i8(
    const float* __restrict__ qrgs_base,
    int32_t qrgs_stride,
    const uint32_t* __restrict__ rel_proj_bf16,
    const uint16_t* __restrict__ k_cache_i8_raw,
    const uint16_t* __restrict__ v_cache_i8_raw,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    const int32_t* __restrict__ pos_ptr,
    int32_t kv_cap,
    int32_t is_global,
    int32_t B,
    uint32_t* __restrict__ out_q8_u32,
    float* __restrict__ out_sc64) {
    pdl_wait();
    flash_attn_tc_q8_b_impl<true, false>(
        qrgs_base, qrgs_stride, rel_proj_bf16, k_cache_i8_raw, v_cache_i8_raw,
        k_scale, v_scale, pos_ptr, kv_cap, is_global, B, out_q8_u32, out_sc64);
}

// Wave-Balanced Long-Context Split-KV Tensor Core FlashAttention (`Grid: (8, ceil(B/64), G_b)`, `Block: 384`)
// Eliminates SM wave-tail quantization (`256` CTAs -> `1,280` CTAs = `16.0` exact waves on 80-SM PRO 4500 & `97%` wave eff on 110-SM PRO 5000)!
extern "C" __global__ __launch_bounds__(384, 1) void ink2_flash_attn_tc_split_b(
    const float* __restrict__ qrgs_base,
    int32_t qrgs_stride,
    const uint32_t* __restrict__ rel_proj_bf16,
    const uint16_t* __restrict__ k_cache_fp16,
    const uint16_t* __restrict__ v_cache_fp16,
    const int32_t* __restrict__ pos_ptr,
    int32_t kv_cap,
    int32_t is_global,
    int32_t B,
    uint32_t* __restrict__ out_part_o_f16x2,
    float2* __restrict__ out_part_ml) {
    pdl_wait();
    flash_attn_tc_q8_b_impl<false, true>(
        qrgs_base, qrgs_stride, rel_proj_bf16, k_cache_fp16, v_cache_fp16,
        nullptr, nullptr, pos_ptr, kv_cap, is_global, B, nullptr, nullptr, out_part_o_f16x2, out_part_ml);
}

extern "C" __global__ __launch_bounds__(384, 1) void ink2_flash_attn_tc_split_b_i8(
    const float* __restrict__ qrgs_base,
    int32_t qrgs_stride,
    const uint32_t* __restrict__ rel_proj_bf16,
    const uint16_t* __restrict__ k_cache_i8_raw,
    const uint16_t* __restrict__ v_cache_i8_raw,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    const int32_t* __restrict__ pos_ptr,
    int32_t kv_cap,
    int32_t is_global,
    int32_t B,
    uint32_t* __restrict__ out_part_o_f16x2,
    float2* __restrict__ out_part_ml) {
    pdl_wait();
    flash_attn_tc_q8_b_impl<true, true>(
        qrgs_base, qrgs_stride, rel_proj_bf16, k_cache_i8_raw, v_cache_i8_raw,
        k_scale, v_scale, pos_ptr, kv_cap, is_global, B, nullptr, nullptr, out_part_o_f16x2, out_part_ml);
}

// Online-Softmax Combine + Sigmoid Gate + `Q8_64` Quantizer (`Grid: (24, B)`, `Block: 64` = `2` warps, `1` warp per `64`-dim half!)
extern "C" __global__ void ink2_flash_attn_split_combine_q8_b(
    const uint32_t* __restrict__ part_o_f16x2, // [n_splits, B, 24, 64]
    const float2* __restrict__ part_ml,        // [n_splits, B, 24]
    const float* __restrict__ qrgs_base,       // [B, qrgs_stride]
    int32_t qrgs_stride,
    int32_t B,
    int32_t n_splits,
    uint32_t* __restrict__ out_q8_u32,         // [B, 768]
    float* __restrict__ out_sc64) {            // [B, 48]
    pdl_wait();
    const int h = blockIdx.x; // 0..23
    const int b = blockIdx.y; // 0..B-1
    if (h >= 24 || b >= B) return;

    const int half64 = threadIdx.x >> 5; // 0..1
    const int lane = threadIdx.x & 31;   // 0..31 (`f16x2` -> `64` dims!)
    const size_t stride_split_qh = static_cast<size_t>(B) * 24u;
    const size_t base_qh = static_cast<size_t>(b) * 24u + h;

    float m_max = -1e30f;
    for (int s = 0; s < n_splits; ++s) {
        const float2 ml = part_ml[static_cast<size_t>(s) * stride_split_qh + base_qh];
        if (ml.y > 0.0f && ml.x > m_max) {
            m_max = ml.x;
        }
    }

    float l_sum = 0.0f;
    float2 o_sum = make_float2(0.0f, 0.0f);
    for (int s = 0; s < n_splits; ++s) {
        const float2 ml = part_ml[static_cast<size_t>(s) * stride_split_qh + base_qh];
        if (ml.y > 0.0f) {
            const float alpha = ptx_exp(ml.x - m_max);
            const float w = ml.y * alpha;
            l_sum += w;
            const uint32_t op = part_o_f16x2[(static_cast<size_t>(s) * stride_split_qh + base_qh) * 64u + half64 * 32 + lane];
            o_sum.x = __fmaf_rn(fp16_bits_to_f32(static_cast<uint16_t>(op & 0xFFFFu)), w, o_sum.x);
            o_sum.y = __fmaf_rn(fp16_bits_to_f32(static_cast<uint16_t>(op >> 16)), w, o_sum.y);
        }
    }

    const float inv_l = ptx_rcp(fmaxf(l_sum, 1e-20f));
    const float2 g2 = reinterpret_cast<const float2*>(
        qrgs_base + static_cast<size_t>(b) * qrgs_stride + 3456 + h * kHd + half64 * 64)[lane];
    const float v0 = (o_sum.x * inv_l) * ptx_sigmoid(g2.x);
    const float v1 = (o_sum.y * inv_l) * ptx_sigmoid(g2.y);

    float amax = warp_max(fmaxf(fabsf(v0), fabsf(v1)));
    const float inv_q = (amax > 0.0f) ? (127.0f * ptx_rcp(amax)) : 0.0f;
    int q0 = __float2int_rn(v0 * inv_q);
    int q1 = __float2int_rn(v1 * inv_q);
    if (q0 < -127) q0 = -127;
    if (q0 > 127) q0 = 127;
    if (q1 < -127) q1 = -127;
    if (q1 > 127) q1 = 127;

    const uint32_t pair16 =
        static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(q0))) |
        (static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(q1))) << 8);
    const uint32_t pack32 = pair16 | (__shfl_xor_sync(0xFFFFFFFFu, pair16, 1) << 16);
    const int blk64_idx = h * 2 + half64;
    if ((lane & 1) == 0) {
        out_q8_u32[static_cast<size_t>(b) * 768u + blk64_idx * 16 + (lane >> 1)] = pack32;
    }
    if (lane == 0) {
        out_sc64[static_cast<size_t>(b) * 48u + blk64_idx] = amax * (1.0f / 127.0f);
    }
}

// 18.6 BF16 Tensor Core Router & Loop-Controller Down-Projection (`mma.m16n8k16.f32.bf16.bf16.f32`)
// Eliminates `35.5 ms` of SIMT BF16 GEMMs (`ctrl_down_r8_b` + `bf16_gemm_router_b`) at 2K!
extern "C" __global__ void ink2_f32_to_bf16_b(
    const float4* __restrict__ src_f4,   // [n_vec8 * 2]
    uint4* __restrict__ dst_bf16_u128,   // [n_vec8]
    int32_t n_vec8) {
    pdl_wait();
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vec8) return;
    const float4 a = src_f4[2 * idx];
    const float4 b = src_f4[2 * idx + 1];
    dst_bf16_u128[idx] = make_uint4(
        pack_bf16x2_rn(a.x, a.y),
        pack_bf16x2_rn(a.z, a.w),
        pack_bf16x2_rn(b.x, b.y),
        pack_bf16x2_rn(b.z, b.w));
}

// Grid: `(n_ctrls * B, 1)`, Block: `256` — 1-Pass SMEM+Register `(cur, msg)` BF16 normalizer (`0-ULP` exact, `-50%` DRAM reads)!
extern "C" __global__ void ink2_ctrl_norm_bf16_b(
    const float4* __restrict__ cur4_base,    // [n_ctrls * B, 576]
    const float4* __restrict__ msg4_base,    // [n_ctrls * B, 576]
    uint4* __restrict__ out_bf16_u128) {     // [n_ctrls * B, 576]
    pdl_wait();
    __shared__ float4 sm_vec4[576];          // 9,216 B (`6` CTAs/SM = `48` warps/SM preserved!)
    __shared__ float sm_red_a[8];
    __shared__ float sm_red_b[8];
    __shared__ float inv_ab[2];

    const int cb = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const float4* cur4 = cur4_base + static_cast<size_t>(cb) * 576u;
    const float4* msg4 = msg4_base + static_cast<size_t>(cb) * 576u;

    float4 vb_reg[3];
    float sq_a = 0.0f;
    float sq_b = 0.0f;
    #pragma unroll
    for (int p = 0; p < 3; ++p) {
        const int i = p * 256 + tid;
        if (i < 576) {
            const float4 va = cur4[i];
            const float4 vb = msg4[i];
            sm_vec4[i] = va;
            vb_reg[p] = vb;
            sq_a += va.x * va.x + va.y * va.y + va.z * va.z + va.w * va.w;
            sq_b += vb.x * vb.x + vb.y * vb.y + vb.z * vb.z + vb.w * vb.w;
        }
    }
    sq_a = warp_sum(sq_a);
    sq_b = warp_sum(sq_b);
    if (lane == 0) {
        sm_red_a[warp_id] = sq_a;
        sm_red_b[warp_id] = sq_b;
    }
    __syncthreads();
    if (warp_id == 0) {
        float sa = (lane < 8) ? sm_red_a[lane] : 0.0f;
        float sb = (lane < 8) ? sm_red_b[lane] : 0.0f;
        sa = warp_sum(sa);
        sb = warp_sum(sb);
        if (lane == 0) {
            inv_ab[0] = ptx_rsqrt(sa * (1.0f / static_cast<float>(kD)) + 1e-6f);
            inv_ab[1] = ptx_rsqrt(sb * (1.0f / static_cast<float>(kD)) + 1e-6f);
        }
    }
    __syncthreads();

    const float sc_a = inv_ab[0];
    const float sc_b = inv_ab[1];
    uint4* dst = out_bf16_u128 + static_cast<size_t>(cb) * 576u;

    for (int c8 = tid; c8 < 288; c8 += 256) {
        const float4 a0 = sm_vec4[2 * c8];
        const float4 a1 = sm_vec4[2 * c8 + 1];
        dst[c8] = make_uint4(
            pack_bf16x2_rn(a0.x * sc_a, a0.y * sc_a),
            pack_bf16x2_rn(a0.z * sc_a, a0.w * sc_a),
            pack_bf16x2_rn(a1.x * sc_a, a1.y * sc_a),
            pack_bf16x2_rn(a1.z * sc_a, a1.w * sc_a));
    }
    __syncthreads();

    #pragma unroll
    for (int p = 0; p < 3; ++p) {
        const int i = p * 256 + tid;
        if (i < 576) {
            sm_vec4[i] = vb_reg[p];
        }
    }
    __syncthreads();

    for (int c8 = tid; c8 < 288; c8 += 256) {
        const float4 b0 = sm_vec4[2 * c8];
        const float4 b1 = sm_vec4[2 * c8 + 1];
        dst[288 + c8] = make_uint4(
            pack_bf16x2_rn(b0.x * sc_b, b0.y * sc_b),
            pack_bf16x2_rn(b0.z * sc_b, b0.w * sc_b),
            pack_bf16x2_rn(b1.x * sc_b, b1.y * sc_b),
            pack_bf16x2_rn(b1.z * sc_b, b1.w * sc_b));
    }
}

// Double-Buffered `cp.async.cg` `64x64` Batched BF16 Tensor Core GEMM (`n_groups = 1` for Router, `n_ctrls` for CtrlDown)
// Grid: `(ceil(rows / 64), ceil(B / 64) * n_groups)`, Block: `256`
// Static SMEM: `20,480` B (`2` CTAs/SM -> `42.0 KiB <= 64 KiB` carveout -> `64 KiB` full L1 preserved!)
extern "C" __global__ void ink2_bf16_tc_gemm_b(
    const uint4* __restrict__ w_bf16_u128,   // [n_groups, rows, cols / 8]
    const uint4* __restrict__ x_bf16_u128,   // [n_groups, B, cols / 8]
    float* __restrict__ y_base,              // [n_groups, B, rows]
    int32_t rows,
    int32_t cols,
    int32_t B,
    int32_t t_blocks_per_group) {
    pdl_wait();
    __shared__ uint4 sm_w[2][64][5]; // 10,240 B (`K_STAGE = 32` BF16 cols = `4` `uint4`s, stride = 20 dwords -> 32-bank bijection!)
    __shared__ uint4 sm_x[2][64][5]; // 10,240 B

    const int grp = blockIdx.y / t_blocks_per_group;
    const int t_blk = blockIdx.y % t_blocks_per_group;
    const int r_cta = blockIdx.x * 64;
    const int t_cta = t_blk * 64;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int wr = warp_id >> 1; // 0..3
    const int wt = warp_id & 1;  // 0..1
    const int g = lane >> 2;     // 0..7
    const int t = lane & 3;      // 0..3

    const int cols8 = cols >> 3;
    const int stages = cols >> 5; // cols / 32 (72 for 2304, 144 for 4608)
    const int lr = tid >> 2;      // 0..63
    const int lq = tid & 3;       // 0..3
    const int gr = r_cta + lr;
    const int gt = t_cta + lr;

    const uint4* w_grp = w_bf16_u128 + static_cast<size_t>(grp) * rows * cols8;
    const uint4* x_grp = x_bf16_u128 + static_cast<size_t>(grp) * B * cols8;
    float* y_grp = y_base + static_cast<size_t>(grp) * B * rows;

    auto issue_stage = [&](int st, int ks) {
        cp_async16_cg_pred(&sm_w[st][lr][lq], w_grp + static_cast<size_t>(gr) * cols8 + ks * 4 + lq, gr < rows);
        cp_async16_cg_pred(&sm_x[st][lr][lq], x_grp + static_cast<size_t>(gt) * cols8 + ks * 4 + lq, gt < B);
        cp_async_commit_grp();
    };

    float acc[4][4] = {};
    issue_stage(0, 0);

    for (int ks = 0; ks < stages; ++ks) {
        if (ks + 1 < stages) {
            issue_stage((ks + 1) & 1, ks + 1);
            cp_async_wait_grp<1>();
        } else {
            cp_async_wait_grp<0>();
        }
        __syncthreads();

        const int st = ks & 1;
        const uint32_t* w0_row = reinterpret_cast<const uint32_t*>(sm_w[st][wr * 16 + g]);
        const uint32_t* w1_row = reinterpret_cast<const uint32_t*>(sm_w[st][wr * 16 + 8 + g]);

        #pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            const unsigned a0 = w0_row[sub * 8 + t];
            const unsigned a1 = w1_row[sub * 8 + t];
            const unsigned a2 = w0_row[sub * 8 + 4 + t];
            const unsigned a3 = w1_row[sub * 8 + 4 + t];
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const uint32_t* x_row = reinterpret_cast<const uint32_t*>(sm_x[st][wt * 32 + nt * 8 + g]);
                mma_bf16_m16n8k16(
                    acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3],
                    a0, a1, a2, a3,
                    x_row[sub * 8 + t], x_row[sub * 8 + 4 + t]);
            }
        }
        __syncthreads();
    }

    const int out_r0 = r_cta + wr * 16 + g;
    const int out_r1 = out_r0 + 8;
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int out_t = t_cta + wt * 32 + nt * 8 + 2 * t + c;
            if (out_t < B) {
                float* dst = y_grp + static_cast<size_t>(out_t) * rows;
                if (out_r0 < rows) dst[out_r0] = acc[nt][c];
                if (out_r1 < rows) dst[out_r1] = acc[nt][2 + c];
            }
        }
    }
}

// 18.7 `2-Term` (`g_hi + g_lo/256`) BF16 Tensor Core Loop-Controller Apply (`ink2_ctrl_apply_tc_b`)
// Grid: `(72, ceil(B / 64) * n_ctrls)`, Block: `256` (`32` channels x `64` tokens, `gain` & `forget` paired in registers!)
// Static SMEM: `27,648` B (`2` CTAs/SM -> `56.0 KiB <= 64 KiB` carveout -> `64 KiB` full L1 preserved!)
extern "C" __global__ void ink2_ctrl_apply_tc_b(
    float* __restrict__ cur_inout_base,         // [n_ctrls, B, D]
    const float* __restrict__ msg_base,         // [n_ctrls, B, D]
    const float4* __restrict__ g_vec4_base,     // [n_ctrls, B, 16]
    const uint4* __restrict__ gain_bf16_u128,   // [n_ctrls, D, 8]
    const uint4* __restrict__ forget_bf16_u128, // [n_ctrls, D, 8]
    int32_t B,
    int32_t t_blocks_per_ctrl) {
    pdl_wait();
    __shared__ uint4 sm_gf[64][9];    // 9,216 B (`32` gain + `32` forget rows x `8` `uint4`s, stride = 36 dwords -> 32-bank bijection!)
    __shared__ uint4 sm_ghi[64][9];   // 9,216 B
    __shared__ uint4 sm_glo[64][9];   // 9,216 B

    const int grp = blockIdx.y / t_blocks_per_ctrl;
    const int t_blk = blockIdx.y % t_blocks_per_ctrl;
    const int c_cta = blockIdx.x * 32; // 0..2272 (`72 * 32 = 2304`)
    const int t_cta = t_blk * 64;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int wr = warp_id >> 1; // 0..3 (`8` channels = `8` gain + `8` forget rows per warp!)
    const int wt = warp_id & 1;  // 0..1 (`32` tokens per warp!)
    const int g = lane >> 2;     // 0..7
    const int t = lane & 3;      // 0..3

    // 1. Async-load `32` `gain` rows and `32` `forget` rows (`64 x 8 = 512` `uint4`s = `2` per thread!)
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int lin = p * 256 + tid;
        const int sr = lin >> 3;   // 0..63
        const int col8 = lin & 7;  // 0..7
        const int wr_s = sr >> 4;
        const int is_forget = (sr >> 3) & 1;
        const int ch_local = wr_s * 8 + (sr & 7);
        const size_t row_idx = static_cast<size_t>(grp) * kD + c_cta + ch_local;
        const uint4* src_w = is_forget ? forget_bf16_u128 : gain_bf16_u128;
        cp_async16_cg(&sm_gf[sr][col8], src_w + row_idx * 8u + col8);
    }

    // 2. Load & split `64` token `g_vec` rows into `2-term` `(g_hi, g_lo * 256)` BF16 (`64 x 8 = 512` `uint4`s = `2` per thread!)
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int lin = p * 256 + tid;
        const int tr = lin >> 3;   // 0..63
        const int col8 = lin & 7;  // 0..7
        const int gt = t_cta + tr;
        uint4 hi4 = make_uint4(0u, 0u, 0u, 0u);
        uint4 lo4 = make_uint4(0u, 0u, 0u, 0u);
        if (gt < B) {
            const float4* gv4 = g_vec4_base + (static_cast<size_t>(grp) * B + gt) * 16u + col8 * 2;
            const float4 a = gv4[0];
            const float4 b = gv4[1];
            hi4 = make_uint4(
                pack_bf16x2_rn(a.x, a.y),
                pack_bf16x2_rn(a.z, a.w),
                pack_bf16x2_rn(b.x, b.y),
                pack_bf16x2_rn(b.z, b.w));
            auto res_pair = [](float v0, float v1, uint32_t h2) -> uint32_t {
                const float r0 = (v0 - bf16_to_f32(static_cast<uint16_t>(h2 & 0xFFFFu))) * 256.0f;
                const float r1 = (v1 - bf16_to_f32(static_cast<uint16_t>(h2 >> 16))) * 256.0f;
                return pack_bf16x2_rn(r0, r1);
            };
            lo4 = make_uint4(
                res_pair(a.x, a.y, hi4.x),
                res_pair(a.z, a.w, hi4.y),
                res_pair(b.x, b.y, hi4.z),
                res_pair(b.z, b.w, hi4.w));
        }
        sm_ghi[tr][col8] = hi4;
        sm_glo[tr][col8] = lo4;
    }

    cp_async_commit_grp();
    cp_async_wait_grp<0>();
    __syncthreads();

    float acc_hi[4][4] = {};
    float acc_lo[4][4] = {};
    const uint32_t* w_gain = reinterpret_cast<const uint32_t*>(sm_gf[wr * 16 + g]);
    const uint32_t* w_forg = reinterpret_cast<const uint32_t*>(sm_gf[wr * 16 + 8 + g]);

    #pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        const unsigned a0 = w_gain[sub * 8 + t];
        const unsigned a1 = w_forg[sub * 8 + t];
        const unsigned a2 = w_gain[sub * 8 + 4 + t];
        const unsigned a3 = w_forg[sub * 8 + 4 + t];
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int tok_row = wt * 32 + nt * 8 + g;
            const uint32_t* xh = reinterpret_cast<const uint32_t*>(sm_ghi[tok_row]);
            const uint32_t* xl = reinterpret_cast<const uint32_t*>(sm_glo[tok_row]);
            mma_bf16_m16n8k16(
                acc_hi[nt][0], acc_hi[nt][1], acc_hi[nt][2], acc_hi[nt][3],
                a0, a1, a2, a3, xh[sub * 8 + t], xh[sub * 8 + 4 + t]);
            mma_bf16_m16n8k16(
                acc_lo[nt][0], acc_lo[nt][1], acc_lo[nt][2], acc_lo[nt][3],
                a0, a1, a2, a3, xl[sub * 8 + t], xl[sub * 8 + 4 + t]);
        }
    }

    // Transpose `ptx_tanh(gain_acc)` and `ptx_tanh(forg_acc)` through dead `sm_ghi`/`sm_glo` (`8,448 B <= 9,216 B` each!)
    // so `lane = 0..31` reads & writes `32` contiguous `128B`-aligned channels of `cur_inout` & `msg` (`4x` fewer DRAM sectors, `0-ULP` exact)!
    __syncthreads();
    float (*sm_tg)[33] = reinterpret_cast<float (*)[33]>(sm_ghi);
    float (*sm_tf)[33] = reinterpret_cast<float (*)[33]>(sm_glo);
    const int ch_local = wr * 8 + g;
    constexpr float kInv256 = 1.0f / 256.0f;
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        #pragma unroll
        for (int c_sub = 0; c_sub < 2; ++c_sub) {
            const int tr = wt * 32 + nt * 8 + 2 * t + c_sub;
            const float gain_acc = acc_hi[nt][c_sub] + acc_lo[nt][c_sub] * kInv256;
            const float forg_acc = acc_hi[nt][2 + c_sub] + acc_lo[nt][2 + c_sub] * kInv256;
            sm_tg[tr][ch_local] = ptx_tanh(gain_acc);
            sm_tf[tr][ch_local] = ptx_tanh(forg_acc);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        const int tr = p * 8 + warp_id;
        const int gt = t_cta + tr;
        if (gt < B) {
            const size_t idx = (static_cast<size_t>(grp) * B + gt) * kD + c_cta + lane;
            const float cur_v = cur_inout_base[idx];
            cur_inout_base[idx] = cur_v - sm_tf[tr][lane] * cur_v + sm_tg[tr][lane] * msg_base[idx];
        }
    }
}

// 18.8 `0.0000-ULP` `float4`-Vectorized `Top-8` MoE Combine (`Grid: (3, B)`, `Block: 192` -> `3 * 192 = 576` `float4`s = `2304`!)
extern "C" __global__ void ink2_moe_combine_top8_vec4_b(
    const float4* __restrict__ pair_y4,  // [B * 8, 576]
    const float4* __restrict__ sel_wts4, // [B * 2]
    float4* __restrict__ moe_out4,       // [B, 576]
    int32_t B) {
    pdl_wait();
    pdl_trigger();
    const int d4 = blockIdx.x * 192 + threadIdx.x;
    const int b = blockIdx.y;
    if (d4 >= 576 || b >= B) return;

    const float4 w03 = sel_wts4[2 * b];
    const float4 w47 = sel_wts4[2 * b + 1];
    const float w[8] = {w03.x, w03.y, w03.z, w03.w, w47.x, w47.y, w47.z, w47.w};

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const size_t base_pair = static_cast<size_t>(b) * 8u;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float4 y = pair_y4[(base_pair + k) * 576u + d4];
        const float wk = w[k];
        sum.x = __fmaf_rn(y.x, wk, sum.x);
        sum.y = __fmaf_rn(y.y, wk, sum.y);
        sum.z = __fmaf_rn(y.z, wk, sum.z);
        sum.w = __fmaf_rn(y.w, wk, sum.w);
    }
    moe_out4[static_cast<size_t>(b) * 576u + d4] = sum;
}

// Speculative verify: `ink2_moe_combine_top8_vec4_b` + the MLP `ink2_short_conv4_b` (no dynamic
// taps, residual add into `cur`) per channel in one launch. The top-8 sum keeps the `k = 0..7`
// FMA order per channel and the conv step is the same `eval_conv4_step`, so both outputs are
// bit-identical to the two kernels. `Grid (ceil(D / 256), 1)`, `Block 256`.
extern "C" __global__ void ink2_moe_combine_conv4_b(
    const float* __restrict__ pair_y,     // [B * 8, D]
    const float* __restrict__ sel_wts,    // [B, 8]
    float* __restrict__ moe_out,          // [B, D]
    const float4* __restrict__ w_conv4,   // [D]
    float* __restrict__ hist3,            // [D * 3]
    float* cur,                           // [B, D] residual in, conv output out
    float* __restrict__ seq_out,          // nullable (`[3 + B, D]`)
    int32_t B) {
    pdl_wait();
    pdl_trigger();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kD) return;
    float* h = hist3 + c * 3;
    float h0 = h[0];
    float h1 = h[1];
    float h2 = h[2];
    if (seq_out != nullptr) {
        seq_out[c] = h0;
        seq_out[kD + c] = h1;
        seq_out[2 * kD + c] = h2;
    }
    const float4 w = w_conv4[c];
    const float4 no_dyn = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int b = 0; b < B; ++b) {
        float x = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            x = __fmaf_rn(pair_y[(static_cast<size_t>(b) * 8u + k) * kD + c], sel_wts[b * 8 + k], x);
        }
        moe_out[static_cast<size_t>(b) * kD + c] = x;
        if (seq_out != nullptr) {
            seq_out[static_cast<size_t>(3 + b) * kD + c] = x;
        }
        const float res = cur[static_cast<size_t>(b) * kD + c];
        const float y = eval_conv4_step(w, h0, h1, h2, x, 0, no_dyn, 1, res);
        h0 = h1;
        h1 = h2;
        h2 = x;
        cur[static_cast<size_t>(b) * kD + c] = y;
    }
    h[0] = h0;
    h[1] = h1;
    h[2] = h2;
}

// 18.9 `0-ULP` Multi-Query (`n_q = 1..4`) `100%` Occupancy (`<= 40` regs, `48` warps/SM) `float4` `depth_mix`
// Eliminates `3x–4x` redundant `blocks_buf` DRAM sweeps across `n_q = 3` (Global Attn) and `n_q = 4` (Loop Msgs)!
// Grid: `(1, B_out)`, Block: `256`
extern "C" __global__ void ink2_depth_mix_multi_b(
    const float* __restrict__ blocks_buf, // [max_blocks, stride_B, D]
    int32_t n_blocks,
    int32_t stride_B,
    int32_t b_offset,
    int32_t B_out,
    const float* __restrict__ cur_opt,    // [stride_B, D] if `append_cur == 1`
    int32_t append_cur,
    const float* __restrict__ queries,    // [n_q, D] (`<= 36 KB` -> 100% L1 cache resident!)
    float* __restrict__ out_mixed,        // [n_q, B_out, D] (or `[3, B_out, D]` if `broadcast3`)
    int32_t broadcast3,
    int32_t n_q) {
    pdl_wait();
    pdl_trigger();
    __shared__ float smem_sq[6][8];
    __shared__ float smem_dq[4][6][8];
    __shared__ float smem_probs[4][6];
    __shared__ float smem_inv[6];
    __shared__ float smem_scores[6];

    const int b_out = blockIdx.y;
    const int b_src = b_offset + b_out;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int total_states = n_blocks + append_cur;
    constexpr float inv_sqrt_d = 0.020833333333333332f;

    for (int s = 0; s < total_states; ++s) {
        // Same element -> thread mapping and accumulation order as `ink2_depth_mix` (#88).
        const float* st = (s < n_blocks)
            ? (blocks_buf + (static_cast<size_t>(s) * stride_B + b_src) * kD)
            : (cur_opt + static_cast<size_t>(b_src) * kD);
        float sq = 0.0f;
        float dq0 = 0.0f, dq1 = 0.0f, dq2 = 0.0f, dq3 = 0.0f;
        #pragma unroll
        for (int t = 0; t < 9; ++t) {
            const int i = tid + t * 256;
            const float v = st[i];
            sq += v * v;
            dq0 += v * queries[i];
            if (n_q > 1) dq1 += v * queries[kD + i];
            if (n_q > 2) dq2 += v * queries[2 * kD + i];
            if (n_q > 3) dq3 += v * queries[3 * kD + i];
        }
        sq = warp_sum(sq);
        dq0 = warp_sum(dq0);
        if (n_q > 1) dq1 = warp_sum(dq1);
        if (n_q > 2) dq2 = warp_sum(dq2);
        if (n_q > 3) dq3 = warp_sum(dq3);
        if (lane == 0) {
            smem_sq[s][warp_id] = sq;
            smem_dq[0][s][warp_id] = dq0;
            if (n_q > 1) smem_dq[1][s][warp_id] = dq1;
            if (n_q > 2) smem_dq[2][s][warp_id] = dq2;
            if (n_q > 3) smem_dq[3][s][warp_id] = dq3;
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        for (int s = 0; s < total_states; ++s) {
            float total_sq = (lane < 8) ? smem_sq[s][lane] : 0.0f;
            total_sq = warp_sum(total_sq);
            if (lane == 0) {
                smem_inv[s] = ptx_rsqrt(total_sq * (1.0f / static_cast<float>(kD)) + 1e-6f);
            }
        }
        __syncwarp();
        for (int qi = 0; qi < n_q; ++qi) {
            for (int s = 0; s < total_states; ++s) {
                float total_dq = (lane < 8) ? smem_dq[qi][s][lane] : 0.0f;
                total_dq = warp_sum(total_dq);
                if (lane == 0) {
                    smem_scores[s] = total_dq * smem_inv[s] * inv_sqrt_d;
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
                    smem_scores[s] = e;
                    z += e;
                }
                const float inv_z = ptx_rcp(z);
                for (int s = 0; s < total_states; ++s) {
                    smem_probs[qi][s] = smem_scores[s] * inv_z * smem_inv[s];
                }
            }
            __syncwarp();
        }
    }
    __syncthreads();

    const size_t plane_stride4 = static_cast<size_t>(B_out) * 576u;
    float4* dst_base4 = reinterpret_cast<float4*>(out_mixed + static_cast<size_t>(b_out) * kD);
    #pragma unroll
    for (int p = 0; p < 3; ++p) {
        const int i4 = p * 256 + tid;
        if (i4 < 576) {
            float4 acc0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 acc1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 acc2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 acc3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int s = 0; s < total_states; ++s) {
                const float4* st4 = reinterpret_cast<const float4*>((s < n_blocks)
                    ? (blocks_buf + (static_cast<size_t>(s) * stride_B + b_src) * kD)
                    : (cur_opt + static_cast<size_t>(b_src) * kD));
                const float4 sv = st4[i4];
                const float pr0 = smem_probs[0][s];
                acc0.x += pr0 * sv.x; acc0.y += pr0 * sv.y; acc0.z += pr0 * sv.z; acc0.w += pr0 * sv.w;
                if (n_q > 1) {
                    const float pr1 = smem_probs[1][s];
                    acc1.x += pr1 * sv.x; acc1.y += pr1 * sv.y; acc1.z += pr1 * sv.z; acc1.w += pr1 * sv.w;
                }
                if (n_q > 2) {
                    const float pr2 = smem_probs[2][s];
                    acc2.x += pr2 * sv.x; acc2.y += pr2 * sv.y; acc2.z += pr2 * sv.z; acc2.w += pr2 * sv.w;
                }
                if (n_q > 3) {
                    const float pr3 = smem_probs[3][s];
                    acc3.x += pr3 * sv.x; acc3.y += pr3 * sv.y; acc3.z += pr3 * sv.z; acc3.w += pr3 * sv.w;
                }
            }
            dst_base4[i4] = acc0;
            if (broadcast3) {
                dst_base4[plane_stride4 + i4] = acc0;
                dst_base4[2u * plane_stride4 + i4] = acc0;
            } else {
                if (n_q > 1) dst_base4[plane_stride4 + i4] = acc1;
                if (n_q > 2) dst_base4[2u * plane_stride4 + i4] = acc2;
                if (n_q > 3) dst_base4[3u * plane_stride4 + i4] = acc3;
            }
        }
    }
}

// 18.10 `FP16` `uint4` (`8` `FP16`s = `16B` `LDG.E.128`) Ordered `Top-8` MoE Combine (`Grid: (1, B)`, `Block: 288` -> `288 * 8 = 2304`!)
extern "C" __global__ void ink2_moe_combine_top8_f16_b(
    const uint4* __restrict__ pair_y_f16_u128, // [B * 8, 288]
    const float4* __restrict__ sel_wts4,       // [B * 2]
    float4* __restrict__ moe_out4,             // [B, 576]
    int32_t B) {
    pdl_wait();
    const int d8 = threadIdx.x; // 0..287
    const int b = blockIdx.y;
    if (d8 >= 288 || b >= B) return;

    const float4 w03 = sel_wts4[2 * b];
    const float4 w47 = sel_wts4[2 * b + 1];
    const float w[8] = {w03.x, w03.y, w03.z, w03.w, w47.x, w47.y, w47.z, w47.w};

    float4 sum0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const size_t base_pair = static_cast<size_t>(b) * 8u;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const uint4 u = pair_y_f16_u128[(base_pair + k) * 288u + d8];
        const float wk = w[k];
        sum0.x = __fmaf_rn(fp16_bits_to_f32(static_cast<uint16_t>(u.x & 0xFFFFu)), wk, sum0.x);
        sum0.y = __fmaf_rn(fp16_bits_to_f32(static_cast<uint16_t>(u.x >> 16)), wk, sum0.y);
        sum0.z = __fmaf_rn(fp16_bits_to_f32(static_cast<uint16_t>(u.y & 0xFFFFu)), wk, sum0.z);
        sum0.w = __fmaf_rn(fp16_bits_to_f32(static_cast<uint16_t>(u.y >> 16)), wk, sum0.w);
        sum1.x = __fmaf_rn(fp16_bits_to_f32(static_cast<uint16_t>(u.z & 0xFFFFu)), wk, sum1.x);
        sum1.y = __fmaf_rn(fp16_bits_to_f32(static_cast<uint16_t>(u.z >> 16)), wk, sum1.y);
        sum1.z = __fmaf_rn(fp16_bits_to_f32(static_cast<uint16_t>(u.w & 0xFFFFu)), wk, sum1.z);
        sum1.w = __fmaf_rn(fp16_bits_to_f32(static_cast<uint16_t>(u.w >> 16)), wk, sum1.w);
    }
    float4* dst = moe_out4 + static_cast<size_t>(b) * 576u + 2 * d8;
    dst[0] = sum0;
    dst[1] = sum1;
}

// 18.11 `0-ULP` Fused `depth_mix_multi_b` + `ctrl_norm_bf16_b` (`Grid: (1, B_out)`, `Block: 256`)
// Writes `out_mixed` (`[n_ctrls, B_out, D]`) AND normalized `(cur, msg)` BF16 (`[n_ctrls, B_out, 4608]`) in ONE launch,
// eliminating `60` `ctrl_norm_bf16_b` launches per prefill chunk AND avoiding re-reading `out_mixed` from DRAM!
extern "C" __global__ void ink2_depth_mix_multi_ctrl_norm_bf16_b(
    const float* __restrict__ blocks_buf, // [max_blocks, stride_B, D]
    int32_t n_blocks,
    int32_t stride_B,
    int32_t b_offset,
    int32_t B_out,
    const float* __restrict__ cur_opt,    // [stride_B, D]
    int32_t append_cur,
    const float* __restrict__ queries,    // [n_q, D] (`n_q = 1` or `3`)
    float* __restrict__ out_mixed,        // [n_ctrls, B_out, D] (`n_ctrls = broadcast3 ? 3 : n_q`)
    int32_t broadcast3,
    int32_t n_q,
    const float4* __restrict__ msg4_base, // [n_ctrls, B_out, 576]
    uint4* __restrict__ out_bf16_u128) {  // [n_ctrls, B_out, 576]
    pdl_wait();
    __shared__ float4 sm_vec4[576];       // 9,216 B (`5` CTAs/SM = `40` warps/SM, `64 KiB` L1!)
    __shared__ float smem_sq[6][8];
    __shared__ float smem_dq[3][6][8];
    __shared__ float smem_probs[3][6];
    __shared__ float smem_inv[6];
    __shared__ float smem_scores[6];
    __shared__ float sm_red_a[3][8];
    __shared__ float sm_red_b[3][8];
    __shared__ float sm_sc_a[3];
    __shared__ float sm_sc_b[3];

    const int b_out = blockIdx.y;
    const int b_src = b_offset + b_out;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int total_states = n_blocks + append_cur;
    const int n_ctrls = broadcast3 ? 3 : n_q;
    constexpr float inv_sqrt_d = 0.020833333333333332f;

    for (int s = 0; s < total_states; ++s) {
        const float* st = (s < n_blocks)
            ? (blocks_buf + (static_cast<size_t>(s) * stride_B + b_src) * kD)
            : (cur_opt + static_cast<size_t>(b_src) * kD);
        float sq = 0.0f;
        float dq0 = 0.0f, dq1 = 0.0f, dq2 = 0.0f;
        #pragma unroll
        for (int t = 0; t < 9; ++t) {
            const int i = tid + t * 256;
            const float v = st[i];
            sq += v * v;
            dq0 += v * queries[i];
            if (n_q > 1) dq1 += v * queries[kD + i];
            if (n_q > 2) dq2 += v * queries[2 * kD + i];
        }
        sq = warp_sum(sq);
        dq0 = warp_sum(dq0);
        if (n_q > 1) dq1 = warp_sum(dq1);
        if (n_q > 2) dq2 = warp_sum(dq2);
        if (lane == 0) {
            smem_sq[s][warp_id] = sq;
            smem_dq[0][s][warp_id] = dq0;
            if (n_q > 1) smem_dq[1][s][warp_id] = dq1;
            if (n_q > 2) smem_dq[2][s][warp_id] = dq2;
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        for (int s = 0; s < total_states; ++s) {
            float total_sq = (lane < 8) ? smem_sq[s][lane] : 0.0f;
            total_sq = warp_sum(total_sq);
            if (lane == 0) {
                smem_inv[s] = ptx_rsqrt(total_sq * (1.0f / static_cast<float>(kD)) + 1e-6f);
            }
        }
        __syncwarp();
        for (int qi = 0; qi < n_q; ++qi) {
            for (int s = 0; s < total_states; ++s) {
                float total_dq = (lane < 8) ? smem_dq[qi][s][lane] : 0.0f;
                total_dq = warp_sum(total_dq);
                if (lane == 0) {
                    smem_scores[s] = total_dq * smem_inv[s] * inv_sqrt_d;
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
                    smem_scores[s] = e;
                    z += e;
                }
                const float inv_z = ptx_rcp(z);
                for (int s = 0; s < total_states; ++s) {
                    smem_probs[qi][s] = smem_scores[s] * inv_z * smem_inv[s];
                }
            }
            __syncwarp();
        }
    }
    __syncthreads();

    const size_t plane_stride4 = static_cast<size_t>(B_out) * 576u;
    float4* dst_base4 = reinterpret_cast<float4*>(out_mixed + static_cast<size_t>(b_out) * kD);

    float4 acc0_r[3] = {};
    float4 acc1_r[3] = {};
    float4 acc2_r[3] = {};
    float sq_a0 = 0.0f, sq_a1 = 0.0f, sq_a2 = 0.0f;
    #pragma unroll
    for (int p = 0; p < 3; ++p) {
        const int i4 = p * 256 + tid;
        if (i4 < 576) {
            float4 acc0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 acc1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 acc2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int s = 0; s < total_states; ++s) {
                const float4* st4 = reinterpret_cast<const float4*>((s < n_blocks)
                    ? (blocks_buf + (static_cast<size_t>(s) * stride_B + b_src) * kD)
                    : (cur_opt + static_cast<size_t>(b_src) * kD));
                const float4 sv = st4[i4];
                const float pr0 = smem_probs[0][s];
                acc0.x += pr0 * sv.x; acc0.y += pr0 * sv.y; acc0.z += pr0 * sv.z; acc0.w += pr0 * sv.w;
                if (n_q > 1) {
                    const float pr1 = smem_probs[1][s];
                    acc1.x += pr1 * sv.x; acc1.y += pr1 * sv.y; acc1.z += pr1 * sv.z; acc1.w += pr1 * sv.w;
                }
                if (n_q > 2) {
                    const float pr2 = smem_probs[2][s];
                    acc2.x += pr2 * sv.x; acc2.y += pr2 * sv.y; acc2.z += pr2 * sv.z; acc2.w += pr2 * sv.w;
                }
            }
            dst_base4[i4] = acc0;
            sq_a0 += acc0.x * acc0.x + acc0.y * acc0.y + acc0.z * acc0.z + acc0.w * acc0.w;
            acc0_r[p] = acc0;
            if (broadcast3) {
                dst_base4[plane_stride4 + i4] = acc0;
                dst_base4[2u * plane_stride4 + i4] = acc0;
                acc1_r[p] = acc0;
                acc2_r[p] = acc0;
            } else {
                if (n_q > 1) {
                    dst_base4[plane_stride4 + i4] = acc1;
                    sq_a1 += acc1.x * acc1.x + acc1.y * acc1.y + acc1.z * acc1.z + acc1.w * acc1.w;
                    acc1_r[p] = acc1;
                }
                if (n_q > 2) {
                    dst_base4[2u * plane_stride4 + i4] = acc2;
                    sq_a2 += acc2.x * acc2.x + acc2.y * acc2.y + acc2.z * acc2.z + acc2.w * acc2.w;
                    acc2_r[p] = acc2;
                }
            }
        }
    }

    sq_a0 = warp_sum(sq_a0);
    if (broadcast3) {
        sq_a1 = sq_a0;
        sq_a2 = sq_a0;
    } else {
        if (n_q > 1) sq_a1 = warp_sum(sq_a1);
        if (n_q > 2) sq_a2 = warp_sum(sq_a2);
    }
    if (lane == 0) {
        sm_red_a[0][warp_id] = sq_a0;
        if (n_ctrls > 1) {
            sm_red_a[1][warp_id] = sq_a1;
            sm_red_a[2][warp_id] = sq_a2;
        }
    }

    for (int ci = 0; ci < n_ctrls; ++ci) {
        const size_t cb = static_cast<size_t>(ci) * B_out + b_out;
        const float4* msg4 = msg4_base + cb * 576u;
        float4 vb_reg[3];
        float sq_b = 0.0f;
        #pragma unroll
        for (int p = 0; p < 3; ++p) {
            const int i4 = p * 256 + tid;
            if (i4 < 576) {
                const float4 vb = msg4[i4];
                vb_reg[p] = vb;
                sq_b += vb.x * vb.x + vb.y * vb.y + vb.z * vb.z + vb.w * vb.w;
            }
        }
        sq_b = warp_sum(sq_b);
        if (lane == 0) {
            sm_red_b[ci][warp_id] = sq_b;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sa = (lane < 8) ? sm_red_a[ci][lane] : 0.0f;
            float sb = (lane < 8) ? sm_red_b[ci][lane] : 0.0f;
            sa = warp_sum(sa);
            sb = warp_sum(sb);
            if (lane == 0) {
                sm_sc_a[ci] = ptx_rsqrt(sa * (1.0f / static_cast<float>(kD)) + 1e-6f);
                sm_sc_b[ci] = ptx_rsqrt(sb * (1.0f / static_cast<float>(kD)) + 1e-6f);
            }
        }
        #pragma unroll
        for (int p = 0; p < 3; ++p) {
            const int i4 = p * 256 + tid;
            if (i4 < 576) {
                sm_vec4[i4] = (ci == 0) ? acc0_r[p] : ((ci == 1) ? acc1_r[p] : acc2_r[p]);
            }
        }
        __syncthreads();

        const float sc_a = sm_sc_a[ci];
        const float sc_b = sm_sc_b[ci];
        uint4* dst = out_bf16_u128 + cb * 576u;
        for (int c8 = tid; c8 < 288; c8 += 256) {
            const float4 a0 = sm_vec4[2 * c8];
            const float4 a1 = sm_vec4[2 * c8 + 1];
            dst[c8] = make_uint4(
                pack_bf16x2_rn(a0.x * sc_a, a0.y * sc_a),
                pack_bf16x2_rn(a0.z * sc_a, a0.w * sc_a),
                pack_bf16x2_rn(a1.x * sc_a, a1.y * sc_a),
                pack_bf16x2_rn(a1.z * sc_a, a1.w * sc_a));
        }
        __syncthreads();

        #pragma unroll
        for (int p = 0; p < 3; ++p) {
            const int i4 = p * 256 + tid;
            if (i4 < 576) {
                sm_vec4[i4] = vb_reg[p];
            }
        }
        __syncthreads();

        for (int c8 = tid; c8 < 288; c8 += 256) {
            const float4 b0 = sm_vec4[2 * c8];
            const float4 b1 = sm_vec4[2 * c8 + 1];
            dst[288 + c8] = make_uint4(
                pack_bf16x2_rn(b0.x * sc_b, b0.y * sc_b),
                pack_bf16x2_rn(b0.z * sc_b, b0.w * sc_b),
                pack_bf16x2_rn(b1.x * sc_b, b1.y * sc_b),
                pack_bf16x2_rn(b1.z * sc_b, b1.w * sc_b));
        }
        __syncthreads();
    }
}

// Experimental lossless device representation of low-alphabet E4M3 expert weights.
// Each block expands part of one expert and preserves the original byte-addressed layout.
extern "C" __global__ void ink2_expand_mxfp8_indices(
    const unsigned char* __restrict__ plane, ushort4* __restrict__ output,
    const int* __restrict__ selected, unsigned vectors_per_expert) {
    pdl_wait();
    __shared__ unsigned short pair[256];
    const unsigned lane = threadIdx.x;
    pair[lane] = plane[lane & 15] | (unsigned(plane[lane >> 4]) << 8);
    __syncthreads();
    const unsigned expert = selected ? unsigned(selected[blockIdx.y]) : blockIdx.y;
    const unsigned offset = expert * vectors_per_expert;
    const uchar4* indices = reinterpret_cast<const uchar4*>(plane + 16);
    for (unsigned i = blockIdx.x * blockDim.x + lane; i < vectors_per_expert; i += blockDim.x * gridDim.x) {
        const uchar4 v = indices[offset + i];
        output[offset + i] = make_ushort4(pair[v.x], pair[v.y], pair[v.z], pair[v.w]);
    }
}
