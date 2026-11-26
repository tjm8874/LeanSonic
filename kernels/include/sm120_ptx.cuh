// LeanSonic — Blackwell (`sm_120`) Inline PTX & Zero-ULP Primitives
#pragma once

#include <cuda_fp16.h>
#include <stdint.h>
#include "leansonic_verified.cuh"

namespace leansonic::ptx {

// ----------------------------------------------------------------------------
// 1. Programmatic Dependent Launch (PDL — `sm_90+` / `sm_120`)
//    NOTE: Never annotate upstream producer outputs read before `pdl_wait()`
//    with `__restrict__` or `__ldg` so reads stay on coherent L2 (`LDG.E`).
// ----------------------------------------------------------------------------
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

// ----------------------------------------------------------------------------
// 2. 16B `cp.async.cg` Global -> Shared Streaming (Bypasses L1 Pollution)
// ----------------------------------------------------------------------------
__device__ __forceinline__ void cp_async_cg_16B(uint32_t smem_addr, const void* gmem_ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_ptr)
        : "memory");
#else
    uint4* dst = reinterpret_cast<uint4*>(static_cast<uintptr_t>(smem_addr));
    *dst = *reinterpret_cast<const uint4*>(gmem_ptr);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

template <int kKeepStages>
__device__ __forceinline__ void cp_async_wait_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(kKeepStages) : "memory");
#endif
}

// ----------------------------------------------------------------------------
// 3. Fast Inline PTX Transcendentals (Eliminates SASS `CALL.REL.NOINC`)
// ----------------------------------------------------------------------------
__device__ __forceinline__ float tanh_approx_f32(float x) {
#if defined(__CUDA_ARCH__)
    float y;
    asm("tanh.approx.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
#else
    return tanhf(x);
#endif
}

__device__ __forceinline__ float ex2_approx_ftz_f32(float x) {
#if defined(__CUDA_ARCH__)
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
#else
    return exp2f(x);
#endif
}

__device__ __forceinline__ float rcp_approx_ftz_f32(float x) {
#if defined(__CUDA_ARCH__)
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
#else
    return 1.0f / x;
#endif
}

// ----------------------------------------------------------------------------
// 4. 22-Bit (`hi + lo / 2048`) FP16 Query Split for `mma.m16n8k16` Decode
// ----------------------------------------------------------------------------
struct Half22Pair {
    half hi;
    half lo;
};

__device__ __forceinline__ Half22Pair split_fp32_to_22bit_fp16(float q_f32) {
    half hi = __float2half_rn(q_f32);
    float rem = (q_f32 - __half2float(hi)) * 2048.0f;
    half lo = __float2half_rn(rem);
    return {hi, lo};
}

// ----------------------------------------------------------------------------
// 5. Verified 3-Op `__byte_perm` LUT: Packed 4-bit `E2M1` -> Exact `2x int8`
//    Proven lossless across all 16 codes in `LeanSonic.Proofs.e2m1_all_16_codes_exact`.
// ----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t decode_4x_e2m1_low_nibbles_to_2x_s8(uint32_t nibbles4) {
    // Each byte of `nibbles4` holds a 4-bit E2M1 code in `0..15`.
    // Selector bit 2 (0..7) indexes `{Lo, Hi}`, while bit 3 selects positive vs negative LUT!
    uint32_t pos = __byte_perm(
        verified::kE2M1LutPosLo,
        verified::kE2M1LutPosHi,
        nibbles4 & 0x07070707u);
    uint32_t neg = __byte_perm(
        verified::kE2M1LutNegLo,
        verified::kE2M1LutNegHi,
        nibbles4 & 0x07070707u);
    // Byte-wise sign select using bit 3 (`0x08`) expanded to `0xFF` mask per byte
    uint32_t sign_bit = (nibbles4 & 0x08080808u) >> 3;
    uint32_t sign_mask = sign_bit * 0xFFu;
    return (pos & ~sign_mask) | (neg & sign_mask);
}

// ----------------------------------------------------------------------------
// 6. Non-Associative `0-ULP` Butterfly Tree (`simt_and_mma_16bx2_0ulp_eq`)
// ----------------------------------------------------------------------------
__device__ __forceinline__ float butterfly_fadd4_rn(float p0, float p1, float p2, float p3) {
    float lo = __fadd_rn(p0, p1);
    float hi = __fadd_rn(p2, p3);
    return __fadd_rn(lo, hi);
}

// ----------------------------------------------------------------------------
// 7. Tensor Core `mma.sync.aligned` Wrappers (`m16n8k16.f16` & `m16n8k32.s8`)
// ----------------------------------------------------------------------------
__device__ __forceinline__ void mma_m16n8k16_f32_f16(
    float (&d)[4],
    const uint32_t (&a)[4],
    const uint32_t (&b)[2],
    const float (&c)[4]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#endif
}

__device__ __forceinline__ void mma_m16n8k32_s32_s8(
    int32_t (&d)[4],
    const uint32_t (&a)[4],
    const uint32_t (&b)[2],
    const int32_t (&c)[4]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
#endif
}

} // namespace leansonic::ptx
