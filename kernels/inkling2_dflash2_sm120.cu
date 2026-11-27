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

// x[t] = rms_norm(embed[tok_t]) * embed_norm with tok_0 = anchor, tok_t = mask_tok.
// Grid `8`, Block `256`.
extern "C" __global__ void df2_block_init(
    const uint32_t* __restrict__ embed_bf16_pairs,      // [vocab, D / 2]
    const uint32_t* __restrict__ embed_norm_bf16_pairs, // [D / 2]
    int32_t anchor,
    int32_t mask_tok,
    float* __restrict__ x) {                            // [8, D]
    pdl_wait();
    pdl_trigger();
    __shared__ float smem_red[8];
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int tok = (t == 0) ? anchor : mask_tok;
    const uint32_t* row = embed_bf16_pairs + static_cast<size_t>(tok) * (kD / 2);
    float* out = x + static_cast<size_t>(t) * kD;
    float sq = 0.0f;
    for (int idx = tid; idx < kD / 2; idx += 256) {
        const uint32_t pair = row[idx];
        const float v0 = bf16_to_f32(static_cast<uint16_t>(pair & 0xFFFFu));
        const float v1 = bf16_to_f32(static_cast<uint16_t>(pair >> 16));
        out[idx * 2] = v0;
        out[idx * 2 + 1] = v1;
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
        const uint32_t nw = embed_norm_bf16_pairs[idx];
        out[idx * 2] *= scale * bf16_to_f32(static_cast<uint16_t>(nw & 0xFFFFu));
        out[idx * 2 + 1] *= scale * bf16_to_f32(static_cast<uint16_t>(nw >> 16));
    }
}

// y[r] = rms_norm(x[r]) * w (`eps = 1e-6`, FP32 weights). Grid `rows`, Block `256`.
extern "C" __global__ void df2_rmsnorm(
    const float* __restrict__ x,  // [rows, D]
    const float* __restrict__ w,  // [D]
    float* __restrict__ y) {      // [rows, D]
    pdl_wait();
    pdl_trigger();
    __shared__ float smem_red[8];
    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    const float* in = x + static_cast<size_t>(r) * kD;
    float sq = 0.0f;
    for (int c = tid; c < kD; c += 256) sq += in[c] * in[c];
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
    float* out = y + static_cast<size_t>(r) * kD;
    for (int c = tid; c < kD; c += 256) out[c] = in[c] * scale * w[c];
}

// Dynamic two-tap group-16 causal convolution over the 8 block rows. Taps `k0 = 2 * post`
// (current row) and `k0 + 1` (previous row) of `base [4, D]` and `dyn [8, 4, 144]`:
//   a = (base[k0] + dyn[k0][g]) * z[t] + (t > 0 ? (base[k0+1] + dyn[k0+1][g]) * z[t-1] : 0)
// `post = 0`: out = a; `post = 1`: out += a (residual). Grid `8`, Block `256`.
extern "C" __global__ void df2_conv2(
    const float* __restrict__ z,     // [8, D]
    const float* __restrict__ dyn,   // [8, 576]
    const float* __restrict__ base,  // [4, D]
    int32_t post,
    float* __restrict__ out) {       // [8, D]
    pdl_wait();
    pdl_trigger();
    const int t = blockIdx.x;
    const int k0 = 2 * post;
    const float* dr = dyn + static_cast<size_t>(t) * kDf2Dyn;
    for (int c = threadIdx.x; c < kD; c += 256) {
        const int g = c >> 4;
        float a = (base[k0 * kD + c] + dr[k0 * kDf2Groups + g]) * z[static_cast<size_t>(t) * kD + c];
        if (t > 0) {
            a += (base[(k0 + 1) * kD + c] + dr[(k0 + 1) * kDf2Groups + g]) *
                 z[static_cast<size_t>(t - 1) * kD + c];
        }
        if (post) out[static_cast<size_t>(t) * kD + c] += a;
        else out[static_cast<size_t>(t) * kD + c] = a;
    }
}

// Per-head RMSNorm + NeoX RoPE (`theta = 1e4`) for `rows` rows at positions `pos0 + r`:
// q heads in place (`do_q`), K (normed, rotated) and V into ring cell `(pos0 + r) % cap`.
// Grid `(do_q ? 40 : 8, rows)`, Block `128`: `blockIdx.x < 8` = KV head, else Q head - 8.
extern "C" __global__ void df2_qk_prep(
    float* __restrict__ q,                    // [rows, 4096]
    const float* __restrict__ k,              // [rows, k_stride]
    const float* __restrict__ v,              // [rows, k_stride]
    int32_t k_stride,
    const uint32_t* __restrict__ qn_bf16_p,   // [64]
    const uint32_t* __restrict__ kn_bf16_p,   // [64]
    const float* __restrict__ inv_freq64,     // [64]
    int32_t pos0,
    int32_t cap,
    float* __restrict__ k_ring,               // [cap, 1024]
    float* __restrict__ v_ring) {             // [cap, 1024]
    pdl_wait();
    pdl_trigger();
    __shared__ float smem_hd[128];
    const int hh = blockIdx.x;
    const int r = blockIdx.y;
    const int d = threadIdx.x;
    const int pos = pos0 + r;
    if (hh < 8) {
        const size_t src = static_cast<size_t>(r) * k_stride + hh * 128 + d;
        const float kr = head_norm_neox_rope128(k[src], kn_bf16_p, inv_freq64, pos, smem_hd);
        const size_t cell = static_cast<size_t>(pos % cap) * kDf2Kv + hh * 128 + d;
        k_ring[cell] = kr;
        v_ring[cell] = v[src];
    } else {
        float* p = q + static_cast<size_t>(r) * kDf2Q + (hh - 8) * 128;
        p[d] = head_norm_neox_rope128(p[d], qn_bf16_p, inv_freq64, pos, smem_hd);
    }
}

// Non-causal block attention: the 8 rows attend every ring key in `[lo, hi)` (cell `p % cap`).
// 32 query heads over 8 KV heads. Grid `(32, 8)`, Block `256` (8 warps split the keys).
extern "C" __global__ void df2_attn(
    const float* __restrict__ q,       // [8, 4096]
    const float* __restrict__ k_ring,  // [cap, 1024]
    const float* __restrict__ v_ring,  // [cap, 1024]
    int32_t lo,
    int32_t hi,
    int32_t cap,
    float* __restrict__ out) {         // [8, 4096]
    pdl_wait();
    pdl_trigger();
    __shared__ float smem_m[8];
    __shared__ float smem_l[8];
    __shared__ float smem_o[8][128];
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int kvh = h >> 2;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const float4 q4 = reinterpret_cast<const float4*>(q + static_cast<size_t>(t) * kDf2Q + h * 128)[lane];
    const float scale = 0.08838834764831845f; // 1 / sqrt(128)
    float m = -1e30f, l = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    for (int p = lo + warp; p < hi; p += 8) {
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
        for (int w = 1; w < 8; ++w) m_max = fmaxf(m_max, smem_m[w]);
        float den = 0.0f, num = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            const float s = (smem_l[w] > 0.0f) ? ptx_exp(smem_m[w] - m_max) : 0.0f;
            den += smem_l[w] * s;
            num += smem_o[w][threadIdx.x] * s;
        }
        out[static_cast<size_t>(t) * kDf2Q + h * 128 + threadIdx.x] = num * ptx_rcp(fmaxf(den, 1e-20f));
    }
}

// act = silu(gate) * up over `[rows, 2 * ff]` (gate rows first). Grid `ceil(rows * ff / 256)`.
extern "C" __global__ void df2_silu_mul(
    const float* __restrict__ gu,  // [rows, 2 * ff]
    float* __restrict__ act,       // [rows, ff]
    int32_t rows,
    int32_t ff) {
    pdl_wait();
    pdl_trigger();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * ff) return;
    const int r = i / ff;
    const int j = i - r * ff;
    const float g = gu[static_cast<size_t>(r) * 2 * ff + j];
    const float u = gu[static_cast<size_t>(r) * 2 * ff + ff + j];
    act[i] = (g * ptx_sigmoid(g)) * u;
}

// Gathers target taps `[4, tap_stride / D, D]` rows `row0 .. row0 + rows` into `[rows, 4 * D]`.
// Grid `rows`, Block `256`.
extern "C" __global__ void df2_tap_gather(
    const float* __restrict__ taps,
    int64_t tap_stride,
    int32_t row0,
    float* __restrict__ out) {     // [rows, 9216]
    pdl_wait();
    pdl_trigger();
    const int r = blockIdx.x;
    for (int i = threadIdx.x; i < 4 * kD; i += 256) {
        const int k = i / kD;
        const int c = i - k * kD;
        out[static_cast<size_t>(r) * 4 * kD + i] =
            taps[static_cast<size_t>(k) * tap_stride + static_cast<size_t>(row0 + r) * kD + c];
    }
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

} // namespace leansonic::inkling2_df2
