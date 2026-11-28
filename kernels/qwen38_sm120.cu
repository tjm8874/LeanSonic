// ============================================================================
// LeanSonic — Full-GPU `Qwen3.8-Flash-Next` / `Qwen4Exp` SM120 Kernel Suite
// Target: NVIDIA Blackwell (`sm_120a` — RTX PRO 4500 32GB & RTX 5070 Ti 16GB)
//
// Architecture (`48` layers, `D = 2560`, `hc_count = 4`, `512` experts `Top-10`):
//   - Group-32 4-bit affine (`uint4` `LDG.E.128` = 32 nibbles + BF16 scale/bias)
//   - 4-Stream Hyper-Connection (`hc_norm` + lowrank-320 `down/up` + `block_inject`)
//   - Layer 1 (`ple_layer_ids = [2]`) 3-Gram Host `Engram`/`PLE` Async HtoD + Dilated-3 Conv4
//   - 36-Layer Gated DeltaNet (`linear_attn`: `16` K-heads, `48` V-heads, `128` dim, FP32 state)
//   - 12-Layer QSA (`self_attn` + `indexer`: `24` Q-heads, `2` KV-heads, `256` dim, Top-512 blocks)
//   - 512-Expert `Top-10` + Gated Shared Expert (`k = 0..10` unified SwiGLU MoE, Resident/Spilled)
// ============================================================================

#include "include/leansonic_verified.cuh"

namespace leansonic::qwen38 {

constexpr int kD = 2560;
constexpr int kHcD = 10240;
constexpr int kExp = 512;
constexpr int kTopK = 10;
constexpr int kMoeFf = 640;
constexpr int kVocab = 248320;

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

// Decodes one 32-weight group (`1` `uint4` = `128` bits = `32` nibbles) against `8` `float4`s!
__device__ __forceinline__ float dot_q4_group32_word(uint32_t w, float sc, float bi, float4 x0, float4 x1) {
    float acc = 0.0f;
    acc = __fmaf_rn(__fmaf_rn(static_cast<float>(w & 0xFu), sc, bi), x0.x, acc);
    acc = __fmaf_rn(__fmaf_rn(static_cast<float>((w >> 4) & 0xFu), sc, bi), x0.y, acc);
    acc = __fmaf_rn(__fmaf_rn(static_cast<float>((w >> 8) & 0xFu), sc, bi), x0.z, acc);
    acc = __fmaf_rn(__fmaf_rn(static_cast<float>((w >> 12) & 0xFu), sc, bi), x0.w, acc);
    acc = __fmaf_rn(__fmaf_rn(static_cast<float>((w >> 16) & 0xFu), sc, bi), x1.x, acc);
    acc = __fmaf_rn(__fmaf_rn(static_cast<float>((w >> 20) & 0xFu), sc, bi), x1.y, acc);
    acc = __fmaf_rn(__fmaf_rn(static_cast<float>((w >> 24) & 0xFu), sc, bi), x1.z, acc);
    acc = __fmaf_rn(__fmaf_rn(static_cast<float>((w >> 28) & 0xFu), sc, bi), x1.w, acc);
    return acc;
}

__device__ __forceinline__ float dot_q4_group32(
    uint4 pack,
    float sc,
    float bi,
    const float4* __restrict__ x4_grp) {
    float s = dot_q4_group32_word(pack.x, sc, bi, x4_grp[0], x4_grp[1]);
    s += dot_q4_group32_word(pack.y, sc, bi, x4_grp[2], x4_grp[3]);
    s += dot_q4_group32_word(pack.z, sc, bi, x4_grp[4], x4_grp[5]);
    s += dot_q4_group32_word(pack.w, sc, bi, x4_grp[6], x4_grp[7]);
    return s;
}

__device__ __forceinline__ float load_bf16_from_u32_array(const uint32_t* __restrict__ p_u32, int idx) {
    uint32_t pair = p_u32[idx >> 1];
    uint16_t u = ((idx & 1) == 0)
        ? static_cast<uint16_t>(pair & 0xFFFFu)
        : static_cast<uint16_t>(pair >> 16);
    return bf16_to_f32(u);
}

__device__ __forceinline__ float warp_dot_q4_row(
    const uint4* __restrict__ row_w,
    const uint32_t* __restrict__ row_sc_u32,
    const uint32_t* __restrict__ row_bi_u32,
    const float4* __restrict__ smem_x4,
    int n_groups,
    int lane) {
    float acc = 0.0f;
    for (int g = lane; g < n_groups; g += 32) {
        uint4 pk = row_w[g];
        float sc = load_bf16_from_u32_array(row_sc_u32, g);
        float bi = load_bf16_from_u32_array(row_bi_u32, g);
        acc += dot_q4_group32(pk, sc, bi, smem_x4 + g * 8);
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
    const int32_t* __restrict__ token_ptr,
    float* __restrict__ out_hc) {          // [10240]
    const int d = blockIdx.x * 256 + threadIdx.x;
    if (d >= kD) return;

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

// ----------------------------------------------------------------------------
// 2. SMEM-Staged Group-32 4-Bit Affine GEMV (`cols <= 6144`, `8` rows/CTA)
//    `post_mode`: `0` = linear, `1` = `silu(y * 0.25f)`, `2` = `sigmoid(y)`
// ----------------------------------------------------------------------------
extern "C" __global__ void q38_q4_gemv(
    const uint4* __restrict__ w_u128,      // [rows, cols / 32]
    const uint32_t* __restrict__ sc_u32,   // [rows, cols / 64]
    const uint32_t* __restrict__ bi_u32,   // [rows, cols / 64]
    const float4* __restrict__ x4,         // [cols / 4]
    float* __restrict__ y,                 // [rows]
    int32_t rows,
    int32_t cols,
    int32_t post_mode) {
    __shared__ float4 smem_x4[1536];       // max `6144 / 4 = 1536` (`24,576` B SMEM)

    const int tid = threadIdx.x;
    const int cols4 = cols >> 2;
    for (int i = tid; i < cols4; i += 256) {
        smem_x4[i] = x4[i];
    }
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + warp_id;
    if (row >= rows) return;

    const int n_groups = cols >> 5;
    const int sc_stride_u32 = cols >> 6;
    float v = warp_dot_q4_row(
        w_u128 + static_cast<size_t>(row) * n_groups,
        sc_u32 + static_cast<size_t>(row) * sc_stride_u32,
        bi_u32 + static_cast<size_t>(row) * sc_stride_u32,
        smem_x4,
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

// Wide `cols = 10240` variant (`40,960` B SMEM) for `hc` `down` (`320` rows) & `inject` (`4` rows)
extern "C" __global__ void q38_q4_gemv_10k(
    const uint4* __restrict__ w_u128,      // [rows, 320]
    const uint32_t* __restrict__ sc_u32,   // [rows, 160]
    const uint32_t* __restrict__ bi_u32,   // [rows, 160]
    const float4* __restrict__ x4,         // [2560]
    float* __restrict__ y,                 // [rows]
    int32_t rows,
    int32_t post_mode) {
    __shared__ float4 smem_x4[2560];       // `40,960` B SMEM

    const int tid = threadIdx.x;
    for (int i = tid; i < 2560; i += 256) {
        smem_x4[i] = x4[i];
    }
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + warp_id;
    if (row >= rows) return;

    float v = warp_dot_q4_row(
        w_u128 + static_cast<size_t>(row) * 320u,
        sc_u32 + static_cast<size_t>(row) * 160u,
        bi_u32 + static_cast<size_t>(row) * 160u,
        smem_x4,
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

// ----------------------------------------------------------------------------
// 3. Layer 1 (`ple_layer_ids = [2]`) Fused PLE Gate + Dilated-3 Conv4 + Inject
//    Grid: `(4, 1)`, Block: `256` (`1` CTA per Hyper-Connection stream `s = 0..3`)
// ----------------------------------------------------------------------------
extern "C" __global__ void q38_ple_inject(
    float* __restrict__ hc_inout,             // [4, 2560]
    const float* __restrict__ key_raw,        // [4, 2560] (`key_proj(ple_emb)`)
    const float* __restrict__ val_raw,        // [2560]    (`value_proj(ple_emb)`)
    const float* __restrict__ norm_key_w,     // [4, 2560] (`f32` effective `1+w` or `w`)
    const float* __restrict__ norm_query_w,   // [4, 2560]
    const float* __restrict__ norm_conv_w,    // [4, 2560]
    const float4* __restrict__ conv_w4,       // [4, 2560] (`[w0, w1, w2, w3]`)
    float* __restrict__ ple_hist9,            // [9, 10240] ring buffer
    const int32_t* __restrict__ pos_ptr) {
    __shared__ float smem_a[8];
    __shared__ float smem_b[8];
    __shared__ float smem_c[8];
    __shared__ float smem_scalars[4]; // [inv_k, inv_q, sig_gate, inv_gv]

    const int s = blockIdx.x; // 0..3
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int pos = *pos_ptr;

    float* hc_s = hc_inout + s * kD;
    const float* kr_s = key_raw + s * kD;
    const float* nk_s = norm_key_w + s * kD;
    const float* nq_s = norm_query_w + s * kD;
    const float* nc_s = norm_conv_w + s * kD;
    const float4* cw_s = conv_w4 + s * kD;

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
            smem_scalars[3] = tv * (1.0f / 2560.0f); // mean(val^2)
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
            float g = td * 0.01976423537605237f; // 1 / sqrt(2560)
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
}

// ----------------------------------------------------------------------------
// 4. 4-Stream Hyper-Connection Norm, Mix, and Gated Residual Injection
// ----------------------------------------------------------------------------
// Grid: `(4, 1)`, Block: `256`
extern "C" __global__ void q38_hc_norm4(
    const float* __restrict__ hc_in,      // [4, 2560]
    const float* __restrict__ hc_gamma,   // [4, 2560] (`f32` effective gamma)
    float* __restrict__ hc_normed) {      // [4, 2560]
    __shared__ float smem_sq[8];
    __shared__ float smem_inv;

    const int s = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const float* src = hc_in + s * kD;
    const float* gam = hc_gamma + s * kD;
    float* dst = hc_normed + s * kD;

    float sq = 0.0f;
    for (int d = tid; d < kD; d += 256) {
        float v = src[d];
        sq += v * v;
    }
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
    for (int d = tid; d < kD; d += 256) {
        dst[d] = src[d] * inv * gam[d];
    }
}

// Fused HC pre-branch front: `q38_hc_norm4` + `hc.down` (`q38_q4_gemv_10k`, `silu(y / 4)`) +
// `hc.inject` (`q38_q4_gemv_10k`, linear) in one launch. Every CTA normalizes all `4` streams with
// the exact `q38_hc_norm4` reduction order into SMEM (`0-ULP` with the three-launch path), writes
// its share of `hc_normed` (still read by `q38_hc_mix4`), then `8` rows: `[0, rows_down)` down,
// `[rows_down, rows_down + n_inj)` inject.
// Grid: `(ceil((rows_down + n_inj) / 8), 1)`, Block: `256`, `40,992` B SMEM
extern "C" __global__ void q38_hc_norm_down_inject(
    const float* __restrict__ hc_in,       // [4, 2560]
    const float* __restrict__ hc_gamma,    // [4, 2560]
    float* __restrict__ hc_normed,         // [4, 2560]
    const uint4* __restrict__ down_w,      // [rows_down, 320]
    const uint32_t* __restrict__ down_sc,  // [rows_down, 160]
    const uint32_t* __restrict__ down_bi,  // [rows_down, 160]
    int32_t rows_down,
    float* __restrict__ y_low,             // [rows_down]
    const uint4* __restrict__ inj_w,       // [n_inj, 320] (nullable when `n_inj == 0`)
    const uint32_t* __restrict__ inj_sc,   // [n_inj, 160]
    const uint32_t* __restrict__ inj_bi,   // [n_inj, 160]
    int32_t n_inj,
    float* __restrict__ y_inj) {           // [n_inj]
    __shared__ float4 smem_x4[2560];
    __shared__ float smem_sq[8];
    __shared__ float smem_inv[4];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    for (int s = 0; s < 4; ++s) {
        const float* src = hc_in + s * kD;
        float sq = 0.0f;
        for (int d = tid; d < kD; d += 256) {
            float v = src[d];
            sq += v * v;
        }
        sq = warp_sum(sq);
        if (lane == 0) smem_sq[warp_id] = sq;
        __syncthreads();
        if (warp_id == 0) {
            float tot = (lane < 8) ? smem_sq[lane] : 0.0f;
            tot = warp_sum(tot);
            if (lane == 0) smem_inv[s] = ptx_rsqrt(tot * (1.0f / 2560.0f) + 1e-6f);
        }
        __syncthreads();
    }
    float* smem_x = reinterpret_cast<float*>(smem_x4);
    const int share = static_cast<int>(blockIdx.x);
    for (int i = tid; i < kHcD; i += 256) {
        const float v = hc_in[i] * smem_inv[i / kD] * hc_gamma[i];
        smem_x[i] = v;
        if ((i >> 8) % static_cast<int>(gridDim.x) == share) hc_normed[i] = v;
    }
    __syncthreads();

    const int row = blockIdx.x * 8 + warp_id;
    if (row < rows_down) {
        float v = warp_dot_q4_row(
            down_w + static_cast<size_t>(row) * 320u,
            down_sc + static_cast<size_t>(row) * 160u,
            down_bi + static_cast<size_t>(row) * 160u,
            smem_x4, 320, lane);
        if (lane == 0) y_low[row] = ptx_silu(v * 0.25f);
    } else if (row < rows_down + n_inj) {
        const int r = row - rows_down;
        float v = warp_dot_q4_row(
            inj_w + static_cast<size_t>(r) * 320u,
            inj_sc + static_cast<size_t>(r) * 160u,
            inj_bi + static_cast<size_t>(r) * 160u,
            smem_x4, 320, lane);
        if (lane == 0) y_inj[r] = v;
    }
}

// Grid: `(10, 1)`, Block: `256` (`2,560` dims)
extern "C" __global__ void q38_hc_mix4(
    const float* __restrict__ hc_normed,  // [4, 2560]
    const float* __restrict__ hc_up_sig,  // [4, 2560] (`sigmoid(up)`)
    float* __restrict__ mixed_out) {      // [2560]
    const int d = blockIdx.x * 256 + threadIdx.x;
    if (d >= kD) return;
    float sum = hc_up_sig[d] * hc_normed[d]
              + hc_up_sig[kD + d] * hc_normed[kD + d]
              + hc_up_sig[2 * kD + d] * hc_normed[2 * kD + d]
              + hc_up_sig[3 * kD + d] * hc_normed[3 * kD + d];
    mixed_out[d] = sum * 0.25f;
}

// Grid: `(40, 1)`, Block: `256` (`10,240` dims)
extern "C" __global__ void q38_hc_inject4(
    float* __restrict__ hc_inout,         // [4, 2560]
    const float* __restrict__ branch_out, // [2560]
    const float* __restrict__ inj_raw4) { // [4]
    const int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= kHcD) return;
    const int s = idx / kD;
    const int d = idx - s * kD;
    const float w = 2.0f * ptx_sigmoid(inj_raw4[s] * 0.25f);
    hc_inout[idx] += branch_out[d] * w;
}

// ----------------------------------------------------------------------------
// 5. Gated DeltaNet (`linear_attn`, 36 Layers): Causal Conv4+SiLU & Recurrent Rule
// ----------------------------------------------------------------------------
// Grid: `(40, 1)`, Block: `256` (`10,240` channels)
extern "C" __global__ void q38_gdn_conv_silu(
    const float* __restrict__ qkv_raw,    // [10240]
    const float4* __restrict__ conv_w4,   // [10240] (`[w0, w1, w2, w3]`)
    float* __restrict__ conv_hist3,       // [10240 * 3]
    float* __restrict__ qkv_conv_out) {   // [10240]
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= 10240) return;

    float* h = conv_hist3 + c * 3;
    float h0 = h[0], h1 = h[1], h2 = h[2];
    float x = qkv_raw[c];
    h[0] = h1;
    h[1] = h2;
    h[2] = x;

    float4 w = conv_w4[c];
    qkv_conv_out[c] = ptx_silu(w.x * h0 + w.y * h1 + w.z * h2 + w.w * x);
}

// Grid: `(48, 1)`, Block: `128` (`1` CTA per V-head `vh = 0..47`, `1` thread per `dv = 0..127`!)
// State layout `[48, k_dim=128, v_dim=128]` -> `thread dv` accesses `S[vh, dk, dv]` coalesced!
extern "C" __global__ void q38_gdn_recurrent_step(
    const float* __restrict__ qkv_conv,   // [10240]: `Q[2048], K[2048], V[6144]`
    const float* __restrict__ zba_buf,    // [6240]:  `Z[6144], B[48], A[48]`
    const float* __restrict__ a_log,      // [48] (`f32`)
    const float* __restrict__ dt_bias,    // [48] (`f32`)
    const float* __restrict__ norm_w,     // [128] (`f32`, 1-centered `RMSNormGated`)
    int32_t use_sigmoid_gate,             // `1` = `sigmoid(z)` (`output_gate_type = "sigmoid"`), `0` = `silu(z)`
    float* __restrict__ state_f32,        // [48 * 128 * 128]
    float* __restrict__ gdn_out) {        // [6144]
    __shared__ float smem_q[128];
    __shared__ float smem_k[128];
    __shared__ float smem_red[4];
    __shared__ float smem_gate[3]; // [decay, beta, inv_rms_out]

    const int vh = blockIdx.x;     // 0..47
    const int kh = vh / 3;         // 0..15 (`3` V-heads per K-head!)
    const int dv = threadIdx.x;    // 0..127
    const int warp_id = dv >> 5;
    const int lane = dv & 31;

    // 1. Load & L2-normalize `Q[kh]` (+ `* 1/sqrt(128)`) and `K[kh]`
    float qv = qkv_conv[kh * 128 + dv];
    float kv = qkv_conv[2048 + kh * 128 + dv];
    float q_sq = warp_sum(qv * qv);
    float k_sq = warp_sum(kv * kv);
    if (lane == 0) smem_red[warp_id] = q_sq;
    __syncthreads();
    float q_tot = smem_red[0] + smem_red[1] + smem_red[2] + smem_red[3];
    float q_inv = ptx_rsqrt(q_tot + 1e-6f) * 0.08838834764831845f; // * (1 / sqrt(128))
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
        float g = -ptx_exp(a_log[vh]) * ptx_softplus(a_raw + dt_bias[vh]);
        smem_gate[0] = ptx_exp(g);         // decay
        smem_gate[1] = ptx_sigmoid(b_raw); // beta
    }
    __syncthreads();

    const float decay = smem_gate[0];
    const float beta = smem_gate[1];
    float* S_vh = state_f32 + static_cast<size_t>(vh) * 16384u;

    // 2. Coalesced Pass 1: `kv_mem = sum_dk (S[dk, dv] * decay) * k[dk]`
    float kv_mem = 0.0f;
    for (int dk = 0; dk < 128; ++dk) {
        kv_mem += (S_vh[dk * 128 + dv] * decay) * smem_k[dk];
    }
    const float v_val = qkv_conv[4096 + vh * 128 + dv];
    const float delta = (v_val - kv_mem) * beta;

    // 3. Coalesced Pass 2: `S_new = S * decay + k * delta; out = sum_dk S_new * q[dk]`
    float out_v = 0.0f;
    for (int dk = 0; dk < 128; ++dk) {
        float s_new = S_vh[dk * 128 + dv] * decay + smem_k[dk] * delta;
        S_vh[dk * 128 + dv] = s_new;
        out_v += s_new * smem_q[dk];
    }

    // 4. Fused `Qwen4ExpTextRMSNormGated(out_v, z)` (`eps = 1e-6`, `sigmoid` or `silu` gate)
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
    gdn_out[vh * 128 + dv] = out_v * smem_gate[2] * norm_w[dv] * gate_act;
}

// ----------------------------------------------------------------------------
// 6. QSA (`self_attn` + `indexer`, 12 Layers): Prep/RoPE/Block-Pool + Sparse GQA
// ----------------------------------------------------------------------------
// Grid: `(31, 1)`, Block: `256`
extern "C" __global__ void q38_qsa_prep_and_cache(
    const float* __restrict__ qsa_proj,        // [13952]
    const float* __restrict__ q_norm_w,        // [256] (`f32` effective `1+w`)
    const float* __restrict__ k_norm_w,        // [256]
    const float* __restrict__ idx_q_norm_w,    // [128]
    const float* __restrict__ idx_k_norm_w,    // [128]
    const float* __restrict__ inv_freq32,      // [32] (`10_000_000^(-2i/64)`)
    const int32_t* __restrict__ pos_ptr,
    float* __restrict__ q_rot_out,             // [24, 256]
    float* __restrict__ q_gate_out,            // [24, 256]
    uint16_t* __restrict__ k_cache_fp16,       // [max_ctx, 512]
    uint16_t* __restrict__ v_cache_fp16,       // [max_ctx, 512]
    float* __restrict__ idx_q_out,             // [4, 128]
    float* __restrict__ idx_raw_k_cache,       // [max_ctx, 128]
    float* __restrict__ idx_block_k_cache) {   // [max_ctx / 4, 128]
    __shared__ float smem_vec[256];
    __shared__ float smem_red[8];

    const int blk = blockIdx.x;
    const int tid = threadIdx.x; // 0..255
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int pos = *pos_ptr;

    if (blk < 24) {
        // Main Q head `h = blk` (`q_proj` `[24, 512]` chunked into `[0..256]` Q and `[256..512]` gate)
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
        // Main KV head `kh = blk - 24` (`0..1`), also writes `v_cache_fp16`!
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
        // Indexer Q head `ih = blk - 26` (`0..3`, `128` dims)
        if (tid >= 128) return;
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
    } else {
        // `blk == 30`: Indexer raw K store + immutable 4-token block key cache when `(pos & 3) == 3`!
        if (tid >= 128) return;
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

// Long-context (`n_blocks > 512`) QSA Block Scorer + Top-512 Mask Builder
// Grid: `(ceil(n_blocks / 8), 1)`, Block: `256` (`1` warp per block `b`)
extern "C" __global__ void q38_qsa_score_blocks(
    const float4* __restrict__ idx_q4,          // [4, 32]
    const float4* __restrict__ block_k4_cache,  // [n_blocks, 32]
    int32_t n_blocks,
    float* __restrict__ block_scores) {         // [n_blocks]
    __shared__ float4 smem_q4[4][32];
    const int tid = threadIdx.x;
    if (tid < 128) {
        smem_q4[tid >> 5][tid & 31] = idx_q4[tid];
    }
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int b = blockIdx.x * 8 + warp_id;
    if (b >= n_blocks) return;

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

// Grid: `(1, 1)`, Block: `256` — marks exact Top-512 blocks in `block_mask [n_blocks]`
extern "C" __global__ void q38_qsa_select_top512(
    float* __restrict__ scores_scratch, // [n_blocks] (selected entries masked in-place)
    int32_t n_blocks,
    int32_t* __restrict__ block_mask) { // [n_blocks]
    __shared__ float smem_val[8];
    __shared__ int smem_idx[8];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

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

// 8-Warp Split-KV QSA GQA (`Grid (24, 1)`, `Block 256`, `head_dim = 256`, `128`-bit `uint4` FP16 loads!)
extern "C" __global__ void q38_qsa_attn(
    const float* __restrict__ q_rot,            // [24, 256]
    const float* __restrict__ q_gate,           // [24, 256]
    const uint4* __restrict__ k_cache_u128,     // [max_ctx, 2, 32] (`256` FP16 = `32` `uint4`)
    const uint4* __restrict__ v_cache_u128,     // [max_ctx, 2, 32]
    const int32_t* __restrict__ block_mask,     // nullable when `use_mask == 0`
    const int32_t* __restrict__ pos_ptr,
    int32_t use_mask,
    float* __restrict__ attn_out) {             // [24, 256]
    __shared__ float smem_m[8];
    __shared__ float smem_l[8];
    __shared__ float smem_o[8][256];

    const int h = blockIdx.x;   // 0..23
    const int kvh = h / 12;     // 0..1 (`12:1` GQA!)
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
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
}

// ----------------------------------------------------------------------------
// 7. `512`-Expert BF16 Router + Exact Softmax Top-10 + Unified `k = 0..10` MoE
// ----------------------------------------------------------------------------
// Grid: `(64, 1)`, Block: `256` (`8` rows/CTA -> `512` BF16 router logits)
extern "C" __global__ void q38_bf16_router_512(
    const uint4* __restrict__ w_bf16_u128, // [512, 320]
    const float4* __restrict__ x4,         // [640]
    float* __restrict__ logits512) {       // [512]
    __shared__ float4 smem_x4[640];
    const int tid = threadIdx.x;
    for (int i = tid; i < 640; i += 256) {
        smem_x4[i] = x4[i];
    }
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + warp_id;
    if (row >= kExp) return;

    const uint4* w_row = w_bf16_u128 + static_cast<size_t>(row) * 320u;
    float acc = 0.0f;
    for (int c8 = lane; c8 < 320; c8 += 32) {
        uint4 pk = w_row[c8];
        float4 u = smem_x4[c8 * 2];
        float4 v = smem_x4[c8 * 2 + 1];
        acc += bf16_to_f32(static_cast<uint16_t>(pk.x & 0xFFFFu)) * u.x
             + bf16_to_f32(static_cast<uint16_t>(pk.x >> 16)) * u.y
             + bf16_to_f32(static_cast<uint16_t>(pk.y & 0xFFFFu)) * u.z
             + bf16_to_f32(static_cast<uint16_t>(pk.y >> 16)) * u.w
             + bf16_to_f32(static_cast<uint16_t>(pk.z & 0xFFFFu)) * v.x
             + bf16_to_f32(static_cast<uint16_t>(pk.z >> 16)) * v.y
             + bf16_to_f32(static_cast<uint16_t>(pk.w & 0xFFFFu)) * v.z
             + bf16_to_f32(static_cast<uint16_t>(pk.w >> 16)) * v.w;
    }
    acc = warp_sum(acc);
    if (lane == 0) logits512[row] = acc;
}

// Grid: `(1, 1)`, Block: `256` (`2` experts/thread -> Softmax + Top-10 + `norm_topk_prob`)
extern "C" __global__ void q38_softmax_topk10(
    const float* __restrict__ logits512,
    int32_t* __restrict__ sel_ids10,  // [10]
    float* __restrict__ sel_wts11) {  // [11] (`0..9` routed, `[10]` filled by shared_gate)
    __shared__ float smem_p[512];
    __shared__ float smem_red[8];
    __shared__ int smem_idx[8];
    __shared__ float smem_topw[10];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

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
}

// Unified `k = 0..10` (`10` Routed + `1` Shared Expert) SwiGLU Up + Shared Gate!
// Grid: `(160, 11)`, Block: `256` (`4` pairs of `(gate, up)` warps = `4` `m` outputs/CTA -> `160 * 4 = 640`!)
extern "C" __global__ void q38_moe_up_swiglu11(
    const uint4* __restrict__ sw_gate_w,     // [512, 640, 80]
    const uint32_t* __restrict__ sw_gate_sc, // [512, 640, 40]
    const uint32_t* __restrict__ sw_gate_bi, // [512, 640, 40]
    const uint4* __restrict__ sw_up_w,       // [512, 640, 80]
    const uint32_t* __restrict__ sw_up_sc,   // [512, 640, 40]
    const uint32_t* __restrict__ sw_up_bi,   // [512, 640, 40]
    const uint4* __restrict__ sh_gu_gate_w,  // [1281, 80]: `[gate(640); up(640); sh_gate(1)]`
    const uint32_t* __restrict__ sh_gu_sc,   // [1281, 40]
    const uint32_t* __restrict__ sh_gu_bi,   // [1281, 40]
    const float4* __restrict__ x4,           // [640] (`2,560` floats)
    const int32_t* __restrict__ sel_ids10,   // [10]
    int32_t is_compact,                      // `0` = index `sel_ids10[k]`, `1` = staged `[0..9]`
    float* __restrict__ sel_wts11,           // [11] (`sel_wts11[10]` written here!)
    float* __restrict__ act11_out) {         // [11, 640]
    __shared__ float4 smem_x4[640];          // `10,240` B SMEM!
    __shared__ float smem_gu[8];

    const int tid = threadIdx.x;
    for (int i = tid; i < 640; i += 256) {
        smem_x4[i] = x4[i];
    }
    __syncthreads();

    const int warp_id = tid >> 5; // 0..7
    const int lane = tid & 31;
    const int k = blockIdx.y;     // 0..10
    const int m = blockIdx.x * 4 + (warp_id >> 1); // 0..639
    const int is_up = warp_id & 1;

    if (k < kTopK) {
        const int ex = is_compact ? k : sel_ids10[k];
        const size_t row = static_cast<size_t>(ex) * kMoeFf + m;
        const uint4* w_ptr = (is_up ? sw_up_w : sw_gate_w) + row * 80u;
        const uint32_t* sc_ptr = (is_up ? sw_up_sc : sw_gate_sc) + row * 40u;
        const uint32_t* bi_ptr = (is_up ? sw_up_bi : sw_gate_bi) + row * 40u;
        float dot = warp_dot_q4_row(w_ptr, sc_ptr, bi_ptr, smem_x4, 80, lane);
        if (lane == 0) smem_gu[warp_id] = dot;
    } else {
        const size_t row = static_cast<size_t>(is_up * kMoeFf + m);
        float dot = warp_dot_q4_row(
            sh_gu_gate_w + row * 80u,
            sh_gu_sc + row * 40u,
            sh_gu_bi + row * 40u,
            smem_x4,
            80,
            lane);
        if (lane == 0) smem_gu[warp_id] = dot;

        // Warp 0 of `blockIdx.x == 0, k == 10` also evaluates row `1280` (`shared_expert_gate`)!
        if (blockIdx.x == 0 && warp_id == 0) {
            float sg = warp_dot_q4_row(
                sh_gu_gate_w + 1280u * 80u,
                sh_gu_sc + 1280u * 40u,
                sh_gu_bi + 1280u * 40u,
                smem_x4,
                80,
                lane);
            if (lane == 0) sel_wts11[10] = ptx_sigmoid(sg);
        }
    }
    __syncthreads();

    if (lane == 0 && is_up == 0) {
        float g = smem_gu[warp_id];
        float u = smem_gu[warp_id + 1];
        act11_out[k * kMoeFf + m] = ptx_silu(g) * u;
    }
}

// Unified `k = 0..10` Down + Weighted Combine (`Grid (320, 1)`, `Block 256` -> `2,560` rows!)
extern "C" __global__ void q38_moe_down_combine11(
    const uint4* __restrict__ sw_down_w,     // [512, 2560, 20]
    const uint32_t* __restrict__ sw_down_sc, // [512, 2560, 10]
    const uint32_t* __restrict__ sw_down_bi, // [512, 2560, 10]
    const uint4* __restrict__ sh_down_w,     // [2560, 20]
    const uint32_t* __restrict__ sh_down_sc, // [2560, 10]
    const uint32_t* __restrict__ sh_down_bi, // [2560, 10]
    const float4* __restrict__ act11_x4,     // [11, 160] (`28,160` B SMEM!)
    const int32_t* __restrict__ sel_ids10,   // [10]
    int32_t is_compact,                      // `0` = index `sel_ids10[k]`, `1` = staged `[0..9]`
    const float* __restrict__ sel_wts11,     // [11]
    float* __restrict__ moe_out) {           // [2560]
    __shared__ float4 smem_a4[11 * 160];     // `28,160` B SMEM

    const int tid = threadIdx.x;
    for (int i = tid; i < 11 * 160; i += 256) {
        smem_a4[i] = act11_x4[i];
    }
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + warp_id;
    if (row >= kD) return;

    float sum_y = 0.0f;
    for (int k = 0; k < kTopK; ++k) {
        const int ex = is_compact ? k : sel_ids10[k];
        const size_t r_idx = static_cast<size_t>(ex) * kD + row;
        float dot = warp_dot_q4_row(
            sw_down_w + r_idx * 20u,
            sw_down_sc + r_idx * 10u,
            sw_down_bi + r_idx * 10u,
            smem_a4 + k * 160,
            20,
            lane);
        sum_y += dot * sel_wts11[k];
    }
    {
        float sh_dot = warp_dot_q4_row(
            sh_down_w + static_cast<size_t>(row) * 20u,
            sh_down_sc + static_cast<size_t>(row) * 10u,
            sh_down_bi + static_cast<size_t>(row) * 10u,
            smem_a4 + 10 * 160,
            20,
            lane);
        sum_y += sh_dot * sel_wts11[10];
    }
    if (lane == 0) {
        moe_out[row] = sum_y;
    }
}

// ----------------------------------------------------------------------------
// 8. Greedy ArgMax (`248,320` Vocab) + `*pos_ptr += 1` (`Grid (1, 1)`, `Block 256`)
// ----------------------------------------------------------------------------
extern "C" __global__ void q38_argmax_and_inc_pos(
    const float* __restrict__ logits,
    int32_t* __restrict__ out_tok,
    int32_t* __restrict__ pos_ptr) {
    __shared__ float smem_val[8];
    __shared__ int smem_idx[8];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

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

extern "C" __global__ void q38_inc_pos(int32_t* __restrict__ pos_ptr) {
    if (threadIdx.x == 0) {
        *pos_ptr += 1;
    }
}
