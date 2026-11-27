//! `Inkling-2-Mini` Lossless Speculative Drafters (`inkling2_mtp.rs`):
//! 1. **`DsparkDrafterGpu`**: Full-GPU `mtp.safetensors` (`DSpark`) 3-layer SwiGLU + `R = 256`
//!    Markov-RNN Drafter using Planar `Q8_64` `__dp4a` projections, shared target `unembed_q8`,
//!    and Host-Pinned `markov_w1` (`512 B` per step -> `~380 MiB` total VRAM!).
//! 2. **`SuffixDrafter`**: Zero-VRAM N-Gram Suffix Matcher fallback when `mtp.safetensors` is absent.

use core::ffi::c_void;
use std::path::Path;
use std::sync::Arc;

use crate::cuda_ffi::{
    check_cu, CUdeviceptr, CUfunction, CUstream, CudaContextHandle, DeviceBuf, PinnedHostBuf,
};
use crate::inkling2_gpu::{repack_bf16_to_planar_q8_64, PlanarQ8Gpu, D, PADDED_VOCAB, VOCAB};
use crate::safetensors::MappedSafeTensors;

const B_DRAFT: usize = 8;
const R_MARKOV: usize = 256;
const HD: usize = 128;
const DFF: usize = 9216;
const LOCAL_RING: usize = 2048;
const LOCAL_WIN: usize = 1025;

#[inline]
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn concat_slices(slices: &[&[u8]]) -> Vec<u8> {
    let total: usize = slices.iter().map(|s| s.len()).sum();
    let mut out = Vec::with_capacity(total);
    for s in slices {
        out.extend_from_slice(s);
    }
    out
}

pub struct MtpKernelFns {
    pub rmsnorm_d_b: CUfunction,
    pub rmsnorm_quant_q8_64_b: CUfunction,
    pub quant_q8_64_b: CUfunction,
    pub q8_dp4a_gemm_b: CUfunction,
    pub q8_dp4a_gemm_b8_fast: Option<CUfunction>,
    pub q8_dp4a_gemv: CUfunction,
    pub q8_dp4a_gemm_ktiled: CUfunction,
    pub tapmix: CUfunction,
    pub copy_anchor: CUfunction,
    pub ctx_store: CUfunction,
    pub init_draft: CUfunction,
    pub qk_norm_rope: CUfunction,
    pub attn: CUfunction,
    pub add_inplace: CUfunction,
    pub silu_mul: CUfunction,
    pub concat_frnn: CUfunction,
    pub add_bias_rows: CUfunction,
    pub markov_raw: CUfunction,
    pub markov_step_q8: CUfunction,
    pub fold_mv_q8: CUfunction,
    pub fold_fin: CUfunction,
}

unsafe impl Send for MtpKernelFns {}
unsafe impl Sync for MtpKernelFns {}

struct MtpLayerGpu {
    attn_norm: DeviceBuf,
    ffn_norm: DeviceBuf,
    dflare_norm: DeviceBuf,
    q_norm: DeviceBuf,
    k_norm: DeviceBuf,
    k_norm_ctx: DeviceBuf,
    qkv_q8: PlanarQ8Gpu,
    kvc_q8: PlanarQ8Gpu,
    wo_q8: PlanarQ8Gpu,
    gu_q8: PlanarQ8Gpu,
    down_q8: PlanarQ8Gpu,
    kc: DeviceBuf,
    vc: DeviceBuf,
    ring: usize, // `0` for global linear (`layer 1`), `LOCAL_RING` for `layers 0, 2`
    win: usize,
}

/// One sequence's drafter context (#107): the three layers' context KV, the anchor taps and the
/// acceptance calibration. The weights and draft scratch buffers are shared.
struct DraftSeqState {
    kv: Vec<(DeviceBuf, DeviceBuf)>,
    anchor: DeviceBuf,
    ctx_end: usize,
    cal: (f32, f32),
    sent: Option<(f32, usize)>,
}

pub struct DsparkDrafterGpu {
    ctx: Arc<CudaContextHandle>,
    fns: MtpKernelFns,
    layers: Vec<MtpLayerGpu>,
    norm: DeviceBuf,
    markov_norm: DeviceBuf,
    d_mask_embed: DeviceBuf,
    d_inv_freq: DeviceBuf,
    alpha: [[f32; 4]; 4],
    jctx_q8: PlanarQ8Gpu,
    jb: DeviceBuf,
    j01: DeviceBuf,
    w1_pinned: PinnedHostBuf,
    fold_q8: PlanarQ8Gpu,
    d_fold_b: DeviceBuf,
    // Persistent draft scratch buffers
    d_mix: DeviceBuf,
    d_kv: DeviceBuf,
    d_anchor: DeviceBuf,
    d_x: DeviceBuf,
    d_qkv: DeviceBuf,
    d_att: DeviceBuf,
    d_o: DeviceBuf,
    d_gu: DeviceBuf,
    d_act: DeviceBuf,
    d_hidden: DeviceBuf,
    d_base: DeviceBuf,
    d_frnn: DeviceBuf,
    d_x2: DeviceBuf,
    d_ctxm: DeviceBuf,
    d_state: DeviceBuf,
    d_raw: DeviceBuf,
    d_lat_q8: DeviceBuf,
    d_lat_sc64: DeviceBuf,
    d_fpart: DeviceBuf,
    d_res: DeviceBuf,
    d_q8: DeviceBuf,
    d_sc64: DeviceBuf,
    pub ctx_end: usize,
    /// Parked states of the other sequences (`LEANSONIC_SEQS > 1`); `cur_seq`'s is live.
    seq_states: Vec<Option<DraftSeqState>>,
    cur_seq: usize,
    base_tau: f32,
    cal: (f32, f32),
    sent: Option<(f32, usize)>,
}

impl DsparkDrafterGpu {
    pub fn load_from_file(
        ctx: &Arc<CudaContextHandle>,
        fns: MtpKernelFns,
        mtp_path: &Path,
        unembed_q8: &PlanarQ8Gpu,
        max_ctx: usize,
    ) -> Result<Self, String> {
        let st = MappedSafeTensors::open(mtp_path)
            .map_err(|e| format!("open {}: {e}", mtp_path.display()))?;

        let base_tau = std::env::var("LEANSONIC_SPEC_TAU")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.35);

        let mut inv_freq_bytes = Vec::with_capacity(64 * 4);
        for i in 0..64 {
            let freq = 1_000_000.0_f64.powf(-2.0 * (i as f64) / (HD as f64)) as f32;
            inv_freq_bytes.extend_from_slice(&freq.to_le_bytes());
        }
        let d_inv_freq = DeviceBuf::from_host_bytes(ctx, &inv_freq_bytes)?;

        let fuse_u16 = st.require("fuse_weight", "BF16", &[4, 4]).as_u16_slice();
        let mut alpha = [[0.0f32; 4]; 4];
        for (r, alpha_row) in alpha.iter_mut().enumerate() {
            let mut row = [0.0f32; 4];
            let mut m = f32::NEG_INFINITY;
            for (j, slot) in row.iter_mut().enumerate() {
                let v = bf16_to_f32(fuse_u16[r * 4 + j]);
                *slot = v;
                if v > m {
                    m = v;
                }
            }
            let mut z = 0.0f32;
            for slot in &mut row {
                *slot = (*slot - m).exp();
                z += *slot;
            }
            for (dst, &ev) in alpha_row.iter_mut().zip(row.iter()) {
                *dst = ev / z;
            }
        }

        let mut layers = Vec::with_capacity(3);
        for li in 0..3 {
            let pfx = format!("layers.{li}.");
            let (kv_rows, ring, win) = if li == 1 {
                (max_ctx, 0usize, max_ctx)
            } else {
                (LOCAL_RING, LOCAL_RING, LOCAL_WIN)
            };
            let wq = st
                .require(&format!("{pfx}self_attn.q_proj.weight"), "BF16", &[3072, D])
                .data;
            let wk = st
                .require(&format!("{pfx}self_attn.k_proj.weight"), "BF16", &[1024, D])
                .data;
            let wv = st
                .require(&format!("{pfx}self_attn.v_proj.weight"), "BF16", &[1024, D])
                .data;
            let wkc = st
                .require(
                    &format!("{pfx}self_attn.k_proj_ctx.weight"),
                    "BF16",
                    &[1024, D],
                )
                .data;
            let wvc = st
                .require(
                    &format!("{pfx}self_attn.v_proj_ctx.weight"),
                    "BF16",
                    &[1024, D],
                )
                .data;
            let wo = st
                .require(&format!("{pfx}self_attn.o_proj.weight"), "BF16", &[D, 3072])
                .data;
            let wgate = st
                .require(&format!("{pfx}mlp.gate_proj.weight"), "BF16", &[DFF, D])
                .data;
            let wup = st
                .require(&format!("{pfx}mlp.up_proj.weight"), "BF16", &[DFF, D])
                .data;
            let wdown = st
                .require(&format!("{pfx}mlp.down_proj.weight"), "BF16", &[D, DFF])
                .data;

            layers.push(MtpLayerGpu {
                attn_norm: DeviceBuf::from_host_bytes(
                    ctx,
                    st.require(&format!("{pfx}input_layernorm.weight"), "BF16", &[D])
                        .data,
                )?,
                ffn_norm: DeviceBuf::from_host_bytes(
                    ctx,
                    st.require(
                        &format!("{pfx}post_attention_layernorm.weight"),
                        "BF16",
                        &[D],
                    )
                    .data,
                )?,
                dflare_norm: DeviceBuf::from_host_bytes(
                    ctx,
                    st.require(&format!("{pfx}dflare_norm.weight"), "BF16", &[D])
                        .data,
                )?,
                q_norm: DeviceBuf::from_host_bytes(
                    ctx,
                    st.require(&format!("{pfx}self_attn.q_norm.weight"), "BF16", &[HD])
                        .data,
                )?,
                k_norm: DeviceBuf::from_host_bytes(
                    ctx,
                    st.require(&format!("{pfx}self_attn.k_norm.weight"), "BF16", &[HD])
                        .data,
                )?,
                k_norm_ctx: DeviceBuf::from_host_bytes(
                    ctx,
                    st.require(&format!("{pfx}self_attn.k_norm_ctx.weight"), "BF16", &[HD])
                        .data,
                )?,
                qkv_q8: repack_bf16_to_planar_q8_64(ctx, &concat_slices(&[wq, wk, wv]), 5120, D)?,
                kvc_q8: repack_bf16_to_planar_q8_64(ctx, &concat_slices(&[wkc, wvc]), 2048, D)?,
                wo_q8: repack_bf16_to_planar_q8_64(ctx, wo, D, 3072)?,
                gu_q8: repack_bf16_to_planar_q8_64(ctx, &concat_slices(&[wgate, wup]), 2 * DFF, D)?,
                down_q8: repack_bf16_to_planar_q8_64(ctx, wdown, D, DFF)?,
                kc: DeviceBuf::alloc_zeroed(ctx, kv_rows * 1024 * 2)?,
                vc: DeviceBuf::alloc_zeroed(ctx, kv_rows * 1024 * 2)?,
                ring,
                win,
            });
        }

        let norm = DeviceBuf::from_host_bytes(ctx, st.require("norm.weight", "BF16", &[D]).data)?;
        let markov_norm = DeviceBuf::from_host_bytes(
            ctx,
            st.require("markov_head.dflare_norm.weight", "BF16", &[D])
                .data,
        )?;

        let mask_u16 = st.require("mask_embed", "BF16", &[D]).as_u16_slice();
        let mut mask_bytes = Vec::with_capacity(D * 4);
        for &u in mask_u16 {
            mask_bytes.extend_from_slice(&bf16_to_f32(u).to_le_bytes());
        }
        let d_mask_embed = DeviceBuf::from_host_bytes(ctx, &mask_bytes)?;

        // Split `markov_head.rnn_proj.weight [768, 5120]`:
        // - `[768, 0..512]` -> `j01` (`f32`)
        // - `[768, 512..5120]` -> `jctx_q8` (`PlanarQ8Gpu` `768 x 4608`)
        let rnn_u16 = st
            .require("markov_head.rnn_proj.weight", "BF16", &[768, 5120])
            .as_u16_slice();
        let mut j01_bytes = Vec::with_capacity(768 * 512 * 4);
        let mut jctx_bf16_bytes = Vec::with_capacity(768 * 4608 * 2);
        for o in 0..768 {
            let row = &rnn_u16[o * 5120..(o + 1) * 5120];
            for &u in &row[..512] {
                j01_bytes.extend_from_slice(&bf16_to_f32(u).to_le_bytes());
            }
            for &u in &row[512..] {
                jctx_bf16_bytes.extend_from_slice(&u.to_le_bytes());
            }
        }
        let j01 = DeviceBuf::from_host_bytes(ctx, &j01_bytes)?;
        let jctx_q8 = repack_bf16_to_planar_q8_64(ctx, &jctx_bf16_bytes, 768, 4608)?;

        let jb_u16 = st
            .require("markov_head.rnn_proj.bias", "BF16", &[768])
            .as_u16_slice();
        let mut jb_bytes = Vec::with_capacity(768 * 4);
        for &u in jb_u16 {
            jb_bytes.extend_from_slice(&bf16_to_f32(u).to_le_bytes());
        }
        let jb = DeviceBuf::from_host_bytes(ctx, &jb_bytes)?;

        let w1_raw = st
            .require(
                "markov_head.markov_w1.weight",
                "BF16",
                &[PADDED_VOCAB, R_MARKOV],
            )
            .data;
        let w1_pinned = PinnedHostBuf::from_bytes(ctx, w1_raw)?;

        // Precompute `fold_q8 [VOCAB, 256] = unembed_q8 @ hidden_out.weight` and
        // `d_fold_b [VOCAB] = unembed_q8 @ hidden_out.bias` on GPU in ~60 ms!
        let ho_u16 = st
            .require("markov_head.hidden_out.weight", "BF16", &[D, R_MARKOV])
            .as_u16_slice();
        let hb_u16 = st
            .require("markov_head.hidden_out.bias", "BF16", &[D])
            .as_u16_slice();
        let (fold_q8, d_fold_b) =
            Self::precompute_fold_on_gpu(ctx, &fns, unembed_q8, ho_u16, hb_u16)?;

        let nb_fold = VOCAB.div_ceil(64);
        Ok(Self {
            d_mix: DeviceBuf::alloc_zeroed(ctx, B_DRAFT * D * 4)?,
            d_kv: DeviceBuf::alloc_zeroed(ctx, B_DRAFT * 2048 * 4)?,
            d_anchor: DeviceBuf::alloc_zeroed(ctx, 4 * D * 4)?,
            seq_states: Vec::new(),
            cur_seq: 0,
            d_x: DeviceBuf::alloc_zeroed(ctx, B_DRAFT * D * 4)?,
            d_qkv: DeviceBuf::alloc_zeroed(ctx, B_DRAFT * 5120 * 4)?,
            d_att: DeviceBuf::alloc_zeroed(ctx, B_DRAFT * 3072 * 4)?,
            d_o: DeviceBuf::alloc_zeroed(ctx, B_DRAFT * D * 4)?,
            d_gu: DeviceBuf::alloc_zeroed(ctx, B_DRAFT * 2 * DFF * 4)?,
            d_act: DeviceBuf::alloc_zeroed(ctx, B_DRAFT * DFF * 4)?,
            d_hidden: DeviceBuf::alloc_zeroed(ctx, B_DRAFT * D * 4)?,
            d_base: DeviceBuf::alloc_zeroed(ctx, 7 * VOCAB * 4)?,
            d_frnn: DeviceBuf::alloc_zeroed(ctx, D * 4)?,
            d_x2: DeviceBuf::alloc_zeroed(ctx, 7 * 2 * D * 4)?,
            d_ctxm: DeviceBuf::alloc_zeroed(ctx, 7 * 768 * 4)?,
            d_state: DeviceBuf::alloc_zeroed(ctx, R_MARKOV * 4)?,
            d_raw: DeviceBuf::alloc_zeroed(ctx, 768 * 4)?,
            d_lat_q8: DeviceBuf::alloc_zeroed(ctx, R_MARKOV)?,
            d_lat_sc64: DeviceBuf::alloc_zeroed(ctx, (R_MARKOV / 64) * 4)?,
            d_fpart: DeviceBuf::alloc_zeroed(ctx, nb_fold * 3 * 4)?,
            d_res: DeviceBuf::alloc_zeroed(ctx, 2 * B_DRAFT * 4)?,
            d_q8: DeviceBuf::alloc_zeroed(ctx, B_DRAFT * DFF)?,
            d_sc64: DeviceBuf::alloc_zeroed(ctx, B_DRAFT * (DFF / 64) * 4)?,
            ctx: Arc::clone(ctx),
            fns,
            layers,
            norm,
            markov_norm,
            d_mask_embed,
            d_inv_freq,
            alpha,
            jctx_q8,
            jb,
            j01,
            w1_pinned,
            fold_q8,
            d_fold_b,
            ctx_end: 0,
            base_tau,
            cal: (1.0, 1.0),
            sent: None,
        })
    }

    fn precompute_fold_on_gpu(
        ctx: &Arc<CudaContextHandle>,
        fns: &MtpKernelFns,
        unembed_q8: &PlanarQ8Gpu,
        ho_u16: &[u16],
        hb_u16: &[u16],
    ) -> Result<(PlanarQ8Gpu, DeviceBuf), String> {
        // Transpose `ho [D, 256]` -> `ho_t [256, D]` in BF16 and quantize to Q8_64 `[257, D]`
        let mut ho_b_bf16 = Vec::with_capacity((R_MARKOV + 1) * D * 2);
        for k in 0..R_MARKOV {
            for d in 0..D {
                ho_b_bf16.extend_from_slice(&ho_u16[d * R_MARKOV + k].to_le_bytes());
            }
        }
        for &u in hb_u16 {
            ho_b_bf16.extend_from_slice(&u.to_le_bytes());
        }
        let ho_q8 = repack_bf16_to_planar_q8_64(ctx, &ho_b_bf16, R_MARKOV + 1, D)?;

        // 1. Compute `d_fold_b [VOCAB] = unembed_q8 @ hb_q8` (`row 256` of `ho_q8`)
        let d_fold_b = DeviceBuf::alloc_zeroed(ctx, VOCAB * 4)?;
        {
            let mut pw = unembed_q8.codes.ptr;
            let mut pws = unembed_q8.scales64.ptr;
            let mut px = ho_q8.codes.ptr + (R_MARKOV * D) as u64;
            let mut pxs = ho_q8.scales64.ptr + (R_MARKOV * (D / 64) * 4) as u64;
            let mut py = d_fold_b.ptr;
            let mut r = VOCAB as i32;
            let mut c = D as i32;
            let mut sc = 0.0f32;
            let mut args: [*mut c_void; 8] = [
                (&mut pw as *mut _) as *mut c_void,
                (&mut pws as *mut _) as *mut c_void,
                (&mut px as *mut _) as *mut c_void,
                (&mut pxs as *mut _) as *mut c_void,
                (&mut py as *mut _) as *mut c_void,
                (&mut r as *mut _) as *mut c_void,
                (&mut c as *mut _) as *mut c_void,
                (&mut sc as *mut _) as *mut c_void,
            ];
            let gx = (VOCAB as u32).div_ceil(16);
            unsafe {
                check_cu(
                    (ctx.api.cuLaunchKernel)(
                        fns.q8_dp4a_gemv,
                        gx,
                        1,
                        1,
                        256,
                        1,
                        1,
                        0,
                        ctx.compute_stream,
                        args.as_mut_ptr(),
                        core::ptr::null_mut(),
                    ),
                    "fold_b gemv",
                )?;
            }
        }

        // 2. Compute `fold [VOCAB, 256]` in 4 `64`-col passes (`B = 64` in `ink2_q8_dp4a_gemm_b`)
        //    over `16,384`-vocab-row slices (`16 MiB` temporary GPU buffer)!
        const V_STEP: usize = 16_384;
        let d_tile = DeviceBuf::alloc_zeroed(ctx, 64 * V_STEP * 4)?;
        let mut h_tile = vec![0.0f32; 64 * V_STEP];
        let mut fold_codes = vec![0u8; VOCAB * R_MARKOV];
        let mut fold_scales = vec![0u8; VOCAB * (R_MARKOV / 64) * 4];

        for v0 in (0..VOCAB).step_by(V_STEP) {
            let v_rows = V_STEP.min(VOCAB - v0);
            for k_blk in 0..4usize {
                let k0 = k_blk * 64;
                let mut pw = unembed_q8.codes.ptr + (v0 * D) as u64;
                let mut pws = unembed_q8.scales64.ptr + (v0 * (D / 64) * 4) as u64;
                let mut px = ho_q8.codes.ptr + (k0 * D) as u64;
                let mut pxs = ho_q8.scales64.ptr + (k0 * (D / 64) * 4) as u64;
                let mut py = d_tile.ptr;
                let mut r = v_rows as i32;
                let mut c = D as i32;
                let mut bi = 64i32;
                let mut ys = v_rows as i32;
                let mut sc = 0.0f32;
                let mut args: [*mut c_void; 10] = [
                    (&mut pw as *mut _) as *mut c_void,
                    (&mut pws as *mut _) as *mut c_void,
                    (&mut px as *mut _) as *mut c_void,
                    (&mut pxs as *mut _) as *mut c_void,
                    (&mut py as *mut _) as *mut c_void,
                    (&mut r as *mut _) as *mut c_void,
                    (&mut c as *mut _) as *mut c_void,
                    (&mut bi as *mut _) as *mut c_void,
                    (&mut ys as *mut _) as *mut c_void,
                    (&mut sc as *mut _) as *mut c_void,
                ];
                let gx = (v_rows as u32).div_ceil(16);
                unsafe {
                    check_cu(
                        (ctx.api.cuLaunchKernel)(
                            fns.q8_dp4a_gemm_b,
                            gx,
                            8,
                            1,
                            256,
                            1,
                            1,
                            0,
                            ctx.compute_stream,
                            args.as_mut_ptr(),
                            core::ptr::null_mut(),
                        ),
                        "fold_tile gemm_b",
                    )?;
                }
                ctx.sync_compute()?;
                d_tile.copy_to_host_f32(&mut h_tile[..64 * v_rows])?;

                // `h_tile` has layout `[64, v_rows]` (`h_tile[k_sub * v_rows + vr]`).
                // Each `(vr, k_blk)` is one exact `64`-element `Q8_64` block!
                for vr in 0..v_rows {
                    let gv = v0 + vr;
                    let mut vals = [0.0f32; 64];
                    let mut amax = 0.0f32;
                    for (ks, slot) in vals.iter_mut().enumerate() {
                        let fv = h_tile[ks * v_rows + vr];
                        *slot = fv;
                        let av = fv.abs();
                        if av > amax {
                            amax = av;
                        }
                    }
                    let scale = amax * (1.0 / 127.0);
                    let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
                    let sc_off = (gv * 4 + k_blk) * 4;
                    fold_scales[sc_off..sc_off + 4].copy_from_slice(&scale.to_le_bytes());
                    let c_off = gv * R_MARKOV + k0;
                    for (ks, &fv) in vals.iter().enumerate() {
                        let q = (fv * inv).round().clamp(-127.0, 127.0) as i8;
                        fold_codes[c_off + ks] = q as u8;
                    }
                }
            }
        }

        let fold_q8 = PlanarQ8Gpu {
            codes: DeviceBuf::from_host_bytes(ctx, &fold_codes)?,
            scales64: DeviceBuf::from_host_bytes(ctx, &fold_scales)?,
            cols: R_MARKOV as i32,
        };
        Ok((fold_q8, d_fold_b))
    }

    /// Makes sequence `seq`'s drafter context live (a new sequence starts empty, as after
    /// `reset`). Drafting and context updates then apply to that sequence only.
    pub fn select_seq(&mut self, seq: usize) -> Result<(), String> {
        if seq == self.cur_seq {
            return Ok(());
        }
        if self.seq_states.len() <= seq.max(self.cur_seq) {
            self.seq_states
                .resize_with(seq.max(self.cur_seq) + 1, || None);
        }
        let mut st = match self.seq_states[seq].take() {
            Some(st) => st,
            None => {
                let mut kv = Vec::with_capacity(self.layers.len());
                for l in &self.layers {
                    kv.push((
                        DeviceBuf::alloc_zeroed(&self.ctx, l.kc.byte_len)?,
                        DeviceBuf::alloc_zeroed(&self.ctx, l.vc.byte_len)?,
                    ));
                }
                DraftSeqState {
                    kv,
                    anchor: DeviceBuf::alloc_zeroed(&self.ctx, self.d_anchor.byte_len)?,
                    ctx_end: 0,
                    cal: (1.0, 1.0),
                    sent: None,
                }
            }
        };
        for (l, (kc, vc)) in self.layers.iter_mut().zip(st.kv.iter_mut()) {
            std::mem::swap(&mut l.kc, kc);
            std::mem::swap(&mut l.vc, vc);
        }
        std::mem::swap(&mut self.d_anchor, &mut st.anchor);
        std::mem::swap(&mut self.ctx_end, &mut st.ctx_end);
        std::mem::swap(&mut self.cal, &mut st.cal);
        std::mem::swap(&mut self.sent, &mut st.sent);
        self.seq_states[self.cur_seq] = Some(st);
        self.cur_seq = seq;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.ctx_end = 0;
        self.cal = (1.0, 1.0);
        self.sent = None;
    }

    fn launch(
        &self,
        stream: CUstream,
        func: CUfunction,
        grid: (u32, u32),
        block: u32,
        params: &mut [*mut c_void],
    ) -> Result<(), String> {
        unsafe {
            check_cu(
                (self.ctx.api.cuLaunchKernel)(
                    func,
                    grid.0,
                    grid.1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    stream,
                    params.as_mut_ptr(),
                    core::ptr::null_mut(),
                ),
                "mtp cuLaunchKernel",
            )
        }
    }

    fn enqueue_rmsnorm_quant_q8(
        &self,
        stream: CUstream,
        inp_ptr: CUdeviceptr,
        gamma_ptr: CUdeviceptr,
        b_len: usize,
    ) -> Result<(), String> {
        let mut pin = inp_ptr;
        let mut pg = gamma_ptr;
        let mut pq = self.d_q8.ptr;
        let mut ps = self.d_sc64.ptr;
        let mut bi = b_len as i32;
        let mut args: [*mut c_void; 5] = [
            (&mut pin as *mut _) as *mut c_void,
            (&mut pg as *mut _) as *mut c_void,
            (&mut pq as *mut _) as *mut c_void,
            (&mut ps as *mut _) as *mut c_void,
            (&mut bi as *mut _) as *mut c_void,
        ];
        self.launch(
            stream,
            self.fns.rmsnorm_quant_q8_64_b,
            (b_len as u32, 1),
            256,
            &mut args,
        )
    }

    fn enqueue_quant_q8(
        &self,
        stream: CUstream,
        x_ptr: CUdeviceptr,
        cols: usize,
        b_len: usize,
    ) -> Result<(), String> {
        let mut px = x_ptr;
        let mut pq = self.d_q8.ptr;
        let mut ps = self.d_sc64.ptr;
        let mut c = cols as i32;
        let mut args: [*mut c_void; 4] = [
            (&mut px as *mut _) as *mut c_void,
            (&mut pq as *mut _) as *mut c_void,
            (&mut ps as *mut _) as *mut c_void,
            (&mut c as *mut _) as *mut c_void,
        ];
        self.launch(
            stream,
            self.fns.quant_q8_64_b,
            ((cols / 64) as u32, b_len as u32),
            64,
            &mut args,
        )
    }

    fn enqueue_q8_ktiled(
        &self,
        stream: CUstream,
        w: &PlanarQ8Gpu,
        rows: i32,
        b_len: usize,
        y_ptr: CUdeviceptr,
    ) -> Result<(), String> {
        let mut pw = w.codes.ptr;
        let mut pws = w.scales64.ptr;
        let mut px = self.d_q8.ptr;
        let mut pxs = self.d_sc64.ptr;
        let mut py = y_ptr;
        let mut r = rows;
        let mut c = w.cols;
        let mut bi = b_len as i32;
        let mut ys = rows;
        let mut args: [*mut c_void; 9] = [
            (&mut pw as *mut _) as *mut c_void,
            (&mut pws as *mut _) as *mut c_void,
            (&mut px as *mut _) as *mut c_void,
            (&mut pxs as *mut _) as *mut c_void,
            (&mut py as *mut _) as *mut c_void,
            (&mut r as *mut _) as *mut c_void,
            (&mut c as *mut _) as *mut c_void,
            (&mut bi as *mut _) as *mut c_void,
            (&mut ys as *mut _) as *mut c_void,
        ];
        let gx = (rows as u32).div_ceil(16);
        if w.cols <= 3072 && b_len <= 8 {
            if let Some(f_b8) = self.fns.q8_dp4a_gemm_b8_fast {
                let mut sc = 0.0f32;
                let mut args10: [*mut c_void; 10] = [
                    (&mut pw as *mut _) as *mut c_void,
                    (&mut pws as *mut _) as *mut c_void,
                    (&mut px as *mut _) as *mut c_void,
                    (&mut pxs as *mut _) as *mut c_void,
                    (&mut py as *mut _) as *mut c_void,
                    (&mut r as *mut _) as *mut c_void,
                    (&mut c as *mut _) as *mut c_void,
                    (&mut bi as *mut _) as *mut c_void,
                    (&mut ys as *mut _) as *mut c_void,
                    (&mut sc as *mut _) as *mut c_void,
                ];
                return self.launch(stream, f_b8, (gx, 1), 256, &mut args10);
            }
        }
        self.launch(
            stream,
            self.fns.q8_dp4a_gemm_ktiled,
            (gx, 1),
            256,
            &mut args,
        )
    }

    /// Appends `n_toks` newly verified/prefilled target tokens (`taps [4, tap_cap, D]`) into DSpark's KV context.
    pub fn enqueue_add_ctx(
        &mut self,
        stream: CUstream,
        d_taps_ptr: CUdeviceptr,
        tap_cap: usize,
        n_toks: usize,
    ) -> Result<(), String> {
        if n_toks == 0 {
            return Ok(());
        }
        if let Some((ps, k)) = self.sent.take() {
            let a = n_toks.saturating_sub(1).min(k) as f32;
            self.cal = (0.9 * self.cal.0 + 0.1 * ps, 0.9 * self.cal.1 + 0.1 * a);
        }

        let mut c0 = 0usize;
        while c0 < n_toks {
            let m = (n_toks - c0).min(B_DRAFT);
            for li in 0..3 {
                let a = self.alpha[li];
                {
                    let mut pt = d_taps_ptr + (c0 * D * 4) as u64;
                    let mut ts = (tap_cap * D) as i32;
                    let mut a0 = a[0];
                    let mut a1 = a[1];
                    let mut a2 = a[2];
                    let mut a3 = a[3];
                    let mut nf = (m * D) as i32;
                    let mut pout = self.d_mix.ptr;
                    let mut args: [*mut c_void; 8] = [
                        (&mut pt as *mut _) as *mut c_void,
                        (&mut ts as *mut _) as *mut c_void,
                        (&mut a0 as *mut _) as *mut c_void,
                        (&mut a1 as *mut _) as *mut c_void,
                        (&mut a2 as *mut _) as *mut c_void,
                        (&mut a3 as *mut _) as *mut c_void,
                        (&mut nf as *mut _) as *mut c_void,
                        (&mut pout as *mut _) as *mut c_void,
                    ];
                    let gx = ((m * D) as u32).div_ceil(256);
                    self.launch(stream, self.fns.tapmix, (gx, 1), 256, &mut args)?;
                }
                let dflare_ptr = self.layers[li].dflare_norm.ptr;
                self.enqueue_rmsnorm_quant_q8(stream, self.d_mix.ptr, dflare_ptr, m)?;
                self.enqueue_q8_ktiled(stream, &self.layers[li].kvc_q8, 2048, m, self.d_kv.ptr)?;
                {
                    let l = &self.layers[li];
                    let mut pkv = self.d_kv.ptr;
                    let mut pknc = l.k_norm_ctx.ptr;
                    let mut pinv = self.d_inv_freq.ptr;
                    let mut p0 = (self.ctx_end + c0) as i32;
                    let mut rng = l.ring as i32;
                    let mut pkc = l.kc.ptr;
                    let mut pvc = l.vc.ptr;
                    let mut args: [*mut c_void; 7] = [
                        (&mut pkv as *mut _) as *mut c_void,
                        (&mut pknc as *mut _) as *mut c_void,
                        (&mut pinv as *mut _) as *mut c_void,
                        (&mut p0 as *mut _) as *mut c_void,
                        (&mut rng as *mut _) as *mut c_void,
                        (&mut pkc as *mut _) as *mut c_void,
                        (&mut pvc as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, self.fns.ctx_store, (8, m as u32), 128, &mut args)?;
                }
            }
            c0 += m;
        }

        {
            let mut pt = d_taps_ptr;
            let mut tc = tap_cap as i32;
            let mut ti = (n_toks - 1) as i32;
            let mut pa = self.d_anchor.ptr;
            let mut args: [*mut c_void; 4] = [
                (&mut pt as *mut _) as *mut c_void,
                (&mut tc as *mut _) as *mut c_void,
                (&mut ti as *mut _) as *mut c_void,
                (&mut pa as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.copy_anchor, (9, 4), 256, &mut args)?;
        }
        self.ctx_end += n_toks;
        Ok(())
    }

    /// Proposes up to `max_k` (`<= 7`) draft tokens following `next_tok` at position `self.ctx_end`.
    pub fn draft_tokens(
        &mut self,
        stream: CUstream,
        embed_ptr: CUdeviceptr,
        embed_norm_ptr: CUdeviceptr,
        unembed_q8: &PlanarQ8Gpu,
        next_tok: u32,
        max_k: usize,
    ) -> Result<Vec<u32>, String> {
        let max_k = max_k.min(B_DRAFT - 1);
        if max_k == 0 {
            return Ok(Vec::new());
        }
        let tau = self.base_tau * (self.cal.0 / self.cal.1.max(1e-3)).clamp(0.5, 2.0);
        let p = self.ctx_end;

        // 0. Init 8-row block `[embed_rmsnorm(next_tok), 7 x mask_embed]`, zero `d_state`, set `res[0..2]`
        {
            let mut pe = embed_ptr;
            let mut pn = embed_norm_ptr;
            let mut pm = self.d_mask_embed.ptr;
            let mut nt = next_tok as i32;
            let mut px = self.d_x.ptr;
            let mut pst = self.d_state.ptr;
            let mut pres = self.d_res.ptr;
            let mut args: [*mut c_void; 7] = [
                (&mut pe as *mut _) as *mut c_void,
                (&mut pn as *mut _) as *mut c_void,
                (&mut pm as *mut _) as *mut c_void,
                (&mut nt as *mut _) as *mut c_void,
                (&mut px as *mut _) as *mut c_void,
                (&mut pst as *mut _) as *mut c_void,
                (&mut pres as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.init_draft, (1, 1), 256, &mut args)?;
        }

        // 1. 3 Dense SwiGLU + Bidirectional Block/Context Attention Layers (`B = 8`)
        for li in 0..3 {
            let attn_norm_ptr = self.layers[li].attn_norm.ptr;
            self.enqueue_rmsnorm_quant_q8(stream, self.d_x.ptr, attn_norm_ptr, B_DRAFT)?;
            self.enqueue_q8_ktiled(
                stream,
                &self.layers[li].qkv_q8,
                5120,
                B_DRAFT,
                self.d_qkv.ptr,
            )?;
            {
                let l = &self.layers[li];
                let mut pqkv = self.d_qkv.ptr;
                let mut pqn = l.q_norm.ptr;
                let mut pkn = l.k_norm.ptr;
                let mut pinv = self.d_inv_freq.ptr;
                let mut p0 = p as i32;
                let mut args: [*mut c_void; 5] = [
                    (&mut pqkv as *mut _) as *mut c_void,
                    (&mut pqn as *mut _) as *mut c_void,
                    (&mut pkn as *mut _) as *mut c_void,
                    (&mut pinv as *mut _) as *mut c_void,
                    (&mut p0 as *mut _) as *mut c_void,
                ];
                self.launch(stream, self.fns.qk_norm_rope, (32, 8), 128, &mut args)?;
            }
            {
                let l = &self.layers[li];
                let lo = p.saturating_sub(l.win);
                let mut pqkv = self.d_qkv.ptr;
                let mut pkc = l.kc.ptr;
                let mut pvc = l.vc.ptr;
                let mut rng = l.ring as i32;
                let mut loi = lo as i32;
                let mut hii = p as i32;
                let mut patt = self.d_att.ptr;
                let mut args: [*mut c_void; 7] = [
                    (&mut pqkv as *mut _) as *mut c_void,
                    (&mut pkc as *mut _) as *mut c_void,
                    (&mut pvc as *mut _) as *mut c_void,
                    (&mut rng as *mut _) as *mut c_void,
                    (&mut loi as *mut _) as *mut c_void,
                    (&mut hii as *mut _) as *mut c_void,
                    (&mut patt as *mut _) as *mut c_void,
                ];
                self.launch(stream, self.fns.attn, (24, 8), 256, &mut args)?;
            }
            self.enqueue_quant_q8(stream, self.d_att.ptr, 3072, B_DRAFT)?;
            self.enqueue_q8_ktiled(
                stream,
                &self.layers[li].wo_q8,
                D as i32,
                B_DRAFT,
                self.d_o.ptr,
            )?;
            self.enqueue_add_inplace(stream, self.d_x.ptr, self.d_o.ptr, B_DRAFT * D)?;

            let ffn_norm_ptr = self.layers[li].ffn_norm.ptr;
            self.enqueue_rmsnorm_quant_q8(stream, self.d_x.ptr, ffn_norm_ptr, B_DRAFT)?;
            self.enqueue_q8_ktiled(
                stream,
                &self.layers[li].gu_q8,
                (2 * DFF) as i32,
                B_DRAFT,
                self.d_gu.ptr,
            )?;
            {
                let mut pgu = self.d_gu.ptr;
                let mut pact = self.d_act.ptr;
                let mut args: [*mut c_void; 2] = [
                    (&mut pgu as *mut _) as *mut c_void,
                    (&mut pact as *mut _) as *mut c_void,
                ];
                let gx = ((B_DRAFT * DFF) as u32).div_ceil(256);
                self.launch(stream, self.fns.silu_mul, (gx, 1), 256, &mut args)?;
            }
            self.enqueue_quant_q8(stream, self.d_act.ptr, DFF, B_DRAFT)?;
            self.enqueue_q8_ktiled(
                stream,
                &self.layers[li].down_q8,
                D as i32,
                B_DRAFT,
                self.d_o.ptr,
            )?;
            self.enqueue_add_inplace(stream, self.d_x.ptr, self.d_o.ptr, B_DRAFT * D)?;
        }

        // 2. Final Norm + Shared `unembed_q8` Base Logits (`rows 1..7`) + `frnn` + `ctxm`
        {
            let mut pin = self.d_x.ptr;
            let mut pg = self.norm.ptr;
            let mut pout = self.d_hidden.ptr;
            let mut hg = 1i32;
            let mut args: [*mut c_void; 4] = [
                (&mut pin as *mut _) as *mut c_void,
                (&mut pg as *mut _) as *mut c_void,
                (&mut pout as *mut _) as *mut c_void,
                (&mut hg as *mut _) as *mut c_void,
            ];
            self.launch(
                stream,
                self.fns.rmsnorm_d_b,
                (B_DRAFT as u32, 1),
                256,
                &mut args,
            )?;
        }
        let hidden_1_7 = self.d_hidden.ptr + (D * 4) as u64;
        self.enqueue_quant_q8(stream, hidden_1_7, D, 7)?;
        {
            let mut pw = unembed_q8.codes.ptr;
            let mut pws = unembed_q8.scales64.ptr;
            let mut px = self.d_q8.ptr;
            let mut pxs = self.d_sc64.ptr;
            let mut py = self.d_base.ptr;
            let mut r = VOCAB as i32;
            let mut c = D as i32;
            let mut bi = 7i32;
            let mut ys = VOCAB as i32;
            let mut sc = 0.0f32;
            let mut args: [*mut c_void; 10] = [
                (&mut pw as *mut _) as *mut c_void,
                (&mut pws as *mut _) as *mut c_void,
                (&mut px as *mut _) as *mut c_void,
                (&mut pxs as *mut _) as *mut c_void,
                (&mut py as *mut _) as *mut c_void,
                (&mut r as *mut _) as *mut c_void,
                (&mut c as *mut _) as *mut c_void,
                (&mut bi as *mut _) as *mut c_void,
                (&mut ys as *mut _) as *mut c_void,
                (&mut sc as *mut _) as *mut c_void,
            ];
            let gx = (VOCAB as u32).div_ceil(16);
            let f_unembed = self
                .fns
                .q8_dp4a_gemm_b8_fast
                .unwrap_or(self.fns.q8_dp4a_gemm_b);
            self.launch(stream, f_unembed, (gx, 1), 256, &mut args)?;
        }

        let a3 = self.alpha[3];
        {
            let mut pt = self.d_anchor.ptr;
            let mut ts = D as i32;
            let mut v0 = a3[0];
            let mut v1 = a3[1];
            let mut v2 = a3[2];
            let mut v3 = a3[3];
            let mut nf = D as i32;
            let mut pout = self.d_frnn.ptr;
            let mut args: [*mut c_void; 8] = [
                (&mut pt as *mut _) as *mut c_void,
                (&mut ts as *mut _) as *mut c_void,
                (&mut v0 as *mut _) as *mut c_void,
                (&mut v1 as *mut _) as *mut c_void,
                (&mut v2 as *mut _) as *mut c_void,
                (&mut v3 as *mut _) as *mut c_void,
                (&mut nf as *mut _) as *mut c_void,
                (&mut pout as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.tapmix, (9, 1), 256, &mut args)?;
        }
        {
            let mut pin = self.d_frnn.ptr;
            let mut pg = self.markov_norm.ptr;
            let mut pout = self.d_frnn.ptr;
            let mut hg = 1i32;
            let mut args: [*mut c_void; 4] = [
                (&mut pin as *mut _) as *mut c_void,
                (&mut pg as *mut _) as *mut c_void,
                (&mut pout as *mut _) as *mut c_void,
                (&mut hg as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.rmsnorm_d_b, (1, 1), 256, &mut args)?;
        }
        {
            let mut ph = hidden_1_7;
            let mut pf = self.d_frnn.ptr;
            let mut px2 = self.d_x2.ptr;
            let mut args: [*mut c_void; 3] = [
                (&mut ph as *mut _) as *mut c_void,
                (&mut pf as *mut _) as *mut c_void,
                (&mut px2 as *mut _) as *mut c_void,
            ];
            let gx = ((7 * 4608) as u32).div_ceil(256);
            self.launch(stream, self.fns.concat_frnn, (gx, 1), 256, &mut args)?;
        }
        self.enqueue_quant_q8(stream, self.d_x2.ptr, 4608, 7)?;
        self.enqueue_q8_ktiled(stream, &self.jctx_q8, 768, 7, self.d_ctxm.ptr)?;
        {
            let mut py = self.d_ctxm.ptr;
            let mut pb = self.jb.ptr;
            let mut cols = 768i32;
            let mut tot = 7 * 768i32;
            let mut args: [*mut c_void; 4] = [
                (&mut py as *mut _) as *mut c_void,
                (&mut pb as *mut _) as *mut c_void,
                (&mut cols as *mut _) as *mut c_void,
                (&mut tot as *mut _) as *mut c_void,
            ];
            let gx = ((7 * 768) as u32).div_ceil(256);
            self.launch(stream, self.fns.add_bias_rows, (gx, 1), 256, &mut args)?;
        }

        // 3. 7-Step Autoregressive Markov RNN Head (100% on GPU, single 64B DtoH at the end!)
        let nb = VOCAB.div_ceil(64) as i32;
        let cum_ptr = self.d_res.ptr + 4;
        for step in 0..max_k {
            let prev_ptr = self.d_res.ptr + (step * 8) as u64;
            let out_best_ptr = self.d_res.ptr + ((step + 1) * 8) as u64;
            let out_sumexp_ptr = out_best_ptr + 4;

            {
                let mut pctxm = self.d_ctxm.ptr + (step * 768 * 4) as u64;
                let mut pj01 = self.j01.ptr;
                let mut pw1 = self.w1_pinned.dev_ptr;
                let mut pst = self.d_state.ptr;
                let mut pprev = prev_ptr;
                let mut praw = self.d_raw.ptr;
                let mut pcum = cum_ptr;
                let mut args: [*mut c_void; 7] = [
                    (&mut pctxm as *mut _) as *mut c_void,
                    (&mut pj01 as *mut _) as *mut c_void,
                    (&mut pw1 as *mut _) as *mut c_void,
                    (&mut pst as *mut _) as *mut c_void,
                    (&mut pprev as *mut _) as *mut c_void,
                    (&mut praw as *mut _) as *mut c_void,
                    (&mut pcum as *mut _) as *mut c_void,
                ];
                self.launch(stream, self.fns.markov_raw, (96, 1), 256, &mut args)?;
            }
            {
                let mut praw = self.d_raw.ptr;
                let mut pst = self.d_state.ptr;
                let mut plq = self.d_lat_q8.ptr;
                let mut pls = self.d_lat_sc64.ptr;
                let mut pcum = cum_ptr;
                let mut args: [*mut c_void; 5] = [
                    (&mut praw as *mut _) as *mut c_void,
                    (&mut pst as *mut _) as *mut c_void,
                    (&mut plq as *mut _) as *mut c_void,
                    (&mut pls as *mut _) as *mut c_void,
                    (&mut pcum as *mut _) as *mut c_void,
                ];
                self.launch(stream, self.fns.markov_step_q8, (1, 1), 256, &mut args)?;
            }
            {
                let mut pfq = self.fold_q8.codes.ptr;
                let mut pfs = self.fold_q8.scales64.ptr;
                let mut plq = self.d_lat_q8.ptr;
                let mut pls = self.d_lat_sc64.ptr;
                let mut pfb = self.d_fold_b.ptr;
                let mut pbase = self.d_base.ptr + (step * VOCAB * 4) as u64;
                let mut ppart = self.d_fpart.ptr;
                let mut pcum = cum_ptr;
                let mut args: [*mut c_void; 8] = [
                    (&mut pfq as *mut _) as *mut c_void,
                    (&mut pfs as *mut _) as *mut c_void,
                    (&mut plq as *mut _) as *mut c_void,
                    (&mut pls as *mut _) as *mut c_void,
                    (&mut pfb as *mut _) as *mut c_void,
                    (&mut pbase as *mut _) as *mut c_void,
                    (&mut ppart as *mut _) as *mut c_void,
                    (&mut pcum as *mut _) as *mut c_void,
                ];
                self.launch(stream, self.fns.fold_mv_q8, (nb as u32, 1), 256, &mut args)?;
            }
            {
                let mut ppart = self.d_fpart.ptr;
                let mut nbi = nb;
                let mut pbest = out_best_ptr;
                let mut psum = out_sumexp_ptr;
                let mut pcum = cum_ptr;
                let mut tf = tau;
                let mut ilast = i32::from(step + 1 >= max_k);
                let mut args: [*mut c_void; 7] = [
                    (&mut ppart as *mut _) as *mut c_void,
                    (&mut nbi as *mut _) as *mut c_void,
                    (&mut pbest as *mut _) as *mut c_void,
                    (&mut psum as *mut _) as *mut c_void,
                    (&mut pcum as *mut _) as *mut c_void,
                    (&mut tf as *mut _) as *mut c_void,
                    (&mut ilast as *mut _) as *mut c_void,
                ];
                self.launch(stream, self.fns.fold_fin, (1, 1), 256, &mut args)?;
            }
        }

        self.ctx.sync_compute()?;
        let mut raw_res = [0u32; 2 * B_DRAFT];
        unsafe {
            check_cu(
                (self.ctx.api.cuMemcpyDtoH_v2)(
                    raw_res.as_mut_ptr().cast(),
                    self.d_res.ptr,
                    2 * B_DRAFT * 4,
                ),
                "DtoH(mtp_res)",
            )?;
        }

        let mut cum = 1.0f32;
        let mut ps = 0.0f32;
        let mut out = Vec::with_capacity(max_k);
        for step in 0..max_k {
            let z = f32::from_bits(raw_res[2 * (step + 1) + 1]).max(1.0);
            cum /= z;
            if cum < tau || out.len() == max_k {
                break;
            }
            out.push(raw_res[2 * (step + 1)]);
            ps += cum;
        }
        self.sent = Some((ps, out.len()));
        Ok(out)
    }

    fn enqueue_add_inplace(
        &self,
        stream: CUstream,
        x_ptr: CUdeviceptr,
        o_ptr: CUdeviceptr,
        n: usize,
    ) -> Result<(), String> {
        let mut px = x_ptr;
        let mut po = o_ptr;
        let mut ni = n as i32;
        let mut args: [*mut c_void; 3] = [
            (&mut px as *mut _) as *mut c_void,
            (&mut po as *mut _) as *mut c_void,
            (&mut ni as *mut _) as *mut c_void,
        ];
        let gx = (n as u32).div_ceil(256);
        self.launch(stream, self.fns.add_inplace, (gx, 1), 256, &mut args)
    }
}

/// Zero-VRAM N-Gram Suffix Drafter fallback (used when `mtp.safetensors` is absent).
pub struct SuffixDrafter;

impl SuffixDrafter {
    /// Searches backwards in `history` (prompt + accepted tokens + `next_tok`) for the longest
    /// suffix match (`n = 4, 3, 2`) and returns up to `max_k` continuation tokens.
    pub fn draft_from_history(history: &[u32], next_tok: u32, max_k: usize) -> Vec<u32> {
        if max_k == 0 || history.len() < 4 {
            return Vec::new();
        }
        let mut full = Vec::with_capacity(history.len() + 1);
        full.extend_from_slice(history);
        full.push(next_tok);
        let n_tot = full.len();
        let search_limit = n_tot.saturating_sub(max_k + 1);

        for gram in [4usize, 3, 2] {
            if n_tot <= gram + max_k {
                continue;
            }
            let needle = &full[n_tot - gram..];
            let start_min = search_limit.saturating_sub(2048);
            let mut pos = search_limit;
            while pos >= start_min + gram {
                pos -= 1;
                if &full[pos + 1 - gram..=pos] == needle {
                    let avail = (n_tot - 1 - pos).min(max_k);
                    return full[pos + 1..pos + 1 + avail].to_vec();
                }
            }
        }
        Vec::new()
    }
}
