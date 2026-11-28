//! Qwen3.8-Flash-Next built-in MTP drafter (`language_model.mtp.*`, 76 tensors, #248 Track 3).
//!
//! The MTP head is one full-attention decoder layer that sees the target's hyper-connection
//! state after its last layer (`hc_hidden`, `[4, 2560]`, before the output mixer) together with
//! the next token:
//!
//! ```text
//! e      = fc_embedding(GemmaRMSNorm_2560(embed(tok_{t+1})))
//! h      = GemmaRMSNorm_10240(hc_hidden_t)            (all four streams together)
//! hc_s   = fc_hidden(h_s) + e                          (s = 0..3)
//! hc     = layer(hc, pos)                              (QSA attention + 512-expert MoE)
//! logits = lm_head(hyper_connection_mixer(hc))         (no final norm, shared lm_head)
//! ```
//!
//! and drafts `tok_{t+2}`. The layer reuses the `kernels/qwen38_sm120.cu` kernels through the same
//! C ABI; only the input fusion lives in `kernels/qwen38_mtp_sm120.cu`. The drafter keeps its own
//! KV cache and position, and borrows the target's embedding table and LM head (`Q4Ref`), which
//! must live in the same CUDA context.

use core::ffi::c_void;
use std::ffi::CString;
use std::sync::Arc;

use crate::cuda_ffi::{
    check_cu, CUdeviceptr, CUfunction, CUmodule, CUstream, CudaContextHandle, DeviceBuf,
    PinnedHostBuf,
};
use crate::qwen38_gpu::{
    Q38_D, Q38_EXP, Q38_HC, Q38_HC_D, Q38_HC_LOW, Q38_MOE_FF, Q38_TOP_K, Q38_VOCAB,
};
use crate::safetensors::ShardedSafeTensors;

macro_rules! kargs {
    ($($arg:ident),+ $(,)?) => {
        [ $(core::ptr::from_mut(&mut $arg).cast::<c_void>()),+ ]
    };
}

const MTP: &str = "language_model.mtp";

/// A target-owned group-32 Q4 matrix: `w [rows, cols / 32]` u128, `sc` / `bi` `[rows, cols / 64]`
/// u32 (two BF16 each).
#[derive(Clone, Copy, Debug)]
pub struct Q4Ref {
    pub w: CUdeviceptr,
    pub sc: CUdeviceptr,
    pub bi: CUdeviceptr,
    pub rows: i32,
    pub cols: i32,
}

struct Q4 {
    w: DeviceBuf,
    sc: DeviceBuf,
    bi: DeviceBuf,
    rows: i32,
    cols: i32,
}

impl Q4 {
    fn load(
        ctx: &Arc<CudaContextHandle>,
        st: &ShardedSafeTensors,
        parts: &[(&str, usize)],
        cols: usize,
    ) -> Result<Self, String> {
        let (mut w, mut sc, mut bi, mut rows) = (Vec::new(), Vec::new(), Vec::new(), 0usize);
        for &(prefix, r) in parts {
            w.extend_from_slice(
                st.require(&format!("{prefix}.weight"), "U32", &[r, cols / 8])
                    .data,
            );
            sc.extend_from_slice(
                st.require(&format!("{prefix}.scales"), "BF16", &[r, cols / 32])
                    .data,
            );
            bi.extend_from_slice(
                st.require(&format!("{prefix}.biases"), "BF16", &[r, cols / 32])
                    .data,
            );
            rows += r;
        }
        Ok(Self {
            w: DeviceBuf::from_host_bytes(ctx, &w)?,
            sc: DeviceBuf::from_host_bytes(ctx, &sc)?,
            bi: DeviceBuf::from_host_bytes(ctx, &bi)?,
            rows: rows as i32,
            cols: cols as i32,
        })
    }

    fn r(&self) -> Q4Ref {
        Q4Ref {
            w: self.w.ptr,
            sc: self.sc.ptr,
            bi: self.bi.ptr,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

fn f32_norm(
    ctx: &Arc<CudaContextHandle>,
    st: &ShardedSafeTensors,
    name: &str,
    n: usize,
    add_one: bool,
) -> Result<DeviceBuf, String> {
    let off = if add_one { 1.0f32 } else { 0.0 };
    let v: Vec<u8> = st
        .require(name, "BF16", &[n])
        .to_u16_vec()
        .into_iter()
        .flat_map(|u| (off + f32::from_bits(u32::from(u) << 16)).to_le_bytes())
        .collect();
    DeviceBuf::from_host_bytes(ctx, &v)
}

struct Hc {
    norm: DeviceBuf,
    down: Q4,
    up: Q4,
    inject: Option<Q4>,
}

fn load_hc(
    ctx: &Arc<CudaContextHandle>,
    st: &ShardedSafeTensors,
    p: &str,
    zero_centered: bool,
    inject: bool,
) -> Result<Hc, String> {
    Ok(Hc {
        norm: f32_norm(
            ctx,
            st,
            &format!("{p}.hc_norm.weight"),
            Q38_HC_D,
            zero_centered,
        )?,
        down: Q4::load(
            ctx,
            st,
            &[(&format!("{p}.input_mix_weight_down"), Q38_HC_LOW)],
            Q38_HC_D,
        )?,
        up: Q4::load(
            ctx,
            st,
            &[(&format!("{p}.input_mix_weight_up"), Q38_HC_D)],
            Q38_HC_LOW,
        )?,
        inject: if inject {
            Some(Q4::load(
                ctx,
                st,
                &[(&format!("{p}.block_inject_weight"), Q38_HC)],
                Q38_HC_D,
            )?)
        } else {
            None
        },
    })
}

/// Byte layout of one `switch_mlp` slab, as in the target engine.
const SW_W: usize = Q38_EXP * Q38_MOE_FF * Q38_D / 2; // 400 MiB
const SW_SB: usize = SW_W / 8; // 50 MiB
const SW_PROJ: usize = SW_W + 2 * SW_SB;
const SW_LAYER: usize = 3 * SW_PROJ;

enum Slab {
    Vram(DeviceBuf),
    Pinned(PinnedHostBuf),
}

impl Slab {
    fn ptr(&self) -> CUdeviceptr {
        match self {
            Slab::Vram(b) => b.ptr,
            Slab::Pinned(b) => b.dev_ptr,
        }
    }
}

struct Fns {
    _q38: CUmodule,
    _mtp: CUmodule,
    embed: CUfunction,
    q4: CUfunction,
    q4_10k: CUfunction,
    hc_norm4: CUfunction,
    hc_mix4: CUfunction,
    hc_inject4: CUfunction,
    qsa_prep: CUfunction,
    qsa_score: CUfunction,
    qsa_select: CUfunction,
    qsa_attn: CUfunction,
    router: CUfunction,
    topk: CUfunction,
    moe_up: CUfunction,
    moe_down: CUfunction,
    argmax: CUfunction,
    rmsnorm: CUfunction,
    add_bcast4: CUfunction,
}

fn load_module(ctx: &Arc<CudaContextHandle>, env: &str, src: &str) -> Result<CUmodule, String> {
    let image = match std::env::var(env) {
        Ok(p) => {
            let mut raw = std::fs::read(&p).map_err(|e| format!("read {p}: {e}"))?;
            raw.push(0);
            raw
        }
        Err(_) => crate::cuda_ffi::compile_cuda_to_ptx_nvrtc(
            &src.replace(
                "#include \"include/leansonic_verified.cuh\"",
                include_str!("../../../kernels/include/leansonic_verified.cuh"),
            ),
            "compute_120a",
        )?,
    };
    let mut m: CUmodule = core::ptr::null_mut();
    unsafe {
        check_cu(
            (ctx.api.cuModuleLoadData)(&mut m, image.as_ptr().cast()),
            env,
        )?
    };
    Ok(m)
}

fn get(ctx: &Arc<CudaContextHandle>, m: CUmodule, name: &str) -> Result<CUfunction, String> {
    let c = CString::new(name).unwrap();
    let mut f: CUfunction = core::ptr::null_mut();
    unsafe { check_cu((ctx.api.cuModuleGetFunction)(&mut f, m, c.as_ptr()), name)? };
    Ok(f)
}

/// The MTP drafter (one decoder layer + its own KV cache and position).
pub struct Qwen38MtpDrafter {
    ctx: Arc<CudaContextHandle>,
    fns: Fns,
    max_ctx: usize,
    pos: usize,
    // Input fusion.
    norm_emb: DeviceBuf,
    norm_hid: DeviceBuf,
    fc_emb: Q4,
    fc_hid: Q4,
    // Layer.
    attn_hc: Hc,
    mlp_hc: Hc,
    qkv_idx: Q4,
    o_proj: Q4,
    q_norm: DeviceBuf,
    k_norm: DeviceBuf,
    iq_norm: DeviceBuf,
    ik_norm: DeviceBuf,
    k_cache: DeviceBuf,
    v_cache: DeviceBuf,
    idx_raw_k: DeviceBuf,
    idx_block_k: DeviceBuf,
    router: DeviceBuf,
    sh_gu: Q4,
    sh_down: Q4,
    switch: Slab,
    mixer: Hc,
    inv_freq32: DeviceBuf,
    // Scratch.
    d_tok: DeviceBuf,
    d_pos: DeviceBuf,
    d_out_tok: DeviceBuf,
    d_embed_hc: DeviceBuf,
    d_e: DeviceBuf,
    d_e_proj: DeviceBuf,
    d_h: DeviceBuf,
    d_hc: DeviceBuf,
    d_hc_normed: DeviceBuf,
    d_hc_low: DeviceBuf,
    d_hc_up: DeviceBuf,
    d_hc_inj4: DeviceBuf,
    d_mixed: DeviceBuf,
    d_branch: DeviceBuf,
    d_qsa_proj: DeviceBuf,
    d_q_rot: DeviceBuf,
    d_q_gate: DeviceBuf,
    d_idx_q: DeviceBuf,
    d_scores: DeviceBuf,
    d_mask: DeviceBuf,
    d_attn_out: DeviceBuf,
    d_router_logits: DeviceBuf,
    d_sel_ids: DeviceBuf,
    d_sel_wts: DeviceBuf,
    d_act11: DeviceBuf,
    d_logits: DeviceBuf,
    /// Whether the 1.5 GB MoE slab is in VRAM (`false`: pinned host memory).
    pub moe_in_vram: bool,
}

impl Qwen38MtpDrafter {
    /// Loads `language_model.mtp.*` from the target's shards into `ctx` (the target engine's
    /// context). `zero_centered` must match the target (`1 + w` RMSNorm weights). The MoE slab
    /// goes to VRAM when `vram_reserve` bytes stay free after it, else to pinned host memory.
    pub fn load(
        ctx: &Arc<CudaContextHandle>,
        st: &ShardedSafeTensors,
        max_ctx: usize,
        zero_centered: bool,
        vram_reserve: usize,
    ) -> Result<Self, String> {
        ctx.bind_current()?;
        let q38 = load_module(
            ctx,
            "LEANSONIC_QWEN38_CUBIN_PATH",
            include_str!("../../../kernels/qwen38_sm120.cu"),
        )?;
        let mtp = load_module(
            ctx,
            "LEANSONIC_QWEN38_MTP_CUBIN_PATH",
            include_str!("../../../kernels/qwen38_mtp_sm120.cu"),
        )?;
        let fns = Fns {
            embed: get(ctx, q38, "q38_embed_init_hc4")?,
            q4: get(ctx, q38, "q38_q4_gemv")?,
            q4_10k: get(ctx, q38, "q38_q4_gemv_10k")?,
            hc_norm4: get(ctx, q38, "q38_hc_norm4")?,
            hc_mix4: get(ctx, q38, "q38_hc_mix4")?,
            hc_inject4: get(ctx, q38, "q38_hc_inject4")?,
            qsa_prep: get(ctx, q38, "q38_qsa_prep_and_cache")?,
            qsa_score: get(ctx, q38, "q38_qsa_score_blocks")?,
            qsa_select: get(ctx, q38, "q38_qsa_select_top512")?,
            qsa_attn: get(ctx, q38, "q38_qsa_attn")?,
            router: get(ctx, q38, "q38_bf16_router_512")?,
            topk: get(ctx, q38, "q38_softmax_topk10")?,
            moe_up: get(ctx, q38, "q38_moe_up_swiglu11")?,
            moe_down: get(ctx, q38, "q38_moe_down_combine11")?,
            argmax: get(ctx, q38, "q38_argmax_and_inc_pos")?,
            rmsnorm: get(ctx, mtp, "q38_mtp_rmsnorm")?,
            add_bcast4: get(ctx, mtp, "q38_mtp_add_bcast4")?,
            _q38: q38,
            _mtp: mtp,
        };
        let z = zero_centered;
        let lp = format!("{MTP}.layers.0");
        let sp = format!("{lp}.self_attn");
        let max_blocks = max_ctx / 4 + 16;
        let zeros = |n: usize| DeviceBuf::alloc_zeroed(ctx, n);

        // `switch_mlp`: same slab layout as the target (`gate`, `up`, `down` x `w, sc, bi`).
        let swp = format!("{lp}.mlp.switch_mlp");
        let parts = [
            ("gate_proj", [512usize, 640, 320], [512usize, 640, 80]),
            ("up_proj", [512, 640, 320], [512, 640, 80]),
            ("down_proj", [512, 2560, 80], [512, 2560, 20]),
        ];
        let (free, _) = ctx.mem_info()?;
        let moe_in_vram = free > SW_LAYER + vram_reserve;
        let switch = if moe_in_vram {
            Slab::Vram(zeros(SW_LAYER)?)
        } else {
            Slab::Pinned(PinnedHostBuf::alloc_uninit(ctx, SW_LAYER)?)
        };
        for (i, (name, wshape, sshape)) in parts.iter().enumerate() {
            let base = i * SW_PROJ;
            let w = st
                .require(&format!("{swp}.{name}.weight"), "U32", wshape)
                .data;
            let sc = st
                .require(&format!("{swp}.{name}.scales"), "BF16", sshape)
                .data;
            let bi = st
                .require(&format!("{swp}.{name}.biases"), "BF16", sshape)
                .data;
            for (off, bytes) in [(base, w), (base + SW_W, sc), (base + SW_W + SW_SB, bi)] {
                match &switch {
                    Slab::Vram(b) => b.copy_from_host_at(off, bytes)?,
                    Slab::Pinned(b) => unsafe {
                        core::ptr::copy_nonoverlapping(
                            bytes.as_ptr(),
                            b.host_ptr.add(off),
                            bytes.len(),
                        )
                    },
                }
            }
        }

        let inv_f: Vec<u8> = (0..32)
            .flat_map(|i| (10_000_000.0_f64.powf(-(2.0 * (i as f64)) / 64.0) as f32).to_le_bytes())
            .collect();
        Ok(Self {
            norm_emb: f32_norm(
                ctx,
                st,
                &format!("{MTP}.pre_fc_norm_embedding.weight"),
                Q38_D,
                z,
            )?,
            norm_hid: f32_norm(
                ctx,
                st,
                &format!("{MTP}.pre_fc_norm_hidden.weight"),
                Q38_HC_D,
                z,
            )?,
            fc_emb: Q4::load(ctx, st, &[(&format!("{MTP}.fc_embedding"), Q38_D)], Q38_D)?,
            fc_hid: Q4::load(ctx, st, &[(&format!("{MTP}.fc_hidden"), Q38_D)], Q38_D)?,
            attn_hc: load_hc(ctx, st, &format!("{lp}.attn_hyper_connection"), z, true)?,
            mlp_hc: load_hc(ctx, st, &format!("{lp}.mlp_hyper_connection"), z, true)?,
            qkv_idx: Q4::load(
                ctx,
                st,
                &[
                    (&format!("{sp}.q_proj"), 12_288),
                    (&format!("{sp}.k_proj"), 512),
                    (&format!("{sp}.v_proj"), 512),
                    (&format!("{sp}.indexer.index_qk_proj"), 640),
                ],
                Q38_D,
            )?,
            o_proj: Q4::load(ctx, st, &[(&format!("{sp}.o_proj"), Q38_D)], 6_144)?,
            q_norm: f32_norm(ctx, st, &format!("{sp}.q_norm.weight"), 256, z)?,
            k_norm: f32_norm(ctx, st, &format!("{sp}.k_norm.weight"), 256, z)?,
            iq_norm: f32_norm(ctx, st, &format!("{sp}.indexer.q_layernorm.weight"), 128, z)?,
            ik_norm: f32_norm(ctx, st, &format!("{sp}.indexer.k_layernorm.weight"), 128, z)?,
            k_cache: zeros(max_ctx * 512 * 2)?,
            v_cache: zeros(max_ctx * 512 * 2)?,
            idx_raw_k: zeros(max_ctx * 128 * 4)?,
            idx_block_k: zeros(max_blocks * 128 * 4)?,
            router: DeviceBuf::from_host_bytes(
                ctx,
                st.require(&format!("{lp}.mlp.gate.weight"), "BF16", &[Q38_EXP, Q38_D])
                    .data,
            )?,
            sh_gu: Q4::load(
                ctx,
                st,
                &[
                    (&format!("{lp}.mlp.shared_expert.gate_proj"), Q38_MOE_FF),
                    (&format!("{lp}.mlp.shared_expert.up_proj"), Q38_MOE_FF),
                    (&format!("{lp}.mlp.shared_expert_gate"), 1),
                ],
                Q38_D,
            )?,
            sh_down: Q4::load(
                ctx,
                st,
                &[(&format!("{lp}.mlp.shared_expert.down_proj"), Q38_D)],
                Q38_MOE_FF,
            )?,
            switch,
            mixer: load_hc(ctx, st, &format!("{MTP}.hyper_connection_mixer"), z, false)?,
            inv_freq32: DeviceBuf::from_host_bytes(ctx, &inv_f)?,
            d_tok: zeros(16)?,
            d_pos: zeros(16)?,
            d_out_tok: zeros(16)?,
            d_embed_hc: zeros(Q38_HC_D * 4)?,
            d_e: zeros(Q38_D * 4)?,
            d_e_proj: zeros(Q38_D * 4)?,
            d_h: zeros(Q38_HC_D * 4)?,
            d_hc: zeros(Q38_HC_D * 4)?,
            d_hc_normed: zeros(Q38_HC_D * 4)?,
            d_hc_low: zeros(Q38_HC_LOW * 4)?,
            d_hc_up: zeros(Q38_HC_D * 4)?,
            d_hc_inj4: zeros(16)?,
            d_mixed: zeros(Q38_D * 4)?,
            d_branch: zeros(Q38_D * 4)?,
            d_qsa_proj: zeros(13_952 * 4)?,
            d_q_rot: zeros(6_144 * 4)?,
            d_q_gate: zeros(6_144 * 4)?,
            d_idx_q: zeros(512 * 4)?,
            d_scores: zeros(max_blocks * 4)?,
            d_mask: zeros(max_blocks * 4)?,
            d_attn_out: zeros(6_144 * 4)?,
            d_router_logits: zeros(Q38_EXP * 4)?,
            d_sel_ids: zeros(Q38_TOP_K * 4)?,
            d_sel_wts: zeros((Q38_TOP_K + 1) * 4)?,
            d_act11: zeros((Q38_TOP_K + 1) * Q38_MOE_FF * 4)?,
            d_logits: zeros(Q38_VOCAB * 4)?,
            moe_in_vram,
            ctx: Arc::clone(ctx),
            fns,
            max_ctx,
            pos: 0,
        })
    }

    /// MTP positions consumed so far (one per `step`).
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Restarts the drafter's own context (KV cache contents are overwritten as positions
    /// are re-fed).
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Rewinds to `pos` (`<= position()`): later KV rows are simply rewritten.
    pub fn rewind(&mut self, pos: usize) {
        self.pos = self.pos.min(pos);
    }

    fn launch(
        &self,
        stream: CUstream,
        f: CUfunction,
        grid: (u32, u32),
        block: u32,
        args: &mut [*mut c_void],
        name: &str,
    ) -> Result<(), String> {
        unsafe {
            check_cu(
                (self.ctx.api.cuLaunchKernel)(
                    f,
                    grid.0,
                    grid.1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    stream,
                    args.as_mut_ptr(),
                    core::ptr::null_mut(),
                ),
                name,
            )
        }
    }

    fn q4(
        &self,
        s: CUstream,
        m: Q4Ref,
        x: CUdeviceptr,
        y: CUdeviceptr,
        mode: i32,
    ) -> Result<(), String> {
        let (mut w, mut sc, mut bi, mut px, mut py) = (m.w, m.sc, m.bi, x, y);
        let (mut rows, mut cols, mut md) = (m.rows, m.cols, mode);
        let grid = ((m.rows as u32).div_ceil(8), 1);
        if m.cols == Q38_HC_D as i32 {
            self.launch(
                s,
                self.fns.q4_10k,
                grid,
                256,
                &mut kargs![w, sc, bi, px, py, rows, md],
                "q38_q4_gemv_10k",
            )
        } else {
            self.launch(
                s,
                self.fns.q4,
                grid,
                256,
                &mut kargs![w, sc, bi, px, py, rows, cols, md],
                "q38_q4_gemv",
            )
        }
    }

    fn hc_pre(&self, s: CUstream, hc: &Hc) -> Result<(), String> {
        let (mut hin, mut gam, mut nrm) = (self.d_hc.ptr, hc.norm.ptr, self.d_hc_normed.ptr);
        self.launch(
            s,
            self.fns.hc_norm4,
            (4, 1),
            256,
            &mut kargs![hin, gam, nrm],
            "q38_hc_norm4",
        )?;
        self.q4(s, hc.down.r(), self.d_hc_normed.ptr, self.d_hc_low.ptr, 1)?;
        self.q4(s, hc.up.r(), self.d_hc_low.ptr, self.d_hc_up.ptr, 2)?;
        let (mut up, mut mixed) = (self.d_hc_up.ptr, self.d_mixed.ptr);
        self.launch(
            s,
            self.fns.hc_mix4,
            (10, 1),
            256,
            &mut kargs![nrm, up, mixed],
            "q38_hc_mix4",
        )?;
        if let Some(inj) = &hc.inject {
            self.q4(s, inj.r(), self.d_hc_normed.ptr, self.d_hc_inj4.ptr, 0)?;
        }
        Ok(())
    }

    fn hc_post(&self, s: CUstream) -> Result<(), String> {
        let (mut hc, mut br, mut inj) = (self.d_hc.ptr, self.d_branch.ptr, self.d_hc_inj4.ptr);
        self.launch(
            s,
            self.fns.hc_inject4,
            (40, 1),
            256,
            &mut kargs![hc, br, inj],
            "q38_hc_inject4",
        )
    }

    /// One MTP step at the drafter's next position: fuses `hc_hidden` (the target's `[4, 2560]`
    /// state after its last layer for position `t`) with `next_tok` (`tok_{t+1}`), runs the layer
    /// and the output head, and returns the greedy draft for `tok_{t+2}`.
    pub fn step(
        &mut self,
        hc_hidden: CUdeviceptr,
        next_tok: u32,
        embed: Q4Ref,
        lm_head: Q4Ref,
    ) -> Result<u32, String> {
        if self.pos >= self.max_ctx {
            return Err(format!("MTP drafter reached max_ctx {}", self.max_ctx));
        }
        self.ctx.bind_current()?;
        let s = self.ctx.compute_stream;
        let tok = next_tok as i32;
        let pos = self.pos as i32;
        unsafe {
            check_cu(
                (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                    self.d_tok.ptr,
                    (&tok as *const i32).cast(),
                    4,
                    s,
                ),
                "HtoD(mtp tok)",
            )?;
            check_cu(
                (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                    self.d_pos.ptr,
                    (&pos as *const i32).cast(),
                    4,
                    s,
                ),
                "HtoD(mtp pos)",
            )?;
        }
        // Input fusion. `q38_embed_init_hc4` tiles the embedding into four streams; stream 0 is
        // `embed(tok)`.
        let (mut ew, mut esc, mut ebi, mut ptok, mut pehc) = (
            embed.w,
            embed.sc,
            embed.bi,
            self.d_tok.ptr,
            self.d_embed_hc.ptr,
        );
        self.launch(
            s,
            self.fns.embed,
            (10, 1),
            256,
            &mut kargs![ew, esc, ebi, ptok, pehc],
            "q38_embed_init_hc4",
        )?;
        let (mut x, mut w, mut n, mut out) = (
            self.d_embed_hc.ptr,
            self.norm_emb.ptr,
            Q38_D as i32,
            self.d_e.ptr,
        );
        self.launch(
            s,
            self.fns.rmsnorm,
            (1, 1),
            256,
            &mut kargs![x, w, n, out],
            "q38_mtp_rmsnorm(emb)",
        )?;
        self.q4(s, self.fc_emb.r(), self.d_e.ptr, self.d_e_proj.ptr, 0)?;
        let (mut x, mut w, mut n, mut out) =
            (hc_hidden, self.norm_hid.ptr, Q38_HC_D as i32, self.d_h.ptr);
        self.launch(
            s,
            self.fns.rmsnorm,
            (1, 1),
            256,
            &mut kargs![x, w, n, out],
            "q38_mtp_rmsnorm(hid)",
        )?;
        for st in 0..Q38_HC {
            let off = (st * Q38_D * 4) as u64;
            self.q4(
                s,
                self.fc_hid.r(),
                self.d_h.ptr + off,
                self.d_hc.ptr + off,
                0,
            )?;
        }
        let (mut hc, mut e) = (self.d_hc.ptr, self.d_e_proj.ptr);
        self.launch(
            s,
            self.fns.add_bcast4,
            (40, 1),
            256,
            &mut kargs![hc, e],
            "q38_mtp_add_bcast4",
        )?;

        // Attention branch (full attention with the QSA indexer), as a target QSA layer.
        self.hc_pre(s, &self.attn_hc)?;
        self.q4(
            s,
            self.qkv_idx.r(),
            self.d_mixed.ptr,
            self.d_qsa_proj.ptr,
            0,
        )?;
        let (mut qp, mut qn, mut kn, mut iqn, mut ikn, mut invf, mut pp) = (
            self.d_qsa_proj.ptr,
            self.q_norm.ptr,
            self.k_norm.ptr,
            self.iq_norm.ptr,
            self.ik_norm.ptr,
            self.inv_freq32.ptr,
            self.d_pos.ptr,
        );
        let (mut qr, mut qg, mut kc, mut vc, mut iq, mut irk, mut ibk) = (
            self.d_q_rot.ptr,
            self.d_q_gate.ptr,
            self.k_cache.ptr,
            self.v_cache.ptr,
            self.d_idx_q.ptr,
            self.idx_raw_k.ptr,
            self.idx_block_k.ptr,
        );
        self.launch(
            s,
            self.fns.qsa_prep,
            (31, 1),
            256,
            &mut kargs![qp, qn, kn, iqn, ikn, invf, pp, qr, qg, kc, vc, iq, irk, ibk],
            "q38_qsa_prep_and_cache",
        )?;
        let n_blocks = ((self.pos + 1) / 4) as i32;
        let mut use_mask: i32 = i32::from(n_blocks > 512);
        if use_mask == 1 {
            let (mut nb, mut sc, mut mk) = (n_blocks, self.d_scores.ptr, self.d_mask.ptr);
            self.launch(
                s,
                self.fns.qsa_score,
                ((n_blocks as u32).div_ceil(8), 1),
                256,
                &mut kargs![iq, ibk, nb, sc],
                "q38_qsa_score_blocks",
            )?;
            self.launch(
                s,
                self.fns.qsa_select,
                (1, 1),
                256,
                &mut kargs![sc, nb, mk],
                "q38_qsa_select_top512",
            )?;
        }
        let (mut mk, mut ao) = (self.d_mask.ptr, self.d_attn_out.ptr);
        self.launch(
            s,
            self.fns.qsa_attn,
            (24, 1),
            256,
            &mut kargs![qr, qg, kc, vc, mk, pp, use_mask, ao],
            "q38_qsa_attn",
        )?;
        self.q4(
            s,
            self.o_proj.r(),
            self.d_attn_out.ptr,
            self.d_branch.ptr,
            0,
        )?;
        self.hc_post(s)?;

        // MoE branch.
        self.hc_pre(s, &self.mlp_hc)?;
        let (mut rw, mut mx, mut rl, mut ids, mut wts) = (
            self.router.ptr,
            self.d_mixed.ptr,
            self.d_router_logits.ptr,
            self.d_sel_ids.ptr,
            self.d_sel_wts.ptr,
        );
        self.launch(
            s,
            self.fns.router,
            (64, 1),
            256,
            &mut kargs![rw, mx, rl],
            "q38_bf16_router_512",
        )?;
        self.launch(
            s,
            self.fns.topk,
            (1, 1),
            256,
            &mut kargs![rl, ids, wts],
            "q38_softmax_topk10",
        )?;
        let b = self.switch.ptr();
        let (mut gw, mut gsc, mut gbi) = (b, b + SW_W as u64, b + (SW_W + SW_SB) as u64);
        let u = b + SW_PROJ as u64;
        let (mut uw, mut usc, mut ubi) = (u, u + SW_W as u64, u + (SW_W + SW_SB) as u64);
        let d = b + 2 * SW_PROJ as u64;
        let (mut dw, mut dsc, mut dbi) = (d, d + SW_W as u64, d + (SW_W + SW_SB) as u64);
        let (mut shw, mut shs, mut shb) = (self.sh_gu.w.ptr, self.sh_gu.sc.ptr, self.sh_gu.bi.ptr);
        let (mut sdw, mut sds, mut sdb) =
            (self.sh_down.w.ptr, self.sh_down.sc.ptr, self.sh_down.bi.ptr);
        let (mut act, mut out, mut cmp) = (self.d_act11.ptr, self.d_branch.ptr, 0i32);
        self.launch(
            s,
            self.fns.moe_up,
            (160, 11),
            256,
            &mut kargs![gw, gsc, gbi, uw, usc, ubi, shw, shs, shb, mx, ids, cmp, wts, act],
            "q38_moe_up_swiglu11",
        )?;
        self.launch(
            s,
            self.fns.moe_down,
            (320, 1),
            256,
            &mut kargs![dw, dsc, dbi, sdw, sds, sdb, act, ids, cmp, wts, out],
            "q38_moe_down_combine11",
        )?;
        self.hc_post(s)?;

        // Output: own mixer, shared LM head, greedy.
        self.hc_pre(s, &self.mixer)?;
        self.q4(s, lm_head, self.d_mixed.ptr, self.d_logits.ptr, 0)?;
        let (mut lg, mut ot, mut pp2) = (self.d_logits.ptr, self.d_out_tok.ptr, self.d_pos.ptr);
        self.launch(
            s,
            self.fns.argmax,
            (1, 1),
            256,
            &mut kargs![lg, ot, pp2],
            "q38_argmax_and_inc_pos",
        )?;
        self.ctx.sync_compute()?;
        let mut draft = 0i32;
        unsafe {
            check_cu(
                (self.ctx.api.cuMemcpyDtoH_v2)(
                    (&mut draft as *mut i32).cast(),
                    self.d_out_tok.ptr,
                    4,
                ),
                "DtoH(mtp draft)",
            )?;
        }
        self.pos += 1;
        Ok(draft as u32)
    }

    /// The drafter's last logits (`[vocab]`), for checks against a reference.
    pub fn read_logits(&self) -> Result<Vec<f32>, String> {
        self.ctx.sync_compute()?;
        let mut v = vec![0.0f32; Q38_VOCAB];
        self.d_logits.copy_to_host_f32(&mut v)?;
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen38_gpu::Qwen38GpuEngine;

    /// Exact-size memcheck for the two MTP fusion kernels (`--padding 65536`): inputs and outputs
    /// of exactly `n` floats; `short` drops the last input element (the planted fault).
    fn mtp_kernel_cases(short: usize) -> Result<(), String> {
        let ctx = CudaContextHandle::init(0)?;
        let m = load_module(
            &ctx,
            "LEANSONIC_QWEN38_MTP_CUBIN_PATH",
            include_str!("../../../kernels/qwen38_mtp_sm120.cu"),
        )?;
        let fl = |n: usize| {
            DeviceBuf::from_host_bytes(
                &ctx,
                &(0..n)
                    .flat_map(|i| ((i % 97) as f32 * 0.01 - 0.4).to_le_bytes())
                    .collect::<Vec<u8>>(),
            )
        };
        let only = std::env::var("LEANSONIC_EXACT_ONLY").ok();
        let run = |name: &str, grid: u32, args: &mut [*mut c_void]| -> Result<(), String> {
            if only.as_deref().is_some_and(|k| k != name) {
                return Ok(());
            }
            let f = get(&ctx, m, name)?;
            unsafe {
                check_cu(
                    (ctx.api.cuLaunchKernel)(
                        f,
                        grid,
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
                    "launch",
                )?;
            }
            ctx.sync_compute()
        };
        for n in [Q38_D, Q38_HC_D] {
            let (x, w, out) = (
                fl(n - short)?,
                fl(n)?,
                DeviceBuf::alloc_zeroed(&ctx, n * 4)?,
            );
            let (mut px, mut pw, mut nn, mut po) = (x.ptr, w.ptr, n as i32, out.ptr);
            run("q38_mtp_rmsnorm", 1, &mut kargs![px, pw, nn, po])?;
        }
        let (hc, e) = (fl(Q38_HC_D - short)?, fl(Q38_D)?);
        let (mut ph, mut pe) = (hc.ptr, e.ptr);
        run("q38_mtp_add_bcast4", 40, &mut kargs![ph, pe])
    }

    #[test]
    #[ignore = "requires a CUDA device; run under compute-sanitizer --tool memcheck --padding 65536"]
    fn exact_size_mtp_kernels() -> Result<(), String> {
        mtp_kernel_cases(0)
    }

    #[test]
    #[ignore = "planted fault: memcheck --padding 65536 must report an out-of-bounds access"]
    fn exact_size_mtp_red_control() -> Result<(), String> {
        let _ = mtp_kernel_cases(1);
        Ok(())
    }

    /// Draft acceptance of the MTP head against the target's own greedy continuation
    /// (`LEANSONIC_Q38_MODEL=<dir>`, `LEANSONIC_Q38_MTP_TOKENS=<n>`): after the target emits
    /// `tok_{t+1}` from position `t`, the drafter sees `(hc_t, tok_{t+1})` and predicts `tok_{t+2}`;
    /// the target's next step decides. `LEANSONIC_Q38_MTP_POS=next` places the MTP row at `t + 1`
    /// instead of `t`.
    #[test]
    #[ignore = "requires the Qwen3.8-Flash-Next checkpoint and a CUDA device"]
    fn mtp_greedy_acceptance() -> Result<(), String> {
        let dir = std::env::var("LEANSONIC_Q38_MODEL").map_err(|_| "set LEANSONIC_Q38_MODEL")?;
        let n: usize = std::env::var("LEANSONIC_Q38_MTP_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(128);
        let prompt: Vec<u32> = std::env::var("LEANSONIC_Q38_MTP_PROMPT")
            .ok()
            .map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect())
            .unwrap_or_else(|| (0..32).map(|i| ((i * 7 + 100) % 50_000) as u32).collect());
        let mut eng = Qwen38GpuEngine::load_from_dir_with_opts(&dir, prompt.len() + n + 64, true)?;
        let (ctx, hc_ptr, embed, lm_head, st) = eng.mtp_view();
        let probe = st
            .require(
                "language_model.model.layers.0.attn_hyper_connection.hc_norm.weight",
                "BF16",
                &[Q38_HC_D],
            )
            .to_u16_vec();
        let zero_centered = probe
            .iter()
            .map(|&u| f32::from_bits(u32::from(u) << 16))
            .sum::<f32>()
            / (Q38_HC_D as f32)
            < 0.5;
        let mut mtp =
            Qwen38MtpDrafter::load(&ctx, st, prompt.len() + n + 64, zero_centered, 512 << 20)?;
        eprintln!(
            "MTP MoE in VRAM: {}, zero_centered {zero_centered}",
            mtp.moe_in_vram
        );
        let next_pos = std::env::var("LEANSONIC_Q38_MTP_POS").as_deref() == Ok("next");
        if next_pos {
            mtp.pos = 1;
        }
        // Feed the prompt one token at a time so `d_hc` is available at every position.
        let mut toks = prompt.clone();
        let mut cur = 0u32;
        let (mut hits, mut tries) = (0usize, 0usize);
        // `drafts[j]`: the MTP draft for the token at target position `j`.
        let total = prompt.len() + n;
        let mut drafts: Vec<Option<u32>> = vec![None; total + 2];
        let (mut target_s, mut mtp_s) = (0.0f64, 0.0f64);
        for i in 0..total {
            let tok = if i < prompt.len() { prompt[i] } else { cur };
            if i >= prompt.len() {
                toks.push(tok);
                if let Some(d) = drafts[i] {
                    tries += 1;
                    hits += usize::from(d == tok);
                }
            }
            let t0 = std::time::Instant::now();
            cur = eng.forward_greedy_token(tok)?;
            let t1 = std::time::Instant::now();
            target_s += (t1 - t0).as_secs_f64();
            // The token after position `i`: the prompt's own while inside it, else the target's.
            let next = if i + 1 < prompt.len() {
                prompt[i + 1]
            } else {
                cur
            };
            let draft = mtp.step(hc_ptr, next, embed, lm_head)?;
            mtp_s += t1.elapsed().as_secs_f64();
            drafts[i + 2] = Some(draft);
        }
        if std::env::var_os("LEANSONIC_MTP_REPLAY_IDS").is_some() {
            eprintln!("target_ids={toks:?}");
            let predicted: Vec<u32> = drafts.iter().flatten().copied().collect();
            eprintln!("draft_ids={predicted:?}");
        }
        eprintln!(
            "MTP greedy acceptance: {hits}/{tries} = {:.1}% (next_pos={next_pos})",
            100.0 * hits as f64 / tries.max(1) as f64
        );
        eprintln!(
            "per position: target forward {:.2} ms, MTP step {:.2} ms",
            1e3 * target_s / total as f64,
            1e3 * mtp_s / total as f64
        );
        Ok(())
    }
}
