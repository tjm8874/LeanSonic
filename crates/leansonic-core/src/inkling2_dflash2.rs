//! DFlash 2 block-diffusion drafter (#103) for Inkling-2-Mini.
//!
//! The drafter reads the same four target taps as DSpark (`[4, tap_cap, D]`):
//!
//! - **Context**: every committed position `p` gets `h = rms_norm(fc([tap0 | tap1 | tap2 | tap3]),
//!   hidden_norm)`, and each of the five layers stores `k = rope(rms_head(wk h) * k_norm, p)` and
//!   `v = wv h` in a sliding-window ring (cell `p % RING`, window 2048).
//! - **Block**: the anchor (the pending token at `p`) and seven mask tokens run non-causally through
//!   five layers. Each layer wraps attention and the SwiGLU MLP in dynamic two-tap group-16 causal
//!   convolutions. The block rows attend every ring key in `[max(lo, p - 2047), p + 8)`.
//! - **Head**: rows `1..=n` are normed and scored by the target's Q8 output head. The Top-16
//!   candidates of each row (plus the softcapped tail log-sum-exp) feed a rank-256 bilinear
//!   selector that walks one path and returns a calibrated confidence per step.
//!
//! Drafts only propose tokens; the target verifies them, so the emitted IDs do not depend on this
//! arithmetic.

use core::ffi::c_void;
use std::path::Path;
use std::sync::Arc;

use crate::cuda_ffi::{
    check_cu, compile_cuda_to_ptx_nvrtc, CUdeviceptr, CUfunction, CUmodule, CUstream,
    CudaContextHandle, DeviceBuf,
};
use crate::inkling2_gpu::{repack_bf16_to_planar_q8_64, PlanarQ8Gpu, D, PADDED_VOCAB, VOCAB};
use crate::safetensors::MappedSafeTensors;

/// Block rows: the anchor plus seven drafted positions.
pub const DF2_BLOCK: usize = 8;
const DF2_LAYERS: usize = 5;
const DF2_Q: usize = 4096;
const DF2_KV: usize = 1024;
const DF2_FF: usize = 6912;
const DF2_DYN: usize = 576;
const DF2_RANK: usize = 256;
const DF2_TAPS: usize = 4;
const DF2_WINDOW: usize = 2048;
/// Ring cells per layer: the window plus the block and slack.
const DF2_RING: usize = DF2_WINDOW + 64;
const DF2_SOFTCAP: f32 = 30.0;
const DF2_MASK_TOKEN: i32 = 200_056;
/// Vocabulary chunks of the first Top-16 stage (`kDf2Chunks`).
const DF2_CHUNKS: usize = 40;

/// The drafter's own kernel module (`kernels/inkling2_dflash2_sm120.cu`). It is a separate
/// translation unit so that adding the drafter never perturbs the target kernels' code
/// generation; it is compiled only when a DFlash 2 checkpoint is selected.
pub const DF2_KERNEL_SRC: &str = include_str!("../../../kernels/inkling2_dflash2_sm120.cu");

/// Entry points of [`DF2_KERNEL_SRC`], in [`load_dflash2_kernels`] order.
pub const DF2_KERNEL_NAMES: [&str; 10] = [
    "df2_block_init",
    "df2_rmsnorm",
    "df2_conv2",
    "df2_qk_prep",
    "df2_attn",
    "df2_silu_mul",
    "df2_tap_gather",
    "df2_top16_part",
    "df2_top16_merge",
    "df2_select",
];

/// Loads the drafter module from `LEANSONIC_DF2_CUBIN_PATH`, or compiles [`DF2_KERNEL_SRC`] with
/// NVRTC (cached like the target module), and returns the [`DF2_KERNEL_NAMES`] handles.
pub fn load_dflash2_kernels(ctx: &CudaContextHandle) -> Result<[CUfunction; 10], String> {
    let image = match std::env::var("LEANSONIC_DF2_CUBIN_PATH") {
        Ok(path) => std::fs::read(&path).map_err(|e| format!("read {path}: {e}"))?,
        Err(_) => compile_cuda_to_ptx_nvrtc(DF2_KERNEL_SRC, "compute_120")?,
    };
    let mut module: CUmodule = core::ptr::null_mut();
    unsafe {
        check_cu(
            (ctx.api.cuModuleLoadData)(&mut module, image.as_ptr().cast()),
            "cuModuleLoadData(dflash2)",
        )?;
    }
    let mut fns: [CUfunction; 10] = [core::ptr::null_mut(); 10];
    for (f, name) in fns.iter_mut().zip(DF2_KERNEL_NAMES) {
        let c = std::ffi::CString::new(name).unwrap();
        unsafe {
            check_cu((ctx.api.cuModuleGetFunction)(f, module, c.as_ptr()), name)?;
        }
    }
    Ok(fns)
}

/// Kernel handles for the drafter: the target module's quantiser and GEMMs plus the
/// [`load_dflash2_kernels`] entry points.
#[derive(Clone, Copy)]
pub struct Dflash2KernelFns {
    pub pdl_all: bool,
    pub quant_q8_64_b: CUfunction,
    pub q8_gemm_ktiled: CUfunction,
    pub q8_gemm_b8_fast: Option<CUfunction>,
    pub block_init: CUfunction,
    pub rmsnorm: CUfunction,
    pub conv2: CUfunction,
    pub qk_prep: CUfunction,
    pub attn: CUfunction,
    pub silu_mul: CUfunction,
    pub tap_gather: CUfunction,
    pub top16_part: CUfunction,
    pub top16_merge: CUfunction,
    pub select: CUfunction,
}

unsafe impl Send for Dflash2KernelFns {}
unsafe impl Sync for Dflash2KernelFns {}

struct Df2Layer {
    attn_norm: DeviceBuf,
    post_norm: DeviceBuf,
    q_norm: DeviceBuf,
    k_norm: DeviceBuf,
    base_a: DeviceBuf,
    base_f: DeviceBuf,
    conv_a: PlanarQ8Gpu,
    conv_f: PlanarQ8Gpu,
    wq: PlanarQ8Gpu,
    wkv: PlanarQ8Gpu,
    wo: PlanarQ8Gpu,
    gu: PlanarQ8Gpu,
    down: PlanarQ8Gpu,
    k_ring: DeviceBuf,
    v_ring: DeviceBuf,
}

struct Df2Scratch {
    x: DeviceBuf,
    xn: DeviceBuf,
    y: DeviceBuf,
    z: DeviceBuf,
    dyn_: DeviceBuf,
    q: DeviceBuf,
    kv: DeviceBuf,
    att: DeviceBuf,
    gu: DeviceBuf,
    act: DeviceBuf,
    tap: DeviceBuf,
    craw: DeviceBuf,
    hd: DeviceBuf,
    logits: DeviceBuf,
    g: DeviceBuf,
    part: DeviceBuf,
    cand: DeviceBuf,
    tail: DeviceBuf,
    ids: DeviceBuf,
    conf: DeviceBuf,
    xq: DeviceBuf,
    xs: DeviceBuf,
}

impl Df2Scratch {
    fn alloc(ctx: &Arc<CudaContextHandle>) -> Result<Self, String> {
        let f = |n: usize| DeviceBuf::alloc_zeroed(ctx, n * 4);
        let rows = DF2_BLOCK;
        Ok(Self {
            x: f(rows * D)?,
            xn: f(rows * D)?,
            y: f(rows * D)?,
            z: f(rows * D)?,
            dyn_: f(rows * DF2_DYN)?,
            q: f(rows * DF2_Q)?,
            kv: f(rows * 2 * DF2_KV)?,
            att: f(rows * DF2_Q)?,
            gu: f(rows * 2 * DF2_FF)?,
            act: f(rows * DF2_FF)?,
            tap: f(rows * DF2_TAPS * D)?,
            craw: f(rows * D)?,
            hd: f(rows * D)?,
            logits: f(rows * VOCAB)?,
            g: f(rows * DF2_RANK)?,
            part: f(rows * DF2_CHUNKS * 34)?,
            cand: f(rows * 32)?,
            tail: f(rows * 2)?,
            ids: f(rows)?,
            conf: f(rows)?,
            xq: DeviceBuf::alloc_zeroed(ctx, rows * DF2_TAPS * D)?,
            xs: f(rows * DF2_TAPS * D / 64)?,
        })
    }
}

pub struct Dflash2DrafterGpu {
    ctx: Arc<CudaContextHandle>,
    fns: Dflash2KernelFns,
    layers: Vec<Df2Layer>,
    fc: PlanarQ8Gpu,
    hidden_norm: DeviceBuf,
    norm: DeviceBuf,
    sel_h: PlanarQ8Gpu,
    pred: DeviceBuf,
    succ: DeviceBuf,
    inv_freq: DeviceBuf,
    scr: Df2Scratch,
    /// Next context position (= the anchor position of the next draft).
    pub ctx_end: usize,
    /// Earliest committed position still valid in the rings.
    lo: usize,
    tau: f32,
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn f32_buf(ctx: &Arc<CudaContextHandle>, src: &[u16]) -> Result<DeviceBuf, String> {
    let mut bytes = Vec::with_capacity(src.len() * 4);
    for &u in src {
        bytes.extend_from_slice(&bf16_to_f32(u).to_le_bytes());
    }
    DeviceBuf::from_host_bytes(ctx, &bytes)
}

fn q8(
    ctx: &Arc<CudaContextHandle>,
    st: &MappedSafeTensors,
    names: &[&str],
    rows: &[usize],
    cols: usize,
) -> Result<PlanarQ8Gpu, String> {
    let total: usize = rows.iter().sum();
    let mut bytes = Vec::with_capacity(total * cols * 2);
    for (name, &r) in names.iter().zip(rows) {
        bytes.extend_from_slice(st.require(name, "BF16", &[r, cols]).data);
    }
    repack_bf16_to_planar_q8_64(ctx, &bytes, total, cols)
}

/// Locates the DFlash 2 checkpoint: `LEANSONIC_DFLASH2` (a directory or a `.safetensors` file),
/// else `<model>/dflash2/model.safetensors` or `<model>/dflash2.safetensors`.
pub fn find_dflash2(model_dir: &Path) -> Option<std::path::PathBuf> {
    let candidates = std::env::var("LEANSONIC_DFLASH2")
        .ok()
        .map(std::path::PathBuf::from)
        .map(|p| vec![p.join("model.safetensors"), p])
        .unwrap_or_else(|| {
            vec![
                model_dir.join("dflash2").join("model.safetensors"),
                model_dir.join("dflash2.safetensors"),
            ]
        });
    candidates.into_iter().find(|p| p.is_file())
}

impl Dflash2DrafterGpu {
    pub fn load_from_file(
        ctx: &Arc<CudaContextHandle>,
        fns: Dflash2KernelFns,
        path: &Path,
    ) -> Result<Self, String> {
        let st =
            MappedSafeTensors::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let vec_f32 = |name: &str, n: usize| -> Result<DeviceBuf, String> {
            f32_buf(ctx, st.require(name, "BF16", &[n]).as_u16_slice())
        };
        let raw = |name: &str, shape: &[usize]| -> Result<DeviceBuf, String> {
            DeviceBuf::from_host_bytes(ctx, st.require(name, "BF16", shape).data)
        };
        let mut layers = Vec::with_capacity(DF2_LAYERS);
        for l in 0..DF2_LAYERS {
            let p = |s: &str| format!("layers.{l}.{s}");
            layers.push(Df2Layer {
                attn_norm: vec_f32(&p("input_layernorm.weight"), D)?,
                post_norm: vec_f32(&p("post_attention_layernorm.weight"), D)?,
                q_norm: raw(&p("self_attn.q_norm.weight"), &[128])?,
                k_norm: raw(&p("self_attn.k_norm.weight"), &[128])?,
                base_a: f32_buf(
                    ctx,
                    st.require(&p("attention_conv.base_kernel"), "BF16", &[2, 2, D])
                        .as_u16_slice(),
                )?,
                base_f: f32_buf(
                    ctx,
                    st.require(&p("mlp_conv.base_kernel"), "BF16", &[2, 2, D])
                        .as_u16_slice(),
                )?,
                conv_a: q8(
                    ctx,
                    &st,
                    &[&p("attention_conv.kernel_projection.weight")],
                    &[DF2_DYN],
                    D,
                )?,
                conv_f: q8(
                    ctx,
                    &st,
                    &[&p("mlp_conv.kernel_projection.weight")],
                    &[DF2_DYN],
                    D,
                )?,
                wq: q8(ctx, &st, &[&p("self_attn.q_proj.weight")], &[DF2_Q], D)?,
                wkv: q8(
                    ctx,
                    &st,
                    &[&p("self_attn.k_proj.weight"), &p("self_attn.v_proj.weight")],
                    &[DF2_KV, DF2_KV],
                    D,
                )?,
                wo: q8(ctx, &st, &[&p("self_attn.o_proj.weight")], &[D], DF2_Q)?,
                gu: q8(
                    ctx,
                    &st,
                    &[&p("mlp.gate_proj.weight"), &p("mlp.up_proj.weight")],
                    &[DF2_FF, DF2_FF],
                    D,
                )?,
                down: q8(ctx, &st, &[&p("mlp.down_proj.weight")], &[D], DF2_FF)?,
                k_ring: DeviceBuf::alloc_zeroed(ctx, DF2_RING * DF2_KV * 4)?,
                v_ring: DeviceBuf::alloc_zeroed(ctx, DF2_RING * DF2_KV * 4)?,
            });
        }
        let mut inv_freq = Vec::with_capacity(64 * 4);
        for i in 0..64 {
            let f = 10_000.0_f64.powf(-2.0 * (i as f64) / 128.0) as f32;
            inv_freq.extend_from_slice(&f.to_le_bytes());
        }
        let tau = std::env::var("LEANSONIC_DF2_TAU")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.2);
        Ok(Self {
            ctx: Arc::clone(ctx),
            fns,
            fc: q8(ctx, &st, &["fc.weight"], &[D], DF2_TAPS * D)?,
            hidden_norm: vec_f32("hidden_norm.weight", D)?,
            norm: vec_f32("norm.weight", D)?,
            sel_h: q8(
                ctx,
                &st,
                &["candidate_selector.hidden_projection.weight"],
                &[DF2_RANK],
                D,
            )?,
            pred: raw(
                "candidate_selector.predecessor_codebook",
                &[PADDED_VOCAB, DF2_RANK],
            )?,
            succ: raw(
                "candidate_selector.successor_codebook",
                &[PADDED_VOCAB, DF2_RANK],
            )?,
            inv_freq: DeviceBuf::from_host_bytes(ctx, &inv_freq)?,
            scr: Df2Scratch::alloc(ctx)?,
            layers,
            ctx_end: 0,
            lo: 0,
            tau,
        })
    }

    pub fn reset(&mut self) {
        self.ctx_end = 0;
        self.lo = 0;
    }

    fn launch(
        &self,
        stream: CUstream,
        func: CUfunction,
        grid: (u32, u32),
        block: u32,
        params: &mut [*mut c_void],
    ) -> Result<(), String> {
        if self.fns.pdl_all {
            return crate::inkling2_gpu::launch_pdl_3d(
                &self.ctx,
                stream,
                func,
                (grid.0, grid.1, 1),
                block,
                0,
                params,
            );
        }
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
                "dflash2 cuLaunchKernel",
            )
        }
    }

    /// Quantizes `rows` FP32 rows of `cols` into the shared Q8_64 input scratch.
    fn quant(
        &self,
        stream: CUstream,
        x: CUdeviceptr,
        cols: usize,
        rows: usize,
    ) -> Result<(), String> {
        let (mut px, mut pq, mut ps, mut c) = (x, self.scr.xq.ptr, self.scr.xs.ptr, cols as i32);
        let mut args: [*mut c_void; 4] = [
            (&mut px as *mut _) as *mut c_void,
            (&mut pq as *mut _) as *mut c_void,
            (&mut ps as *mut _) as *mut c_void,
            (&mut c as *mut _) as *mut c_void,
        ];
        self.launch(
            stream,
            self.fns.quant_q8_64_b,
            ((cols / 64) as u32, rows as u32),
            64,
            &mut args,
        )
    }

    /// `y[rows_in, w.rows] = x_q8 @ w^T` for the quantized input scratch (`rows_in <= 8`).
    fn gemm(
        &self,
        stream: CUstream,
        w: &PlanarQ8Gpu,
        out_rows: usize,
        rows_in: usize,
        y: CUdeviceptr,
    ) -> Result<(), String> {
        let (mut pw, mut pws, mut px, mut pxs, mut py) = (
            w.codes.ptr,
            w.scales64.ptr,
            self.scr.xq.ptr,
            self.scr.xs.ptr,
            y,
        );
        let (mut r, mut c, mut b, mut ys) =
            (out_rows as i32, w.cols, rows_in as i32, out_rows as i32);
        let gx = (out_rows as u32).div_ceil(16);
        if w.cols <= 3072 {
            if let Some(f) = self.fns.q8_gemm_b8_fast {
                let mut sc = 0.0f32;
                let mut args: [*mut c_void; 10] = [
                    (&mut pw as *mut _) as *mut c_void,
                    (&mut pws as *mut _) as *mut c_void,
                    (&mut px as *mut _) as *mut c_void,
                    (&mut pxs as *mut _) as *mut c_void,
                    (&mut py as *mut _) as *mut c_void,
                    (&mut r as *mut _) as *mut c_void,
                    (&mut c as *mut _) as *mut c_void,
                    (&mut b as *mut _) as *mut c_void,
                    (&mut ys as *mut _) as *mut c_void,
                    (&mut sc as *mut _) as *mut c_void,
                ];
                return self.launch(stream, f, (gx, 1), 256, &mut args);
            }
        }
        let mut args: [*mut c_void; 9] = [
            (&mut pw as *mut _) as *mut c_void,
            (&mut pws as *mut _) as *mut c_void,
            (&mut px as *mut _) as *mut c_void,
            (&mut pxs as *mut _) as *mut c_void,
            (&mut py as *mut _) as *mut c_void,
            (&mut r as *mut _) as *mut c_void,
            (&mut c as *mut _) as *mut c_void,
            (&mut b as *mut _) as *mut c_void,
            (&mut ys as *mut _) as *mut c_void,
        ];
        self.launch(stream, self.fns.q8_gemm_ktiled, (gx, 1), 256, &mut args)
    }

    fn rmsnorm(
        &self,
        stream: CUstream,
        x: CUdeviceptr,
        w: &DeviceBuf,
        y: CUdeviceptr,
        rows: usize,
    ) -> Result<(), String> {
        let (mut px, mut pw, mut py) = (x, w.ptr, y);
        let mut args: [*mut c_void; 3] = [
            (&mut px as *mut _) as *mut c_void,
            (&mut pw as *mut _) as *mut c_void,
            (&mut py as *mut _) as *mut c_void,
        ];
        self.launch(stream, self.fns.rmsnorm, (rows as u32, 1), 256, &mut args)
    }

    fn conv2(
        &self,
        stream: CUstream,
        z: CUdeviceptr,
        base: &DeviceBuf,
        post: bool,
        out: CUdeviceptr,
    ) -> Result<(), String> {
        let (mut pz, mut pd, mut pb, mut po, mut pt) =
            (z, self.scr.dyn_.ptr, base.ptr, post as i32, out);
        let mut args: [*mut c_void; 5] = [
            (&mut pz as *mut _) as *mut c_void,
            (&mut pd as *mut _) as *mut c_void,
            (&mut pb as *mut _) as *mut c_void,
            (&mut po as *mut _) as *mut c_void,
            (&mut pt as *mut _) as *mut c_void,
        ];
        self.launch(
            stream,
            self.fns.conv2,
            (DF2_BLOCK as u32, 1),
            256,
            &mut args,
        )
    }

    /// Per-head norm + RoPE of `rows` rows at `pos0..`: block queries in place when `with_q`,
    /// K / V of `scr.kv` (stride `2 * DF2_KV`) into layer `li`'s rings.
    fn qk_prep(
        &self,
        stream: CUstream,
        li: usize,
        with_q: bool,
        pos0: usize,
        rows: usize,
    ) -> Result<(), String> {
        let l = &self.layers[li];
        let mut pq = self.scr.q.ptr;
        let mut pk = self.scr.kv.ptr;
        let mut pv = self.scr.kv.ptr + (DF2_KV * 4) as u64;
        let mut ks = (2 * DF2_KV) as i32;
        let (mut pqn, mut pkn, mut pf) = (l.q_norm.ptr, l.k_norm.ptr, self.inv_freq.ptr);
        let (mut p0, mut cap) = (pos0 as i32, DF2_RING as i32);
        let (mut kr, mut vr) = (l.k_ring.ptr, l.v_ring.ptr);
        let mut args: [*mut c_void; 11] = [
            (&mut pq as *mut _) as *mut c_void,
            (&mut pk as *mut _) as *mut c_void,
            (&mut pv as *mut _) as *mut c_void,
            (&mut ks as *mut _) as *mut c_void,
            (&mut pqn as *mut _) as *mut c_void,
            (&mut pkn as *mut _) as *mut c_void,
            (&mut pf as *mut _) as *mut c_void,
            (&mut p0 as *mut _) as *mut c_void,
            (&mut cap as *mut _) as *mut c_void,
            (&mut kr as *mut _) as *mut c_void,
            (&mut vr as *mut _) as *mut c_void,
        ];
        let gx = if with_q { 40 } else { 8 };
        self.launch(stream, self.fns.qk_prep, (gx, rows as u32), 128, &mut args)
    }

    /// Commits `n_toks` verified / prefilled target positions (`taps [4, tap_cap, D]`, rows
    /// `0..n_toks`) at positions `ctx_end..`. Only the last `DF2_WINDOW` rows can be attended.
    pub fn enqueue_add_ctx(
        &mut self,
        stream: CUstream,
        taps: CUdeviceptr,
        tap_cap: usize,
        n_toks: usize,
    ) -> Result<(), String> {
        if n_toks == 0 {
            return Ok(());
        }
        let skip = n_toks.saturating_sub(DF2_WINDOW);
        let mut r0 = skip;
        while r0 < n_toks {
            let m = (n_toks - r0).min(DF2_BLOCK);
            {
                let (mut pt, mut ts, mut row0, mut po) =
                    (taps, (tap_cap * D) as i64, r0 as i32, self.scr.tap.ptr);
                let mut args: [*mut c_void; 4] = [
                    (&mut pt as *mut _) as *mut c_void,
                    (&mut ts as *mut _) as *mut c_void,
                    (&mut row0 as *mut _) as *mut c_void,
                    (&mut po as *mut _) as *mut c_void,
                ];
                self.launch(stream, self.fns.tap_gather, (m as u32, 1), 256, &mut args)?;
            }
            self.quant(stream, self.scr.tap.ptr, DF2_TAPS * D, m)?;
            self.gemm(stream, &self.fc, D, m, self.scr.craw.ptr)?;
            self.rmsnorm(
                stream,
                self.scr.craw.ptr,
                &self.hidden_norm,
                self.scr.hd.ptr,
                m,
            )?;
            self.quant(stream, self.scr.hd.ptr, D, m)?;
            for li in 0..DF2_LAYERS {
                self.gemm(stream, &self.layers[li].wkv, 2 * DF2_KV, m, self.scr.kv.ptr)?;
                self.qk_prep(stream, li, false, self.ctx_end + r0, m)?;
            }
            r0 += m;
        }
        if skip > 0 {
            self.lo = self.ctx_end + skip;
        }
        self.ctx_end += n_toks;
        self.lo = self
            .lo
            .max(self.ctx_end.saturating_sub(DF2_RING - DF2_BLOCK));
        Ok(())
    }

    /// Drafts up to `max_k` (<= 7) tokens after `next_tok` (the pending token at `ctx_end`) and
    /// keeps the prefix whose cumulative calibrated confidence stays above `tau`.
    pub fn draft_tokens(
        &mut self,
        stream: CUstream,
        embed_ptr: CUdeviceptr,
        embed_norm_ptr: CUdeviceptr,
        unembed_q8: &PlanarQ8Gpu,
        next_tok: u32,
        max_k: usize,
    ) -> Result<Vec<u32>, String> {
        let n = max_k.min(DF2_BLOCK - 1);
        if n == 0 || self.ctx_end == 0 {
            return Ok(Vec::new());
        }
        let pos = self.ctx_end;
        let s = &self.scr;
        let (x, xn, y, z) = (s.x.ptr, s.xn.ptr, s.y.ptr, s.z.ptr);
        {
            let (mut pe, mut pn, mut a, mut mt, mut px) = (
                embed_ptr,
                embed_norm_ptr,
                next_tok as i32,
                DF2_MASK_TOKEN,
                x,
            );
            let mut args: [*mut c_void; 5] = [
                (&mut pe as *mut _) as *mut c_void,
                (&mut pn as *mut _) as *mut c_void,
                (&mut a as *mut _) as *mut c_void,
                (&mut mt as *mut _) as *mut c_void,
                (&mut px as *mut _) as *mut c_void,
            ];
            self.launch(
                stream,
                self.fns.block_init,
                (DF2_BLOCK as u32, 1),
                256,
                &mut args,
            )?;
        }
        let lo = self.lo.max((pos + 1).saturating_sub(DF2_WINDOW));
        for li in 0..DF2_LAYERS {
            let l = &self.layers[li];
            // Attention sublayer.
            self.rmsnorm(stream, x, &l.attn_norm, xn, DF2_BLOCK)?;
            self.quant(stream, xn, D, DF2_BLOCK)?;
            self.gemm(stream, &l.conv_a, DF2_DYN, DF2_BLOCK, s.dyn_.ptr)?;
            self.conv2(stream, xn, &l.base_a, false, y)?;
            self.quant(stream, y, D, DF2_BLOCK)?;
            self.gemm(stream, &l.wq, DF2_Q, DF2_BLOCK, s.q.ptr)?;
            self.gemm(stream, &l.wkv, 2 * DF2_KV, DF2_BLOCK, s.kv.ptr)?;
            self.qk_prep(stream, li, true, pos, DF2_BLOCK)?;
            {
                let (mut pq, mut kr, mut vr) = (s.q.ptr, l.k_ring.ptr, l.v_ring.ptr);
                let (mut plo, mut phi, mut cap) =
                    (lo as i32, (pos + DF2_BLOCK) as i32, DF2_RING as i32);
                let mut po = s.att.ptr;
                let mut args: [*mut c_void; 7] = [
                    (&mut pq as *mut _) as *mut c_void,
                    (&mut kr as *mut _) as *mut c_void,
                    (&mut vr as *mut _) as *mut c_void,
                    (&mut plo as *mut _) as *mut c_void,
                    (&mut phi as *mut _) as *mut c_void,
                    (&mut cap as *mut _) as *mut c_void,
                    (&mut po as *mut _) as *mut c_void,
                ];
                self.launch(
                    stream,
                    self.fns.attn,
                    (32, DF2_BLOCK as u32),
                    256,
                    &mut args,
                )?;
            }
            self.quant(stream, s.att.ptr, DF2_Q, DF2_BLOCK)?;
            self.gemm(stream, &l.wo, D, DF2_BLOCK, z)?;
            self.conv2(stream, z, &l.base_a, true, x)?;
            // MLP sublayer.
            self.rmsnorm(stream, x, &l.post_norm, xn, DF2_BLOCK)?;
            self.quant(stream, xn, D, DF2_BLOCK)?;
            self.gemm(stream, &l.conv_f, DF2_DYN, DF2_BLOCK, s.dyn_.ptr)?;
            self.conv2(stream, xn, &l.base_f, false, y)?;
            self.quant(stream, y, D, DF2_BLOCK)?;
            self.gemm(stream, &l.gu, 2 * DF2_FF, DF2_BLOCK, s.gu.ptr)?;
            {
                let (mut pg, mut pa, mut r, mut ff) =
                    (s.gu.ptr, s.act.ptr, DF2_BLOCK as i32, DF2_FF as i32);
                let mut args: [*mut c_void; 4] = [
                    (&mut pg as *mut _) as *mut c_void,
                    (&mut pa as *mut _) as *mut c_void,
                    (&mut r as *mut _) as *mut c_void,
                    (&mut ff as *mut _) as *mut c_void,
                ];
                let blocks = ((DF2_BLOCK * DF2_FF) as u32).div_ceil(256);
                self.launch(stream, self.fns.silu_mul, (blocks, 1), 256, &mut args)?;
            }
            self.quant(stream, s.act.ptr, DF2_FF, DF2_BLOCK)?;
            self.gemm(stream, &l.down, D, DF2_BLOCK, z)?;
            self.conv2(stream, z, &l.base_f, true, x)?;
        }
        // Head over draft rows 1..=n.
        self.rmsnorm(stream, x + (D * 4) as u64, &self.norm, s.hd.ptr, n)?;
        self.quant(stream, s.hd.ptr, D, n)?;
        self.gemm(stream, unembed_q8, VOCAB, n, s.logits.ptr)?;
        self.gemm(stream, &self.sel_h, DF2_RANK, n, s.g.ptr)?;
        {
            let (mut pl, mut v, mut cap, mut pp) =
                (s.logits.ptr, VOCAB as i32, DF2_SOFTCAP, s.part.ptr);
            let mut args: [*mut c_void; 4] = [
                (&mut pl as *mut _) as *mut c_void,
                (&mut v as *mut _) as *mut c_void,
                (&mut cap as *mut _) as *mut c_void,
                (&mut pp as *mut _) as *mut c_void,
            ];
            self.launch(
                stream,
                self.fns.top16_part,
                (n as u32, DF2_CHUNKS as u32),
                256,
                &mut args,
            )?;
            let (mut pp, mut pc, mut pt) = (s.part.ptr, s.cand.ptr, s.tail.ptr);
            let mut args: [*mut c_void; 3] = [
                (&mut pp as *mut _) as *mut c_void,
                (&mut pc as *mut _) as *mut c_void,
                (&mut pt as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.top16_merge, (n as u32, 1), 256, &mut args)?;
        }
        {
            let (mut a, mut pc, mut pt, mut pg) =
                (next_tok as i32, s.cand.ptr, s.tail.ptr, s.g.ptr);
            let (mut pp, mut ps, mut nn, mut cap) =
                (self.pred.ptr, self.succ.ptr, n as i32, DF2_SOFTCAP);
            let (mut pi, mut pcf) = (s.ids.ptr, s.conf.ptr);
            let mut args: [*mut c_void; 10] = [
                (&mut a as *mut _) as *mut c_void,
                (&mut pc as *mut _) as *mut c_void,
                (&mut pt as *mut _) as *mut c_void,
                (&mut pg as *mut _) as *mut c_void,
                (&mut pp as *mut _) as *mut c_void,
                (&mut ps as *mut _) as *mut c_void,
                (&mut nn as *mut _) as *mut c_void,
                (&mut cap as *mut _) as *mut c_void,
                (&mut pi as *mut _) as *mut c_void,
                (&mut pcf as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.select, (1, 1), 512, &mut args)?;
        }
        self.ctx.sync_stream(stream)?;
        let mut ids = [0i32; DF2_BLOCK];
        let mut conf = [0f32; DF2_BLOCK];
        unsafe {
            check_cu(
                (self.ctx.api.cuMemcpyDtoH_v2)(ids.as_mut_ptr().cast(), s.ids.ptr, n * 4),
                "DtoH(df2 ids)",
            )?;
            check_cu(
                (self.ctx.api.cuMemcpyDtoH_v2)(conf.as_mut_ptr().cast(), s.conf.ptr, n * 4),
                "DtoH(df2 conf)",
            )?;
        }
        let mut out = Vec::with_capacity(n);
        let mut cum = 1.0f32;
        for i in 0..n {
            cum *= conf[i];
            if cum < self.tau || !(0..VOCAB as i32).contains(&ids[i]) {
                break;
            }
            out.push(ids[i] as u32);
        }
        Ok(out)
    }
}
