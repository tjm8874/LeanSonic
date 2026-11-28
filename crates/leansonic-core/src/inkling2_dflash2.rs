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

/// Entry points of [`DF2_KERNEL_SRC`], in [`Df2ModuleFns`] field order.
pub const DF2_KERNEL_NAMES: [&str; 16] = [
    "df2_block_init",
    "df2_norm_quant8",
    "df2_rmsnorm_quant",
    "df2_conv2_quant",
    "df2_conv2_post",
    "df2_gemm_sk",
    "df2_qk_prep",
    "df2_attn_q8",
    "df2_silu_quant",
    "df2_tap_quant",
    "df2_top16_part",
    "df2_top16_merge",
    "df2_select",
    "df2_q8_to_q4",
    "df2_q4_head",
    "df2_gemm_sk_q4",
];

/// Handles of the drafter module's [`DF2_KERNEL_NAMES`].
#[derive(Clone, Copy)]
pub struct Df2ModuleFns {
    pub block_init: CUfunction,
    pub norm_quant8: CUfunction,
    pub rmsnorm_quant: CUfunction,
    pub conv2_quant: CUfunction,
    pub conv2_post: CUfunction,
    pub gemm_sk: CUfunction,
    pub qk_prep: CUfunction,
    pub attn_q8: CUfunction,
    pub silu_quant: CUfunction,
    pub tap_quant: CUfunction,
    pub top16_part: CUfunction,
    pub top16_merge: CUfunction,
    pub select: CUfunction,
    pub q8_to_q4: CUfunction,
    pub q4_head: CUfunction,
    pub gemm_sk_q4: CUfunction,
}

/// Loads the drafter module from `LEANSONIC_DF2_CUBIN_PATH`, or compiles [`DF2_KERNEL_SRC`] with
/// NVRTC (cached like the target module), and returns its [`DF2_KERNEL_NAMES`] handles.
pub fn load_dflash2_kernels(ctx: &CudaContextHandle) -> Result<Df2ModuleFns, String> {
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
    let mut f: [CUfunction; 16] = [core::ptr::null_mut(); 16];
    for (h, name) in f.iter_mut().zip(DF2_KERNEL_NAMES) {
        let c = std::ffi::CString::new(name).unwrap();
        unsafe {
            check_cu((ctx.api.cuModuleGetFunction)(h, module, c.as_ptr()), name)?;
        }
    }
    Ok(Df2ModuleFns {
        block_init: f[0],
        norm_quant8: f[1],
        rmsnorm_quant: f[2],
        conv2_quant: f[3],
        conv2_post: f[4],
        gemm_sk: f[5],
        qk_prep: f[6],
        attn_q8: f[7],
        silu_quant: f[8],
        tap_quant: f[9],
        top16_part: f[10],
        top16_merge: f[11],
        select: f[12],
        q8_to_q4: f[13],
        q4_head: f[14],
        gemm_sk_q4: f[15],
    })
}

/// Kernel handles for the drafter: the target module's planar Q8_64 GEMMs plus the drafter
/// module ([`load_dflash2_kernels`]).
#[derive(Clone, Copy)]
pub struct Dflash2KernelFns {
    pub pdl_all: bool,
    pub q8_gemm_ktiled: CUfunction,
    pub q8_gemm_b8_fast: Option<CUfunction>,
    pub m: Df2ModuleFns,
}

unsafe impl Send for Dflash2KernelFns {}
unsafe impl Sync for Dflash2KernelFns {}

struct Df2Layer {
    attn_norm: DeviceBuf,
    post_norm: DeviceBuf,
    q_norm: DeviceBuf,
    base_a: DeviceBuf,
    base_f: DeviceBuf,
    conv_a: PlanarQ8Gpu,
    conv_f: PlanarQ8Gpu,
    wo: PlanarQ8Gpu,
    gu: Df2Weight,
    down: Df2Weight,
}

/// Signed 4-bit planar matrix (`df2_q8_to_q4` layout): codes `[rows, cols / 2]`, FP32 scales
/// `[rows, cols / 64]`.
struct Q4Mat {
    codes: DeviceBuf,
    scales: DeviceBuf,
    rows: usize,
    cols: usize,
}

/// An MLP matrix of the drafter: Q4 (`LEANSONIC_DF2_MLP=q4`, default) or Q8_64.
enum Df2Weight {
    Q8(PlanarQ8Gpu, usize),
    Q4(Q4Mat),
}

/// A planar Q8_64 matrix (or a row range of one) by pointer.
#[derive(Clone, Copy)]
struct Q8View {
    codes: CUdeviceptr,
    scales: CUdeviceptr,
    rows: usize,
    cols: usize,
}

impl Q8View {
    fn of(w: &PlanarQ8Gpu, rows: usize) -> Self {
        Self {
            codes: w.codes.ptr,
            scales: w.scales64.ptr,
            rows,
            cols: w.cols as usize,
        }
    }

    fn rows(self, r0: usize, rows: usize) -> Self {
        Self {
            codes: self.codes + (r0 * self.cols) as u64,
            scales: self.scales + (r0 * (self.cols / 64) * 4) as u64,
            rows,
            cols: self.cols,
        }
    }
}

/// Split-K slices of the drafter GEMMs (`K = 2304`: conv 4, qkv / gate-up 2; `wo` 4, `down` 4,
/// context `fc` 6).
const DF2_SK_CONV: usize = 4;
const DF2_SK_QKV: usize = 2;
const DF2_SK_GU: usize = 2;
const DF2_SK_WO: usize = 4;
const DF2_SK_DOWN: usize = 4;
const DF2_SK_FC: usize = 6;
/// Blocks of the grid-stride Q4 draft head (`LEANSONIC_DF2_HEAD_BLOCKS`, default 880).
fn head_q4_blocks() -> usize {
    std::env::var("LEANSONIC_DF2_HEAD_BLOCKS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(880)
}

/// Rows of one layer's `[q_proj; k_proj; v_proj]`.
const DF2_QKV: usize = DF2_Q + 2 * DF2_KV;

struct Df2Scratch {
    x: DeviceBuf,
    xn: DeviceBuf,
    dynp: DeviceBuf,
    q: DeviceBuf,
    skp: DeviceBuf,
    mask_row: DeviceBuf,
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
            dynp: f(DF2_SK_CONV * rows * DF2_DYN)?,
            q: f(rows * DF2_Q)?,
            skp: f((DF2_SK_FC * D)
                .max(DF2_SK_GU * 2 * DF2_FF)
                .max(DF2_SK_QKV * DF2_LAYERS * 2 * DF2_KV)
                .max(DF2_SK_WO.max(DF2_SK_DOWN) * D)
                * rows)?,
            mask_row: f(D)?,
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
    /// `[q_proj; k_proj; v_proj]` of all five layers (`[5 * 6144, D]`): one split-K GEMM per
    /// layer in a draft, and one over all five K / V blocks per context commit.
    wqkv_all: PlanarQ8Gpu,
    /// Whether `scr.mask_row` holds the normalized mask-token row.
    mask_ready: bool,
    /// K norms of all layers (`[5, 128]` BF16).
    k_norm_all: DeviceBuf,
    /// FP32 K / V rings of all layers (`[5, DF2_RING, 1024]`).
    k_ring_all: DeviceBuf,
    v_ring_all: DeviceBuf,
    /// Per-sequence `(ctx_end, lo)` of the engine's sequences (`select_seq`); the selected
    /// sequence's live values are `ctx_end` / `lo`, its rings `cur * DF2_LAYERS` layers in.
    seq_state: Vec<(usize, usize)>,
    cur: usize,
    /// Q4 copy of the target's output head (`LEANSONIC_DF2_HEAD=q4`, default; `q8` drafts with
    /// the target's own Q8_64 head): codes `[rows16, D / 2]` and scales `[rows16, D / 64]`.
    head_q4: Option<(DeviceBuf, DeviceBuf)>,
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

/// Kernel argument list: `args![a, b, ...]` takes `&mut` of each local.
macro_rules! args {
    ($($v:ident),* $(,)?) => {
        [$((&mut $v as *mut _) as *mut c_void),*]
    };
}

impl Dflash2DrafterGpu {
    pub fn load_from_file(
        ctx: &Arc<CudaContextHandle>,
        fns: Dflash2KernelFns,
        path: &Path,
        unembed_q8: &PlanarQ8Gpu,
        n_seqs: usize,
    ) -> Result<Self, String> {
        let st =
            MappedSafeTensors::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let vec_f32 = |name: &str, n: usize| -> Result<DeviceBuf, String> {
            f32_buf(ctx, st.require(name, "BF16", &[n]).as_u16_slice())
        };
        let raw = |name: &str, shape: &[usize]| -> Result<DeviceBuf, String> {
            DeviceBuf::from_host_bytes(ctx, st.require(name, "BF16", shape).data)
        };
        let mlp_q4 = match std::env::var("LEANSONIC_DF2_MLP").as_deref() {
            Ok("q8") => false,
            Ok("q4") | Err(_) => true,
            Ok(v) => return Err(format!("LEANSONIC_DF2_MLP must be q4 or q8, got {v}")),
        };
        let mut layers = Vec::with_capacity(DF2_LAYERS);
        let mut qkv_names = Vec::with_capacity(3 * DF2_LAYERS);
        let mut k_norms = Vec::with_capacity(DF2_LAYERS * 256);
        for l in 0..DF2_LAYERS {
            let p = |s: &str| format!("layers.{l}.{s}");
            qkv_names.push(p("self_attn.q_proj.weight"));
            qkv_names.push(p("self_attn.k_proj.weight"));
            qkv_names.push(p("self_attn.v_proj.weight"));
            k_norms.extend_from_slice(
                st.require(&p("self_attn.k_norm.weight"), "BF16", &[128])
                    .data,
            );
            layers.push(Df2Layer {
                attn_norm: vec_f32(&p("input_layernorm.weight"), D)?,
                post_norm: vec_f32(&p("post_attention_layernorm.weight"), D)?,
                q_norm: raw(&p("self_attn.q_norm.weight"), &[128])?,
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
                wo: q8(ctx, &st, &[&p("self_attn.o_proj.weight")], &[D], DF2_Q)?,
                gu: Self::mlp_weight(
                    ctx,
                    fns,
                    mlp_q4,
                    q8(
                        ctx,
                        &st,
                        &[&p("mlp.gate_proj.weight"), &p("mlp.up_proj.weight")],
                        &[DF2_FF, DF2_FF],
                        D,
                    )?,
                    2 * DF2_FF,
                )?,
                down: Self::mlp_weight(
                    ctx,
                    fns,
                    mlp_q4,
                    q8(ctx, &st, &[&p("mlp.down_proj.weight")], &[D], DF2_FF)?,
                    D,
                )?,
            });
        }
        let qkv_refs: Vec<&str> = qkv_names.iter().map(String::as_str).collect();
        let qkv_rows: Vec<usize> = (0..DF2_LAYERS)
            .flat_map(|_| [DF2_Q, DF2_KV, DF2_KV])
            .collect();
        let wqkv_all = q8(ctx, &st, &qkv_refs, &qkv_rows, D)?;
        let mut inv_freq = Vec::with_capacity(64 * 4);
        for i in 0..64 {
            let f = 10_000.0_f64.powf(-2.0 * (i as f64) / 128.0) as f32;
            inv_freq.extend_from_slice(&f.to_le_bytes());
        }
        let tau = std::env::var("LEANSONIC_DF2_TAU")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.2);
        let ring = n_seqs * DF2_LAYERS * DF2_RING * DF2_KV * 4;
        let head_q4 = match std::env::var("LEANSONIC_DF2_HEAD").as_deref() {
            Ok("q8") => None,
            Ok("q4") | Err(_) => Some(Self::build_head_q4(ctx, fns, unembed_q8)?),
            Ok(v) => return Err(format!("LEANSONIC_DF2_HEAD must be q4 or q8, got {v}")),
        };
        Ok(Self {
            head_q4,
            ctx: Arc::clone(ctx),
            fns,
            wqkv_all,
            mask_ready: false,
            k_norm_all: DeviceBuf::from_host_bytes(ctx, &k_norms)?,
            k_ring_all: DeviceBuf::alloc_zeroed(ctx, ring)?,
            v_ring_all: DeviceBuf::alloc_zeroed(ctx, ring)?,
            seq_state: vec![(0, 0); n_seqs],
            cur: 0,
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

    /// Requantizes `rows` rows of a planar Q8_64 matrix into a [`Q4Mat`] with `rows16` rows
    /// (the rows past `rows` stay zero).
    fn q4_from_q8(
        ctx: &Arc<CudaContextHandle>,
        fns: Dflash2KernelFns,
        w: &PlanarQ8Gpu,
        rows: usize,
        rows16: usize,
    ) -> Result<Q4Mat, String> {
        let cols = w.cols as usize;
        let codes = DeviceBuf::alloc_zeroed(ctx, rows16 * cols / 2)?;
        let scales = DeviceBuf::alloc_zeroed(ctx, rows16 * (cols / 64) * 4)?;
        let (mut pw, mut ps, mut pc, mut pss, mut c) = (
            w.codes.ptr,
            w.scales64.ptr,
            codes.ptr,
            scales.ptr,
            cols as i32,
        );
        let mut a = args![pw, ps, pc, pss, c];
        unsafe {
            check_cu(
                (ctx.api.cuLaunchKernel)(
                    fns.m.q8_to_q4,
                    rows as u32,
                    (cols / 64) as u32,
                    1,
                    32,
                    1,
                    1,
                    0,
                    ctx.compute_stream,
                    a.as_mut_ptr(),
                    core::ptr::null_mut(),
                ),
                "df2_q8_to_q4",
            )?;
        }
        ctx.sync_stream(ctx.compute_stream)?;
        Ok(Q4Mat {
            codes,
            scales,
            rows: rows16,
            cols,
        })
    }

    /// The Q8_64 head as the Q4 drafting copy (rows padded to a multiple of 16).
    fn build_head_q4(
        ctx: &Arc<CudaContextHandle>,
        fns: Dflash2KernelFns,
        unembed_q8: &PlanarQ8Gpu,
    ) -> Result<(DeviceBuf, DeviceBuf), String> {
        let m = Self::q4_from_q8(ctx, fns, unembed_q8, VOCAB, VOCAB.div_ceil(16) * 16)?;
        Ok((m.codes, m.scales))
    }

    /// An MLP matrix: Q4 when `q4` (the Q8_64 upload is released), else Q8_64.
    fn mlp_weight(
        ctx: &Arc<CudaContextHandle>,
        fns: Dflash2KernelFns,
        q4: bool,
        w: PlanarQ8Gpu,
        rows: usize,
    ) -> Result<Df2Weight, String> {
        if q4 {
            Ok(Df2Weight::Q4(Self::q4_from_q8(ctx, fns, &w, rows, rows)?))
        } else {
            Ok(Df2Weight::Q8(w, rows))
        }
    }

    /// Split-K GEMM of the eight block rows over an MLP matrix into `out`.
    fn gemm_sk_w(
        &self,
        stream: CUstream,
        w: &Df2Weight,
        slices: usize,
        out: CUdeviceptr,
    ) -> Result<(), String> {
        match w {
            Df2Weight::Q8(m, rows) => self.gemm_sk(stream, Q8View::of(m, *rows), slices, out),
            Df2Weight::Q4(m) => {
                let slice32 = m.cols / 32 / slices;
                debug_assert!(
                    slice32 <= 64 && slice32.is_multiple_of(2) && slice32 * slices * 32 == m.cols
                );
                let (mut pw, mut pws, mut px, mut pxs, mut pp) = (
                    m.codes.ptr,
                    m.scales.ptr,
                    self.scr.xq.ptr,
                    self.scr.xs.ptr,
                    out,
                );
                let (mut r, mut c, mut sl) = (m.rows as i32, m.cols as i32, slice32 as i32);
                let mut a = args![pw, pws, px, pxs, pp, r, c, sl];
                let grid = ((m.rows / 16) as u32, slices as u32, 1);
                self.launch(stream, self.fns.m.gemm_sk_q4, grid, 256, &mut a)
            }
        }
    }

    /// Clears the selected sequence's context (like the engine's `reset`).
    pub fn reset(&mut self) {
        // The mask row is a pure function of the weights: it stays valid across resets.
        self.ctx_end = 0;
        self.lo = 0;
    }

    fn launch(
        &self,
        stream: CUstream,
        func: CUfunction,
        grid: (u32, u32, u32),
        block: u32,
        params: &mut [*mut c_void],
    ) -> Result<(), String> {
        if self.fns.pdl_all {
            return crate::inkling2_gpu::launch_pdl_3d(
                &self.ctx, stream, func, grid, block, 0, params,
            );
        }
        unsafe {
            check_cu(
                (self.ctx.api.cuLaunchKernel)(
                    func,
                    grid.0,
                    grid.1,
                    grid.2,
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

    /// Byte offset of the selected sequence's layer `li` ring.
    fn ring_layer(&self, li: usize) -> u64 {
        ((self.cur * DF2_LAYERS + li) * DF2_RING * DF2_KV * 4) as u64
    }

    /// Selects the sequence whose context the next commit / draft uses.
    pub fn select_seq(&mut self, seq: usize) -> Result<(), String> {
        if seq >= self.seq_state.len() {
            return Err(format!(
                "DFlash 2 sequence {seq} >= {}",
                self.seq_state.len()
            ));
        }
        self.seq_state[self.cur] = (self.ctx_end, self.lo);
        (self.ctx_end, self.lo) = self.seq_state[seq];
        self.cur = seq;
        Ok(())
    }

    /// `y[rows_in, w.rows]` (row stride `y_stride`) `= x_q8 @ w^T` for the Q8_64 input scratch.
    fn gemm(
        &self,
        stream: CUstream,
        w: Q8View,
        rows_in: usize,
        y: CUdeviceptr,
        y_stride: usize,
    ) -> Result<(), String> {
        let (mut pw, mut pws, mut px, mut pxs, mut py) =
            (w.codes, w.scales, self.scr.xq.ptr, self.scr.xs.ptr, y);
        let (mut r, mut c, mut b, mut ys) = (
            w.rows as i32,
            w.cols as i32,
            rows_in as i32,
            y_stride as i32,
        );
        let gx = (w.rows as u32).div_ceil(16);
        if w.cols <= 3072 {
            if let Some(f) = self.fns.q8_gemm_b8_fast {
                let mut sc = 0.0f32;
                let mut a = args![pw, pws, px, pxs, py, r, c, b, ys, sc];
                return self.launch(stream, f, (gx, 1, 1), 256, &mut a);
            }
        }
        let mut a = args![pw, pws, px, pxs, py, r, c, b, ys];
        self.launch(stream, self.fns.q8_gemm_ktiled, (gx, 1, 1), 256, &mut a)
    }

    /// Split-K GEMM of the eight block rows into `out` (`[slices, 8, rows]`), row `r` of the
    /// output reading weight row `(r / seg) * seg_stride + seg_off + r % seg` of `w`.
    #[allow(clippy::too_many_arguments)]
    fn gemm_sk_rows(
        &self,
        stream: CUstream,
        w: Q8View,
        rows: usize,
        slices: usize,
        out: CUdeviceptr,
        seg: (usize, usize, usize),
    ) -> Result<(), String> {
        let slice16 = w.cols / 16 / slices;
        debug_assert!(
            slice16 <= 128 && slice16.is_multiple_of(4) && slice16 * slices * 16 == w.cols
        );
        let (mut pw, mut pws, mut px, mut pxs, mut pp) =
            (w.codes, w.scales, self.scr.xq.ptr, self.scr.xs.ptr, out);
        let (mut r, mut c, mut sl) = (rows as i32, w.cols as i32, slice16 as i32);
        let (mut sg, mut sst, mut so) = (seg.0 as i32, seg.1 as i32, seg.2 as i32);
        let mut a = args![pw, pws, px, pxs, pp, r, c, sl, sg, sst, so];
        let grid = ((rows / 16) as u32, slices as u32, 1);
        self.launch(stream, self.fns.m.gemm_sk, grid, 256, &mut a)
    }

    fn gemm_sk(
        &self,
        stream: CUstream,
        w: Q8View,
        slices: usize,
        out: CUdeviceptr,
    ) -> Result<(), String> {
        self.gemm_sk_rows(stream, w, w.rows, slices, out, (w.rows, 0, 0))
    }

    /// Q8_64 of `rms_norm(sum of nsplit partials of x) * w` for `rows` rows into the input scratch.
    fn rmsnorm_quant(
        &self,
        stream: CUstream,
        x: CUdeviceptr,
        nsplit: usize,
        split_stride: usize,
        w: &DeviceBuf,
        rows: usize,
    ) -> Result<(), String> {
        let (mut px, mut ns, mut ss, mut pw, mut pq, mut ps) = (
            x,
            nsplit as i32,
            split_stride as i64,
            w.ptr,
            self.scr.xq.ptr,
            self.scr.xs.ptr,
        );
        let mut a = args![px, ns, ss, pw, pq, ps];
        self.launch(
            stream,
            self.fns.m.rmsnorm_quant,
            (rows as u32, 1, 1),
            256,
            &mut a,
        )
    }

    /// `xn = rms_norm(x) * nw` (FP32 and Q8_64), then the dynamic conv kernels
    /// `dyn = conv . xn` as `DF2_SK_CONV` split-K partials in `scr.dynp`.
    fn norm_dyn(&self, stream: CUstream, nw: &DeviceBuf, conv: &PlanarQ8Gpu) -> Result<(), String> {
        let s = &self.scr;
        let (mut px, mut pn, mut pxn, mut pq, mut ps) =
            (s.x.ptr, nw.ptr, s.xn.ptr, s.xq.ptr, s.xs.ptr);
        let mut a = args![px, pn, pxn, pq, ps];
        let grid = ((DF2_BLOCK * D / 512) as u32, 1, 1);
        self.launch(stream, self.fns.m.norm_quant8, grid, 256, &mut a)?;
        self.gemm_sk(stream, Q8View::of(conv, DF2_DYN), DF2_SK_CONV, s.dynp.ptr)
    }

    /// Pre-conv of `xn` straight into the Q8_64 input scratch.
    fn conv2_quant(&self, stream: CUstream, base: &DeviceBuf) -> Result<(), String> {
        let s = &self.scr;
        let (mut pz, mut pd, mut nd, mut pb, mut pq, mut ps) = (
            s.xn.ptr,
            s.dynp.ptr,
            DF2_SK_CONV as i32,
            base.ptr,
            s.xq.ptr,
            s.xs.ptr,
        );
        let mut a = args![pz, pd, nd, pb, pq, ps];
        self.launch(
            stream,
            self.fns.m.conv2_quant,
            ((DF2_BLOCK * D / 512) as u32, 1, 1),
            256,
            &mut a,
        )
    }

    /// Residual post-conv of the split-K partials in `scr.skp` into `x`.
    fn conv2_post(&self, stream: CUstream, nsplit: usize, base: &DeviceBuf) -> Result<(), String> {
        let s = &self.scr;
        let (mut pz, mut ns, mut pd, mut nd, mut pb, mut px) = (
            s.skp.ptr,
            nsplit as i32,
            s.dynp.ptr,
            DF2_SK_CONV as i32,
            base.ptr,
            s.x.ptr,
        );
        let mut a = args![pz, ns, pd, nd, pb, px];
        self.launch(
            stream,
            self.fns.m.conv2_post,
            ((D / 256) as u32, DF2_BLOCK as u32, 1),
            256,
            &mut a,
        )
    }

    /// Per-head norm + RoPE of `rows` rows at `pos0..` for `layers` layers starting at `li`, over
    /// the `DF2_SK_QKV` split-K partials in `scr.skp` (`row_stride` columns per row; layer `l`'s
    /// K at `k_col + l * layer_cols`, V 1024 columns later): block queries (columns `0..4096`)
    /// into `scr.q` when `with_q`, K / V into the rings.
    #[allow(clippy::too_many_arguments)]
    fn qk_prep(
        &self,
        stream: CUstream,
        li: usize,
        layers: usize,
        with_q: bool,
        row_stride: usize,
        k_col: usize,
        layer_cols: usize,
        pos0: usize,
        rows: usize,
    ) -> Result<(), String> {
        let mut pqkv = self.scr.skp.ptr;
        let (mut ns, mut ss, mut rs) = (
            DF2_SK_QKV as i32,
            (DF2_BLOCK * row_stride) as i32,
            row_stride as i32,
        );
        let (mut kc, mut lc) = (k_col as i32, layer_cols as i32);
        let mut pq = self.scr.q.ptr;
        let mut pqn = self.layers[li].q_norm.ptr;
        let mut pkn = self.k_norm_all.ptr + (li * 256) as u64;
        let mut pf = self.inv_freq.ptr;
        let (mut p0, mut cap, mut rl) = (pos0 as i32, DF2_RING as i32, (DF2_RING * DF2_KV) as i64);
        let mut kr = self.k_ring_all.ptr + self.ring_layer(li);
        let mut vr = self.v_ring_all.ptr + self.ring_layer(li);
        let mut a = args![pqkv, ns, ss, rs, kc, lc, pq, pqn, pkn, pf, p0, cap, rl, kr, vr];
        let gx = if with_q { 40 } else { 8 };
        self.launch(
            stream,
            self.fns.m.qk_prep,
            (gx, rows as u32, layers as u32),
            128,
            &mut a,
        )
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
        let wqkv = Q8View::of(&self.wqkv_all, DF2_LAYERS * DF2_QKV);
        let kv_rows = DF2_LAYERS * 2 * DF2_KV;
        let mut r0 = skip;
        while r0 < n_toks {
            let m = (n_toks - r0).min(DF2_BLOCK);
            {
                let (mut pt, mut ts, mut row0, mut pq, mut ps) = (
                    taps,
                    (tap_cap * D) as i64,
                    r0 as i32,
                    self.scr.xq.ptr,
                    self.scr.xs.ptr,
                );
                let mut a = args![pt, ts, row0, pq, ps];
                let grid = ((DF2_TAPS * D / 512) as u32, m as u32, 1);
                self.launch(stream, self.fns.m.tap_quant, grid, 256, &mut a)?;
            }
            self.gemm_sk(stream, Q8View::of(&self.fc, D), DF2_SK_FC, self.scr.skp.ptr)?;
            self.rmsnorm_quant(
                stream,
                self.scr.skp.ptr,
                DF2_SK_FC,
                DF2_BLOCK * D,
                &self.hidden_norm,
                m,
            )?;
            // The K / V blocks of all five layers: logical row `r` = weight row
            // `(r / 2048) * 6144 + 4096 + r % 2048`.
            self.gemm_sk_rows(
                stream,
                wqkv,
                kv_rows,
                DF2_SK_QKV,
                self.scr.skp.ptr,
                (2 * DF2_KV, DF2_QKV, DF2_Q),
            )?;
            self.qk_prep(
                stream,
                0,
                DF2_LAYERS,
                false,
                kv_rows,
                0,
                2 * DF2_KV,
                self.ctx_end + r0,
                m,
            )?;
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
        {
            let (mut pe, mut pn, mut an, mut mt, mut uc, mut pm, mut px) = (
                embed_ptr,
                embed_norm_ptr,
                next_tok as i32,
                DF2_MASK_TOKEN,
                self.mask_ready as i32,
                s.mask_row.ptr,
                s.x.ptr,
            );
            let mut a = args![pe, pn, an, mt, uc, pm, px];
            self.launch(
                stream,
                self.fns.m.block_init,
                (DF2_BLOCK as u32, 1, 1),
                256,
                &mut a,
            )?;
        }
        let lo = self.lo.max((pos + 1).saturating_sub(DF2_WINDOW));
        let wqkv = Q8View::of(&self.wqkv_all, DF2_LAYERS * DF2_QKV);
        for li in 0..DF2_LAYERS {
            let l = &self.layers[li];
            // Attention sublayer.
            self.norm_dyn(stream, &l.attn_norm, &l.conv_a)?;
            self.conv2_quant(stream, &l.base_a)?;
            let qkv = wqkv.rows(li * DF2_QKV, DF2_QKV);
            self.gemm_sk(stream, qkv, DF2_SK_QKV, s.skp.ptr)?;
            self.qk_prep(stream, li, 1, true, DF2_QKV, DF2_Q, 0, pos, DF2_BLOCK)?;
            {
                let mut pq = s.q.ptr;
                let mut kr = self.k_ring_all.ptr + self.ring_layer(li);
                let mut vr = self.v_ring_all.ptr + self.ring_layer(li);
                let (mut plo, mut phi, mut cap) =
                    (lo as i32, (pos + DF2_BLOCK) as i32, DF2_RING as i32);
                let (mut pxq, mut pxs) = (s.xq.ptr, s.xs.ptr);
                let mut a = args![pq, kr, vr, plo, phi, cap, pxq, pxs];
                self.launch(
                    stream,
                    self.fns.m.attn_q8,
                    (32, DF2_BLOCK as u32, 1),
                    512,
                    &mut a,
                )?;
            }
            self.gemm_sk(stream, Q8View::of(&l.wo, D), DF2_SK_WO, s.skp.ptr)?;
            self.conv2_post(stream, DF2_SK_WO, &l.base_a)?;
            // MLP sublayer.
            self.norm_dyn(stream, &l.post_norm, &l.conv_f)?;
            self.conv2_quant(stream, &l.base_f)?;
            self.gemm_sk_w(stream, &l.gu, DF2_SK_GU, s.skp.ptr)?;
            {
                let (mut pg, mut ns, mut ff, mut pq, mut ps) = (
                    s.skp.ptr,
                    DF2_SK_GU as i32,
                    DF2_FF as i32,
                    s.xq.ptr,
                    s.xs.ptr,
                );
                let mut a = args![pg, ns, ff, pq, ps];
                let grid = ((DF2_BLOCK * DF2_FF / 512) as u32, 1, 1);
                self.launch(stream, self.fns.m.silu_quant, grid, 256, &mut a)?;
            }
            self.gemm_sk_w(stream, &l.down, DF2_SK_DOWN, s.skp.ptr)?;
            self.conv2_post(stream, DF2_SK_DOWN, &l.base_f)?;
        }
        // Head over draft rows 1..=n.
        self.rmsnorm_quant(stream, s.x.ptr + (D * 4) as u64, 1, 0, &self.norm, n)?;
        if let Some((codes, scales)) = &self.head_q4 {
            let (mut pw, mut ps, mut px, mut pxs, mut py) =
                (codes.ptr, scales.ptr, s.xq.ptr, s.xs.ptr, s.logits.ptr);
            let (mut r, mut b, mut ys) = (VOCAB as i32, n as i32, VOCAB as i32);
            let mut a = args![pw, ps, px, pxs, py, r, b, ys];
            let grid = (VOCAB.div_ceil(16).min(head_q4_blocks()) as u32, 1, 1);
            self.launch(stream, self.fns.m.q4_head, grid, 256, &mut a)?;
        } else {
            self.gemm(
                stream,
                Q8View::of(unembed_q8, VOCAB),
                n,
                s.logits.ptr,
                VOCAB,
            )?;
        }
        self.gemm(
            stream,
            Q8View::of(&self.sel_h, DF2_RANK),
            n,
            s.g.ptr,
            DF2_RANK,
        )?;
        {
            let (mut pl, mut v, mut cap, mut pp) =
                (s.logits.ptr, VOCAB as i32, DF2_SOFTCAP, s.part.ptr);
            let mut a = args![pl, v, cap, pp];
            let grid = (n as u32, DF2_CHUNKS as u32, 1);
            self.launch(stream, self.fns.m.top16_part, grid, 256, &mut a)?;
            let (mut pp, mut pc, mut pt) = (s.part.ptr, s.cand.ptr, s.tail.ptr);
            let mut a = args![pp, pc, pt];
            self.launch(
                stream,
                self.fns.m.top16_merge,
                (n as u32, 1, 1),
                256,
                &mut a,
            )?;
        }
        {
            let (mut an, mut pc, mut pt, mut pg) =
                (next_tok as i32, s.cand.ptr, s.tail.ptr, s.g.ptr);
            let (mut pp, mut ps, mut nn, mut cap) =
                (self.pred.ptr, self.succ.ptr, n as i32, DF2_SOFTCAP);
            let (mut pi, mut pcf) = (s.ids.ptr, s.conf.ptr);
            let mut a = args![an, pc, pt, pg, pp, ps, nn, cap, pi, pcf];
            self.launch(stream, self.fns.m.select, (1, 1, 1), 512, &mut a)?;
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
        self.mask_ready = true;
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
