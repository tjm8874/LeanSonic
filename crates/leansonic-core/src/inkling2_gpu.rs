//! Real `Inkling-2-Mini` (`19.5 GB` Official `safetensors`) Full-GPU Blackwell Engine.
//!
//! Wave 1 Optimized Architecture:
//! 1. Load-time 128B-aligned Planar `Q8_64` (`int8` codes + per-64 `f32` block scales) for all
//!    dense projections (`wq/wr/gate/sink`, `wk`, `wv`, `wo`) and `unembed` (`200,058 x 2,304`),
//!    cutting dense + head DRAM bytes by `46.9%` and freeing `~1.10 GiB` VRAM on 16GB GPUs!
//! 2. Fused `8,528 x 2,304` SWA `[wq; wr; gate; sink; wk; wv]` single-launch projection on
//!    `pass == 0` / Prelude / Coda visits (`24` visits reduced from 3 GEMVs to 1 GEMV).
//! 3. SMEM-staged 2-rows/warp `__dp4a` INT8 GEMV & SMEM-staged `Q8_16` `__dp4a` `NVFP4` MoE.
//! 4. Device-driven 48-visit CUDA Graph replay + Visit-Multiplicity-Weighted Spilled Mode.

use core::ffi::c_void;
use std::ffi::CString;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::cuda_ffi::{
    check_cu, compile_cuda_to_ptx_nvrtc, CUdeviceptr, CUfunction, CUmodule, CUstream,
    CapturedGraphExec, CudaContextHandle, DeviceBuf, PinnedHostBuf,
};
use crate::inkling2_mtp::{DsparkDrafterGpu, MtpKernelFns, SuffixDrafter};
use crate::safetensors::MappedSafeTensors;

const CUDA_KERNEL_SRC: &str = include_str!("../../../kernels/inkling2_sm120.cu");

pub const D: usize = 2304;
pub const N_LAYER: usize = 24;
pub const N_SLOT: usize = 48;
pub const N_HEAD: usize = 24;
pub const N_KV: usize = 8;
pub const HD: usize = 128;
pub const N_REL: usize = 16;
pub const EXTENT: usize = 1024;
pub const N_EXP: usize = 128;
pub const TOP_K: usize = 8;
pub const FF: usize = 1152;
pub const RANK: usize = 64;
pub const VOCAB: usize = 200_058;
pub const PADDED_VOCAB: usize = 201_024;
pub const GLOBAL_LAYERS: [usize; 4] = [5, 11, 17, 23];
pub const MAX_BATCH_TOKENS: usize = 1024;
const EXPERT_PAGE: usize = 64;
pub const MAX_SPEC_TOKENS: usize = 8;
pub const TAP_SLOTS: [usize; 3] = [4, 28, 46];
const CONV_SEQ_STEPS: usize = MAX_SPEC_TOKENS + 3; // 11
const CONV_SEQ_SLOT_FLOATS: usize = (1024 + 1024 + D + D) * CONV_SEQ_STEPS; // 6656 * 11 = 73,216
const QRGS_ROWS: usize = N_HEAD * HD + N_REL * N_HEAD + N_HEAD * HD + N_HEAD; // 6552
const SWA_K_OFFSET: usize = QRGS_ROWS; // 6552
const SWA_V_OFFSET: usize = QRGS_ROWS + N_KV * HD; // 7576
const SWA_ALL_ROWS: usize = QRGS_ROWS + 2 * N_KV * HD; // 8600

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BatchHeadMode {
    None,
    LastOnly,
    AllHeads,
}

enum TierBuf {
    Vram(DeviceBuf),
    HostPinned(PinnedHostBuf),
}

impl TierBuf {
    fn ptr(&self) -> CUdeviceptr {
        match self {
            Self::Vram(b) => b.ptr,
            Self::HostPinned(b) => b.dev_ptr,
        }
    }
}

pub struct PlanarQ8Gpu {
    pub codes: DeviceBuf,
    pub scales64: DeviceBuf,
    pub cols: i32,
}

struct CtrlGpu {
    down: DeviceBuf,
    gain: DeviceBuf,
    forget: DeviceBuf,
}

enum ExpertScales {
    Nvfp4 {
        s13: DeviceBuf,
        a13: DeviceBuf,
        s2: DeviceBuf,
        a2: DeviceBuf,
    },
    Mxfp8 {
        s13: DeviceBuf,
        s2: DeviceBuf,
    },
}

struct LayerGpu {
    global: bool,
    attn_query: Option<DeviceBuf>,
    mlp_query: DeviceBuf,
    norms_packed: DeviceBuf,
    /// On SWA layers: `[wq(3072); wr(384); gate(3072); sink(24); wk(1024); wv(1024)]` = `8,600` rows!
    wq_all_q8: PlanarQ8Gpu,
    wk_glob_q8: Option<PlanarQ8Gpu>,
    wv_glob_q8: Option<PlanarQ8Gpu>,
    wo_q8: PlanarQ8Gpu,
    q_clip: DeviceBuf,
    rel_clip: DeviceBuf,
    rel_proj: DeviceBuf,
    k_conv4: DeviceBuf,
    v_conv4: DeviceBuf,
    attn_conv4: DeviceBuf,
    mlp_conv4: DeviceBuf,
    mlp_norm: DeviceBuf,
    router_w: DeviceBuf,
    router_bias: DeviceBuf,
    ctrl3: Option<CtrlGpu>,
    ctrl_mlp: Option<CtrlGpu>,
    msg_query: Option<DeviceBuf>,
    expert_scales: ExpertScales,
}

struct SlotStateGpu {
    kv_cap: usize,
    is_global: bool,
    k_cache: DeviceBuf,
    v_cache: DeviceBuf,
    hist_k: CUdeviceptr,
    hist_v: CUdeviceptr,
    hist_attn: CUdeviceptr,
    hist_mlp: CUdeviceptr,
}

struct Q8GroupKernel {
    function: CUfunction,
    columns_per_block: usize,
    threads: u32,
}

struct KernelFns {
    _module: CUmodule,
    embed_rmsnorm: CUfunction,
    rmsnorm_d: CUfunction,
    rmsnorm_quant_q8_64: CUfunction,
    depth_mix: CUfunction,
    ctrl_down: CUfunction,
    ctrl_apply: CUfunction,
    bf16_gemv: CUfunction,
    quant_q8_64: CUfunction,
    quant_q8_16: Q8GroupKernel,
    q8_dp4a_gemv: CUfunction,
    short_conv4: CUfunction,
    prep_qkv_and_cache: CUfunction,
    gqa_rel_sink_attn: CUfunction,
    gqa_attn_split: Option<CUfunction>,
    gqa_attn_combine: Option<CUfunction>,
    router_topk8: CUfunction,
    moe_nvfp4_up_situ: CUfunction,
    moe_nvfp4_down: CUfunction,
    moe_nvfp4_dp4a_up_situ: CUfunction,
    moe_nvfp4_dp4a_down: CUfunction,
    moe_mxfp8_up_situ: CUfunction,
    moe_mxfp8_down: CUfunction,
    copy_d: CUfunction,
    argmax_and_inc_pos: CUfunction,
    inc_pos: CUfunction,
    // Batched multi-token kernels (up to the selected prefill capacity)
    embed_rmsnorm_b: CUfunction,
    rmsnorm_d_b: CUfunction,
    rmsnorm_quant_q8_64_b: CUfunction,
    quant_q8_64_b: CUfunction,
    depth_mix_b: CUfunction,
    ctrl_down_b: CUfunction,
    ctrl_down_r8_b: Option<CUfunction>,
    ctrl_apply_b: CUfunction,
    ctrl_apply_t8_b: Option<CUfunction>,
    q8_dp4a_gemm_b: CUfunction,
    short_conv4_b: CUfunction,
    prep_qkv_and_cache_b: CUfunction,
    gqa_rel_sink_attn_b: CUfunction,
    bf16_gemm_router_b: CUfunction,
    router_topk8_b: CUfunction,
    build_expert_groups_b: CUfunction,
    build_expert_groups_wide_b: Option<CUfunction>,
    moe_nvfp4_dp4a_grouped_up_situ: CUfunction,
    moe_nvfp4_dp4a_grouped_down: CUfunction,
    moe_mxfp8_grouped_up_situ: CUfunction,
    moe_mxfp8_grouped_down: CUfunction,
    moe_combine_top8_b: CUfunction,
    inc_pos_by: CUfunction,
    argmax_b: CUfunction,
    conv_rewind: CUfunction,
    q8_dp4a_gemm_ktiled: CUfunction,
    mtp_tapmix: CUfunction,
    mtp_copy_anchor: CUfunction,
    mtp_ctx_store: CUfunction,
    mtp_init_draft: CUfunction,
    mtp_qk_norm_rope: CUfunction,
    mtp_attn: CUfunction,
    mtp_add_inplace: CUfunction,
    mtp_silu_mul: CUfunction,
    mtp_concat_frnn: CUfunction,
    mtp_add_bias_rows: CUfunction,
    mtp_markov_raw: CUfunction,
    mtp_markov_step_q8: CUfunction,
    mtp_fold_mv_q8: CUfunction,
    mtp_fold_fin: CUfunction,
    // v0.6.0 0-ULP fused glue & coalesced controller kernels (`Issue #11`)
    depth_mix_rmsnorm_quant_q8_64_b: Option<CUfunction>,
    depth_mix_rmsnorm_d_quant_q8_16_b: Option<CUfunction>,
    rmsnorm_d_quant_q8_16_b: Option<CUfunction>,
    ctrl_apply_coalesced_b: Option<CUfunction>,
    gqa_rel_sink_attn_q8_b: Option<CUfunction>,
    gqa_attn_combine_q8: Option<CUfunction>,
    short_conv4_save: Option<CUfunction>,
    short_conv4_save_b: Option<CUfunction>,
    loop_pass_setup_b: Option<CUfunction>,
    q8_dp4a_qkv3_gemv: Option<CUfunction>,
    short_conv4_kv2: Option<CUfunction>,
}

unsafe impl Send for KernelFns {}
unsafe impl Sync for KernelFns {}

// One compute stream uses this memory in order: attention, MLP hidden +
// router logits, then expert down-projection output. Each phase's simultaneous
// inputs/outputs have disjoint ranges.
struct LayerWorkspace {
    pair_output: DeviceBuf,
    mlp_hidden: CUdeviceptr,
    router_logits: CUdeviceptr,
    qrgs: CUdeviceptr,
    k_raw: CUdeviceptr,
    v_raw: CUdeviceptr,
    k_conv: CUdeviceptr,
    v_conv: CUdeviceptr,
    attn_out: CUdeviceptr,
    wo_out: CUdeviceptr,
}

impl LayerWorkspace {
    fn new(pair_output: DeviceBuf, capacity: usize) -> Self {
        const ATTENTION_FLOATS: usize = SWA_ALL_ROWS + 2 * 1056 + 2 * N_KV * HD + N_HEAD * HD + D;
        const { assert!(ATTENTION_FLOATS <= TOP_K * D && D + N_EXP <= TOP_K * D) };
        let mut next = pair_output.ptr;
        let mut slice = |cols: usize| {
            let ptr = next;
            next += (capacity * cols * size_of::<f32>()) as u64;
            debug_assert!(next - pair_output.ptr <= pair_output.byte_len as u64);
            ptr
        };
        Self {
            qrgs: slice(SWA_ALL_ROWS),
            k_raw: slice(1056),
            v_raw: slice(1056),
            k_conv: slice(N_KV * HD),
            v_conv: slice(N_KV * HD),
            attn_out: slice(N_HEAD * HD),
            wo_out: slice(D),
            mlp_hidden: pair_output.ptr,
            router_logits: pair_output.ptr + (capacity * D * size_of::<f32>()) as u64,
            pair_output,
        }
    }
}

// All these buffers scale linearly with the prefill width. A multiple of 64
// also accommodates every page of the expert occurrence tables.
struct PrefillBuffers {
    capacity: usize,
    bytes_per_row: usize,
    d_taps: Option<DeviceBuf>,
    d_attn_part_ml: DeviceBuf,
    d_attn_part_o: DeviceBuf,
    d_b_cur: DeviceBuf,
    d_b_prelude_x: DeviceBuf,
    d_b_blocks: DeviceBuf,
    d_b_prelude_block0: DeviceBuf,
    d_b_bank: DeviceBuf,
    d_b_msgs: DeviceBuf,
    d_b_ctrl_g: DeviceBuf,
    d_b_sel_ids: DeviceBuf,
    d_b_sel_wts: DeviceBuf,
    d_b_exp_toks: DeviceBuf,
    d_b_exp_pairs: DeviceBuf,
    // Controller inputs, expert up-projection output, then combined MoE output.
    // The down projection has finished reading its input before combine writes.
    d_b_activations: DeviceBuf,
    layer_workspace: LayerWorkspace,
    // Attention and the output head use groups of 64. Experts use groups of 16.
    // Each quantization replaces both planes before the corresponding GEMM.
    d_b_act_codes: DeviceBuf,
    d_b_act_scales: DeviceBuf,
}

impl PrefillBuffers {
    fn new(
        ctx: &Arc<CudaContextHandle>,
        capacity: usize,
        splits: usize,
        has_drafter: bool,
    ) -> Result<Self, String> {
        let mut bytes = 0;
        let mut allocate = |size| {
            bytes += size;
            DeviceBuf::alloc_zeroed(ctx, size)
        };
        Ok(Self {
            capacity,
            d_taps: if has_drafter {
                Some(allocate(capacity * 4 * D * 4)?)
            } else {
                None
            },
            d_attn_part_ml: allocate((capacity * N_HEAD * splits * 2 * 4).max(4))?,
            d_attn_part_o: allocate((capacity * N_HEAD * splits * HD * 4).max(4))?,
            d_b_cur: allocate(capacity * D * 4)?,
            d_b_prelude_x: allocate(capacity * D * 4)?,
            d_b_blocks: allocate(6 * capacity * D * 4)?,
            d_b_prelude_block0: allocate(capacity * D * 4)?,
            d_b_bank: allocate(2 * capacity * D * 4)?,
            d_b_msgs: allocate((48 * MAX_SPEC_TOKENS).max(4 * capacity) * D * 4)?,
            d_b_ctrl_g: allocate(3 * capacity * RANK * 4)?,
            d_b_sel_ids: allocate(capacity * TOP_K * 4)?,
            d_b_sel_wts: allocate(capacity * TOP_K * 4)?,
            d_b_exp_toks: allocate(N_EXP * capacity * 4)?,
            d_b_exp_pairs: allocate(N_EXP * capacity * 4)?,
            d_b_activations: allocate(capacity * (TOP_K * FF).max(3 * D) * 4)?,
            layer_workspace: LayerWorkspace::new(allocate(capacity * TOP_K * D * 4)?, capacity),
            d_b_act_codes: allocate(capacity * (TOP_K * FF).max(3 * N_HEAD * HD))?,
            d_b_act_scales: allocate(capacity * (TOP_K * (FF / 16)).max(3 * N_HEAD * HD / 64) * 4)?,
            bytes_per_row: (bytes - (48 * MAX_SPEC_TOKENS).saturating_sub(4 * capacity) * D * 4)
                / capacity,
        })
    }
}

pub struct Inkling2GpuEngine {
    pub ctx: Arc<CudaContextHandle>,
    fns: KernelFns,
    embed: TierBuf,
    embed_norm: DeviceBuf,
    final_norm: DeviceBuf,
    unembed_q8: PlanarQ8Gpu,
    output_query: DeviceBuf,
    d_all_msg_queries: DeviceBuf,
    layers: Vec<LayerGpu>,
    expert_weights: Vec<[TierBuf; 2]>,
    slots: Vec<SlotStateGpu>,
    conv_history: DeviceBuf,
    pub resident_expert_layers: usize,
    pub spilled_expert_layers: usize,
    pub max_ctx: usize,
    use_moe_dp4a: bool,
    // Persistent device scratch buffers
    d_token_in: DeviceBuf,
    d_prefill_tokens: DeviceBuf,
    d_token_out: DeviceBuf,
    d_pos: DeviceBuf,
    d_cur: DeviceBuf,
    d_prelude_x: DeviceBuf,
    d_blocks: DeviceBuf,
    d_prelude_block0: DeviceBuf,
    d_bank: DeviceBuf,
    d_msgs: DeviceBuf,
    d_inputs: DeviceBuf,
    d_ctrl_g: DeviceBuf,
    /// `[8600]` floats: `[0..6552)` = `q,r,gate,sink`, `[6552..7576)` = `k_swa`, `[7576..8600)` = `v_swa`
    d_qrgs: DeviceBuf,
    d_k_raw: DeviceBuf,
    d_v_raw: DeviceBuf,
    d_k_conv: DeviceBuf,
    d_v_conv: DeviceBuf,
    d_attn_out: DeviceBuf,
    /// Split-KV GQA attention (`LEANSONIC_ATTN_SPLIT=G`, `0` = legacy 24-CTA kernel).
    attn_split_g: usize,
    d_wo_out: DeviceBuf,
    d_mlp_h: DeviceBuf,
    d_router_logits: DeviceBuf,
    d_sel_ids: DeviceBuf,
    d_sel_wts: DeviceBuf,
    d_moe_act: DeviceBuf,
    d_moe_y: DeviceBuf,
    d_logits: DeviceBuf,
    // Quantized activation scratch buffers (`Q8_64` up to 3 vectors & `Q8_16` up to 8 vectors)
    d_act_q8_64: DeviceBuf, // 3 * 3072 B
    d_act_sc_64: DeviceBuf, // 3 * 48 * 4 B
    d_act_q8_16: DeviceBuf, // max 8 * 1152 B
    d_act_sc_16: DeviceBuf, // max 8 * 72 * 4 B
    // Multi-token batch scratch buffers (up to the selected prefill capacity) + Speculative verify buffers (`B <= 8`)
    prefill: PrefillBuffers,
    d_b_exp_counts: DeviceBuf,
    d_spec_logits: DeviceBuf,
    d_spec_tokens_out: DeviceBuf,
    d_conv_seq: DeviceBuf,
    d_conv_hist_ptrs: DeviceBuf,
    pub drafter: Option<DsparkDrafterGpu>,
    pub spec_k: usize,
    pub spec_drafted_total: usize,
    pub spec_accepted_total: usize,
    pub spec_verify_steps: usize,
    host_pos: usize,
    decode_graph: Option<CapturedGraphExec>,
    prefill_graph: Option<CapturedGraphExec>,
    pub enable_cuda_graph: bool,
}

unsafe impl Send for Inkling2GpuEngine {}

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

/// Multi-threaded load-time repack of row-major `BF16 [rows, cols]` (`cols % 64 == 0`)
/// into Planar `Q8_64` (`codes [rows, cols] i8` + `scales [rows, cols / 64] f32`).
pub fn repack_bf16_to_planar_q8_64(
    ctx: &Arc<CudaContextHandle>,
    bf16_bytes: &[u8],
    rows: usize,
    cols: usize,
) -> Result<PlanarQ8Gpu, String> {
    assert_eq!(cols % 64, 0);
    assert_eq!(bf16_bytes.len(), rows * cols * 2);
    let scale_stride = (cols / 64) * 4;
    let result = PlanarQ8Gpu {
        codes: DeviceBuf::alloc_zeroed(ctx, rows * cols)?,
        scales64: DeviceBuf::alloc_zeroed(ctx, rows * scale_stride)?,
        cols: cols as i32,
    };
    if rows == 0 || cols == 0 {
        return Ok(result);
    }

    // Keep the host staging area bounded while packing the vocabulary projection.
    const STAGING_BYTES: usize = 32 * 1024 * 1024;
    let tile_rows = (STAGING_BYTES / (cols + scale_stride)).clamp(1, rows);
    let mut codes = vec![0u8; tile_rows * cols];
    let mut scales = vec![0u8; tile_rows * scale_stride];
    for first in (0..rows).step_by(tile_rows) {
        let count = tile_rows.min(rows - first);
        let tile_codes = &mut codes[..count * cols];
        let tile_scales = &mut scales[..count * scale_stride];
        pack_q8_rows(
            &bf16_bytes[first * cols * 2..(first + count) * cols * 2],
            cols,
            tile_codes,
            tile_scales,
        );
        result.codes.copy_from_host_at(first * cols, tile_codes)?;
        result
            .scales64
            .copy_from_host_at(first * scale_stride, tile_scales)?;
    }
    Ok(result)
}

fn pack_q8_rows(bf16_bytes: &[u8], cols: usize, codes: &mut [u8], scales_bytes: &mut [u8]) {
    let rows = codes.len() / cols;
    let cols64 = cols / 64;
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get().min(16))
        .unwrap_or(8);
    let rows_per_thread = rows.div_ceil(n_threads).max(1);

    std::thread::scope(|s| {
        let mut code_rest = codes;
        let mut scale_rest = scales_bytes;
        let mut row_start = 0usize;
        while row_start < rows {
            let chunk_rows = rows_per_thread.min(rows - row_start);
            let (c_chunk, c_tail) = code_rest.split_at_mut(chunk_rows * cols);
            let (s_chunk, s_tail) = scale_rest.split_at_mut(chunk_rows * cols64 * 4);
            let src_chunk = &bf16_bytes[row_start * cols * 2..(row_start + chunk_rows) * cols * 2];
            code_rest = c_tail;
            scale_rest = s_tail;
            row_start += chunk_rows;

            s.spawn(move || {
                for r in 0..chunk_rows {
                    let w_row = &src_chunk[r * cols * 2..(r + 1) * cols * 2];
                    let dst_c = &mut c_chunk[r * cols..(r + 1) * cols];
                    let dst_s = &mut s_chunk[r * cols64 * 4..(r + 1) * cols64 * 4];
                    for b in 0..cols64 {
                        let sub = &w_row[b * 128..(b + 1) * 128];
                        let mut amax = 0.0f32;
                        let mut fvals = [0.0f32; 64];
                        for (i, &bytes) in sub.as_chunks::<2>().0.iter().enumerate() {
                            let fv = bf16_to_f32(u16::from_le_bytes(bytes));
                            fvals[i] = fv;
                            let av = fv.abs();
                            if av > amax {
                                amax = av;
                            }
                        }
                        let sc = amax * (1.0 / 127.0);
                        let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
                        dst_s[b * 4..(b + 1) * 4].copy_from_slice(&sc.to_le_bytes());
                        for (i, &fv) in fvals.iter().enumerate() {
                            let q = (fv * inv).round().clamp(-127.0, 127.0) as i8;
                            dst_c[b * 64 + i] = q as u8;
                        }
                    }
                }
            });
        }
    });
}

// Prioritize the layers visited three times. At the budget boundary, each expert
// matrix can reside independently because kernels receive separate up/down pointers.
fn plan_expert_placement(expert_views: &[[&[u8]; 2]], mut budget: usize) -> [[bool; 2]; N_LAYER] {
    let mut resident = [[false; 2]; N_LAYER];
    for il in (6..18).chain(0..6).chain(18..N_LAYER) {
        for (on_device, bytes) in resident[il].iter_mut().zip(expert_views[il]) {
            if bytes.len() <= budget {
                *on_device = true;
                budget -= bytes.len();
            }
        }
    }
    resident
}

impl Inkling2GpuEngine {
    pub fn load_from_dir(
        model_dir: impl AsRef<Path>,
        max_ctx: usize,
        enable_cuda_graph: bool,
    ) -> Result<Self, String> {
        let default_spec_k = std::env::var("LEANSONIC_SPEC_K")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        Self::load_from_dir_with_spec(model_dir, max_ctx, enable_cuda_graph, default_spec_k)
    }

    pub fn load_from_dir_with_spec(
        model_dir: impl AsRef<Path>,
        max_ctx: usize,
        enable_cuda_graph: bool,
        spec_k: usize,
    ) -> Result<Self, String> {
        Self::load_model_path(model_dir.as_ref(), max_ctx, enable_cuda_graph, spec_k)
    }

    // Share the loader across callers passing different path types.
    fn load_model_path(
        model_dir: &Path,
        max_ctx: usize,
        enable_cuda_graph: bool,
        spec_k: usize,
    ) -> Result<Self, String> {
        let spec_k = spec_k.min(MAX_SPEC_TOKENS - 1);
        let t_load = Instant::now();
        let ctx = CudaContextHandle::init(0)?;
        ctx.bind_current()?;
        // `LEANSONIC_ATTN_SPLIT=G` (`1..=64`) enables Split-KV + GQA-grouped attention across
        // BOTH Solo Decode (`B=1`) and `DSpark` Verify / Prefill (`B>=1`) for `0-ULP` spec parity!
        let attn_split_g = std::env::var("LEANSONIC_ATTN_SPLIT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
            .min(64);
        let use_moe_dp4a = std::env::var("LEANSONIC_MOE_DP4A")
            .map(|v| v != "0")
            .unwrap_or(true);

        let module_image = if let Ok(cubin_path) = std::env::var("LEANSONIC_CUBIN_PATH") {
            std::fs::read(&cubin_path).map_err(|e| format!("read {cubin_path}: {e}"))?
        } else {
            compile_cuda_to_ptx_nvrtc(CUDA_KERNEL_SRC, "compute_120")?
        };

        let fns = Self::load_module_functions(&ctx, &module_image)?;
        if attn_split_g > 0 && (fns.gqa_attn_split.is_none() || fns.gqa_attn_combine.is_none()) {
            return Err(
                "LEANSONIC_ATTN_SPLIT>0 requested, but loaded cubin lacks ink2_gqa_attn_split/combine"
                    .into(),
            );
        }
        let st_path = model_dir.join("model-00001-of-00001.safetensors");
        let st = MappedSafeTensors::open(&st_path)
            .map_err(|e| format!("open {}: {e}", st_path.display()))?;

        let embed_raw = st
            .require("model.llm.embed.weight", "BF16", &[PADDED_VOCAB, D])
            .data;
        let embed = TierBuf::HostPinned(PinnedHostBuf::from_bytes(&ctx, embed_raw)?);
        st.discard_resident_pages();
        let embed_norm = DeviceBuf::from_host_bytes(
            &ctx,
            st.require("model.llm.embed_norm.weight", "BF16", &[D]).data,
        )?;
        let final_norm = DeviceBuf::from_host_bytes(
            &ctx,
            st.require("model.llm.norm.weight", "BF16", &[D]).data,
        )?;
        let unembed_all = st
            .require("model.llm.unembed.weight", "BF16", &[PADDED_VOCAB, D])
            .data;
        // Repack `unembed` (`922 MB` BF16 -> `490 MB` Planar `Q8_64`, freeing `432 MB` VRAM!)
        let unembed_q8 =
            repack_bf16_to_planar_q8_64(&ctx, &unembed_all[..VOCAB * D * 2], VOCAB, D)?;
        let output_query = DeviceBuf::from_host_bytes(
            &ctx,
            st.require("model.llm.attn_res.output_query", "F32", &[D])
                .data,
        )?;

        // Three prior FP32 samples per channel for each of the four convolutions.
        let kv_history_bytes = 3 * N_KV * HD * size_of::<f32>();
        let residual_history_bytes = 3 * D * size_of::<f32>();
        let slot_history_bytes = 2 * kv_history_bytes + 2 * residual_history_bytes;
        let conv_history = DeviceBuf::alloc_zeroed(&ctx, N_SLOT * slot_history_bytes)?;
        let mut slots = Vec::with_capacity(N_SLOT);
        for s_idx in 0..N_SLOT {
            let hist_k = conv_history.ptr + (s_idx * slot_history_bytes) as u64;
            let hist_v = hist_k + kv_history_bytes as u64;
            let hist_attn = hist_v + kv_history_bytes as u64;
            let hist_mlp = hist_attn + residual_history_bytes as u64;
            let il = if s_idx < N_LAYER {
                s_idx
            } else {
                6 + ((s_idx - N_LAYER) % 12)
            };
            let is_global = GLOBAL_LAYERS.contains(&il);
            // A short context ends before the ring wraps. For longer contexts the
            // extra batch protects earlier rows while a whole chunk writes its KV.
            let kv_cap = if is_global {
                max_ctx
            } else {
                max_ctx.min(EXTENT + EXPERT_PAGE)
            };
            slots.push(SlotStateGpu {
                kv_cap,
                is_global,
                k_cache: DeviceBuf::alloc_zeroed(&ctx, kv_cap * N_KV * HD * 2)?,
                v_cache: DeviceBuf::alloc_zeroed(&ctx, kv_cap * N_KV * HD * 2)?,
                hist_k,
                hist_v,
                hist_attn,
                hist_mlp,
            });
        }

        let mut hist_ptr_bytes = Vec::with_capacity(N_SLOT * 4 * 8);
        for s in &slots {
            for p in [s.hist_k, s.hist_v, s.hist_attn, s.hist_mlp] {
                hist_ptr_bytes.extend_from_slice(&p.to_le_bytes());
            }
        }
        let d_conv_hist_ptrs = DeviceBuf::from_host_bytes(&ctx, &hist_ptr_bytes)?;
        let d_conv_seq = if spec_k > 0 {
            DeviceBuf::alloc_zeroed(&ctx, N_SLOT * CONV_SEQ_SLOT_FLOATS * 4)?
        } else {
            DeviceBuf::alloc_zeroed(&ctx, 4)?
        };

        // Load the optional drafter before budgeting memory for tiered expert weights.
        let mtp_path = model_dir.join("mtp.safetensors");
        let drafter = if spec_k > 0 && mtp_path.exists() {
            let mtp_fns = MtpKernelFns {
                rmsnorm_d_b: fns.rmsnorm_d_b,
                rmsnorm_quant_q8_64_b: fns.rmsnorm_quant_q8_64_b,
                quant_q8_64_b: fns.quant_q8_64_b,
                q8_dp4a_gemm_b: fns.q8_dp4a_gemm_b,
                q8_dp4a_gemv: fns.q8_dp4a_gemv,
                q8_dp4a_gemm_ktiled: fns.q8_dp4a_gemm_ktiled,
                tapmix: fns.mtp_tapmix,
                copy_anchor: fns.mtp_copy_anchor,
                ctx_store: fns.mtp_ctx_store,
                init_draft: fns.mtp_init_draft,
                qk_norm_rope: fns.mtp_qk_norm_rope,
                attn: fns.mtp_attn,
                add_inplace: fns.mtp_add_inplace,
                silu_mul: fns.mtp_silu_mul,
                concat_frnn: fns.mtp_concat_frnn,
                add_bias_rows: fns.mtp_add_bias_rows,
                markov_raw: fns.mtp_markov_raw,
                markov_step_q8: fns.mtp_markov_step_q8,
                fold_mv_q8: fns.mtp_fold_mv_q8,
                fold_fin: fns.mtp_fold_fin,
            };
            Some(DsparkDrafterGpu::load_from_file(
                &ctx,
                mtp_fns,
                &mtp_path,
                &unembed_q8,
                max_ctx,
            )?)
        } else {
            None
        };

        let cross_q = st
            .require("model.llm.middle_cycle_cross_pass_queries", "F32", &[24, D])
            .as_f32_slice();
        let cross_kvq = st
            .require(
                "model.llm.middle_cycle_cross_pass_kv_queries",
                "F32",
                &[24, D],
            )
            .as_f32_slice();

        let mut all_mq_bytes = Vec::with_capacity(12 * 4 * D * 4);
        for j in 0..12 {
            let rows = [
                &cross_q[2 * j * D..(2 * j + 1) * D],
                &cross_kvq[2 * j * D..(2 * j + 1) * D],
                &cross_kvq[(2 * j + 1) * D..(2 * j + 2) * D],
                &cross_q[(2 * j + 1) * D..(2 * j + 2) * D],
            ];
            for r in rows {
                for &f in r {
                    all_mq_bytes.extend_from_slice(&f.to_le_bytes());
                }
            }
        }
        let d_all_msg_queries = DeviceBuf::from_host_bytes(&ctx, &all_mq_bytes)?;

        let mut layers = Vec::with_capacity(N_LAYER);
        let mut expert_views = Vec::with_capacity(N_LAYER);
        for il in 0..N_LAYER {
            let global = GLOBAL_LAYERS.contains(&il);
            let pfx = format!("model.llm.layers.{il}.");
            let in_core = (6..18).contains(&il);

            let attn_query = if il > 0 {
                let qname = format!("model.llm.attn_res.attn_queries.{}", il - 1);
                let shape: &[usize] = if global { &[3, D] } else { &[D] };
                Some(DeviceBuf::from_host_bytes(
                    &ctx,
                    st.require(&qname, "F32", shape).data,
                )?)
            } else {
                None
            };
            let mlp_query = DeviceBuf::from_host_bytes(
                &ctx,
                st.require(&format!("model.llm.attn_res.mlp_queries.{il}"), "F32", &[D])
                    .data,
            )?;

            let norms_packed = if global {
                let nq = st
                    .require(&format!("{pfx}q_input_norm.weight"), "BF16", &[D])
                    .data;
                let nk = st
                    .require(&format!("{pfx}k_input_norm.weight"), "BF16", &[D])
                    .data;
                let nv = st
                    .require(&format!("{pfx}v_input_norm.weight"), "BF16", &[D])
                    .data;
                DeviceBuf::from_host_bytes(&ctx, &concat_slices(&[nq, nk, nv]))?
            } else {
                let raw = st
                    .require(&format!("{pfx}attn_norm.weight"), "BF16", &[D])
                    .data;
                DeviceBuf::from_host_bytes(&ctx, &concat_slices(&[raw, raw, raw]))?
            };

            let wq = st
                .require(
                    &format!("{pfx}attn.wq_du.weight"),
                    "BF16",
                    &[N_HEAD * HD, D],
                )
                .data;
            let wr = st
                .require(
                    &format!("{pfx}attn.wr_du.weight"),
                    "BF16",
                    &[N_REL * N_HEAD, D],
                )
                .data;
            let gate_attn = st
                .require(
                    &format!("{pfx}attn.gate_attn.weight"),
                    "BF16",
                    &[N_HEAD * HD, D],
                )
                .data;
            let sink = st
                .require(
                    &format!("{pfx}attn.attn_sink_proj.weight"),
                    "BF16",
                    &[N_HEAD, D],
                )
                .data;
            let wk = st
                .require(&format!("{pfx}attn.wk_dv.weight"), "BF16", &[N_KV * HD, D])
                .data;
            let wv = st
                .require(&format!("{pfx}attn.wv_dv.weight"), "BF16", &[N_KV * HD, D])
                .data;

            let (wq_all_q8, wk_glob_q8, wv_glob_q8) = if global {
                let kd = st
                    .require(
                        &format!("{pfx}attn.k_sconv.weight_proj"),
                        "BF16",
                        &[4 * N_KV, D],
                    )
                    .data;
                let vd = st
                    .require(
                        &format!("{pfx}attn.v_sconv.weight_proj"),
                        "BF16",
                        &[4 * N_KV, D],
                    )
                    .data;
                (
                    repack_bf16_to_planar_q8_64(
                        &ctx,
                        &concat_slices(&[wq, wr, gate_attn, sink]),
                        QRGS_ROWS,
                        D,
                    )?,
                    Some(repack_bf16_to_planar_q8_64(
                        &ctx,
                        &concat_slices(&[wk, kd]),
                        1056,
                        D,
                    )?),
                    Some(repack_bf16_to_planar_q8_64(
                        &ctx,
                        &concat_slices(&[wv, vd]),
                        1056,
                        D,
                    )?),
                )
            } else {
                // Fuse ALL 6 `[wq; wr; gate; sink; wk; wv]` (`8,600` rows!) into one Planar Q8 tensor!
                (
                    repack_bf16_to_planar_q8_64(
                        &ctx,
                        &concat_slices(&[wq, wr, gate_attn, sink, wk, wv]),
                        SWA_ALL_ROWS,
                        D,
                    )?,
                    None,
                    None,
                )
            };

            let wo_q8 = repack_bf16_to_planar_q8_64(
                &ctx,
                st.require(
                    &format!("{pfx}attn.wo_ud.weight"),
                    "BF16",
                    &[D, N_HEAD * HD],
                )
                .data,
                D,
                3072,
            )?;

            let q_clip = DeviceBuf::from_host_bytes(
                &ctx,
                st.require(&format!("{pfx}attn.q_l2_clip.scale"), "BF16", &[N_HEAD, HD])
                    .data,
            )?;
            let rel_clip = DeviceBuf::from_host_bytes(
                &ctx,
                st.require(
                    &format!("{pfx}attn.rel_l2_clip.scale"),
                    "BF16",
                    &[N_HEAD, N_REL],
                )
                .data,
            )?;
            let rel_proj = DeviceBuf::from_host_bytes(
                &ctx,
                st.require(
                    &format!("{pfx}attn.rel_logits_proj.proj"),
                    "BF16",
                    &[N_REL, EXTENT],
                )
                .data,
            )?;

            let conv_to_f32_bytes = |name: &str, ch: usize| -> Vec<u8> {
                let u16s = st.require(name, "BF16", &[ch, 1, 4]).as_u16_slice();
                let mut out = Vec::with_capacity(ch * 16);
                for &u in u16s {
                    out.extend_from_slice(&bf16_to_f32(u).to_le_bytes());
                }
                out
            };
            let static_conv_to_f32_bytes = |name: &str, ch: usize| -> Vec<u8> {
                let u16s = st.require(name, "BF16", &[4, ch]).as_u16_slice();
                let mut out = Vec::with_capacity(ch * 16);
                for c in 0..ch {
                    for &lag in &[3usize, 2, 1, 0] {
                        out.extend_from_slice(&bf16_to_f32(u16s[lag * ch + c]).to_le_bytes());
                    }
                }
                out
            };

            let (k_conv4, v_conv4) = if global {
                (
                    DeviceBuf::from_host_bytes(
                        &ctx,
                        &static_conv_to_f32_bytes(
                            &format!("{pfx}attn.k_sconv.static_w"),
                            N_KV * HD,
                        ),
                    )?,
                    DeviceBuf::from_host_bytes(
                        &ctx,
                        &static_conv_to_f32_bytes(
                            &format!("{pfx}attn.v_sconv.static_w"),
                            N_KV * HD,
                        ),
                    )?,
                )
            } else {
                (
                    DeviceBuf::from_host_bytes(
                        &ctx,
                        &conv_to_f32_bytes(&format!("{pfx}attn.k_sconv.weight"), N_KV * HD),
                    )?,
                    DeviceBuf::from_host_bytes(
                        &ctx,
                        &conv_to_f32_bytes(&format!("{pfx}attn.v_sconv.weight"), N_KV * HD),
                    )?,
                )
            };
            let attn_conv4 = DeviceBuf::from_host_bytes(
                &ctx,
                &conv_to_f32_bytes(&format!("{pfx}attn_sconv.weight"), D),
            )?;
            let mlp_conv4 = DeviceBuf::from_host_bytes(
                &ctx,
                &conv_to_f32_bytes(&format!("{pfx}mlp_sconv.weight"), D),
            )?;

            let mlp_norm = DeviceBuf::from_host_bytes(
                &ctx,
                st.require(&format!("{pfx}mlp_norm.weight"), "BF16", &[D])
                    .data,
            )?;
            // Keep tiny router (`128 x 2304 = 576 KB`) in exact BF16 to prevent top-8 router tie flips!
            let router_w = DeviceBuf::from_host_bytes(
                &ctx,
                st.require(&format!("{pfx}mlp.gate.weight"), "BF16", &[N_EXP, D])
                    .data,
            )?;
            let router_bias = DeviceBuf::from_host_bytes(
                &ctx,
                st.require(&format!("{pfx}mlp.gate.bias"), "F32", &[N_EXP])
                    .data,
            )?;

            let ctrl_slices = |cname: &str| -> (&[u8], &[u8], &[u8]) {
                (
                    st.require(
                        &format!("{pfx}controllers.{cname}.down_weight"),
                        "BF16",
                        &[RANK, 2 * D],
                    )
                    .data,
                    st.require(
                        &format!("{pfx}controllers.{cname}.gain_weight"),
                        "BF16",
                        &[D, RANK],
                    )
                    .data,
                    st.require(
                        &format!("{pfx}controllers.{cname}.forget_weight"),
                        "BF16",
                        &[D, RANK],
                    )
                    .data,
                )
            };

            let (ctrl3, ctrl_mlp) = if in_core {
                let (da, ga, fa) = ctrl_slices("attn");
                let (dk, gk, fk) = ctrl_slices("k");
                let (dv, gv, fv) = ctrl_slices("v");
                let (dm, gm, fm) = ctrl_slices("mlp");
                (
                    Some(CtrlGpu {
                        down: DeviceBuf::from_host_bytes(&ctx, &concat_slices(&[da, dk, dv]))?,
                        gain: DeviceBuf::from_host_bytes(&ctx, &concat_slices(&[ga, gk, gv]))?,
                        forget: DeviceBuf::from_host_bytes(&ctx, &concat_slices(&[fa, fk, fv]))?,
                    }),
                    Some(CtrlGpu {
                        down: DeviceBuf::from_host_bytes(&ctx, dm)?,
                        gain: DeviceBuf::from_host_bytes(&ctx, gm)?,
                        forget: DeviceBuf::from_host_bytes(&ctx, fm)?,
                    }),
                )
            } else {
                (None, None)
            };

            let msg_query = if in_core {
                let j = il - 6;
                let rows = [
                    &cross_q[2 * j * D..(2 * j + 1) * D],
                    &cross_kvq[2 * j * D..(2 * j + 1) * D],
                    &cross_kvq[(2 * j + 1) * D..(2 * j + 2) * D],
                    &cross_q[(2 * j + 1) * D..(2 * j + 2) * D],
                ];
                let mut bytes = Vec::with_capacity(4 * D * 4);
                for r in rows {
                    for &f in r {
                        bytes.extend_from_slice(&f.to_le_bytes());
                    }
                }
                Some(DeviceBuf::from_host_bytes(&ctx, &bytes)?)
            } else {
                None
            };

            let expert_scales = if il < 20 {
                let w13_raw = st
                    .require(
                        &format!("{pfx}mlp.experts.w13_weight"),
                        "U8",
                        &[N_EXP, 2 * FF, D / 2],
                    )
                    .data;
                let w2_raw = st
                    .require(
                        &format!("{pfx}mlp.experts.w2_weight"),
                        "U8",
                        &[N_EXP, D, FF / 2],
                    )
                    .data;
                expert_views.push([w13_raw, w2_raw]);
                ExpertScales::Nvfp4 {
                    s13: DeviceBuf::from_host_bytes(
                        &ctx,
                        st.require(
                            &format!("{pfx}mlp.experts.w13_weight_scale"),
                            "F8_E4M3",
                            &[N_EXP, 2 * FF, D / 16],
                        )
                        .data,
                    )?,
                    a13: DeviceBuf::from_host_bytes(
                        &ctx,
                        st.require(
                            &format!("{pfx}mlp.experts.w13_weight_amax"),
                            "F32",
                            &[N_EXP],
                        )
                        .data,
                    )?,
                    s2: DeviceBuf::from_host_bytes(
                        &ctx,
                        st.require(
                            &format!("{pfx}mlp.experts.w2_weight_scale"),
                            "F8_E4M3",
                            &[N_EXP, D, FF / 16],
                        )
                        .data,
                    )?,
                    a2: DeviceBuf::from_host_bytes(
                        &ctx,
                        st.require(&format!("{pfx}mlp.experts.w2_weight_amax"), "F32", &[N_EXP])
                            .data,
                    )?,
                }
            } else {
                let w13_raw = st
                    .require(
                        &format!("{pfx}mlp.experts.w13_weight"),
                        "F8_E4M3",
                        &[N_EXP, 2 * FF, D],
                    )
                    .data;
                let w2_raw = st
                    .require(
                        &format!("{pfx}mlp.experts.w2_weight"),
                        "F8_E4M3",
                        &[N_EXP, D, FF],
                    )
                    .data;
                expert_views.push([w13_raw, w2_raw]);
                ExpertScales::Mxfp8 {
                    s13: DeviceBuf::from_host_bytes(
                        &ctx,
                        st.require(
                            &format!("{pfx}mlp.experts.w13_weight_scale"),
                            "U8",
                            &[N_EXP, 2 * FF, D / 32],
                        )
                        .data,
                    )?,
                    s2: DeviceBuf::from_host_bytes(
                        &ctx,
                        st.require(
                            &format!("{pfx}mlp.experts.w2_weight_scale"),
                            "U8",
                            &[N_EXP, D, FF / 32],
                        )
                        .data,
                    )?,
                }
            };

            layers.push(LayerGpu {
                global,
                attn_query,
                mlp_query,
                norms_packed,
                wq_all_q8,
                wk_glob_q8,
                wv_glob_q8,
                wo_q8,
                q_clip,
                rel_clip,
                rel_proj,
                k_conv4,
                v_conv4,
                attn_conv4,
                mlp_conv4,
                mlp_norm,
                router_w,
                router_bias,
                ctrl3,
                ctrl_mlp,
                msg_query,
                expert_scales,
            });
        }

        let mut eng = Self {
            d_token_in: DeviceBuf::alloc_zeroed(&ctx, 4)?,
            d_prefill_tokens: DeviceBuf::alloc_zeroed(&ctx, max_ctx * 4)?,
            d_token_out: DeviceBuf::alloc_zeroed(&ctx, 4)?,
            d_pos: DeviceBuf::alloc_zeroed(&ctx, 4)?,
            d_cur: DeviceBuf::alloc_zeroed(&ctx, D * 4)?,
            d_prelude_x: DeviceBuf::alloc_zeroed(&ctx, D * 4)?,
            d_blocks: DeviceBuf::alloc_zeroed(&ctx, 6 * D * 4)?,
            d_prelude_block0: DeviceBuf::alloc_zeroed(&ctx, D * 4)?,
            d_bank: DeviceBuf::alloc_zeroed(&ctx, 2 * D * 4)?,
            d_msgs: DeviceBuf::alloc_zeroed(&ctx, 48 * D * 4)?,
            d_inputs: DeviceBuf::alloc_zeroed(&ctx, 3 * D * 4)?,
            d_ctrl_g: DeviceBuf::alloc_zeroed(&ctx, 3 * RANK * 4)?,
            d_qrgs: DeviceBuf::alloc_zeroed(&ctx, SWA_ALL_ROWS * 4)?,
            d_k_raw: DeviceBuf::alloc_zeroed(&ctx, 1056 * 4)?,
            d_v_raw: DeviceBuf::alloc_zeroed(&ctx, 1056 * 4)?,
            d_k_conv: DeviceBuf::alloc_zeroed(&ctx, 1024 * 4)?,
            d_v_conv: DeviceBuf::alloc_zeroed(&ctx, 1024 * 4)?,
            d_attn_out: DeviceBuf::alloc_zeroed(&ctx, 3072 * 4)?,
            attn_split_g,
            d_wo_out: DeviceBuf::alloc_zeroed(&ctx, D * 4)?,
            d_mlp_h: DeviceBuf::alloc_zeroed(&ctx, D * 4)?,
            d_router_logits: DeviceBuf::alloc_zeroed(&ctx, N_EXP * 4)?,
            d_sel_ids: DeviceBuf::alloc_zeroed(&ctx, TOP_K * 4)?,
            d_sel_wts: DeviceBuf::alloc_zeroed(&ctx, TOP_K * 4)?,
            d_moe_act: DeviceBuf::alloc_zeroed(&ctx, TOP_K * FF * 4)?,
            d_moe_y: DeviceBuf::alloc_zeroed(&ctx, D * 4)?,
            d_logits: DeviceBuf::alloc_zeroed(&ctx, VOCAB * 4)?,
            d_act_q8_64: DeviceBuf::alloc_zeroed(&ctx, 3 * 3072)?,
            d_act_sc_64: DeviceBuf::alloc_zeroed(&ctx, 3 * 48 * 4)?,
            d_act_q8_16: DeviceBuf::alloc_zeroed(&ctx, TOP_K * FF)?,
            d_act_sc_16: DeviceBuf::alloc_zeroed(&ctx, TOP_K * (FF / 16) * 4)?,
            prefill: PrefillBuffers::new(&ctx, EXPERT_PAGE, attn_split_g, drafter.is_some())?,
            d_b_exp_counts: DeviceBuf::alloc_zeroed(&ctx, N_EXP * 4)?,
            d_spec_logits: DeviceBuf::alloc_zeroed(&ctx, MAX_SPEC_TOKENS * VOCAB * 4)?,
            d_spec_tokens_out: DeviceBuf::alloc_zeroed(&ctx, MAX_SPEC_TOKENS * 4)?,
            d_conv_seq,
            d_conv_hist_ptrs,
            drafter,
            spec_k,
            spec_drafted_total: 0,
            spec_accepted_total: 0,
            spec_verify_steps: 0,
            ctx,
            fns,
            conv_history,
            embed,
            embed_norm,
            final_norm,
            unembed_q8,
            output_query,
            d_all_msg_queries,
            layers,
            slots,
            expert_weights: Vec::with_capacity(N_LAYER),
            resident_expert_layers: 0,
            spilled_expert_layers: 0,
            max_ctx,
            use_moe_dp4a,
            host_pos: 0,
            decode_graph: None,
            prefill_graph: None,
            enable_cuda_graph,
        };

        // Scales, dense weights, KV, drafting state and scratch are already allocated.
        // The two graph executions consumed 8 MiB in the 16 GB Blackwell measurements.
        let (free, _) = eng.ctx.mem_info()?;
        let graph_reserve = if enable_cuda_graph {
            16 * 1024 * 1024
        } else {
            0
        };
        let budget = free.saturating_sub(graph_reserve);
        let resident = plan_expert_placement(&expert_views, budget);
        let expert_bytes: usize = expert_views
            .iter()
            .zip(&resident)
            .flat_map(|(views, selected)| views.iter().zip(selected))
            .filter(|(_, selected)| **selected)
            .map(|(view, _)| view.len())
            .sum();
        let limit = match std::env::var("LEANSONIC_PREFILL_CHUNK") {
            Ok(s) => s
                .parse::<usize>()
                .ok()
                .filter(|n| (EXPERT_PAGE..=MAX_BATCH_TOKENS).contains(n) && n.is_power_of_two())
                .ok_or("LEANSONIC_PREFILL_CHUNK must be 64, 128, 256, 512 or 1024")?,
            Err(std::env::VarError::NotPresent) => MAX_BATCH_TOKENS,
            Err(e) => return Err(e.to_string()),
        };
        if eng.fns.build_expert_groups_wide_b.is_some() {
            let local_slots = eng.slots.iter().filter(|s| !s.is_global).count();
            let base_ring = max_ctx.min(EXTENT + EXPERT_PAGE);
            let headroom = budget - expert_bytes;
            for width in [1024, 512, 256, 128] {
                if width > limit || width > max_ctx {
                    continue;
                }
                let ring_rows = max_ctx.min(EXTENT + width) - base_ring;
                let extra = (width - EXPERT_PAGE) * eng.prefill.bytes_per_row
                    + ring_rows * local_slots * 2 * N_KV * HD * 2;
                if extra > headroom {
                    continue;
                }
                match eng.resize_prefill(width) {
                    Ok(()) => {}
                    Err(e) if e.starts_with("CUDA error 2 in `cuMemAlloc_v2`") => continue,
                    Err(e) => return Err(e),
                }
                // Driver allocation granularity can exceed the requested bytes.
                // Preserve the compact profile's exact matrix placement.
                let (now_free, _) = eng.ctx.mem_info()?;
                if plan_expert_placement(&expert_views, now_free.saturating_sub(graph_reserve))
                    == resident
                {
                    break;
                }
                eng.resize_prefill(EXPERT_PAGE)?;
            }
        }
        let place = |on_device: bool, bytes: &[u8]| -> Result<TierBuf, String> {
            if on_device {
                if let Ok(buf) = DeviceBuf::from_host_bytes(&eng.ctx, bytes) {
                    return Ok(TierBuf::Vram(buf));
                }
            }
            Ok(TierBuf::HostPinned(PinnedHostBuf::from_bytes(
                &eng.ctx, bytes,
            )?))
        };
        // The dense tensors now live in their final buffers. Release their source pages
        // before copying the larger expert matrices into VRAM or pinned host memory.
        st.discard_resident_pages();
        let mut mixed_layers = 0;
        for ([up, down], [up_on_device, down_on_device]) in expert_views.into_iter().zip(resident) {
            let pair = [place(up_on_device, up)?, place(down_on_device, down)?];
            match &pair {
                [TierBuf::Vram(_), TierBuf::Vram(_)] => eng.resident_expert_layers += 1,
                [TierBuf::HostPinned(_), TierBuf::HostPinned(_)] => eng.spilled_expert_layers += 1,
                _ => {
                    mixed_layers += 1;
                    eng.spilled_expert_layers += 1;
                }
            }
            eng.expert_weights.push(pair);
            st.discard_resident_pages();
        }

        if enable_cuda_graph {
            let ctx_clone = Arc::clone(&eng.ctx);
            let d_graph = CapturedGraphExec::capture(&ctx_clone, |stream| {
                eng.enqueue_step_impl(stream, false)
            })?;
            let p_graph = CapturedGraphExec::capture(&ctx_clone, |stream| {
                eng.enqueue_step_impl(stream, true)
            })?;
            eng.decode_graph = Some(d_graph);
            eng.prefill_graph = Some(p_graph);
            eng.reset()?;
        }

        let (free_after, total_vram) = eng.ctx.mem_info()?;
        let drafter_tag = if eng.drafter.is_some() {
            format!("DSpark(K={spec_k})")
        } else if spec_k > 0 {
            format!("Suffix(K={spec_k})")
        } else {
            "off".to_string()
        };
        eprintln!(
            "[LeanSonic GPU] {} | VRAM used: {:.2} GiB / {:.2} GiB | Resident expert layers: {}/24 (Spilled: {}) | Mixed VRAM/host layers: {} | Spec: {} | Prefill chunk: {} | Loaded in {:.2}s",
            eng.ctx.device_name,
            (total_vram - free_after) as f64 / (1024.0 * 1024.0 * 1024.0),
            total_vram as f64 / (1024.0 * 1024.0 * 1024.0),
            eng.resident_expert_layers,
            eng.spilled_expert_layers,
            mixed_layers,
            drafter_tag,
            eng.prefill.capacity,
            t_load.elapsed().as_secs_f64()
        );
        Ok(eng)
    }

    fn resize_prefill(&mut self, capacity: usize) -> Result<(), String> {
        let buffers = PrefillBuffers::new(
            &self.ctx,
            capacity,
            self.attn_split_g,
            self.drafter.is_some(),
        )?;
        let mut replacements = Vec::new();
        let cap = self.max_ctx.min(EXTENT + capacity);
        for (index, slot) in self.slots.iter().enumerate() {
            // Global slots always retain the full context.
            if slot.is_global || slot.kv_cap == cap {
                continue;
            }
            let bytes = cap * N_KV * HD * 2;
            let k = DeviceBuf::alloc_zeroed(&self.ctx, bytes)?;
            let v = DeviceBuf::alloc_zeroed(&self.ctx, bytes)?;
            replacements.push((index, k, v));
        }
        // No token has run yet. Commit only after every allocation succeeds.
        self.prefill = buffers;
        for (i, k, v) in replacements {
            self.slots[i].kv_cap = cap;
            self.slots[i].k_cache = k;
            self.slots[i].v_cache = v;
        }
        Ok(())
    }

    fn load_module_functions(
        ctx: &Arc<CudaContextHandle>,
        image: &[u8],
    ) -> Result<KernelFns, String> {
        let mut module: CUmodule = core::ptr::null_mut();
        unsafe {
            check_cu(
                (ctx.api.cuModuleLoadData)(&mut module, image.as_ptr().cast()),
                "cuModuleLoadData",
            )?;
        }
        let get_fn = |name: &str| -> Result<CUfunction, String> {
            let c = CString::new(name).unwrap();
            let mut f: CUfunction = core::ptr::null_mut();
            unsafe {
                check_cu(
                    (ctx.api.cuModuleGetFunction)(&mut f, module, c.as_ptr()),
                    name,
                )?;
            }
            Ok(f)
        };
        Ok(KernelFns {
            _module: module,
            embed_rmsnorm: get_fn("ink2_embed_rmsnorm")?,
            rmsnorm_d: get_fn("ink2_rmsnorm_d")?,
            rmsnorm_quant_q8_64: get_fn("ink2_rmsnorm_quant_q8_64")?,
            depth_mix: get_fn("ink2_depth_mix")?,
            ctrl_down: get_fn("ink2_ctrl_down")?,
            ctrl_apply: get_fn("ink2_ctrl_apply")?,
            bf16_gemv: get_fn("ink2_bf16_gemv")?,
            quant_q8_64: get_fn("ink2_quant_q8_64")?,
            // The original v0.3 external cubin launches one warp per 16 columns.
            quant_q8_16: match get_fn("ink2_quant_q8_16_packed") {
                Ok(function) => Q8GroupKernel {
                    function,
                    columns_per_block: 128,
                    threads: 128,
                },
                Err(_) => Q8GroupKernel {
                    function: get_fn("ink2_quant_q8_16")?,
                    columns_per_block: 16,
                    threads: 32,
                },
            },
            q8_dp4a_gemv: get_fn("ink2_q8_dp4a_gemv")?,
            short_conv4: get_fn("ink2_short_conv4")?,
            prep_qkv_and_cache: get_fn("ink2_prep_qkv_and_cache")?,
            gqa_rel_sink_attn: get_fn("ink2_gqa_rel_sink_attn")?,
            gqa_attn_split: get_fn("ink2_gqa_attn_split").ok(),
            gqa_attn_combine: get_fn("ink2_gqa_attn_combine").ok(),
            router_topk8: get_fn("ink2_router_topk8")?,
            moe_nvfp4_up_situ: get_fn("ink2_moe_nvfp4_up_situ")?,
            moe_nvfp4_down: get_fn("ink2_moe_nvfp4_down")?,
            moe_nvfp4_dp4a_up_situ: get_fn("ink2_moe_nvfp4_dp4a_up_situ")?,
            moe_nvfp4_dp4a_down: get_fn("ink2_moe_nvfp4_dp4a_down")?,
            moe_mxfp8_up_situ: get_fn("ink2_moe_mxfp8_up_situ")?,
            moe_mxfp8_down: get_fn("ink2_moe_mxfp8_down")?,
            copy_d: get_fn("ink2_copy_d")?,
            argmax_and_inc_pos: get_fn("ink2_argmax_and_inc_pos")?,
            inc_pos: get_fn("ink2_inc_pos")?,
            embed_rmsnorm_b: get_fn("ink2_embed_rmsnorm_b")?,
            rmsnorm_d_b: get_fn("ink2_rmsnorm_d_b")?,
            rmsnorm_quant_q8_64_b: get_fn("ink2_rmsnorm_quant_q8_64_b")?,
            quant_q8_64_b: get_fn("ink2_quant_q8_64_b")?,
            depth_mix_b: get_fn("ink2_depth_mix_b")?,
            ctrl_down_b: get_fn("ink2_ctrl_down_b")?,
            ctrl_down_r8_b: get_fn("ink2_ctrl_down_r8_b").ok(),
            ctrl_apply_b: get_fn("ink2_ctrl_apply_b")?,
            ctrl_apply_t8_b: get_fn("ink2_ctrl_apply_t8_b").ok(),
            q8_dp4a_gemm_b: get_fn("ink2_q8_dp4a_gemm_b")?,
            short_conv4_b: get_fn("ink2_short_conv4_b")?,
            prep_qkv_and_cache_b: get_fn("ink2_prep_qkv_and_cache_b")?,
            gqa_rel_sink_attn_b: get_fn("ink2_gqa_rel_sink_attn_b")?,
            bf16_gemm_router_b: get_fn("ink2_bf16_gemm_router_b")?,
            router_topk8_b: get_fn("ink2_router_topk8_b")?,
            build_expert_groups_b: get_fn("ink2_build_expert_groups_b")?,
            build_expert_groups_wide_b: get_fn("ink2_build_expert_groups_wide_b").ok(),
            moe_nvfp4_dp4a_grouped_up_situ: get_fn("ink2_moe_nvfp4_dp4a_grouped_up_situ")?,
            moe_nvfp4_dp4a_grouped_down: get_fn("ink2_moe_nvfp4_dp4a_grouped_down")?,
            moe_mxfp8_grouped_up_situ: get_fn("ink2_moe_mxfp8_grouped_up_situ")?,
            moe_mxfp8_grouped_down: get_fn("ink2_moe_mxfp8_grouped_down")?,
            moe_combine_top8_b: get_fn("ink2_moe_combine_top8_b")?,
            inc_pos_by: get_fn("ink2_inc_pos_by")?,
            argmax_b: get_fn("ink2_argmax_b")?,
            conv_rewind: get_fn("ink2_conv_rewind")?,
            q8_dp4a_gemm_ktiled: get_fn("ink2_q8_dp4a_gemm_ktiled")?,
            mtp_tapmix: get_fn("mtp_tapmix")?,
            mtp_copy_anchor: get_fn("mtp_copy_anchor")?,
            mtp_ctx_store: get_fn("mtp_ctx_store")?,
            mtp_init_draft: get_fn("mtp_init_draft")?,
            mtp_qk_norm_rope: get_fn("mtp_qk_norm_rope")?,
            mtp_attn: get_fn("mtp_attn")?,
            mtp_add_inplace: get_fn("mtp_add_inplace")?,
            mtp_silu_mul: get_fn("mtp_silu_mul")?,
            mtp_concat_frnn: get_fn("mtp_concat_frnn")?,
            mtp_add_bias_rows: get_fn("mtp_add_bias_rows")?,
            mtp_markov_raw: get_fn("mtp_markov_raw")?,
            mtp_markov_step_q8: get_fn("mtp_markov_step_q8")?,
            mtp_fold_mv_q8: get_fn("mtp_fold_mv_q8")?,
            mtp_fold_fin: get_fn("mtp_fold_fin")?,
            depth_mix_rmsnorm_quant_q8_64_b: get_fn("ink2_depth_mix_rmsnorm_quant_q8_64_b").ok(),
            depth_mix_rmsnorm_d_quant_q8_16_b: get_fn("ink2_depth_mix_rmsnorm_d_quant_q8_16_b")
                .ok(),
            rmsnorm_d_quant_q8_16_b: get_fn("ink2_rmsnorm_d_quant_q8_16_b").ok(),
            ctrl_apply_coalesced_b: get_fn("ink2_ctrl_apply_coalesced_b").ok(),
            gqa_rel_sink_attn_q8_b: get_fn("ink2_gqa_rel_sink_attn_q8_b").ok(),
            gqa_attn_combine_q8: get_fn("ink2_gqa_attn_combine_q8").ok(),
            short_conv4_save: get_fn("ink2_short_conv4_save").ok(),
            short_conv4_save_b: get_fn("ink2_short_conv4_save_b").ok(),
            loop_pass_setup_b: get_fn("ink2_loop_pass_setup_b").ok(),
            q8_dp4a_qkv3_gemv: get_fn("ink2_q8_dp4a_qkv3_gemv").ok(),
            short_conv4_kv2: get_fn("ink2_short_conv4_kv2").ok(),
        })
    }

    pub fn reset(&mut self) -> Result<(), String> {
        self.ctx.bind_current()?;
        self.ctx.sync_compute()?;
        self.host_pos = 0;
        self.spec_drafted_total = 0;
        self.spec_accepted_total = 0;
        self.spec_verify_steps = 0;
        if let Some(ref mut d) = self.drafter {
            d.reset();
        }
        unsafe {
            check_cu(
                (self.ctx.api.cuMemsetD8_v2)(self.d_pos.ptr, 0, 4),
                "zero pos",
            )?;
            check_cu(
                (self.ctx.api.cuMemsetD8_v2)(self.conv_history.ptr, 0, self.conv_history.byte_len),
                "zero convolution histories",
            )?;
            check_cu(
                (self.ctx.api.cuStreamSynchronize)(core::ptr::null_mut()),
                "sync default stream",
            )?;
        }
        self.ctx.sync_compute()
    }

    #[allow(clippy::too_many_arguments)]
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
                "cuLaunchKernel",
            )
        }
    }

    fn enqueue_copy_d(
        &self,
        stream: CUstream,
        src: CUdeviceptr,
        dst: CUdeviceptr,
    ) -> Result<(), String> {
        let mut p_src = src;
        let mut p_dst = dst;
        let mut n = D as i32;
        let mut args: [*mut c_void; 3] = [
            (&mut p_src as *mut _) as *mut c_void,
            (&mut p_dst as *mut _) as *mut c_void,
            (&mut n as *mut _) as *mut c_void,
        ];
        self.launch(stream, self.fns.copy_d, (9, 1), 256, &mut args)
    }

    fn enqueue_ctrl_n(
        &self,
        stream: CUstream,
        cur_ptr: CUdeviceptr,
        msg_ptr: CUdeviceptr,
        ctrl: &CtrlGpu,
        n_ctrls: u32,
    ) -> Result<(), String> {
        let mut p_cur = cur_ptr;
        let mut p_msg = msg_ptr;
        let mut p_down = ctrl.down.ptr;
        let mut p_g = self.d_ctrl_g.ptr;
        let mut args1: [*mut c_void; 4] = [
            (&mut p_cur as *mut _) as *mut c_void,
            (&mut p_msg as *mut _) as *mut c_void,
            (&mut p_down as *mut _) as *mut c_void,
            (&mut p_g as *mut _) as *mut c_void,
        ];
        self.launch(stream, self.fns.ctrl_down, (64, n_ctrls), 256, &mut args1)?;

        let mut p_gain = ctrl.gain.ptr;
        let mut p_forget = ctrl.forget.ptr;
        if let Some(f_coal) = self.fns.ctrl_apply_coalesced_b {
            let mut b_one = 1i32;
            let mut args2: [*mut c_void; 6] = [
                (&mut p_cur as *mut _) as *mut c_void,
                (&mut p_msg as *mut _) as *mut c_void,
                (&mut p_g as *mut _) as *mut c_void,
                (&mut p_gain as *mut _) as *mut c_void,
                (&mut p_forget as *mut _) as *mut c_void,
                (&mut b_one as *mut _) as *mut c_void,
            ];
            self.launch(stream, f_coal, (36, n_ctrls), 64, &mut args2)
        } else {
            let mut args2: [*mut c_void; 5] = [
                (&mut p_cur as *mut _) as *mut c_void,
                (&mut p_msg as *mut _) as *mut c_void,
                (&mut p_g as *mut _) as *mut c_void,
                (&mut p_gain as *mut _) as *mut c_void,
                (&mut p_forget as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.ctrl_apply, (9, n_ctrls), 256, &mut args2)
        }
    }

    fn enqueue_rmsnorm_quant_q8_64(
        &self,
        stream: CUstream,
        inp_ptr: CUdeviceptr,
        gamma_ptr: CUdeviceptr,
        count: u32,
    ) -> Result<(), String> {
        let mut pin = inp_ptr;
        let mut pg = gamma_ptr;
        let mut pq = self.d_act_q8_64.ptr;
        let mut ps = self.d_act_sc_64.ptr;
        let mut args: [*mut c_void; 4] = [
            (&mut pin as *mut _) as *mut c_void,
            (&mut pg as *mut _) as *mut c_void,
            (&mut pq as *mut _) as *mut c_void,
            (&mut ps as *mut _) as *mut c_void,
        ];
        self.launch(
            stream,
            self.fns.rmsnorm_quant_q8_64,
            (count, 1),
            256,
            &mut args,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_bf16_gemv(
        &self,
        stream: CUstream,
        w_ptr: CUdeviceptr,
        x_ptr: CUdeviceptr,
        y_ptr: CUdeviceptr,
        rows: i32,
        cols: i32,
        softcap: f32,
    ) -> Result<(), String> {
        let mut pw = w_ptr;
        let mut px = x_ptr;
        let mut py = y_ptr;
        let mut r = rows;
        let mut c = cols;
        let mut sc = softcap;
        let mut args: [*mut c_void; 6] = [
            (&mut pw as *mut _) as *mut c_void,
            (&mut px as *mut _) as *mut c_void,
            (&mut py as *mut _) as *mut c_void,
            (&mut r as *mut _) as *mut c_void,
            (&mut c as *mut _) as *mut c_void,
            (&mut sc as *mut _) as *mut c_void,
        ];
        let grid_x = (rows as u32).div_ceil(8);
        self.launch(stream, self.fns.bf16_gemv, (grid_x, 1), 256, &mut args)
    }

    fn enqueue_quant_q8_64(
        &self,
        stream: CUstream,
        x_ptr: CUdeviceptr,
        cols: usize,
    ) -> Result<(), String> {
        let mut px = x_ptr;
        let mut pq = self.d_act_q8_64.ptr;
        let mut ps = self.d_act_sc_64.ptr;
        let mut args: [*mut c_void; 3] = [
            (&mut px as *mut _) as *mut c_void,
            (&mut pq as *mut _) as *mut c_void,
            (&mut ps as *mut _) as *mut c_void,
        ];
        self.launch(
            stream,
            self.fns.quant_q8_64,
            ((cols / 64) as u32, 1),
            64,
            &mut args,
        )
    }

    fn enqueue_quant_q8_16(
        &self,
        stream: CUstream,
        x_ptr: CUdeviceptr,
        n_vecs: usize,
        cols: usize,
    ) -> Result<(), String> {
        self.enqueue_quant_q8_groups(
            stream,
            x_ptr,
            self.d_act_q8_16.ptr,
            self.d_act_sc_16.ptr,
            n_vecs,
            cols,
        )
    }

    fn enqueue_quant_q8_groups(
        &self,
        stream: CUstream,
        x_ptr: CUdeviceptr,
        mut codes: CUdeviceptr,
        mut scales: CUdeviceptr,
        n_vecs: usize,
        cols: usize,
    ) -> Result<(), String> {
        let mut px = x_ptr;
        let kernel = &self.fns.quant_q8_16;
        let mut c = cols as i32;
        let mut args: [*mut c_void; 4] = [
            (&mut px as *mut _) as *mut c_void,
            (&mut codes as *mut _) as *mut c_void,
            (&mut scales as *mut _) as *mut c_void,
            (&mut c as *mut _) as *mut c_void,
        ];
        self.launch(
            stream,
            kernel.function,
            (
                cols.div_ceil(kernel.columns_per_block) as u32,
                n_vecs as u32,
            ),
            kernel.threads,
            &mut args,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_q8_gemv_slice_at(
        &self,
        stream: CUstream,
        w: &PlanarQ8Gpu,
        row_offset: usize,
        rows: i32,
        act_slot: usize,
        y_ptr: CUdeviceptr,
        softcap: f32,
    ) -> Result<(), String> {
        let cols = w.cols as usize;
        assert!(
            cols > 0 && cols.is_multiple_of(64) && cols <= 3072,
            "ink2_q8_dp4a_gemv requires cols in 64..=3072 (multiple of 64), got {cols}"
        );
        let mut pw = w.codes.ptr + (row_offset * cols) as u64;
        let mut pws = w.scales64.ptr + (row_offset * (cols / 64) * 4) as u64;
        let mut px = self.d_act_q8_64.ptr + (act_slot * cols) as u64;
        let mut pxs = self.d_act_sc_64.ptr + (act_slot * (cols / 64) * 4) as u64;
        let mut py = y_ptr;
        let mut r = rows;
        let mut c = w.cols;
        let mut sc = softcap;
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
        // 16 rows per 256-thread CTA (2 rows/warp = 38 regs <= 42 SM120 full occupancy, 16x SMEM reuse!)
        let grid_x = (rows as u32).div_ceil(16);
        self.launch(stream, self.fns.q8_dp4a_gemv, (grid_x, 1), 256, &mut args)
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_gqa_attn_split_impl(
        &self,
        stream: CUstream,
        qrgs_ptr: CUdeviceptr,
        qrgs_stride: i32,
        rel_proj_ptr: CUdeviceptr,
        k_cache_ptr: CUdeviceptr,
        v_cache_ptr: CUdeviceptr,
        kv_cap: i32,
        is_global: i32,
        attn_out_ptr: CUdeviceptr,
        out_q8_ptr: CUdeviceptr,
        out_sc64_ptr: CUdeviceptr,
        b_len: usize,
    ) -> Result<(), String> {
        let mut pqrgs = qrgs_ptr;
        let mut qstride = qrgs_stride;
        let mut prp = rel_proj_ptr;
        let mut pkc = k_cache_ptr;
        let mut pvc = v_cache_ptr;
        let mut ppos = self.d_pos.ptr;
        let mut cap = kv_cap;
        let mut ig = is_global;
        let mut pml = self.prefill.d_attn_part_ml.ptr;
        let mut po = self.prefill.d_attn_part_o.ptr;
        let mut split_args: [*mut c_void; 10] = [
            (&mut pqrgs as *mut _) as *mut c_void,
            (&mut qstride as *mut _) as *mut c_void,
            (&mut prp as *mut _) as *mut c_void,
            (&mut pkc as *mut _) as *mut c_void,
            (&mut pvc as *mut _) as *mut c_void,
            (&mut ppos as *mut _) as *mut c_void,
            (&mut cap as *mut _) as *mut c_void,
            (&mut ig as *mut _) as *mut c_void,
            (&mut pml as *mut _) as *mut c_void,
            (&mut po as *mut _) as *mut c_void,
        ];
        let g_u32 = self.attn_split_g as u32;
        self.launch(
            stream,
            self.fns.gqa_attn_split.unwrap(),
            ((8 * b_len) as u32, g_u32),
            256,
            &mut split_args,
        )?;
        let mut gi = self.attn_split_g as i32;
        if let Some(f_comb_q8) = self.fns.gqa_attn_combine_q8 {
            let mut pq8 = out_q8_ptr;
            let mut psc = out_sc64_ptr;
            let mut comb_args: [*mut c_void; 7] = [
                (&mut pml as *mut _) as *mut c_void,
                (&mut po as *mut _) as *mut c_void,
                (&mut pqrgs as *mut _) as *mut c_void,
                (&mut qstride as *mut _) as *mut c_void,
                (&mut gi as *mut _) as *mut c_void,
                (&mut pq8 as *mut _) as *mut c_void,
                (&mut psc as *mut _) as *mut c_void,
            ];
            self.launch(stream, f_comb_q8, (24, b_len as u32), 128, &mut comb_args)
        } else {
            let mut pout = attn_out_ptr;
            let mut comb_args: [*mut c_void; 6] = [
                (&mut pml as *mut _) as *mut c_void,
                (&mut po as *mut _) as *mut c_void,
                (&mut pqrgs as *mut _) as *mut c_void,
                (&mut qstride as *mut _) as *mut c_void,
                (&mut gi as *mut _) as *mut c_void,
                (&mut pout as *mut _) as *mut c_void,
            ];
            self.launch(
                stream,
                self.fns.gqa_attn_combine.unwrap(),
                (24, b_len as u32),
                128,
                &mut comb_args,
            )?;
            if b_len == 1 {
                self.enqueue_quant_q8_64(stream, attn_out_ptr, 3072)
            } else {
                self.enqueue_quant_q8_64_b(stream, attn_out_ptr, 3072, b_len)
            }
        }
    }

    fn enqueue_visit(
        &self,
        stream: CUstream,
        il: usize,
        slot_idx: usize,
        n_blocks: &mut usize,
        msg_base_ptr: CUdeviceptr,
    ) -> Result<(), String> {
        let l = &self.layers[il];
        let st = &self.slots[slot_idx];
        let d_bytes = (D * 4) as u64;
        let has_msgs = msg_base_ptr != 0;
        let distinct_k_v = l.global || has_msgs;
        let norm_count: u32 = if distinct_k_v { 3 } else { 1 };

        // 1–3. Attention inputs `[xq, xk, xv]` + optional Loop Controller + `RMSNorm + Q8_64 Quant`!
        if !has_msgs {
            if let Some(ref aq) = l.attn_query {
                let n_q: u32 = if l.global { 3 } else { 1 };
                if let Some(f_fused) = self.fns.depth_mix_rmsnorm_quant_q8_64_b {
                    let mut pb = self.d_blocks.ptr;
                    let mut nb = *n_blocks as i32;
                    let mut sb = 1i32;
                    let mut bo = 0i32;
                    let mut bout = 1i32;
                    let mut pc = self.d_cur.ptr;
                    let mut ac = 1i32;
                    let mut pq = aq.ptr;
                    let mut pg = l.norms_packed.ptr;
                    let mut pq8 = self.d_act_q8_64.ptr;
                    let mut psc = self.d_act_sc_64.ptr;
                    let mut args: [*mut c_void; 11] = [
                        (&mut pb as *mut _) as *mut c_void,
                        (&mut nb as *mut _) as *mut c_void,
                        (&mut sb as *mut _) as *mut c_void,
                        (&mut bo as *mut _) as *mut c_void,
                        (&mut bout as *mut _) as *mut c_void,
                        (&mut pc as *mut _) as *mut c_void,
                        (&mut ac as *mut _) as *mut c_void,
                        (&mut pq as *mut _) as *mut c_void,
                        (&mut pg as *mut _) as *mut c_void,
                        (&mut pq8 as *mut _) as *mut c_void,
                        (&mut psc as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, f_fused, (n_q, 1), 256, &mut args)?;
                } else {
                    let mut pb = self.d_blocks.ptr;
                    let mut nb = *n_blocks as i32;
                    let mut pc = self.d_cur.ptr;
                    let mut ac = 1i32;
                    let mut pq = aq.ptr;
                    let mut pout = self.d_inputs.ptr;
                    let mut b3 = 0i32;
                    let mut args: [*mut c_void; 7] = [
                        (&mut pb as *mut _) as *mut c_void,
                        (&mut nb as *mut _) as *mut c_void,
                        (&mut pc as *mut _) as *mut c_void,
                        (&mut ac as *mut _) as *mut c_void,
                        (&mut pq as *mut _) as *mut c_void,
                        (&mut pout as *mut _) as *mut c_void,
                        (&mut b3 as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, self.fns.depth_mix, (n_q, 1), 256, &mut args)?;
                    self.enqueue_rmsnorm_quant_q8_64(
                        stream,
                        self.d_inputs.ptr,
                        l.norms_packed.ptr,
                        norm_count,
                    )?;
                }
            } else {
                // Layer 0 (`il == 0`): `!l.global && !has_msgs`, quantize `d_cur` directly (`0` `copy_d` launches!)
                debug_assert!(!distinct_k_v);
                self.enqueue_rmsnorm_quant_q8_64(stream, self.d_cur.ptr, l.norms_packed.ptr, 1)?;
            }
        } else {
            let aq = l.attn_query.as_ref().unwrap();
            let n_q: u32 = if l.global { 3 } else { 1 };
            let mut pb = self.d_blocks.ptr;
            let mut nb = *n_blocks as i32;
            let mut pc = self.d_cur.ptr;
            let mut ac = 1i32;
            let mut pq = aq.ptr;
            let mut pout = self.d_inputs.ptr;
            let mut b3 = i32::from(!l.global);
            let mut args: [*mut c_void; 7] = [
                (&mut pb as *mut _) as *mut c_void,
                (&mut nb as *mut _) as *mut c_void,
                (&mut pc as *mut _) as *mut c_void,
                (&mut ac as *mut _) as *mut c_void,
                (&mut pq as *mut _) as *mut c_void,
                (&mut pout as *mut _) as *mut c_void,
                (&mut b3 as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.depth_mix, (n_q, 1), 256, &mut args)?;
            self.enqueue_ctrl_n(
                stream,
                self.d_inputs.ptr,
                msg_base_ptr,
                l.ctrl3.as_ref().unwrap(),
                3,
            )?;
            self.enqueue_rmsnorm_quant_q8_64(
                stream,
                self.d_inputs.ptr,
                l.norms_packed.ptr,
                norm_count,
            )?;
        }

        let k_raw_ptr: CUdeviceptr;
        let v_raw_ptr: CUdeviceptr;

        if !distinct_k_v {
            self.enqueue_q8_gemv_slice_at(
                stream,
                &l.wq_all_q8,
                0,
                SWA_ALL_ROWS as i32,
                0,
                self.d_qrgs.ptr,
                0.0,
            )?;
            k_raw_ptr = self.d_qrgs.ptr + (SWA_K_OFFSET as u64) * 4;
            v_raw_ptr = self.d_qrgs.ptr + (SWA_V_OFFSET as u64) * 4;
        } else if let Some(f_qkv3) = self.fns.q8_dp4a_qkv3_gemv {
            let (wk_codes, wk_sc, rk, wv_codes, wv_sc, rv) = if l.global {
                let wk = l.wk_glob_q8.as_ref().unwrap();
                let wv = l.wv_glob_q8.as_ref().unwrap();
                (
                    wk.codes.ptr,
                    wk.scales64.ptr,
                    1056i32,
                    wv.codes.ptr,
                    wv.scales64.ptr,
                    1056i32,
                )
            } else {
                (
                    l.wq_all_q8.codes.ptr + (SWA_K_OFFSET * D) as u64,
                    l.wq_all_q8.scales64.ptr + (SWA_K_OFFSET * (D / 64) * 4) as u64,
                    1024i32,
                    l.wq_all_q8.codes.ptr + (SWA_V_OFFSET * D) as u64,
                    l.wq_all_q8.scales64.ptr + (SWA_V_OFFSET * (D / 64) * 4) as u64,
                    1024i32,
                )
            };
            let mut pwq = l.wq_all_q8.codes.ptr;
            let mut pwqs = l.wq_all_q8.scales64.ptr;
            let mut rq = QRGS_ROWS as i32;
            let mut pyq = self.d_qrgs.ptr;
            let mut pwk = wk_codes;
            let mut pwks = wk_sc;
            let mut rk_mut = rk;
            let mut pyk = self.d_k_raw.ptr;
            let mut pwv = wv_codes;
            let mut pwvs = wv_sc;
            let mut rv_mut = rv;
            let mut pyv = self.d_v_raw.ptr;
            let mut px = self.d_act_q8_64.ptr;
            let mut pxs = self.d_act_sc_64.ptr;
            let mut args: [*mut c_void; 14] = [
                (&mut pwq as *mut _) as *mut c_void,
                (&mut pwqs as *mut _) as *mut c_void,
                (&mut rq as *mut _) as *mut c_void,
                (&mut pyq as *mut _) as *mut c_void,
                (&mut pwk as *mut _) as *mut c_void,
                (&mut pwks as *mut _) as *mut c_void,
                (&mut rk_mut as *mut _) as *mut c_void,
                (&mut pyk as *mut _) as *mut c_void,
                (&mut pwv as *mut _) as *mut c_void,
                (&mut pwvs as *mut _) as *mut c_void,
                (&mut rv_mut as *mut _) as *mut c_void,
                (&mut pyv as *mut _) as *mut c_void,
                (&mut px as *mut _) as *mut c_void,
                (&mut pxs as *mut _) as *mut c_void,
            ];
            let grid_x = (QRGS_ROWS as u32).div_ceil(16)
                + (rk as u32).div_ceil(16)
                + (rv as u32).div_ceil(16);
            self.launch(stream, f_qkv3, (grid_x, 1), 256, &mut args)?;
            k_raw_ptr = self.d_k_raw.ptr;
            v_raw_ptr = self.d_v_raw.ptr;
        } else if l.global {
            self.enqueue_q8_gemv_slice_at(
                stream,
                &l.wq_all_q8,
                0,
                QRGS_ROWS as i32,
                0,
                self.d_qrgs.ptr,
                0.0,
            )?;
            self.enqueue_q8_gemv_slice_at(
                stream,
                l.wk_glob_q8.as_ref().unwrap(),
                0,
                1056,
                1,
                self.d_k_raw.ptr,
                0.0,
            )?;
            self.enqueue_q8_gemv_slice_at(
                stream,
                l.wv_glob_q8.as_ref().unwrap(),
                0,
                1056,
                2,
                self.d_v_raw.ptr,
                0.0,
            )?;
            k_raw_ptr = self.d_k_raw.ptr;
            v_raw_ptr = self.d_v_raw.ptr;
        } else {
            self.enqueue_q8_gemv_slice_at(
                stream,
                &l.wq_all_q8,
                0,
                QRGS_ROWS as i32,
                0,
                self.d_qrgs.ptr,
                0.0,
            )?;
            self.enqueue_q8_gemv_slice_at(
                stream,
                &l.wq_all_q8,
                SWA_K_OFFSET,
                1024,
                1,
                self.d_k_raw.ptr,
                0.0,
            )?;
            self.enqueue_q8_gemv_slice_at(
                stream,
                &l.wq_all_q8,
                SWA_V_OFFSET,
                1024,
                2,
                self.d_v_raw.ptr,
                0.0,
            )?;
            k_raw_ptr = self.d_k_raw.ptr;
            v_raw_ptr = self.d_v_raw.ptr;
        }

        // 5. K & V Short-Conv4 (fused `(4, 2)` launch when `short_conv4_kv2` is available!)
        if let Some(f_kv2) = self.fns.short_conv4_kv2 {
            let mut pk = k_raw_ptr;
            let mut pwk = l.k_conv4.ptr;
            let mut pdk = if l.global { k_raw_ptr + 1024 * 4 } else { 0 };
            let mut phk = st.hist_k;
            let mut pok = self.d_k_conv.ptr;
            let mut pv = v_raw_ptr;
            let mut pwv = l.v_conv4.ptr;
            let mut pdv = if l.global { v_raw_ptr + 1024 * 4 } else { 0 };
            let mut phv = st.hist_v;
            let mut pov = self.d_v_conv.ptr;
            let mut hd = i32::from(l.global);
            let mut args: [*mut c_void; 11] = [
                (&mut pk as *mut _) as *mut c_void,
                (&mut pwk as *mut _) as *mut c_void,
                (&mut pdk as *mut _) as *mut c_void,
                (&mut phk as *mut _) as *mut c_void,
                (&mut pok as *mut _) as *mut c_void,
                (&mut pv as *mut _) as *mut c_void,
                (&mut pwv as *mut _) as *mut c_void,
                (&mut pdv as *mut _) as *mut c_void,
                (&mut phv as *mut _) as *mut c_void,
                (&mut pov as *mut _) as *mut c_void,
                (&mut hd as *mut _) as *mut c_void,
            ];
            self.launch(stream, f_kv2, (4, 2), 256, &mut args)?;
        } else {
            for (raw_ptr, conv_w, hist_ptr, dst_ptr) in [
                (k_raw_ptr, l.k_conv4.ptr, st.hist_k, self.d_k_conv.ptr),
                (v_raw_ptr, l.v_conv4.ptr, st.hist_v, self.d_v_conv.ptr),
            ] {
                let mut p_cur = raw_ptr;
                let mut p_w = conv_w;
                let mut p_dyn = if l.global { raw_ptr + 1024 * 4 } else { 0 };
                let mut p_hist = hist_ptr;
                let mut p_out = dst_ptr;
                let mut p_res: CUdeviceptr = 0;
                let mut ch = 1024i32;
                let mut hd = i32::from(l.global);
                let mut dr = 0i32;
                let mut args: [*mut c_void; 9] = [
                    (&mut p_cur as *mut _) as *mut c_void,
                    (&mut p_w as *mut _) as *mut c_void,
                    (&mut p_dyn as *mut _) as *mut c_void,
                    (&mut p_hist as *mut _) as *mut c_void,
                    (&mut p_out as *mut _) as *mut c_void,
                    (&mut p_res as *mut _) as *mut c_void,
                    (&mut ch as *mut _) as *mut c_void,
                    (&mut hd as *mut _) as *mut c_void,
                    (&mut dr as *mut _) as *mut c_void,
                ];
                self.launch(stream, self.fns.short_conv4, (4, 1), 256, &mut args)?;
            }
        }

        // 6. Q/R/K Clip + V Head-RMSNorm + Log-Scaling + FP16 KV Cache Write
        let q_ptr = self.d_qrgs.ptr;
        let r_ptr = q_ptr + 3072 * 4;
        let gate_ptr = r_ptr + 384 * 4;
        let sink_ptr = gate_ptr + 3072 * 4;
        {
            let mut pq = q_ptr;
            let mut pr = r_ptr;
            let mut pk = self.d_k_conv.ptr;
            let mut pv = self.d_v_conv.ptr;
            let mut pqc = l.q_clip.ptr;
            let mut prc = l.rel_clip.ptr;
            let mut pkc = st.k_cache.ptr;
            let mut pvc = st.v_cache.ptr;
            let mut ppos = self.d_pos.ptr;
            let mut cap = st.kv_cap as i32;
            let mut ig = i32::from(l.global);
            let mut args: [*mut c_void; 11] = [
                (&mut pq as *mut _) as *mut c_void,
                (&mut pr as *mut _) as *mut c_void,
                (&mut pk as *mut _) as *mut c_void,
                (&mut pv as *mut _) as *mut c_void,
                (&mut pqc as *mut _) as *mut c_void,
                (&mut prc as *mut _) as *mut c_void,
                (&mut pkc as *mut _) as *mut c_void,
                (&mut pvc as *mut _) as *mut c_void,
                (&mut ppos as *mut _) as *mut c_void,
                (&mut cap as *mut _) as *mut c_void,
                (&mut ig as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.prep_qkv_and_cache, (5, 1), 256, &mut args)?;
        }

        // 7. GQA Relative-Bias + Sink Attention + In-Register `Q8_64` Epilogue!
        if self.attn_split_g == 0 {
            if let Some(f_attn_q8) = self.fns.gqa_rel_sink_attn_q8_b {
                let mut pqrgs = self.d_qrgs.ptr;
                let mut qstride = SWA_ALL_ROWS as i32;
                let mut prp = l.rel_proj.ptr;
                let mut pkc = st.k_cache.ptr;
                let mut pvc = st.v_cache.ptr;
                let mut ppos = self.d_pos.ptr;
                let mut cap = st.kv_cap as i32;
                let mut ig = i32::from(l.global);
                let mut pq8 = self.d_act_q8_64.ptr;
                let mut psc = self.d_act_sc_64.ptr;
                let mut args: [*mut c_void; 10] = [
                    (&mut pqrgs as *mut _) as *mut c_void,
                    (&mut qstride as *mut _) as *mut c_void,
                    (&mut prp as *mut _) as *mut c_void,
                    (&mut pkc as *mut _) as *mut c_void,
                    (&mut pvc as *mut _) as *mut c_void,
                    (&mut ppos as *mut _) as *mut c_void,
                    (&mut cap as *mut _) as *mut c_void,
                    (&mut ig as *mut _) as *mut c_void,
                    (&mut pq8 as *mut _) as *mut c_void,
                    (&mut psc as *mut _) as *mut c_void,
                ];
                self.launch(stream, f_attn_q8, (24, 1), 256, &mut args)?;
            } else {
                let mut pq = q_ptr;
                let mut pr = r_ptr;
                let mut pg = gate_ptr;
                let mut ps = sink_ptr;
                let mut prp = l.rel_proj.ptr;
                let mut pkc = st.k_cache.ptr;
                let mut pvc = st.v_cache.ptr;
                let mut ppos = self.d_pos.ptr;
                let mut cap = st.kv_cap as i32;
                let mut ig = i32::from(l.global);
                let mut pout = self.d_attn_out.ptr;
                let mut args: [*mut c_void; 11] = [
                    (&mut pq as *mut _) as *mut c_void,
                    (&mut pr as *mut _) as *mut c_void,
                    (&mut pg as *mut _) as *mut c_void,
                    (&mut ps as *mut _) as *mut c_void,
                    (&mut prp as *mut _) as *mut c_void,
                    (&mut pkc as *mut _) as *mut c_void,
                    (&mut pvc as *mut _) as *mut c_void,
                    (&mut ppos as *mut _) as *mut c_void,
                    (&mut cap as *mut _) as *mut c_void,
                    (&mut ig as *mut _) as *mut c_void,
                    (&mut pout as *mut _) as *mut c_void,
                ];
                self.launch(stream, self.fns.gqa_rel_sink_attn, (24, 1), 256, &mut args)?;
                self.enqueue_quant_q8_64(stream, self.d_attn_out.ptr, 3072)?;
            }
        } else {
            self.enqueue_gqa_attn_split_impl(
                stream,
                self.d_qrgs.ptr,
                SWA_ALL_ROWS as i32,
                l.rel_proj.ptr,
                st.k_cache.ptr,
                st.v_cache.ptr,
                st.kv_cap as i32,
                i32::from(l.global),
                self.d_attn_out.ptr,
                self.d_act_q8_64.ptr,
                self.d_act_sc_64.ptr,
                1,
            )?;
        }

        // 8. `wo` Planar `Q8_64` `__dp4a` Projection + `attn_sconv` (+ fused block-start save!)
        self.enqueue_q8_gemv_slice_at(stream, &l.wo_q8, 0, D as i32, 0, self.d_wo_out.ptr, 0.0)?;

        let is_block_start = il.is_multiple_of(6);
        let save_blk_ptr = if is_block_start {
            let blk_dst = self.d_blocks.ptr + (*n_blocks as u64) * d_bytes;
            *n_blocks += 1;
            blk_dst
        } else {
            0
        };
        if let Some(f_sc_save) = self.fns.short_conv4_save {
            let mut p_cur = self.d_wo_out.ptr;
            let mut p_w = l.attn_conv4.ptr;
            let mut p_dyn: CUdeviceptr = 0;
            let mut p_hist = st.hist_attn;
            let mut p_out = self.d_cur.ptr;
            let mut p_res = if is_block_start { 0 } else { self.d_cur.ptr };
            let mut p_save = save_blk_ptr;
            let mut ch = D as i32;
            let mut hd = 0i32;
            let mut dr = i32::from(!is_block_start);
            let mut args: [*mut c_void; 10] = [
                (&mut p_cur as *mut _) as *mut c_void,
                (&mut p_w as *mut _) as *mut c_void,
                (&mut p_dyn as *mut _) as *mut c_void,
                (&mut p_hist as *mut _) as *mut c_void,
                (&mut p_out as *mut _) as *mut c_void,
                (&mut p_res as *mut _) as *mut c_void,
                (&mut p_save as *mut _) as *mut c_void,
                (&mut ch as *mut _) as *mut c_void,
                (&mut hd as *mut _) as *mut c_void,
                (&mut dr as *mut _) as *mut c_void,
            ];
            self.launch(stream, f_sc_save, (9, 1), 256, &mut args)?;
        } else {
            if save_blk_ptr != 0 {
                self.enqueue_copy_d(stream, self.d_cur.ptr, save_blk_ptr)?;
            }
            let mut p_cur = self.d_wo_out.ptr;
            let mut p_w = l.attn_conv4.ptr;
            let mut p_dyn: CUdeviceptr = 0;
            let mut p_hist = st.hist_attn;
            let mut p_out = self.d_cur.ptr;
            let mut p_res = if is_block_start { 0 } else { self.d_cur.ptr };
            let mut ch = D as i32;
            let mut hd = 0i32;
            let mut dr = i32::from(!is_block_start);
            let mut args: [*mut c_void; 9] = [
                (&mut p_cur as *mut _) as *mut c_void,
                (&mut p_w as *mut _) as *mut c_void,
                (&mut p_dyn as *mut _) as *mut c_void,
                (&mut p_hist as *mut _) as *mut c_void,
                (&mut p_out as *mut _) as *mut c_void,
                (&mut p_res as *mut _) as *mut c_void,
                (&mut ch as *mut _) as *mut c_void,
                (&mut hd as *mut _) as *mut c_void,
                (&mut dr as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.short_conv4, (9, 1), 256, &mut args)?;
        }

        // 9. Fused MLP Prologue (`depth_mix + rmsnorm_d + quant_q8_16` when `!has_msgs`)
        let wants_mlp_q8_16 =
            matches!(l.expert_scales, ExpertScales::Nvfp4 { .. }) && self.use_moe_dp4a;
        let mut mlp_h_ready = false;
        let mut mlp_q8_16_ready = false;
        if !has_msgs {
            if let Some(f_mlp_fused) = self.fns.depth_mix_rmsnorm_d_quant_q8_16_b {
                let mut pb = self.d_blocks.ptr;
                let mut nb = *n_blocks as i32;
                let mut sb = 1i32;
                let mut bo = 0i32;
                let mut pc = self.d_cur.ptr;
                let mut ac = 1i32;
                let mut pq = l.mlp_query.ptr;
                let mut pg = l.mlp_norm.ptr;
                let mut pout = self.d_mlp_h.ptr;
                let mut dq = i32::from(wants_mlp_q8_16);
                let mut pq8 = self.d_act_q8_16.ptr;
                let mut psc = self.d_act_sc_16.ptr;
                let mut args: [*mut c_void; 12] = [
                    (&mut pb as *mut _) as *mut c_void,
                    (&mut nb as *mut _) as *mut c_void,
                    (&mut sb as *mut _) as *mut c_void,
                    (&mut bo as *mut _) as *mut c_void,
                    (&mut pc as *mut _) as *mut c_void,
                    (&mut ac as *mut _) as *mut c_void,
                    (&mut pq as *mut _) as *mut c_void,
                    (&mut pg as *mut _) as *mut c_void,
                    (&mut pout as *mut _) as *mut c_void,
                    (&mut dq as *mut _) as *mut c_void,
                    (&mut pq8 as *mut _) as *mut c_void,
                    (&mut psc as *mut _) as *mut c_void,
                ];
                self.launch(stream, f_mlp_fused, (1, 1), 256, &mut args)?;
                mlp_h_ready = true;
                mlp_q8_16_ready = wants_mlp_q8_16;
            }
        }
        if !mlp_h_ready {
            let mut pb = self.d_blocks.ptr;
            let mut nb = *n_blocks as i32;
            let mut pc = self.d_cur.ptr;
            let mut ac = 1i32;
            let mut pq = l.mlp_query.ptr;
            let mut pout = self.d_mlp_h.ptr;
            let mut b3 = 0i32;
            let mut args: [*mut c_void; 7] = [
                (&mut pb as *mut _) as *mut c_void,
                (&mut nb as *mut _) as *mut c_void,
                (&mut pc as *mut _) as *mut c_void,
                (&mut ac as *mut _) as *mut c_void,
                (&mut pq as *mut _) as *mut c_void,
                (&mut pout as *mut _) as *mut c_void,
                (&mut b3 as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.depth_mix, (1, 1), 256, &mut args)?;
            if has_msgs {
                self.enqueue_ctrl_n(
                    stream,
                    self.d_mlp_h.ptr,
                    msg_base_ptr + 3 * d_bytes,
                    l.ctrl_mlp.as_ref().unwrap(),
                    1,
                )?;
            }
            if wants_mlp_q8_16 {
                if let Some(f_rn_q8) = self.fns.rmsnorm_d_quant_q8_16_b {
                    let mut pin = self.d_mlp_h.ptr;
                    let mut pg = l.mlp_norm.ptr;
                    let mut pout = self.d_mlp_h.ptr;
                    let mut pq8 = self.d_act_q8_16.ptr;
                    let mut psc = self.d_act_sc_16.ptr;
                    let mut args: [*mut c_void; 5] = [
                        (&mut pin as *mut _) as *mut c_void,
                        (&mut pg as *mut _) as *mut c_void,
                        (&mut pout as *mut _) as *mut c_void,
                        (&mut pq8 as *mut _) as *mut c_void,
                        (&mut psc as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, f_rn_q8, (1, 1), 256, &mut args)?;
                    mlp_q8_16_ready = true;
                }
            }
            if !mlp_q8_16_ready {
                let mut pin = self.d_mlp_h.ptr;
                let mut pg = l.mlp_norm.ptr;
                let mut pout = self.d_mlp_h.ptr;
                let mut hg = 1i32;
                let mut args: [*mut c_void; 4] = [
                    (&mut pin as *mut _) as *mut c_void,
                    (&mut pg as *mut _) as *mut c_void,
                    (&mut pout as *mut _) as *mut c_void,
                    (&mut hg as *mut _) as *mut c_void,
                ];
                self.launch(stream, self.fns.rmsnorm_d, (1, 1), 256, &mut args)?;
            }
        }

        // 10. Exact BF16 Router + Warp-Butterfly TopK8 + Wide `uint4` Full-GPU MoE!
        self.enqueue_bf16_gemv(
            stream,
            l.router_w.ptr,
            self.d_mlp_h.ptr,
            self.d_router_logits.ptr,
            N_EXP as i32,
            D as i32,
            0.0,
        )?;
        {
            let mut pl = self.d_router_logits.ptr;
            let mut pb = l.router_bias.ptr;
            let mut pids = self.d_sel_ids.ptr;
            let mut pwts = self.d_sel_wts.ptr;
            let mut args: [*mut c_void; 4] = [
                (&mut pl as *mut _) as *mut c_void,
                (&mut pb as *mut _) as *mut c_void,
                (&mut pids as *mut _) as *mut c_void,
                (&mut pwts as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.router_topk8, (1, 1), 32, &mut args)?;
        }

        let [w13, w2] = &self.expert_weights[il];
        match &l.expert_scales {
            ExpertScales::Nvfp4 { s13, a13, s2, a2 } => {
                if self.use_moe_dp4a {
                    if !mlp_q8_16_ready {
                        self.enqueue_quant_q8_16(stream, self.d_mlp_h.ptr, 1, D)?;
                    }
                    let mut pw13 = w13.ptr();
                    let mut ps13 = s13.ptr;
                    let mut pa13 = a13.ptr;
                    let mut pxq = self.d_act_q8_16.ptr;
                    let mut pxs = self.d_act_sc_16.ptr;
                    let mut pids = self.d_sel_ids.ptr;
                    let mut pact = self.d_moe_act.ptr;
                    let mut up_args: [*mut c_void; 7] = [
                        (&mut pw13 as *mut _) as *mut c_void,
                        (&mut ps13 as *mut _) as *mut c_void,
                        (&mut pa13 as *mut _) as *mut c_void,
                        (&mut pxq as *mut _) as *mut c_void,
                        (&mut pxs as *mut _) as *mut c_void,
                        (&mut pids as *mut _) as *mut c_void,
                        (&mut pact as *mut _) as *mut c_void,
                    ];
                    self.launch(
                        stream,
                        self.fns.moe_nvfp4_dp4a_up_situ,
                        (288, 8),
                        256,
                        &mut up_args,
                    )?;

                    self.enqueue_quant_q8_16(stream, self.d_moe_act.ptr, TOP_K, FF)?;
                    let mut pw2 = w2.ptr();
                    let mut ps2 = s2.ptr;
                    let mut pa2 = a2.ptr;
                    let mut pwts = self.d_sel_wts.ptr;
                    let mut py = self.d_moe_y.ptr;
                    let mut dn_args: [*mut c_void; 8] = [
                        (&mut pw2 as *mut _) as *mut c_void,
                        (&mut ps2 as *mut _) as *mut c_void,
                        (&mut pa2 as *mut _) as *mut c_void,
                        (&mut pxq as *mut _) as *mut c_void,
                        (&mut pxs as *mut _) as *mut c_void,
                        (&mut pids as *mut _) as *mut c_void,
                        (&mut pwts as *mut _) as *mut c_void,
                        (&mut py as *mut _) as *mut c_void,
                    ];
                    self.launch(
                        stream,
                        self.fns.moe_nvfp4_dp4a_down,
                        (288, 1),
                        256,
                        &mut dn_args,
                    )?;
                } else {
                    let mut pw13 = w13.ptr();
                    let mut ps13 = s13.ptr;
                    let mut pa13 = a13.ptr;
                    let mut px = self.d_mlp_h.ptr;
                    let mut pids = self.d_sel_ids.ptr;
                    let mut pact = self.d_moe_act.ptr;
                    let mut up_args: [*mut c_void; 6] = [
                        (&mut pw13 as *mut _) as *mut c_void,
                        (&mut ps13 as *mut _) as *mut c_void,
                        (&mut pa13 as *mut _) as *mut c_void,
                        (&mut px as *mut _) as *mut c_void,
                        (&mut pids as *mut _) as *mut c_void,
                        (&mut pact as *mut _) as *mut c_void,
                    ];
                    self.launch(
                        stream,
                        self.fns.moe_nvfp4_up_situ,
                        (288, 8),
                        256,
                        &mut up_args,
                    )?;

                    let mut pw2 = w2.ptr();
                    let mut ps2 = s2.ptr;
                    let mut pa2 = a2.ptr;
                    let mut pwts = self.d_sel_wts.ptr;
                    let mut py = self.d_moe_y.ptr;
                    let mut dn_args: [*mut c_void; 7] = [
                        (&mut pw2 as *mut _) as *mut c_void,
                        (&mut ps2 as *mut _) as *mut c_void,
                        (&mut pa2 as *mut _) as *mut c_void,
                        (&mut pact as *mut _) as *mut c_void,
                        (&mut pids as *mut _) as *mut c_void,
                        (&mut pwts as *mut _) as *mut c_void,
                        (&mut py as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, self.fns.moe_nvfp4_down, (288, 1), 256, &mut dn_args)?;
                }
            }
            ExpertScales::Mxfp8 { s13, s2 } => {
                let mut pw13 = w13.ptr();
                let mut ps13 = s13.ptr;
                let mut px = self.d_mlp_h.ptr;
                let mut pids = self.d_sel_ids.ptr;
                let mut pact = self.d_moe_act.ptr;
                let mut up_args: [*mut c_void; 5] = [
                    (&mut pw13 as *mut _) as *mut c_void,
                    (&mut ps13 as *mut _) as *mut c_void,
                    (&mut px as *mut _) as *mut c_void,
                    (&mut pids as *mut _) as *mut c_void,
                    (&mut pact as *mut _) as *mut c_void,
                ];
                self.launch(
                    stream,
                    self.fns.moe_mxfp8_up_situ,
                    (288, 8),
                    256,
                    &mut up_args,
                )?;

                let mut pw2 = w2.ptr();
                let mut ps2 = s2.ptr;
                let mut pwts = self.d_sel_wts.ptr;
                let mut py = self.d_moe_y.ptr;
                let mut dn_args: [*mut c_void; 6] = [
                    (&mut pw2 as *mut _) as *mut c_void,
                    (&mut ps2 as *mut _) as *mut c_void,
                    (&mut pact as *mut _) as *mut c_void,
                    (&mut pids as *mut _) as *mut c_void,
                    (&mut pwts as *mut _) as *mut c_void,
                    (&mut py as *mut _) as *mut c_void,
                ];
                self.launch(stream, self.fns.moe_mxfp8_down, (288, 1), 256, &mut dn_args)?;
            }
        }

        // 11. `mlp_sconv` + Residual Add in-place into `d_cur`!
        {
            let mut p_cur = self.d_moe_y.ptr;
            let mut p_w = l.mlp_conv4.ptr;
            let mut p_dyn: CUdeviceptr = 0;
            let mut p_hist = st.hist_mlp;
            let mut p_out = self.d_cur.ptr;
            let mut p_res = self.d_cur.ptr;
            let mut ch = D as i32;
            let mut hd = 0i32;
            let mut dr = 1i32;
            let mut args: [*mut c_void; 9] = [
                (&mut p_cur as *mut _) as *mut c_void,
                (&mut p_w as *mut _) as *mut c_void,
                (&mut p_dyn as *mut _) as *mut c_void,
                (&mut p_hist as *mut _) as *mut c_void,
                (&mut p_out as *mut _) as *mut c_void,
                (&mut p_res as *mut _) as *mut c_void,
                (&mut ch as *mut _) as *mut c_void,
                (&mut hd as *mut _) as *mut c_void,
                (&mut dr as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.short_conv4, (9, 1), 256, &mut args)?;
        }

        if let Some(taps) = &self.prefill.d_taps {
            if let Some(tap_idx) = TAP_SLOTS.iter().position(|&s| s == slot_idx) {
                let dst = taps.ptr + (tap_idx * self.prefill.capacity * D * 4) as u64;
                self.enqueue_copy_d(stream, self.d_cur.ptr, dst)?;
            }
        }

        Ok(())
    }

    fn enqueue_step_impl(&self, stream: CUstream, prefill_only: bool) -> Result<(), String> {
        let d_bytes = (D * 4) as u64;

        // 0. Embed + `embed_norm` (`prefill_only` indexes `d_prefill_tokens[*d_pos]` on GPU!)
        {
            let mut pe = self.embed.ptr();
            let mut pn = self.embed_norm.ptr;
            let mut pt = if prefill_only {
                self.d_prefill_tokens.ptr
            } else {
                self.d_token_in.ptr
            };
            let mut pp = self.d_pos.ptr;
            let mut up = i32::from(prefill_only);
            let mut px = self.d_cur.ptr;
            let mut args: [*mut c_void; 6] = [
                (&mut pe as *mut _) as *mut c_void,
                (&mut pn as *mut _) as *mut c_void,
                (&mut pt as *mut _) as *mut c_void,
                (&mut pp as *mut _) as *mut c_void,
                (&mut up as *mut _) as *mut c_void,
                (&mut px as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.embed_rmsnorm, (1, 1), 256, &mut args)?;
        }

        let mut n_blocks = 0usize;
        for il in 0..6 {
            self.enqueue_visit(stream, il, il, &mut n_blocks, 0)?;
        }
        debug_assert_eq!(n_blocks, 1);
        if let Some(f_setup) = self.fns.loop_pass_setup_b {
            let mut p_cur = self.d_cur.ptr;
            let mut p_blk = self.d_blocks.ptr;
            let mut p_px = self.d_prelude_x.ptr;
            let mut p_pb0 = self.d_prelude_block0.ptr;
            let mut p_bank = self.d_bank.ptr;
            let mut bi = 1i32;
            let mut mode = 0i32;
            let mut args: [*mut c_void; 7] = [
                (&mut p_cur as *mut _) as *mut c_void,
                (&mut p_blk as *mut _) as *mut c_void,
                (&mut p_px as *mut _) as *mut c_void,
                (&mut p_pb0 as *mut _) as *mut c_void,
                (&mut p_bank as *mut _) as *mut c_void,
                (&mut bi as *mut _) as *mut c_void,
                (&mut mode as *mut _) as *mut c_void,
            ];
            self.launch(stream, f_setup, (9, 1), 256, &mut args)?;
        } else {
            self.enqueue_copy_d(stream, self.d_cur.ptr, self.d_prelude_x.ptr)?;
            self.enqueue_copy_d(stream, self.d_blocks.ptr, self.d_prelude_block0.ptr)?;
        }

        for pass in 0..3 {
            if pass > 0 {
                if let Some(f_setup) = self.fns.loop_pass_setup_b {
                    let mut p_cur = self.d_cur.ptr;
                    let mut p_blk = self.d_blocks.ptr;
                    let mut p_px = self.d_prelude_x.ptr;
                    let mut p_pb0 = self.d_prelude_block0.ptr;
                    let mut p_bank = self.d_bank.ptr;
                    let mut bi = 1i32;
                    let mut mode = 1i32;
                    let mut args: [*mut c_void; 7] = [
                        (&mut p_cur as *mut _) as *mut c_void,
                        (&mut p_blk as *mut _) as *mut c_void,
                        (&mut p_px as *mut _) as *mut c_void,
                        (&mut p_pb0 as *mut _) as *mut c_void,
                        (&mut p_bank as *mut _) as *mut c_void,
                        (&mut bi as *mut _) as *mut c_void,
                        (&mut mode as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, f_setup, (9, 1), 256, &mut args)?;
                } else {
                    self.enqueue_copy_d(stream, self.d_blocks.ptr + 2 * d_bytes, self.d_bank.ptr)?;
                    self.enqueue_copy_d(stream, self.d_cur.ptr, self.d_bank.ptr + d_bytes)?;
                    self.enqueue_copy_d(stream, self.d_prelude_x.ptr, self.d_cur.ptr)?;
                    self.enqueue_copy_d(stream, self.d_prelude_block0.ptr, self.d_blocks.ptr)?;
                }
                n_blocks = 1;
                // Precompute ALL `12 * 4 = 48` cross-pass messages for this pass in ONE `48`-CTA launch!
                let mut pb = self.d_bank.ptr;
                let mut nb = 2i32;
                let mut pc: CUdeviceptr = 0;
                let mut ac = 0i32;
                let mut pq = self.d_all_msg_queries.ptr;
                let mut pout = self.d_msgs.ptr;
                let mut b3 = 0i32;
                let mut args: [*mut c_void; 7] = [
                    (&mut pb as *mut _) as *mut c_void,
                    (&mut nb as *mut _) as *mut c_void,
                    (&mut pc as *mut _) as *mut c_void,
                    (&mut ac as *mut _) as *mut c_void,
                    (&mut pq as *mut _) as *mut c_void,
                    (&mut pout as *mut _) as *mut c_void,
                    (&mut b3 as *mut _) as *mut c_void,
                ];
                self.launch(stream, self.fns.depth_mix, (48, 1), 256, &mut args)?;
            }
            for ip in 6..18 {
                let slot = if pass == 0 {
                    ip
                } else {
                    N_LAYER + (pass - 1) * 12 + (ip - 6)
                };
                let msg_ptr = if pass > 0 {
                    self.d_msgs.ptr + ((ip - 6) * 4) as u64 * d_bytes
                } else {
                    0
                };
                self.enqueue_visit(stream, ip, slot, &mut n_blocks, msg_ptr)?;
            }
        }

        for il in 18..24 {
            self.enqueue_visit(stream, il, il, &mut n_blocks, 0)?;
        }
        debug_assert_eq!(n_blocks, 4);

        if prefill_only {
            if let Some(taps) = &self.prefill.d_taps {
                if let Some(f_mlp_fused) = self.fns.depth_mix_rmsnorm_d_quant_q8_16_b {
                    let mut pb = self.d_blocks.ptr;
                    let mut nb = n_blocks as i32;
                    let mut sb = 1i32;
                    let mut bo = 0i32;
                    let mut pc = self.d_cur.ptr;
                    let mut ac = 1i32;
                    let mut pq = self.output_query.ptr;
                    let mut pg = self.final_norm.ptr;
                    let mut p_tap3 = taps.ptr + (3 * self.prefill.capacity * D * 4) as u64;
                    let mut dq = 0i32;
                    let mut z_q8: CUdeviceptr = 0;
                    let mut z_sc: CUdeviceptr = 0;
                    let mut args: [*mut c_void; 12] = [
                        (&mut pb as *mut _) as *mut c_void,
                        (&mut nb as *mut _) as *mut c_void,
                        (&mut sb as *mut _) as *mut c_void,
                        (&mut bo as *mut _) as *mut c_void,
                        (&mut pc as *mut _) as *mut c_void,
                        (&mut ac as *mut _) as *mut c_void,
                        (&mut pq as *mut _) as *mut c_void,
                        (&mut pg as *mut _) as *mut c_void,
                        (&mut p_tap3 as *mut _) as *mut c_void,
                        (&mut dq as *mut _) as *mut c_void,
                        (&mut z_q8 as *mut _) as *mut c_void,
                        (&mut z_sc as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, f_mlp_fused, (1, 1), 256, &mut args)?;
                } else {
                    let mut pb = self.d_blocks.ptr;
                    let mut nb = n_blocks as i32;
                    let mut pc = self.d_cur.ptr;
                    let mut ac = 1i32;
                    let mut pq = self.output_query.ptr;
                    let mut pout = self.d_mlp_h.ptr;
                    let mut b3 = 0i32;
                    let mut args: [*mut c_void; 7] = [
                        (&mut pb as *mut _) as *mut c_void,
                        (&mut nb as *mut _) as *mut c_void,
                        (&mut pc as *mut _) as *mut c_void,
                        (&mut ac as *mut _) as *mut c_void,
                        (&mut pq as *mut _) as *mut c_void,
                        (&mut pout as *mut _) as *mut c_void,
                        (&mut b3 as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, self.fns.depth_mix, (1, 1), 256, &mut args)?;
                    let mut pin = self.d_mlp_h.ptr;
                    let mut pg = self.final_norm.ptr;
                    let mut p_tap3 = taps.ptr + (3 * self.prefill.capacity * D * 4) as u64;
                    let mut hg = 1i32;
                    let mut rn_args: [*mut c_void; 4] = [
                        (&mut pin as *mut _) as *mut c_void,
                        (&mut pg as *mut _) as *mut c_void,
                        (&mut p_tap3 as *mut _) as *mut c_void,
                        (&mut hg as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, self.fns.rmsnorm_d, (1, 1), 256, &mut rn_args)?;
                }
            }
            let mut pp = self.d_pos.ptr;
            let mut args: [*mut c_void; 1] = [(&mut pp as *mut _) as *mut c_void];
            self.launch(stream, self.fns.inc_pos, (1, 1), 32, &mut args)?;
            return Ok(());
        }

        // Output Depth Mix + Fused `final_norm + Q8_64` + Planar `Q8_64` `unembed` (`softcap = 30.0`) + ArgMax
        if let (true, Some(f_fused)) = (
            self.drafter.is_none(),
            self.fns.depth_mix_rmsnorm_quant_q8_64_b,
        ) {
            let mut pb = self.d_blocks.ptr;
            let mut nb = n_blocks as i32;
            let mut sb = 1i32;
            let mut bo = 0i32;
            let mut bout = 1i32;
            let mut pc = self.d_cur.ptr;
            let mut ac = 1i32;
            let mut pq = self.output_query.ptr;
            let mut pg = self.final_norm.ptr;
            let mut pq8 = self.d_act_q8_64.ptr;
            let mut psc = self.d_act_sc_64.ptr;
            let mut args: [*mut c_void; 11] = [
                (&mut pb as *mut _) as *mut c_void,
                (&mut nb as *mut _) as *mut c_void,
                (&mut sb as *mut _) as *mut c_void,
                (&mut bo as *mut _) as *mut c_void,
                (&mut bout as *mut _) as *mut c_void,
                (&mut pc as *mut _) as *mut c_void,
                (&mut ac as *mut _) as *mut c_void,
                (&mut pq as *mut _) as *mut c_void,
                (&mut pg as *mut _) as *mut c_void,
                (&mut pq8 as *mut _) as *mut c_void,
                (&mut psc as *mut _) as *mut c_void,
            ];
            self.launch(stream, f_fused, (1, 1), 256, &mut args)?;
        } else {
            let mut pb = self.d_blocks.ptr;
            let mut nb = n_blocks as i32;
            let mut pc = self.d_cur.ptr;
            let mut ac = 1i32;
            let mut pq = self.output_query.ptr;
            let mut pout = self.d_mlp_h.ptr;
            let mut b3 = 0i32;
            let mut args: [*mut c_void; 7] = [
                (&mut pb as *mut _) as *mut c_void,
                (&mut nb as *mut _) as *mut c_void,
                (&mut pc as *mut _) as *mut c_void,
                (&mut ac as *mut _) as *mut c_void,
                (&mut pq as *mut _) as *mut c_void,
                (&mut pout as *mut _) as *mut c_void,
                (&mut b3 as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.depth_mix, (1, 1), 256, &mut args)?;
            if let Some(taps) = &self.prefill.d_taps {
                let mut pin = self.d_mlp_h.ptr;
                let mut pg = self.final_norm.ptr;
                let mut p_tap3 = taps.ptr + (3 * self.prefill.capacity * D * 4) as u64;
                let mut hg = 1i32;
                let mut rn_args: [*mut c_void; 4] = [
                    (&mut pin as *mut _) as *mut c_void,
                    (&mut pg as *mut _) as *mut c_void,
                    (&mut p_tap3 as *mut _) as *mut c_void,
                    (&mut hg as *mut _) as *mut c_void,
                ];
                self.launch(stream, self.fns.rmsnorm_d, (1, 1), 256, &mut rn_args)?;
            }
            self.enqueue_rmsnorm_quant_q8_64(stream, self.d_mlp_h.ptr, self.final_norm.ptr, 1)?;
        }
        self.enqueue_q8_gemv_slice_at(
            stream,
            &self.unembed_q8,
            0,
            VOCAB as i32,
            0,
            self.d_logits.ptr,
            30.0,
        )?;
        {
            let mut pl = self.d_logits.ptr;
            let mut pt = self.d_token_out.ptr;
            let mut pp = self.d_pos.ptr;
            let mut args: [*mut c_void; 3] = [
                (&mut pl as *mut _) as *mut c_void,
                (&mut pt as *mut _) as *mut c_void,
                (&mut pp as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.argmax_and_inc_pos, (1, 1), 256, &mut args)?;
        }

        Ok(())
    }

    fn enqueue_drafter_context(&mut self, rows: usize) -> Result<(), String> {
        if let Some(drafter) = &mut self.drafter {
            let taps = self
                .prefill
                .d_taps
                .as_ref()
                .ok_or("missing DSpark tap storage")?;
            drafter.enqueue_add_ctx(
                self.ctx.compute_stream,
                taps.ptr,
                self.prefill.capacity,
                rows,
            )?;
        }
        Ok(())
    }

    pub fn forward_greedy_token(&mut self, token: u32) -> Result<u32, String> {
        if self.host_pos >= self.max_ctx {
            return Err(format!("sequence exceeded max_ctx ({})", self.max_ctx));
        }
        let tok_i32 = token as i32;
        unsafe {
            check_cu(
                (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                    self.d_token_in.ptr,
                    (&tok_i32 as *const i32).cast(),
                    4,
                    self.ctx.compute_stream,
                ),
                "HtoD(token_in)",
            )?;
        }
        if let Some(ref graph) = self.decode_graph {
            graph.replay()?;
        } else {
            self.enqueue_step_impl(self.ctx.compute_stream, false)?;
        }
        self.enqueue_drafter_context(1)?;
        self.ctx.sync_compute()?;
        let mut out_tok = 0i32;
        unsafe {
            check_cu(
                (self.ctx.api.cuMemcpyDtoH_v2)(
                    (&mut out_tok as *mut i32).cast(),
                    self.d_token_out.ptr,
                    4,
                ),
                "DtoH(token_out)",
            )?;
        }
        self.host_pos += 1;
        Ok(out_tok as u32)
    }

    pub fn read_last_logits(&self) -> Result<Vec<f32>, String> {
        self.ctx.sync_compute()?;
        let mut host = vec![0.0f32; VOCAB];
        self.d_logits.copy_to_host_f32(&mut host)?;
        Ok(host)
    }

    pub fn prefill_tokens(&mut self, tokens: &[u32]) -> Result<(u32, f64), String> {
        assert!(!tokens.is_empty());
        if self.host_pos + tokens.len() > self.max_ctx {
            return Err(format!("prefill exceeded max_ctx ({})", self.max_ctx));
        }
        let t0 = Instant::now();

        if std::env::var("LEANSONIC_SERIAL_PREFILL")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            let n_prefix = tokens.len() - 1;
            if n_prefix > 0 {
                unsafe {
                    check_cu(
                        (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                            self.d_prefill_tokens.ptr + (self.host_pos * 4) as u64,
                            tokens.as_ptr().cast(),
                            n_prefix * 4,
                            self.ctx.compute_stream,
                        ),
                        "HtoD(prefill_tokens)",
                    )?;
                }
                for _ in 0..n_prefix {
                    if let Some(ref p_graph) = self.prefill_graph {
                        p_graph.replay()?;
                    } else {
                        self.enqueue_step_impl(self.ctx.compute_stream, true)?;
                    }
                    self.enqueue_drafter_context(1)?;
                }
                self.host_pos += n_prefix;
            }
            let next_tok = self.forward_greedy_token(tokens[n_prefix])?;
            return Ok((next_tok, t0.elapsed().as_secs_f64() * 1000.0));
        }

        // Fast Single-Token Shortcut when `tokens.len() == 1`
        if tokens.len() == 1 {
            let next_tok = self.forward_greedy_token(tokens[0])?;
            return Ok((next_tok, t0.elapsed().as_secs_f64() * 1000.0));
        }

        unsafe {
            check_cu(
                (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                    self.d_prefill_tokens.ptr + (self.host_pos * 4) as u64,
                    tokens.as_ptr().cast(),
                    tokens.len() * 4,
                    self.ctx.compute_stream,
                ),
                "HtoD(prefill_tokens_batch)",
            )?;
        }

        let mut offset = 0usize;
        let mut next_tok = 0u32;
        while offset < tokens.len() {
            let rem = tokens.len() - offset;
            let chunk_len = if rem > 1 && rem <= self.prefill.capacity && rem % EXPERT_PAGE == 1 {
                rem - 1
            } else {
                rem.min(self.prefill.capacity)
            };
            let is_last = offset + chunk_len == tokens.len();

            if is_last && chunk_len == 1 {
                next_tok = self.forward_greedy_token(tokens[offset])?;
                break;
            }

            let tok_ptr = self.d_prefill_tokens.ptr + ((self.host_pos) * 4) as u64;
            let head_mode = if is_last {
                BatchHeadMode::LastOnly
            } else {
                BatchHeadMode::None
            };
            self.enqueue_batch_step_impl(
                self.ctx.compute_stream,
                tok_ptr,
                chunk_len,
                head_mode,
                false,
            )?;
            self.enqueue_drafter_context(chunk_len)?;
            self.host_pos += chunk_len;
            offset += chunk_len;

            if is_last {
                self.ctx.sync_compute()?;
                let mut out_tok = 0i32;
                unsafe {
                    check_cu(
                        (self.ctx.api.cuMemcpyDtoH_v2)(
                            (&mut out_tok as *mut i32).cast(),
                            self.d_token_out.ptr,
                            4,
                        ),
                        "DtoH(batch_token_out)",
                    )?;
                }
                next_tok = out_tok as u32;
            }
        }

        Ok((next_tok, t0.elapsed().as_secs_f64() * 1000.0))
    }

    /// Unified Multi-Token Verification Step (`1 <= tokens.len() <= MAX_SPEC_TOKENS`):
    /// Computes greedy argmax predictions for ALL `B` input tokens in one forward pass!
    pub fn forward_verify_batch_tokens(&mut self, tokens: &[u32]) -> Result<Vec<u32>, String> {
        assert!(!tokens.is_empty() && tokens.len() <= MAX_SPEC_TOKENS);
        if self.host_pos + tokens.len() > self.max_ctx {
            return Err(format!("verify batch exceeded max_ctx ({})", self.max_ctx));
        }
        let b_len = tokens.len();
        let tok_ptr = self.d_prefill_tokens.ptr + (self.host_pos * 4) as u64;
        unsafe {
            check_cu(
                (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                    tok_ptr,
                    tokens.as_ptr().cast(),
                    b_len * 4,
                    self.ctx.compute_stream,
                ),
                "HtoD(verify_tokens)",
            )?;
        }
        self.enqueue_batch_step_impl(
            self.ctx.compute_stream,
            tok_ptr,
            b_len,
            BatchHeadMode::AllHeads,
            self.spec_k > 0,
        )?;
        self.ctx.sync_compute()?;
        self.host_pos += b_len;
        let mut out_i32 = vec![0i32; b_len];
        unsafe {
            check_cu(
                (self.ctx.api.cuMemcpyDtoH_v2)(
                    out_i32.as_mut_ptr().cast(),
                    self.d_spec_tokens_out.ptr,
                    b_len * 4,
                ),
                "DtoH(spec_tokens_out)",
            )?;
        }
        Ok(out_i32.into_iter().map(|v| v as u32).collect())
    }

    /// `O(1)` Rewind of all 48 slots' `short_conv4` histories + `*d_pos` / `host_pos` (`keep < b_len`).
    pub fn rewind_speculative(&mut self, keep: usize, b_len: usize) -> Result<(), String> {
        if keep < b_len {
            let mut pseq = self.d_conv_seq.ptr;
            let mut phist = self.d_conv_hist_ptrs.ptr;
            let mut ppos = self.d_pos.ptr;
            let mut kp = keep as i32;
            let mut bi = b_len as i32;
            let mut args: [*mut c_void; 5] = [
                (&mut pseq as *mut _) as *mut c_void,
                (&mut phist as *mut _) as *mut c_void,
                (&mut ppos as *mut _) as *mut c_void,
                (&mut kp as *mut _) as *mut c_void,
                (&mut bi as *mut _) as *mut c_void,
            ];
            self.launch(
                self.ctx.compute_stream,
                self.fns.conv_rewind,
                (9, (N_SLOT * 4) as u32),
                256,
                &mut args,
            )?;
            self.host_pos -= b_len - keep;
        }
        Ok(())
    }

    /// Unified Greedy Generation Loop (`spec_k == 0` -> CUDA Graph Solo; `spec_k > 0` -> DSpark/Suffix + Batch Verify).
    /// Guarantees `0-ULP` lossless greedy token identity (`spec_k == 0` == `spec_k > 0`)!
    pub fn generate_greedy_tokens(
        &mut self,
        first_next: u32,
        n_gen: usize,
        stop_tok: Option<u32>,
        prompt_tokens: &[u32],
        mut on_token: impl FnMut(u32),
    ) -> Result<(Vec<u32>, u32), String> {
        if n_gen == 0 {
            return Ok((Vec::new(), first_next));
        }

        let mut out_ids = Vec::with_capacity(n_gen);
        let mut cur_tok = first_next;

        if self.spec_k == 0 {
            for step in 0..n_gen {
                if Some(cur_tok) == stop_tok {
                    break;
                }
                out_ids.push(cur_tok);
                on_token(cur_tok);
                if stop_tok.is_some() && out_ids.len() >= n_gen {
                    break;
                }
                if step + 1 < n_gen || stop_tok.is_none() {
                    cur_tok = self.forward_greedy_token(cur_tok)?;
                }
            }
            return Ok((out_ids, cur_tok));
        }

        // Speculative Decoding (`DSpark` `mtp.safetensors` GPU Drafter or `SuffixDrafter` fallback)
        let mut full_hist: Vec<u32> = if self.drafter.is_none() {
            let mut h = Vec::with_capacity(prompt_tokens.len() + n_gen + 8);
            h.extend_from_slice(prompt_tokens);
            h
        } else {
            Vec::new()
        };

        let mut steps_done = 0usize;
        while steps_done < n_gen {
            if Some(cur_tok) == stop_tok {
                break;
            }
            let rem_steps = n_gen - steps_done;
            let max_draft = self
                .spec_k
                .min(rem_steps.saturating_sub(1))
                .min(self.max_ctx.saturating_sub(self.host_pos + 1));

            let drafts = if max_draft == 0 {
                Vec::new()
            } else if let Some(ref mut d) = self.drafter {
                d.draft_tokens(
                    self.ctx.compute_stream,
                    self.embed.ptr(),
                    self.embed_norm.ptr,
                    &self.unembed_q8,
                    cur_tok,
                    max_draft,
                )?
            } else {
                SuffixDrafter::draft_from_history(&full_hist, cur_tok, max_draft)
            };

            if drafts.is_empty() {
                out_ids.push(cur_tok);
                on_token(cur_tok);
                if self.drafter.is_none() {
                    full_hist.push(cur_tok);
                }
                steps_done += 1;
                if stop_tok.is_some() && out_ids.len() >= n_gen {
                    break;
                }
                cur_tok = self.forward_greedy_token(cur_tok)?;
                continue;
            }

            let k_len = drafts.len();
            let b_len = k_len + 1;
            let mut verify_in = Vec::with_capacity(b_len);
            verify_in.push(cur_tok);
            verify_in.extend_from_slice(&drafts);

            let preds = self.forward_verify_batch_tokens(&verify_in)?;
            self.spec_verify_steps += 1;
            self.spec_drafted_total += k_len;

            let mut accepted = 0usize;
            while accepted < k_len {
                if stop_tok.is_some() && Some(verify_in[accepted]) == stop_tok {
                    break;
                }
                if preds[accepted] == drafts[accepted] {
                    accepted += 1;
                } else {
                    break;
                }
            }
            self.spec_accepted_total += accepted;
            let keep = accepted + 1; // `1 <= keep <= b_len`

            if keep < b_len {
                self.rewind_speculative(keep, b_len)?;
            }
            self.enqueue_drafter_context(keep)?;

            let mut hit_stop = false;
            for &accepted_tok in &verify_in[..keep] {
                if Some(accepted_tok) == stop_tok {
                    cur_tok = accepted_tok;
                    hit_stop = true;
                    break;
                }
                if out_ids.len() < n_gen {
                    out_ids.push(accepted_tok);
                    on_token(accepted_tok);
                    if self.drafter.is_none() {
                        full_hist.push(accepted_tok);
                    }
                }
            }
            steps_done += keep;
            if hit_stop || (stop_tok.is_some() && out_ids.len() >= n_gen) {
                break;
            }
            cur_tok = preds[accepted];
        }

        Ok((out_ids, cur_tok))
    }

    fn enqueue_copy_n(
        &self,
        stream: CUstream,
        src: CUdeviceptr,
        dst: CUdeviceptr,
        n_floats: usize,
    ) -> Result<(), String> {
        let mut p_src = src;
        let mut p_dst = dst;
        let mut n = n_floats as i32;
        let mut args: [*mut c_void; 3] = [
            (&mut p_src as *mut _) as *mut c_void,
            (&mut p_dst as *mut _) as *mut c_void,
            (&mut n as *mut _) as *mut c_void,
        ];
        let grid_x = (n_floats as u32).div_ceil(256);
        self.launch(stream, self.fns.copy_d, (grid_x, 1), 256, &mut args)
    }

    fn enqueue_ctrl_n_b(
        &self,
        stream: CUstream,
        cur_ptr: CUdeviceptr,
        msg_ptr: CUdeviceptr,
        ctrl: &CtrlGpu,
        n_ctrls: u32,
        b_len: usize,
    ) -> Result<(), String> {
        let b_u32 = b_len as u32;
        let mut b_i32 = b_len as i32;
        let mut p_cur = cur_ptr;
        let mut p_msg = msg_ptr;
        let mut p_down = ctrl.down.ptr;
        let mut p_g = self.prefill.d_b_ctrl_g.ptr;
        let mut args1: [*mut c_void; 5] = [
            (&mut p_cur as *mut _) as *mut c_void,
            (&mut p_msg as *mut _) as *mut c_void,
            (&mut p_down as *mut _) as *mut c_void,
            (&mut p_g as *mut _) as *mut c_void,
            (&mut b_i32 as *mut _) as *mut c_void,
        ];
        // A short batch has too few blocks to amortize an eight-wide tile.
        // Optional symbols also support previously built external cubins.
        let (down, rank_blocks) = match self.fns.ctrl_down_r8_b {
            Some(f) if b_len >= 16 => (f, 8),
            _ => (self.fns.ctrl_down_b, 64),
        };
        self.launch(
            stream,
            down,
            (rank_blocks, n_ctrls * b_u32),
            256,
            &mut args1,
        )?;

        let mut p_gain = ctrl.gain.ptr;
        let mut p_forget = ctrl.forget.ptr;
        let mut args2: [*mut c_void; 6] = [
            (&mut p_cur as *mut _) as *mut c_void,
            (&mut p_msg as *mut _) as *mut c_void,
            (&mut p_g as *mut _) as *mut c_void,
            (&mut p_gain as *mut _) as *mut c_void,
            (&mut p_forget as *mut _) as *mut c_void,
            (&mut b_i32 as *mut _) as *mut c_void,
        ];
        if b_len >= 16 {
            if let Some(f_t8) = self.fns.ctrl_apply_t8_b {
                return self.launch(
                    stream,
                    f_t8,
                    (9, n_ctrls * b_u32.div_ceil(8)),
                    256,
                    &mut args2,
                );
            }
        }
        if let Some(f_coal) = self.fns.ctrl_apply_coalesced_b {
            self.launch(stream, f_coal, (36, n_ctrls * b_u32), 64, &mut args2)
        } else {
            self.launch(
                stream,
                self.fns.ctrl_apply_b,
                (9, n_ctrls * b_u32),
                256,
                &mut args2,
            )
        }
    }

    fn enqueue_rmsnorm_quant_q8_64_b(
        &self,
        stream: CUstream,
        inp_ptr: CUdeviceptr,
        gamma_ptr: CUdeviceptr,
        count: u32,
        b_len: usize,
    ) -> Result<(), String> {
        let mut pin = inp_ptr;
        let mut pg = gamma_ptr;
        let mut pq = self.prefill.d_b_act_codes.ptr;
        let mut ps = self.prefill.d_b_act_scales.ptr;
        let mut b_i32 = b_len as i32;
        let mut args: [*mut c_void; 5] = [
            (&mut pin as *mut _) as *mut c_void,
            (&mut pg as *mut _) as *mut c_void,
            (&mut pq as *mut _) as *mut c_void,
            (&mut ps as *mut _) as *mut c_void,
            (&mut b_i32 as *mut _) as *mut c_void,
        ];
        self.launch(
            stream,
            self.fns.rmsnorm_quant_q8_64_b,
            (b_len as u32, count),
            256,
            &mut args,
        )
    }

    fn enqueue_quant_q8_64_b(
        &self,
        stream: CUstream,
        x_ptr: CUdeviceptr,
        cols: usize,
        b_len: usize,
    ) -> Result<(), String> {
        let mut px = x_ptr;
        let mut pq = self.prefill.d_b_act_codes.ptr;
        let mut ps = self.prefill.d_b_act_scales.ptr;
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

    fn enqueue_quant_q8_16_b(
        &self,
        stream: CUstream,
        x_ptr: CUdeviceptr,
        n_vecs: usize,
        cols: usize,
    ) -> Result<(), String> {
        self.enqueue_quant_q8_groups(
            stream,
            x_ptr,
            self.prefill.d_b_act_codes.ptr,
            self.prefill.d_b_act_scales.ptr,
            n_vecs,
            cols,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_q8_gemm_slice_at(
        &self,
        stream: CUstream,
        w: &PlanarQ8Gpu,
        row_offset: usize,
        rows: i32,
        act_slot: usize,
        b_len: usize,
        y_ptr: CUdeviceptr,
        y_stride: i32,
        softcap: f32,
    ) -> Result<(), String> {
        let cols = w.cols as usize;
        assert!(
            cols > 0 && cols.is_multiple_of(64),
            "ink2_q8_dp4a_gemm_b requires positive cols divisible by 64, got {cols}"
        );
        let mut pw = w.codes.ptr + (row_offset * cols) as u64;
        let mut pws = w.scales64.ptr + (row_offset * (cols / 64) * 4) as u64;
        let mut px = self.prefill.d_b_act_codes.ptr + (act_slot * b_len * cols) as u64;
        let mut pxs = self.prefill.d_b_act_scales.ptr + (act_slot * b_len * (cols / 64) * 4) as u64;
        let mut py = y_ptr;
        let mut r = rows;
        let mut c = w.cols;
        let mut b_i32 = b_len as i32;
        let mut ys = y_stride;
        let mut sc = softcap;
        let mut args: [*mut c_void; 10] = [
            (&mut pw as *mut _) as *mut c_void,
            (&mut pws as *mut _) as *mut c_void,
            (&mut px as *mut _) as *mut c_void,
            (&mut pxs as *mut _) as *mut c_void,
            (&mut py as *mut _) as *mut c_void,
            (&mut r as *mut _) as *mut c_void,
            (&mut c as *mut _) as *mut c_void,
            (&mut b_i32 as *mut _) as *mut c_void,
            (&mut ys as *mut _) as *mut c_void,
            (&mut sc as *mut _) as *mut c_void,
        ];
        let grid_x = (rows as u32).div_ceil(16);
        let grid_y = (b_len as u32).div_ceil(8);
        self.launch(
            stream,
            self.fns.q8_dp4a_gemm_b,
            (grid_x, grid_y),
            256,
            &mut args,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_batch_visit(
        &self,
        stream: CUstream,
        il: usize,
        slot_idx: usize,
        b_len: usize,
        n_blocks: &mut usize,
        msg_base_ptr: CUdeviceptr,
        record_seq: bool,
    ) -> Result<(), String> {
        let l = &self.layers[il];
        let st = &self.slots[slot_idx];
        let b_u32 = b_len as u32;
        let b_i32 = b_len as i32;
        let bd_floats = b_len * D;
        let bd_bytes = (bd_floats * 4) as u64;
        let has_msgs = msg_base_ptr != 0;
        let distinct_k_v = l.global || has_msgs;
        let norm_count: u32 = if distinct_k_v { 3 } else { 1 };
        let slot_seq_base = if record_seq {
            self.d_conv_seq.ptr + (slot_idx * CONV_SEQ_SLOT_FLOATS * 4) as u64
        } else {
            0
        };

        // 1–3. Batched Attention inputs `[xq, xk, xv]` + optional Loop Controller + `RMSNorm + Q8_64`!
        if !has_msgs {
            if let Some(ref aq) = l.attn_query {
                let n_q: u32 = if l.global { 3 } else { 1 };
                if let Some(f_fused) = self.fns.depth_mix_rmsnorm_quant_q8_64_b {
                    let mut pb = self.prefill.d_b_blocks.ptr;
                    let mut nb = *n_blocks as i32;
                    let mut sb = b_i32;
                    let mut bo = 0i32;
                    let mut bout = b_i32;
                    let mut pc = self.prefill.d_b_cur.ptr;
                    let mut ac = 1i32;
                    let mut pq = aq.ptr;
                    let mut pg = l.norms_packed.ptr;
                    let mut pq8 = self.prefill.d_b_act_codes.ptr;
                    let mut psc = self.prefill.d_b_act_scales.ptr;
                    let mut args: [*mut c_void; 11] = [
                        (&mut pb as *mut _) as *mut c_void,
                        (&mut nb as *mut _) as *mut c_void,
                        (&mut sb as *mut _) as *mut c_void,
                        (&mut bo as *mut _) as *mut c_void,
                        (&mut bout as *mut _) as *mut c_void,
                        (&mut pc as *mut _) as *mut c_void,
                        (&mut ac as *mut _) as *mut c_void,
                        (&mut pq as *mut _) as *mut c_void,
                        (&mut pg as *mut _) as *mut c_void,
                        (&mut pq8 as *mut _) as *mut c_void,
                        (&mut psc as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, f_fused, (n_q, b_u32), 256, &mut args)?;
                } else {
                    let mut pb = self.prefill.d_b_blocks.ptr;
                    let mut nb = *n_blocks as i32;
                    let mut sb = b_i32;
                    let mut bo = 0i32;
                    let mut bout = b_i32;
                    let mut pc = self.prefill.d_b_cur.ptr;
                    let mut ac = 1i32;
                    let mut pq = aq.ptr;
                    let mut pout = self.prefill.d_b_activations.ptr;
                    let mut b3 = 0i32;
                    let mut args: [*mut c_void; 10] = [
                        (&mut pb as *mut _) as *mut c_void,
                        (&mut nb as *mut _) as *mut c_void,
                        (&mut sb as *mut _) as *mut c_void,
                        (&mut bo as *mut _) as *mut c_void,
                        (&mut bout as *mut _) as *mut c_void,
                        (&mut pc as *mut _) as *mut c_void,
                        (&mut ac as *mut _) as *mut c_void,
                        (&mut pq as *mut _) as *mut c_void,
                        (&mut pout as *mut _) as *mut c_void,
                        (&mut b3 as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, self.fns.depth_mix_b, (n_q, b_u32), 256, &mut args)?;
                    self.enqueue_rmsnorm_quant_q8_64_b(
                        stream,
                        self.prefill.d_b_activations.ptr,
                        l.norms_packed.ptr,
                        norm_count,
                        b_len,
                    )?;
                }
            } else {
                // Layer 0 (`il == 0`): `!l.global && !has_msgs`, quantize `d_b_cur` directly (`0` `copy_d` launches!)
                debug_assert!(!distinct_k_v);
                self.enqueue_rmsnorm_quant_q8_64_b(
                    stream,
                    self.prefill.d_b_cur.ptr,
                    l.norms_packed.ptr,
                    1,
                    b_len,
                )?;
            }
        } else {
            let aq = l.attn_query.as_ref().unwrap();
            let n_q: u32 = if l.global { 3 } else { 1 };
            let mut pb = self.prefill.d_b_blocks.ptr;
            let mut nb = *n_blocks as i32;
            let mut sb = b_i32;
            let mut bo = 0i32;
            let mut bout = b_i32;
            let mut pc = self.prefill.d_b_cur.ptr;
            let mut ac = 1i32;
            let mut pq = aq.ptr;
            let mut pout = self.prefill.d_b_activations.ptr;
            let mut b3 = i32::from(!l.global);
            let mut args: [*mut c_void; 10] = [
                (&mut pb as *mut _) as *mut c_void,
                (&mut nb as *mut _) as *mut c_void,
                (&mut sb as *mut _) as *mut c_void,
                (&mut bo as *mut _) as *mut c_void,
                (&mut bout as *mut _) as *mut c_void,
                (&mut pc as *mut _) as *mut c_void,
                (&mut ac as *mut _) as *mut c_void,
                (&mut pq as *mut _) as *mut c_void,
                (&mut pout as *mut _) as *mut c_void,
                (&mut b3 as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.depth_mix_b, (n_q, b_u32), 256, &mut args)?;
            self.enqueue_ctrl_n_b(
                stream,
                self.prefill.d_b_activations.ptr,
                msg_base_ptr,
                l.ctrl3.as_ref().unwrap(),
                3,
                b_len,
            )?;
            self.enqueue_rmsnorm_quant_q8_64_b(
                stream,
                self.prefill.d_b_activations.ptr,
                l.norms_packed.ptr,
                norm_count,
                b_len,
            )?;
        }

        let k_raw_ptr: CUdeviceptr;
        let k_in_stride: i32;
        let v_raw_ptr: CUdeviceptr;
        let v_in_stride: i32;

        if !distinct_k_v {
            self.enqueue_q8_gemm_slice_at(
                stream,
                &l.wq_all_q8,
                0,
                SWA_ALL_ROWS as i32,
                0,
                b_len,
                self.prefill.layer_workspace.qrgs,
                SWA_ALL_ROWS as i32,
                0.0,
            )?;
            k_raw_ptr = self.prefill.layer_workspace.qrgs + (SWA_K_OFFSET as u64) * 4;
            k_in_stride = SWA_ALL_ROWS as i32;
            v_raw_ptr = self.prefill.layer_workspace.qrgs + (SWA_V_OFFSET as u64) * 4;
            v_in_stride = SWA_ALL_ROWS as i32;
        } else if l.global {
            self.enqueue_q8_gemm_slice_at(
                stream,
                &l.wq_all_q8,
                0,
                QRGS_ROWS as i32,
                0,
                b_len,
                self.prefill.layer_workspace.qrgs,
                SWA_ALL_ROWS as i32,
                0.0,
            )?;
            self.enqueue_q8_gemm_slice_at(
                stream,
                l.wk_glob_q8.as_ref().unwrap(),
                0,
                1056,
                1,
                b_len,
                self.prefill.layer_workspace.k_raw,
                1056,
                0.0,
            )?;
            self.enqueue_q8_gemm_slice_at(
                stream,
                l.wv_glob_q8.as_ref().unwrap(),
                0,
                1056,
                2,
                b_len,
                self.prefill.layer_workspace.v_raw,
                1056,
                0.0,
            )?;
            k_raw_ptr = self.prefill.layer_workspace.k_raw;
            k_in_stride = 1056;
            v_raw_ptr = self.prefill.layer_workspace.v_raw;
            v_in_stride = 1056;
        } else {
            self.enqueue_q8_gemm_slice_at(
                stream,
                &l.wq_all_q8,
                0,
                QRGS_ROWS as i32,
                0,
                b_len,
                self.prefill.layer_workspace.qrgs,
                SWA_ALL_ROWS as i32,
                0.0,
            )?;
            self.enqueue_q8_gemm_slice_at(
                stream,
                &l.wq_all_q8,
                SWA_K_OFFSET,
                1024,
                1,
                b_len,
                self.prefill.layer_workspace.k_raw,
                1024,
                0.0,
            )?;
            self.enqueue_q8_gemm_slice_at(
                stream,
                &l.wq_all_q8,
                SWA_V_OFFSET,
                1024,
                2,
                b_len,
                self.prefill.layer_workspace.v_raw,
                1024,
                0.0,
            )?;
            k_raw_ptr = self.prefill.layer_workspace.k_raw;
            k_in_stride = 1024;
            v_raw_ptr = self.prefill.layer_workspace.v_raw;
            v_in_stride = 1024;
        }

        // 4. Register-Persistent Causal `short_conv4` for K & V
        for (raw_ptr, raw_stride, conv_w, hist_ptr, dst_ptr, seq_off_floats) in [
            (
                k_raw_ptr,
                k_in_stride,
                l.k_conv4.ptr,
                st.hist_k,
                self.prefill.layer_workspace.k_conv,
                0usize,
            ),
            (
                v_raw_ptr,
                v_in_stride,
                l.v_conv4.ptr,
                st.hist_v,
                self.prefill.layer_workspace.v_conv,
                1024 * CONV_SEQ_STEPS,
            ),
        ] {
            let mut p_cur = raw_ptr;
            let mut ins = raw_stride;
            let mut p_w = conv_w;
            let mut p_dyn = if l.global { raw_ptr + 1024 * 4 } else { 0 };
            let mut df4 = 264i32; // 1056 / 4
            let mut p_hist = hist_ptr;
            let mut p_out = dst_ptr;
            let mut outs = 1024i32;
            let mut p_res: CUdeviceptr = 0;
            let mut ress = 0i32;
            let mut p_seq = if record_seq {
                slot_seq_base + (seq_off_floats * 4) as u64
            } else {
                0
            };
            let mut ch = 1024i32;
            let mut bi = b_i32;
            let mut hd = i32::from(l.global);
            let mut dr = 0i32;
            let mut args: [*mut c_void; 15] = [
                (&mut p_cur as *mut _) as *mut c_void,
                (&mut ins as *mut _) as *mut c_void,
                (&mut p_w as *mut _) as *mut c_void,
                (&mut p_dyn as *mut _) as *mut c_void,
                (&mut df4 as *mut _) as *mut c_void,
                (&mut p_hist as *mut _) as *mut c_void,
                (&mut p_out as *mut _) as *mut c_void,
                (&mut outs as *mut _) as *mut c_void,
                (&mut p_res as *mut _) as *mut c_void,
                (&mut ress as *mut _) as *mut c_void,
                (&mut p_seq as *mut _) as *mut c_void,
                (&mut ch as *mut _) as *mut c_void,
                (&mut bi as *mut _) as *mut c_void,
                (&mut hd as *mut _) as *mut c_void,
                (&mut dr as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.short_conv4_b, (4, 1), 256, &mut args)?;
        }

        // 5. Batched Q/R/K Clip + V RMSNorm + KV Cache Write (Ring-Hazard-Free via `1088 >= 1024 + 64`!)
        {
            let mut pqrgs = self.prefill.layer_workspace.qrgs;
            let mut qstride = SWA_ALL_ROWS as i32;
            let mut pk = self.prefill.layer_workspace.k_conv;
            let mut pv = self.prefill.layer_workspace.v_conv;
            let mut pqc = l.q_clip.ptr;
            let mut prc = l.rel_clip.ptr;
            let mut pkc = st.k_cache.ptr;
            let mut pvc = st.v_cache.ptr;
            let mut ppos = self.d_pos.ptr;
            let mut cap = st.kv_cap as i32;
            let mut ig = i32::from(l.global);
            let mut args: [*mut c_void; 11] = [
                (&mut pqrgs as *mut _) as *mut c_void,
                (&mut qstride as *mut _) as *mut c_void,
                (&mut pk as *mut _) as *mut c_void,
                (&mut pv as *mut _) as *mut c_void,
                (&mut pqc as *mut _) as *mut c_void,
                (&mut prc as *mut _) as *mut c_void,
                (&mut pkc as *mut _) as *mut c_void,
                (&mut pvc as *mut _) as *mut c_void,
                (&mut ppos as *mut _) as *mut c_void,
                (&mut cap as *mut _) as *mut c_void,
                (&mut ig as *mut _) as *mut c_void,
            ];
            self.launch(
                stream,
                self.fns.prep_qkv_and_cache_b,
                (5, b_u32),
                256,
                &mut args,
            )?;
        }

        // 6. Batched GQA Relative-Bias + Sink Attention + In-Register `Q8_64` Epilogue!
        if self.attn_split_g == 0 {
            if let Some(f_attn_q8) = self.fns.gqa_rel_sink_attn_q8_b {
                let mut pqrgs = self.prefill.layer_workspace.qrgs;
                let mut qstride = SWA_ALL_ROWS as i32;
                let mut prp = l.rel_proj.ptr;
                let mut pkc = st.k_cache.ptr;
                let mut pvc = st.v_cache.ptr;
                let mut ppos = self.d_pos.ptr;
                let mut cap = st.kv_cap as i32;
                let mut ig = i32::from(l.global);
                let mut pq8 = self.prefill.d_b_act_codes.ptr;
                let mut psc = self.prefill.d_b_act_scales.ptr;
                let mut args: [*mut c_void; 10] = [
                    (&mut pqrgs as *mut _) as *mut c_void,
                    (&mut qstride as *mut _) as *mut c_void,
                    (&mut prp as *mut _) as *mut c_void,
                    (&mut pkc as *mut _) as *mut c_void,
                    (&mut pvc as *mut _) as *mut c_void,
                    (&mut ppos as *mut _) as *mut c_void,
                    (&mut cap as *mut _) as *mut c_void,
                    (&mut ig as *mut _) as *mut c_void,
                    (&mut pq8 as *mut _) as *mut c_void,
                    (&mut psc as *mut _) as *mut c_void,
                ];
                self.launch(stream, f_attn_q8, (24, b_u32), 256, &mut args)?;
            } else {
                let mut pqrgs = self.prefill.layer_workspace.qrgs;
                let mut qstride = SWA_ALL_ROWS as i32;
                let mut prp = l.rel_proj.ptr;
                let mut pkc = st.k_cache.ptr;
                let mut pvc = st.v_cache.ptr;
                let mut ppos = self.d_pos.ptr;
                let mut cap = st.kv_cap as i32;
                let mut ig = i32::from(l.global);
                let mut pout = self.prefill.layer_workspace.attn_out;
                let mut args: [*mut c_void; 9] = [
                    (&mut pqrgs as *mut _) as *mut c_void,
                    (&mut qstride as *mut _) as *mut c_void,
                    (&mut prp as *mut _) as *mut c_void,
                    (&mut pkc as *mut _) as *mut c_void,
                    (&mut pvc as *mut _) as *mut c_void,
                    (&mut ppos as *mut _) as *mut c_void,
                    (&mut cap as *mut _) as *mut c_void,
                    (&mut ig as *mut _) as *mut c_void,
                    (&mut pout as *mut _) as *mut c_void,
                ];
                self.launch(
                    stream,
                    self.fns.gqa_rel_sink_attn_b,
                    (24, b_u32),
                    256,
                    &mut args,
                )?;
                self.enqueue_quant_q8_64_b(
                    stream,
                    self.prefill.layer_workspace.attn_out,
                    3072,
                    b_len,
                )?;
            }
        } else {
            self.enqueue_gqa_attn_split_impl(
                stream,
                self.prefill.layer_workspace.qrgs,
                SWA_ALL_ROWS as i32,
                l.rel_proj.ptr,
                st.k_cache.ptr,
                st.v_cache.ptr,
                st.kv_cap as i32,
                i32::from(l.global),
                self.prefill.layer_workspace.attn_out,
                self.prefill.d_b_act_codes.ptr,
                self.prefill.d_b_act_scales.ptr,
                b_len,
            )?;
        }

        // 7. `wo` Batched `Q8_64` GEMM + `attn_sconv` (+ fused block-start save!)
        self.enqueue_q8_gemm_slice_at(
            stream,
            &l.wo_q8,
            0,
            D as i32,
            0,
            b_len,
            self.prefill.layer_workspace.wo_out,
            D as i32,
            0.0,
        )?;

        let is_block_start = il.is_multiple_of(6);
        let save_blk_ptr = if is_block_start {
            let blk_dst = self.prefill.d_b_blocks.ptr + (*n_blocks as u64) * bd_bytes;
            *n_blocks += 1;
            blk_dst
        } else {
            0
        };
        if let Some(f_sc_save_b) = self.fns.short_conv4_save_b {
            let mut p_cur = self.prefill.layer_workspace.wo_out;
            let mut ins = D as i32;
            let mut p_w = l.attn_conv4.ptr;
            let mut p_dyn: CUdeviceptr = 0;
            let mut df4 = 0i32;
            let mut p_hist = st.hist_attn;
            let mut p_out = self.prefill.d_b_cur.ptr;
            let mut outs = D as i32;
            let mut p_res = if is_block_start {
                0
            } else {
                self.prefill.d_b_cur.ptr
            };
            let mut ress = D as i32;
            let mut p_save = save_blk_ptr;
            let mut p_seq = if record_seq {
                slot_seq_base + (2048 * CONV_SEQ_STEPS * 4) as u64
            } else {
                0
            };
            let mut ch = D as i32;
            let mut bi = b_i32;
            let mut hd = 0i32;
            let mut dr = i32::from(!is_block_start);
            let mut args: [*mut c_void; 16] = [
                (&mut p_cur as *mut _) as *mut c_void,
                (&mut ins as *mut _) as *mut c_void,
                (&mut p_w as *mut _) as *mut c_void,
                (&mut p_dyn as *mut _) as *mut c_void,
                (&mut df4 as *mut _) as *mut c_void,
                (&mut p_hist as *mut _) as *mut c_void,
                (&mut p_out as *mut _) as *mut c_void,
                (&mut outs as *mut _) as *mut c_void,
                (&mut p_res as *mut _) as *mut c_void,
                (&mut ress as *mut _) as *mut c_void,
                (&mut p_save as *mut _) as *mut c_void,
                (&mut p_seq as *mut _) as *mut c_void,
                (&mut ch as *mut _) as *mut c_void,
                (&mut bi as *mut _) as *mut c_void,
                (&mut hd as *mut _) as *mut c_void,
                (&mut dr as *mut _) as *mut c_void,
            ];
            self.launch(stream, f_sc_save_b, (9, 1), 256, &mut args)?;
        } else {
            if save_blk_ptr != 0 {
                self.enqueue_copy_n(stream, self.prefill.d_b_cur.ptr, save_blk_ptr, bd_floats)?;
            }
            let mut p_cur = self.prefill.layer_workspace.wo_out;
            let mut ins = D as i32;
            let mut p_w = l.attn_conv4.ptr;
            let mut p_dyn: CUdeviceptr = 0;
            let mut df4 = 0i32;
            let mut p_hist = st.hist_attn;
            let mut p_out = self.prefill.d_b_cur.ptr;
            let mut outs = D as i32;
            let mut p_res = if is_block_start {
                0
            } else {
                self.prefill.d_b_cur.ptr
            };
            let mut ress = D as i32;
            let mut p_seq = if record_seq {
                slot_seq_base + (2048 * CONV_SEQ_STEPS * 4) as u64
            } else {
                0
            };
            let mut ch = D as i32;
            let mut bi = b_i32;
            let mut hd = 0i32;
            let mut dr = i32::from(!is_block_start);
            let mut args: [*mut c_void; 15] = [
                (&mut p_cur as *mut _) as *mut c_void,
                (&mut ins as *mut _) as *mut c_void,
                (&mut p_w as *mut _) as *mut c_void,
                (&mut p_dyn as *mut _) as *mut c_void,
                (&mut df4 as *mut _) as *mut c_void,
                (&mut p_hist as *mut _) as *mut c_void,
                (&mut p_out as *mut _) as *mut c_void,
                (&mut outs as *mut _) as *mut c_void,
                (&mut p_res as *mut _) as *mut c_void,
                (&mut ress as *mut _) as *mut c_void,
                (&mut p_seq as *mut _) as *mut c_void,
                (&mut ch as *mut _) as *mut c_void,
                (&mut bi as *mut _) as *mut c_void,
                (&mut hd as *mut _) as *mut c_void,
                (&mut dr as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.short_conv4_b, (9, 1), 256, &mut args)?;
        }

        // 8. Fused Batched MLP Prologue (`depth_mix + rmsnorm_d + quant_q8_16` when `!has_msgs`)
        let wants_mlp_q8_16 = matches!(l.expert_scales, ExpertScales::Nvfp4 { .. });
        let mut mlp_h_ready = false;
        let mut mlp_q8_16_ready = false;
        if !has_msgs {
            if let Some(f_mlp_fused) = self.fns.depth_mix_rmsnorm_d_quant_q8_16_b {
                let mut pb = self.prefill.d_b_blocks.ptr;
                let mut nb = *n_blocks as i32;
                let mut sb = b_i32;
                let mut bo = 0i32;
                let mut pc = self.prefill.d_b_cur.ptr;
                let mut ac = 1i32;
                let mut pq = l.mlp_query.ptr;
                let mut pg = l.mlp_norm.ptr;
                let mut pout = self.prefill.layer_workspace.mlp_hidden;
                let mut dq = i32::from(wants_mlp_q8_16);
                let mut pq8 = self.prefill.d_b_act_codes.ptr;
                let mut psc = self.prefill.d_b_act_scales.ptr;
                let mut args: [*mut c_void; 12] = [
                    (&mut pb as *mut _) as *mut c_void,
                    (&mut nb as *mut _) as *mut c_void,
                    (&mut sb as *mut _) as *mut c_void,
                    (&mut bo as *mut _) as *mut c_void,
                    (&mut pc as *mut _) as *mut c_void,
                    (&mut ac as *mut _) as *mut c_void,
                    (&mut pq as *mut _) as *mut c_void,
                    (&mut pg as *mut _) as *mut c_void,
                    (&mut pout as *mut _) as *mut c_void,
                    (&mut dq as *mut _) as *mut c_void,
                    (&mut pq8 as *mut _) as *mut c_void,
                    (&mut psc as *mut _) as *mut c_void,
                ];
                self.launch(stream, f_mlp_fused, (1, b_u32), 256, &mut args)?;
                mlp_h_ready = true;
                mlp_q8_16_ready = wants_mlp_q8_16;
            }
        }
        if !mlp_h_ready {
            let mut pb = self.prefill.d_b_blocks.ptr;
            let mut nb = *n_blocks as i32;
            let mut sb = b_i32;
            let mut bo = 0i32;
            let mut bout = b_i32;
            let mut pc = self.prefill.d_b_cur.ptr;
            let mut ac = 1i32;
            let mut pq = l.mlp_query.ptr;
            let mut pout = self.prefill.layer_workspace.mlp_hidden;
            let mut b3 = 0i32;
            let mut args: [*mut c_void; 10] = [
                (&mut pb as *mut _) as *mut c_void,
                (&mut nb as *mut _) as *mut c_void,
                (&mut sb as *mut _) as *mut c_void,
                (&mut bo as *mut _) as *mut c_void,
                (&mut bout as *mut _) as *mut c_void,
                (&mut pc as *mut _) as *mut c_void,
                (&mut ac as *mut _) as *mut c_void,
                (&mut pq as *mut _) as *mut c_void,
                (&mut pout as *mut _) as *mut c_void,
                (&mut b3 as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.depth_mix_b, (1, b_u32), 256, &mut args)?;
            if has_msgs {
                self.enqueue_ctrl_n_b(
                    stream,
                    self.prefill.layer_workspace.mlp_hidden,
                    msg_base_ptr + 3 * bd_bytes,
                    l.ctrl_mlp.as_ref().unwrap(),
                    1,
                    b_len,
                )?;
            }
            if wants_mlp_q8_16 {
                if let Some(f_rn_q8) = self.fns.rmsnorm_d_quant_q8_16_b {
                    let mut pin = self.prefill.layer_workspace.mlp_hidden;
                    let mut pg = l.mlp_norm.ptr;
                    let mut pout = self.prefill.layer_workspace.mlp_hidden;
                    let mut pq8 = self.prefill.d_b_act_codes.ptr;
                    let mut psc = self.prefill.d_b_act_scales.ptr;
                    let mut args: [*mut c_void; 5] = [
                        (&mut pin as *mut _) as *mut c_void,
                        (&mut pg as *mut _) as *mut c_void,
                        (&mut pout as *mut _) as *mut c_void,
                        (&mut pq8 as *mut _) as *mut c_void,
                        (&mut psc as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, f_rn_q8, (1, b_u32), 256, &mut args)?;
                    mlp_q8_16_ready = true;
                }
            }
            if !mlp_q8_16_ready {
                let mut pin = self.prefill.layer_workspace.mlp_hidden;
                let mut pg = l.mlp_norm.ptr;
                let mut pout = self.prefill.layer_workspace.mlp_hidden;
                let mut hg = 1i32;
                let mut args: [*mut c_void; 4] = [
                    (&mut pin as *mut _) as *mut c_void,
                    (&mut pg as *mut _) as *mut c_void,
                    (&mut pout as *mut _) as *mut c_void,
                    (&mut hg as *mut _) as *mut c_void,
                ];
                self.launch(stream, self.fns.rmsnorm_d_b, (b_u32, 1), 256, &mut args)?;
            }
        }

        // 9. Batched BF16 Router + TopK8 + Zero-Atomic Grouped-Expert MoE + Combine!
        {
            let mut pw = l.router_w.ptr;
            let mut px = self.prefill.layer_workspace.mlp_hidden;
            let mut py = self.prefill.layer_workspace.router_logits;
            let mut bi = b_i32;
            let mut args: [*mut c_void; 4] = [
                (&mut pw as *mut _) as *mut c_void,
                (&mut px as *mut _) as *mut c_void,
                (&mut py as *mut _) as *mut c_void,
                (&mut bi as *mut _) as *mut c_void,
            ];
            self.launch(
                stream,
                self.fns.bf16_gemm_router_b,
                (16, b_u32.div_ceil(4)),
                256,
                &mut args,
            )?;
        }
        {
            let mut pl = self.prefill.layer_workspace.router_logits;
            let mut pb = l.router_bias.ptr;
            let mut pids = self.prefill.d_b_sel_ids.ptr;
            let mut pwts = self.prefill.d_b_sel_wts.ptr;
            let mut args: [*mut c_void; 4] = [
                (&mut pl as *mut _) as *mut c_void,
                (&mut pb as *mut _) as *mut c_void,
                (&mut pids as *mut _) as *mut c_void,
                (&mut pwts as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.router_topk8_b, (b_u32, 1), 32, &mut args)?;
        }
        {
            let mut pids = self.prefill.d_b_sel_ids.ptr;
            let mut pcnt = self.d_b_exp_counts.ptr;
            let mut ptok = self.prefill.d_b_exp_toks.ptr;
            let mut ppair = self.prefill.d_b_exp_pairs.ptr;
            let mut bi = b_i32;
            let mut args: [*mut c_void; 5] = [
                (&mut pids as *mut _) as *mut c_void,
                (&mut pcnt as *mut _) as *mut c_void,
                (&mut ptok as *mut _) as *mut c_void,
                (&mut ppair as *mut _) as *mut c_void,
                (&mut bi as *mut _) as *mut c_void,
            ];
            let (kernel, grid_x) = if b_len > EXPERT_PAGE {
                (
                    self.fns
                        .build_expert_groups_wide_b
                        .ok_or("wide expert grouping unavailable")?,
                    N_EXP as u32 / 4,
                )
            } else {
                (self.fns.build_expert_groups_b, 1)
            };
            self.launch(stream, kernel, (grid_x, 1), 128, &mut args)?;
        }

        let [w13, w2] = &self.expert_weights[il];
        match &l.expert_scales {
            ExpertScales::Nvfp4 { s13, a13, s2, a2 } => {
                if !mlp_q8_16_ready {
                    self.enqueue_quant_q8_16_b(
                        stream,
                        self.prefill.layer_workspace.mlp_hidden,
                        b_len,
                        D,
                    )?;
                }
                let mut pw13 = w13.ptr();
                let mut ps13 = s13.ptr;
                let mut pa13 = a13.ptr;
                let mut pxq = self.prefill.d_b_act_codes.ptr;
                let mut pxs = self.prefill.d_b_act_scales.ptr;
                let mut pcnt = self.d_b_exp_counts.ptr;
                let mut ptok = self.prefill.d_b_exp_toks.ptr;
                let mut ppair = self.prefill.d_b_exp_pairs.ptr;
                let mut pact = self.prefill.d_b_activations.ptr;
                let mut up_args: [*mut c_void; 9] = [
                    (&mut pw13 as *mut _) as *mut c_void,
                    (&mut ps13 as *mut _) as *mut c_void,
                    (&mut pa13 as *mut _) as *mut c_void,
                    (&mut pxq as *mut _) as *mut c_void,
                    (&mut pxs as *mut _) as *mut c_void,
                    (&mut pcnt as *mut _) as *mut c_void,
                    (&mut ptok as *mut _) as *mut c_void,
                    (&mut ppair as *mut _) as *mut c_void,
                    (&mut pact as *mut _) as *mut c_void,
                ];
                self.launch(
                    stream,
                    self.fns.moe_nvfp4_dp4a_grouped_up_situ,
                    (72, 128),
                    256,
                    &mut up_args,
                )?;

                self.enqueue_quant_q8_16_b(
                    stream,
                    self.prefill.d_b_activations.ptr,
                    b_len * TOP_K,
                    FF,
                )?;
                let mut pw2 = w2.ptr();
                let mut ps2 = s2.ptr;
                let mut pa2 = a2.ptr;
                let mut ppy = self.prefill.layer_workspace.pair_output.ptr;
                let mut dn_args: [*mut c_void; 8] = [
                    (&mut pw2 as *mut _) as *mut c_void,
                    (&mut ps2 as *mut _) as *mut c_void,
                    (&mut pa2 as *mut _) as *mut c_void,
                    (&mut pxq as *mut _) as *mut c_void,
                    (&mut pxs as *mut _) as *mut c_void,
                    (&mut pcnt as *mut _) as *mut c_void,
                    (&mut ppair as *mut _) as *mut c_void,
                    (&mut ppy as *mut _) as *mut c_void,
                ];
                self.launch(
                    stream,
                    self.fns.moe_nvfp4_dp4a_grouped_down,
                    (72, 128),
                    256,
                    &mut dn_args,
                )?;
            }
            ExpertScales::Mxfp8 { s13, s2 } => {
                let mut pw13 = w13.ptr();
                let mut ps13 = s13.ptr;
                let mut px = self.prefill.layer_workspace.mlp_hidden;
                let mut pcnt = self.d_b_exp_counts.ptr;
                let mut ptok = self.prefill.d_b_exp_toks.ptr;
                let mut ppair = self.prefill.d_b_exp_pairs.ptr;
                let mut pact = self.prefill.d_b_activations.ptr;
                let mut up_args: [*mut c_void; 7] = [
                    (&mut pw13 as *mut _) as *mut c_void,
                    (&mut ps13 as *mut _) as *mut c_void,
                    (&mut px as *mut _) as *mut c_void,
                    (&mut pcnt as *mut _) as *mut c_void,
                    (&mut ptok as *mut _) as *mut c_void,
                    (&mut ppair as *mut _) as *mut c_void,
                    (&mut pact as *mut _) as *mut c_void,
                ];
                self.launch(
                    stream,
                    self.fns.moe_mxfp8_grouped_up_situ,
                    (72, 128),
                    256,
                    &mut up_args,
                )?;

                let mut pw2 = w2.ptr();
                let mut ps2 = s2.ptr;
                let mut ppy = self.prefill.layer_workspace.pair_output.ptr;
                let mut dn_args: [*mut c_void; 6] = [
                    (&mut pw2 as *mut _) as *mut c_void,
                    (&mut ps2 as *mut _) as *mut c_void,
                    (&mut pact as *mut _) as *mut c_void,
                    (&mut pcnt as *mut _) as *mut c_void,
                    (&mut ppair as *mut _) as *mut c_void,
                    (&mut ppy as *mut _) as *mut c_void,
                ];
                self.launch(
                    stream,
                    self.fns.moe_mxfp8_grouped_down,
                    (72, 128),
                    256,
                    &mut dn_args,
                )?;
            }
        }

        {
            let mut ppy = self.prefill.layer_workspace.pair_output.ptr;
            let mut pwts = self.prefill.d_b_sel_wts.ptr;
            let mut pout = self.prefill.d_b_activations.ptr;
            let mut bi = b_i32;
            let mut args: [*mut c_void; 4] = [
                (&mut ppy as *mut _) as *mut c_void,
                (&mut pwts as *mut _) as *mut c_void,
                (&mut pout as *mut _) as *mut c_void,
                (&mut bi as *mut _) as *mut c_void,
            ];
            self.launch(
                stream,
                self.fns.moe_combine_top8_b,
                (9, b_u32),
                256,
                &mut args,
            )?;
        }

        // 10. `mlp_sconv` + Residual Add into `d_b_cur`
        {
            let mut p_cur = self.prefill.d_b_activations.ptr;
            let mut ins = D as i32;
            let mut p_w = l.mlp_conv4.ptr;
            let mut p_dyn: CUdeviceptr = 0;
            let mut df4 = 0i32;
            let mut p_hist = st.hist_mlp;
            let mut p_out = self.prefill.d_b_cur.ptr;
            let mut outs = D as i32;
            let mut p_res = self.prefill.d_b_cur.ptr;
            let mut ress = D as i32;
            let mut p_seq = if record_seq {
                slot_seq_base + (4352 * CONV_SEQ_STEPS * 4) as u64
            } else {
                0
            };
            let mut ch = D as i32;
            let mut bi = b_i32;
            let mut hd = 0i32;
            let mut dr = 1i32;
            let mut args: [*mut c_void; 15] = [
                (&mut p_cur as *mut _) as *mut c_void,
                (&mut ins as *mut _) as *mut c_void,
                (&mut p_w as *mut _) as *mut c_void,
                (&mut p_dyn as *mut _) as *mut c_void,
                (&mut df4 as *mut _) as *mut c_void,
                (&mut p_hist as *mut _) as *mut c_void,
                (&mut p_out as *mut _) as *mut c_void,
                (&mut outs as *mut _) as *mut c_void,
                (&mut p_res as *mut _) as *mut c_void,
                (&mut ress as *mut _) as *mut c_void,
                (&mut p_seq as *mut _) as *mut c_void,
                (&mut ch as *mut _) as *mut c_void,
                (&mut bi as *mut _) as *mut c_void,
                (&mut hd as *mut _) as *mut c_void,
                (&mut dr as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.short_conv4_b, (9, 1), 256, &mut args)?;
        }

        if let Some(taps) = &self.prefill.d_taps {
            if let Some(tap_idx) = TAP_SLOTS.iter().position(|&s| s == slot_idx) {
                let dst = taps.ptr + (tap_idx * self.prefill.capacity * D * 4) as u64;
                self.enqueue_copy_n(stream, self.prefill.d_b_cur.ptr, dst, bd_floats)?;
            }
        }

        Ok(())
    }

    fn enqueue_batch_step_impl(
        &self,
        stream: CUstream,
        tokens_dev_ptr: CUdeviceptr,
        b_len: usize,
        head_mode: BatchHeadMode,
        record_seq: bool,
    ) -> Result<(), String> {
        let b_u32 = b_len as u32;
        let b_i32 = b_len as i32;
        let bd_floats = b_len * D;
        let bd_bytes = (bd_floats * 4) as u64;

        // 0. Batched Embed + `embed_norm`
        {
            let mut pe = self.embed.ptr();
            let mut pn = self.embed_norm.ptr;
            let mut pt = tokens_dev_ptr;
            let mut px = self.prefill.d_b_cur.ptr;
            let mut args: [*mut c_void; 4] = [
                (&mut pe as *mut _) as *mut c_void,
                (&mut pn as *mut _) as *mut c_void,
                (&mut pt as *mut _) as *mut c_void,
                (&mut px as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.embed_rmsnorm_b, (b_u32, 1), 256, &mut args)?;
        }

        let mut n_blocks = 0usize;
        for il in 0..6 {
            self.enqueue_batch_visit(stream, il, il, b_len, &mut n_blocks, 0, record_seq)?;
        }
        debug_assert_eq!(n_blocks, 1);
        if let Some(f_setup) = self.fns.loop_pass_setup_b {
            let mut p_cur = self.prefill.d_b_cur.ptr;
            let mut p_blk = self.prefill.d_b_blocks.ptr;
            let mut p_px = self.prefill.d_b_prelude_x.ptr;
            let mut p_pb0 = self.prefill.d_b_prelude_block0.ptr;
            let mut p_bank = self.prefill.d_b_bank.ptr;
            let mut bi = b_i32;
            let mut mode = 0i32;
            let mut args: [*mut c_void; 7] = [
                (&mut p_cur as *mut _) as *mut c_void,
                (&mut p_blk as *mut _) as *mut c_void,
                (&mut p_px as *mut _) as *mut c_void,
                (&mut p_pb0 as *mut _) as *mut c_void,
                (&mut p_bank as *mut _) as *mut c_void,
                (&mut bi as *mut _) as *mut c_void,
                (&mut mode as *mut _) as *mut c_void,
            ];
            self.launch(stream, f_setup, (9, b_u32), 256, &mut args)?;
        } else {
            self.enqueue_copy_n(
                stream,
                self.prefill.d_b_cur.ptr,
                self.prefill.d_b_prelude_x.ptr,
                bd_floats,
            )?;
            self.enqueue_copy_n(
                stream,
                self.prefill.d_b_blocks.ptr,
                self.prefill.d_b_prelude_block0.ptr,
                bd_floats,
            )?;
        }

        let precompute_all_msgs = b_len <= MAX_SPEC_TOKENS;
        for pass in 0..3 {
            if pass > 0 {
                if let Some(f_setup) = self.fns.loop_pass_setup_b {
                    let mut p_cur = self.prefill.d_b_cur.ptr;
                    let mut p_blk = self.prefill.d_b_blocks.ptr;
                    let mut p_px = self.prefill.d_b_prelude_x.ptr;
                    let mut p_pb0 = self.prefill.d_b_prelude_block0.ptr;
                    let mut p_bank = self.prefill.d_b_bank.ptr;
                    let mut bi = b_i32;
                    let mut mode = 1i32;
                    let mut args: [*mut c_void; 7] = [
                        (&mut p_cur as *mut _) as *mut c_void,
                        (&mut p_blk as *mut _) as *mut c_void,
                        (&mut p_px as *mut _) as *mut c_void,
                        (&mut p_pb0 as *mut _) as *mut c_void,
                        (&mut p_bank as *mut _) as *mut c_void,
                        (&mut bi as *mut _) as *mut c_void,
                        (&mut mode as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, f_setup, (9, b_u32), 256, &mut args)?;
                } else {
                    self.enqueue_copy_n(
                        stream,
                        self.prefill.d_b_blocks.ptr + 2 * bd_bytes,
                        self.prefill.d_b_bank.ptr,
                        bd_floats,
                    )?;
                    self.enqueue_copy_n(
                        stream,
                        self.prefill.d_b_cur.ptr,
                        self.prefill.d_b_bank.ptr + bd_bytes,
                        bd_floats,
                    )?;
                    self.enqueue_copy_n(
                        stream,
                        self.prefill.d_b_prelude_x.ptr,
                        self.prefill.d_b_cur.ptr,
                        bd_floats,
                    )?;
                    self.enqueue_copy_n(
                        stream,
                        self.prefill.d_b_prelude_block0.ptr,
                        self.prefill.d_b_blocks.ptr,
                        bd_floats,
                    )?;
                }
                n_blocks = 1;
                if precompute_all_msgs {
                    let mut pb = self.prefill.d_b_bank.ptr;
                    let mut nb = 2i32;
                    let mut sb = b_i32;
                    let mut bo = 0i32;
                    let mut bout = b_i32;
                    let mut pc: CUdeviceptr = 0;
                    let mut ac = 0i32;
                    let mut pq = self.d_all_msg_queries.ptr;
                    let mut pout = self.prefill.d_b_msgs.ptr;
                    let mut b3 = 0i32;
                    let mut args: [*mut c_void; 10] = [
                        (&mut pb as *mut _) as *mut c_void,
                        (&mut nb as *mut _) as *mut c_void,
                        (&mut sb as *mut _) as *mut c_void,
                        (&mut bo as *mut _) as *mut c_void,
                        (&mut bout as *mut _) as *mut c_void,
                        (&mut pc as *mut _) as *mut c_void,
                        (&mut ac as *mut _) as *mut c_void,
                        (&mut pq as *mut _) as *mut c_void,
                        (&mut pout as *mut _) as *mut c_void,
                        (&mut b3 as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, self.fns.depth_mix_b, (48, b_u32), 256, &mut args)?;
                }
            }
            for ip in 6..18 {
                let slot = if pass == 0 {
                    ip
                } else {
                    N_LAYER + (pass - 1) * 12 + (ip - 6)
                };
                let msg_ptr = if pass == 0 {
                    0
                } else if precompute_all_msgs {
                    self.prefill.d_b_msgs.ptr + ((ip - 6) * 4) as u64 * bd_bytes
                } else {
                    let mq = self.layers[ip].msg_query.as_ref().unwrap();
                    let mut pb = self.prefill.d_b_bank.ptr;
                    let mut nb = 2i32;
                    let mut sb = b_i32;
                    let mut bo = 0i32;
                    let mut bout = b_i32;
                    let mut pc: CUdeviceptr = 0;
                    let mut ac = 0i32;
                    let mut pq = mq.ptr;
                    let mut pout = self.prefill.d_b_msgs.ptr;
                    let mut b3 = 0i32;
                    let mut args: [*mut c_void; 10] = [
                        (&mut pb as *mut _) as *mut c_void,
                        (&mut nb as *mut _) as *mut c_void,
                        (&mut sb as *mut _) as *mut c_void,
                        (&mut bo as *mut _) as *mut c_void,
                        (&mut bout as *mut _) as *mut c_void,
                        (&mut pc as *mut _) as *mut c_void,
                        (&mut ac as *mut _) as *mut c_void,
                        (&mut pq as *mut _) as *mut c_void,
                        (&mut pout as *mut _) as *mut c_void,
                        (&mut b3 as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, self.fns.depth_mix_b, (4, b_u32), 256, &mut args)?;
                    self.prefill.d_b_msgs.ptr
                };
                self.enqueue_batch_visit(
                    stream,
                    ip,
                    slot,
                    b_len,
                    &mut n_blocks,
                    msg_ptr,
                    record_seq,
                )?;
            }
        }

        for il in 18..24 {
            self.enqueue_batch_visit(stream, il, il, b_len, &mut n_blocks, 0, record_seq)?;
        }
        debug_assert_eq!(n_blocks, 4);

        if let Some(taps) = &self.prefill.d_taps {
            // Compute `tap 3` (`final_norm(output_query_mix)`) across all `b_len` tokens!
            let p_tap3_dst = taps.ptr + (3 * self.prefill.capacity * D * 4) as u64;
            if let Some(f_mlp_fused) = self.fns.depth_mix_rmsnorm_d_quant_q8_16_b {
                let mut pb = self.prefill.d_b_blocks.ptr;
                let mut nb = n_blocks as i32;
                let mut sb = b_i32;
                let mut bo = 0i32;
                let mut pc = self.prefill.d_b_cur.ptr;
                let mut ac = 1i32;
                let mut pq = self.output_query.ptr;
                let mut pg = self.final_norm.ptr;
                let mut p_tap3 = p_tap3_dst;
                let mut dq = 0i32;
                let mut z_q8: CUdeviceptr = 0;
                let mut z_sc: CUdeviceptr = 0;
                let mut args: [*mut c_void; 12] = [
                    (&mut pb as *mut _) as *mut c_void,
                    (&mut nb as *mut _) as *mut c_void,
                    (&mut sb as *mut _) as *mut c_void,
                    (&mut bo as *mut _) as *mut c_void,
                    (&mut pc as *mut _) as *mut c_void,
                    (&mut ac as *mut _) as *mut c_void,
                    (&mut pq as *mut _) as *mut c_void,
                    (&mut pg as *mut _) as *mut c_void,
                    (&mut p_tap3 as *mut _) as *mut c_void,
                    (&mut dq as *mut _) as *mut c_void,
                    (&mut z_q8 as *mut _) as *mut c_void,
                    (&mut z_sc as *mut _) as *mut c_void,
                ];
                self.launch(stream, f_mlp_fused, (1, b_u32), 256, &mut args)?;
            } else {
                let mut pb = self.prefill.d_b_blocks.ptr;
                let mut nb = n_blocks as i32;
                let mut sb = b_i32;
                let mut bo = 0i32;
                let mut bout = b_i32;
                let mut pc = self.prefill.d_b_cur.ptr;
                let mut ac = 1i32;
                let mut pq = self.output_query.ptr;
                let mut pout = self.prefill.layer_workspace.mlp_hidden;
                let mut b3 = 0i32;
                let mut args: [*mut c_void; 10] = [
                    (&mut pb as *mut _) as *mut c_void,
                    (&mut nb as *mut _) as *mut c_void,
                    (&mut sb as *mut _) as *mut c_void,
                    (&mut bo as *mut _) as *mut c_void,
                    (&mut bout as *mut _) as *mut c_void,
                    (&mut pc as *mut _) as *mut c_void,
                    (&mut ac as *mut _) as *mut c_void,
                    (&mut pq as *mut _) as *mut c_void,
                    (&mut pout as *mut _) as *mut c_void,
                    (&mut b3 as *mut _) as *mut c_void,
                ];
                self.launch(stream, self.fns.depth_mix_b, (1, b_u32), 256, &mut args)?;
                let mut pin = self.prefill.layer_workspace.mlp_hidden;
                let mut pg = self.final_norm.ptr;
                let mut p_tap3 = p_tap3_dst;
                let mut hg = 1i32;
                let mut rn_args: [*mut c_void; 4] = [
                    (&mut pin as *mut _) as *mut c_void,
                    (&mut pg as *mut _) as *mut c_void,
                    (&mut p_tap3 as *mut _) as *mut c_void,
                    (&mut hg as *mut _) as *mut c_void,
                ];
                self.launch(stream, self.fns.rmsnorm_d_b, (b_u32, 1), 256, &mut rn_args)?;
            }
        }

        match head_mode {
            BatchHeadMode::None => {}
            BatchHeadMode::LastOnly => {
                if let Some(f_fused) = self.fns.depth_mix_rmsnorm_quant_q8_64_b {
                    let mut pb = self.prefill.d_b_blocks.ptr;
                    let mut nb = n_blocks as i32;
                    let mut sb = b_i32;
                    let mut bo = (b_len - 1) as i32;
                    let mut bout = 1i32;
                    let mut pc = self.prefill.d_b_cur.ptr;
                    let mut ac = 1i32;
                    let mut pq = self.output_query.ptr;
                    let mut pg = self.final_norm.ptr;
                    let mut pq8 = self.d_act_q8_64.ptr;
                    let mut psc = self.d_act_sc_64.ptr;
                    let mut args: [*mut c_void; 11] = [
                        (&mut pb as *mut _) as *mut c_void,
                        (&mut nb as *mut _) as *mut c_void,
                        (&mut sb as *mut _) as *mut c_void,
                        (&mut bo as *mut _) as *mut c_void,
                        (&mut bout as *mut _) as *mut c_void,
                        (&mut pc as *mut _) as *mut c_void,
                        (&mut ac as *mut _) as *mut c_void,
                        (&mut pq as *mut _) as *mut c_void,
                        (&mut pg as *mut _) as *mut c_void,
                        (&mut pq8 as *mut _) as *mut c_void,
                        (&mut psc as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, f_fused, (1, 1), 256, &mut args)?;
                } else {
                    let mut pb = self.prefill.d_b_blocks.ptr;
                    let mut nb = n_blocks as i32;
                    let mut sb = b_i32;
                    let mut bo = (b_len - 1) as i32;
                    let mut bout = 1i32;
                    let mut pc = self.prefill.d_b_cur.ptr;
                    let mut ac = 1i32;
                    let mut pq = self.output_query.ptr;
                    let mut pout = self.d_mlp_h.ptr;
                    let mut b3 = 0i32;
                    let mut args: [*mut c_void; 10] = [
                        (&mut pb as *mut _) as *mut c_void,
                        (&mut nb as *mut _) as *mut c_void,
                        (&mut sb as *mut _) as *mut c_void,
                        (&mut bo as *mut _) as *mut c_void,
                        (&mut bout as *mut _) as *mut c_void,
                        (&mut pc as *mut _) as *mut c_void,
                        (&mut ac as *mut _) as *mut c_void,
                        (&mut pq as *mut _) as *mut c_void,
                        (&mut pout as *mut _) as *mut c_void,
                        (&mut b3 as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, self.fns.depth_mix_b, (1, 1), 256, &mut args)?;
                    self.enqueue_rmsnorm_quant_q8_64(
                        stream,
                        self.d_mlp_h.ptr,
                        self.final_norm.ptr,
                        1,
                    )?;
                }
                self.enqueue_q8_gemv_slice_at(
                    stream,
                    &self.unembed_q8,
                    0,
                    VOCAB as i32,
                    0,
                    self.d_logits.ptr,
                    30.0,
                )?;
                let mut pl = self.d_logits.ptr;
                let mut pt = self.d_token_out.ptr;
                let mut am_args: [*mut c_void; 2] = [
                    (&mut pl as *mut _) as *mut c_void,
                    (&mut pt as *mut _) as *mut c_void,
                ];
                self.launch(stream, self.fns.argmax_b, (1, 1), 256, &mut am_args)?;
            }
            BatchHeadMode::AllHeads => {
                if let Some(f_fused) = self.fns.depth_mix_rmsnorm_quant_q8_64_b {
                    let mut pb = self.prefill.d_b_blocks.ptr;
                    let mut nb = n_blocks as i32;
                    let mut sb = b_i32;
                    let mut bo = 0i32;
                    let mut bout = b_i32;
                    let mut pc = self.prefill.d_b_cur.ptr;
                    let mut ac = 1i32;
                    let mut pq = self.output_query.ptr;
                    let mut pg = self.final_norm.ptr;
                    let mut pq8 = self.prefill.d_b_act_codes.ptr;
                    let mut psc = self.prefill.d_b_act_scales.ptr;
                    let mut args: [*mut c_void; 11] = [
                        (&mut pb as *mut _) as *mut c_void,
                        (&mut nb as *mut _) as *mut c_void,
                        (&mut sb as *mut _) as *mut c_void,
                        (&mut bo as *mut _) as *mut c_void,
                        (&mut bout as *mut _) as *mut c_void,
                        (&mut pc as *mut _) as *mut c_void,
                        (&mut ac as *mut _) as *mut c_void,
                        (&mut pq as *mut _) as *mut c_void,
                        (&mut pg as *mut _) as *mut c_void,
                        (&mut pq8 as *mut _) as *mut c_void,
                        (&mut psc as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, f_fused, (1, b_u32), 256, &mut args)?;
                } else {
                    let mut pb = self.prefill.d_b_blocks.ptr;
                    let mut nb = n_blocks as i32;
                    let mut sb = b_i32;
                    let mut bo = 0i32;
                    let mut bout = b_i32;
                    let mut pc = self.prefill.d_b_cur.ptr;
                    let mut ac = 1i32;
                    let mut pq = self.output_query.ptr;
                    let mut pout = self.prefill.layer_workspace.mlp_hidden;
                    let mut b3 = 0i32;
                    let mut args: [*mut c_void; 10] = [
                        (&mut pb as *mut _) as *mut c_void,
                        (&mut nb as *mut _) as *mut c_void,
                        (&mut sb as *mut _) as *mut c_void,
                        (&mut bo as *mut _) as *mut c_void,
                        (&mut bout as *mut _) as *mut c_void,
                        (&mut pc as *mut _) as *mut c_void,
                        (&mut ac as *mut _) as *mut c_void,
                        (&mut pq as *mut _) as *mut c_void,
                        (&mut pout as *mut _) as *mut c_void,
                        (&mut b3 as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, self.fns.depth_mix_b, (1, b_u32), 256, &mut args)?;
                    self.enqueue_rmsnorm_quant_q8_64_b(
                        stream,
                        self.prefill.layer_workspace.mlp_hidden,
                        self.final_norm.ptr,
                        1,
                        b_len,
                    )?;
                }
                self.enqueue_q8_gemm_slice_at(
                    stream,
                    &self.unembed_q8,
                    0,
                    VOCAB as i32,
                    0,
                    b_len,
                    self.d_spec_logits.ptr,
                    VOCAB as i32,
                    30.0,
                )?;
                let mut pl = self.d_spec_logits.ptr;
                let mut pt = self.d_spec_tokens_out.ptr;
                let mut am_args: [*mut c_void; 2] = [
                    (&mut pl as *mut _) as *mut c_void,
                    (&mut pt as *mut _) as *mut c_void,
                ];
                self.launch(stream, self.fns.argmax_b, (b_u32, 1), 256, &mut am_args)?;
            }
        }

        let mut pp = self.d_pos.ptr;
        let mut delta = b_i32;
        let mut pos_args: [*mut c_void; 2] = [
            (&mut pp as *mut _) as *mut c_void,
            (&mut delta as *mut _) as *mut c_void,
        ];
        self.launch(stream, self.fns.inc_pos_by, (1, 1), 32, &mut pos_args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planar_q8_packs_unaligned_bf16_rows_in_order() {
        let cols = 128;
        let mut bytes = vec![0; 1 + 3 * cols * 2];
        let mut expected_codes = Vec::new();
        let mut expected_scales = Vec::new();
        for (block, target) in bytes[1..].as_chunks_mut::<128>().0.iter_mut().enumerate() {
            let peak = [0.0_f32, 1.0, 2.0, 0.5, 4.0, 8.0][block];
            expected_scales.extend_from_slice(&(peak * (1.0 / 127.0)).to_le_bytes());
            for (i, word) in target.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                let value = peak * sign;
                word.copy_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
                let code = if peak == 0.0 { 0 } else { (127.0 * sign) as i8 };
                expected_codes.push(code as u8);
            }
        }
        let mut codes = vec![0; 3 * cols];
        let mut scales = vec![0; 24];
        pack_q8_rows(&bytes[1..], cols, &mut codes, &mut scales);
        assert_eq!(codes, expected_codes);
        assert_eq!(scales, expected_scales);
    }

    #[test]
    fn loop_layers_take_priority_and_a_remaining_matrix_can_fit() {
        let up = [0u8; 2];
        let down = [0u8; 1];
        let tensors = [[up.as_slice(), down.as_slice()]; N_LAYER];
        let placement = plan_expert_placement(&tensors, 12 * 3 + 1);
        assert!(placement[6..18].iter().all(|pair| *pair == [true, true]));
        assert_eq!(placement[0], [false, true]);
        assert!(placement[1..6]
            .iter()
            .chain(&placement[18..])
            .all(|pair| *pair == [false, false]));
        assert_eq!(plan_expert_placement(&tensors, 0), [[false; 2]; N_LAYER]);
        assert_eq!(
            plan_expert_placement(&tensors, N_LAYER * 3),
            [[true; 2]; N_LAYER]
        );
    }
}
