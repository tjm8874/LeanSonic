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
    check_cu, compile_cuda_to_ptx_nvrtc, CUdeviceptr, CUevent, CUfunction, CUlaunchAttribute,
    CUlaunchConfig, CUmodule, CUstream, CapturedGraphExec, CudaContextHandle, DeviceBuf,
    PinnedHostBuf, CU_EVENT_DISABLE_TIMING,
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
/// #80: GPUs with at most this much VRAM (16 GB cards report ~15.5-15.9 GiB) default to INT8
/// global KV when `max_ctx > AUTO_INT8_KV_MIN_CTX` and `LEANSONIC_KV` is unset.
const AUTO_INT8_KV_MAX_VRAM: usize = 16_896 << 20; // 16.5 GiB
const AUTO_INT8_KV_MIN_CTX: usize = 8_192;
pub const MAX_BATCH_TOKENS: usize = 2048;
/// Rows of one multi-sequence decode round (#107) when `LEANSONIC_SEQS` > 1.
pub const MAX_MS_ROWS: usize = 64;
const NVFP4_LAYERS: usize = 20;
const EXPERT_PAGE: usize = 64;
pub const MAX_SPEC_TOKENS: usize = 8;
pub const TAP_SLOTS: [usize; 3] = [4, 28, 46];
const CONV_SEQ_STEPS: usize = MAX_SPEC_TOKENS + 3; // 11
const CONV_SEQ_SLOT_FLOATS: usize = (1024 + 1024 + D + D) * CONV_SEQ_STEPS; // 6656 * 11 = 73,216
/// Verify-round `short_conv4` inputs per slot: `(1024 + 1024 + D + D) * MAX_MS_ROWS` floats.
const CONV_XREC_SLOT_FLOATS: usize = (1024 + 1024 + D + D) * MAX_MS_ROWS;
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

enum NativeQ8Codes<'a> {
    Raw(&'a [u8]),
    Encoded(crate::inkling_mxfp8::Matrix<'a>),
}

impl<'a> NativeQ8Codes<'a> {
    fn lookup(
        st: &'a MappedSafeTensors,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Option<Self>, String> {
        match (st.get(name), st.get(&format!("{name}.rans_v1"))) {
            (None, None) => Ok(None),
            (Some(t), None)
                if t.dtype == "I8" && t.shape == [rows, cols] && t.data.len() == rows * cols =>
            {
                Ok(Some(Self::Raw(t.data)))
            }
            (None, Some(t)) if t.dtype == "U8" && t.shape == [t.data.len()] => {
                crate::inkling_mxfp8::Matrix::parse(t.data, rows * cols)
                    .map(|m| Some(Self::Encoded(m)))
                    .map_err(|e| format!("{name}: {e}"))
            }
            _ => Err(format!("invalid or conflicting native codes for {name}")),
        }
    }

    fn upload(&self, dst: &DeviceBuf, offset: usize) -> Result<(), String> {
        match self {
            Self::Raw(bytes) => dst.copy_from_host_at(offset, bytes),
            Self::Encoded(matrix) => {
                matrix.for_each_chunk(|pos, bytes| dst.copy_from_host_at(offset + pos, bytes))
            }
        }
    }
}

fn native_q8_scales<'a>(
    st: &'a MappedSafeTensors,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Option<std::borrow::Cow<'a, [u8]>>, String> {
    let blocks = cols / 64;
    match (st.get(name), st.get(&format!("{name}.dict_v1"))) {
        (None, None) => Ok(None),
        (Some(s), None)
            if s.dtype == "F32"
                && s.shape == [rows, blocks]
                && s.data.len() == rows * blocks * 4 =>
        {
            Ok(Some(std::borrow::Cow::Borrowed(s.data)))
        }
        (None, Some(s)) if s.dtype == "U8" && s.shape == [s.data.len()] => {
            let raw = crate::inkling_q8_scales::unpack(s.data, rows * blocks)
                .map_err(|error| format!("{name}: {error}"))?;
            Ok(Some(std::borrow::Cow::Owned(raw)))
        }
        _ => Err(format!("invalid or conflicting Q8 scale planes for {name}")),
    }
}

fn load_output_head(
    ctx: &Arc<CudaContextHandle>,
    st: &MappedSafeTensors,
) -> Result<PlanarQ8Gpu, String> {
    let dense = st.get("model.llm.unembed.weight");
    let codes = NativeQ8Codes::lookup(st, "leansonic.unembed_q8_64.codes", VOCAB, D)?;
    let scales = native_q8_scales(st, "leansonic.unembed_q8_64.scales", VOCAB, D)?;
    match (dense, codes, scales) {
        (Some(w), None, None)
            if w.dtype == "BF16"
                && w.shape == [PADDED_VOCAB, D]
                && w.data.len() == PADDED_VOCAB * D * 2 =>
        {
            repack_bf16_to_planar_q8_64(ctx, &w.data[..VOCAB * D * 2], VOCAB, D)
        }
        (None, Some(c), Some(s)) => {
            let matrix = allocate_planar_q8(ctx, VOCAB, D)?;
            c.upload(&matrix.codes, 0)?;
            matrix.scales64.copy_from_host_bytes(&s)?;
            Ok(matrix)
        }
        _ => Err(
            "invalid Inkling output head: expected the BF16 weight or both native Q8_64 planes"
                .into(),
        ),
    }
}

// Components occupy successive rows of the GPU matrix in kernel ABI order.
fn load_attention_matrix(
    ctx: &Arc<CudaContextHandle>,
    st: &MappedSafeTensors,
    prefix: &str,
    parts: &[(&str, usize)],
    cols: usize,
) -> Result<PlanarQ8Gpu, String> {
    let total_rows = parts.iter().map(|(_, rows)| rows).sum();
    let result = allocate_planar_q8(ctx, total_rows, cols)?;
    let scale_stride = cols / 64 * 4;
    let mut row = 0;
    for &(suffix, rows) in parts {
        let name = format!("{prefix}attn.{suffix}");
        let dense = st.get(&name);
        let codes = NativeQ8Codes::lookup(st, &format!("{name}.q8_64.codes"), rows, cols)?;
        let scales = native_q8_scales(st, &format!("{name}.q8_64.scales"), rows, cols)?;
        match (dense, codes, scales) {
            (Some(w), None, None)
                if w.dtype == "BF16"
                    && w.shape == [rows, cols]
                    && w.data.len() == rows * cols * 2 =>
            {
                upload_bf16_rows(&result, w.data, row, rows, cols)?;
            }
            (None, Some(c), Some(s)) => {
                c.upload(&result.codes, row * cols)?;
                result.scales64.copy_from_host_at(row * scale_stride, &s)?;
            }
            _ => {
                return Err(format!(
                "invalid Inkling projection {name}: expected BF16 or a complete native Q8_64 pair"
            ))
            }
        }
        row += rows;
    }
    Ok(result)
}

struct CtrlGpu {
    down: DeviceBuf,
    gain: DeviceBuf,
    forget: DeviceBuf,
}

enum ExpertScales {
    Nvfp4 {
        s13: ExpertScalePlane,
        a13: DeviceBuf,
        s2: ExpertScalePlane,
        a2: DeviceBuf,
    },
    Mxfp8 {
        s13: ExpertScalePlane,
        s2: ExpertScalePlane,
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
    /// `[kv_cap, 8]` FP32 absmax scales when this slot stores INT8 KV (`LEANSONIC_KV=int8`).
    k_scale: Option<DeviceBuf>,
    v_scale: Option<DeviceBuf>,
    hist_k: CUdeviceptr,
    hist_v: CUdeviceptr,
    hist_attn: CUdeviceptr,
    hist_mlp: CUdeviceptr,
    /// Bytes between two sequences' K (or V) caches / scale planes / conv histories
    /// (`LEANSONIC_SEQS` > 1 stores one copy per sequence back to back).
    seq_kv_bytes: u64,
    seq_scale_bytes: u64,
    seq_hist_bytes: u64,
}

impl SlotStateGpu {
    /// `(k_scale, v_scale)` device pointers of sequence `seq`; `0` (kernel `nullptr`) selects
    /// FP16 KV.
    fn kv_scale_ptrs_at(&self, seq: usize) -> (CUdeviceptr, CUdeviceptr) {
        match (&self.k_scale, &self.v_scale) {
            (Some(k), Some(v)) => {
                let off = seq as u64 * self.seq_scale_bytes;
                (k.ptr + off, v.ptr + off)
            }
            _ => (0, 0),
        }
    }
    fn k_at(&self, seq: usize) -> CUdeviceptr {
        self.k_cache.ptr + seq as u64 * self.seq_kv_bytes
    }
    fn v_at(&self, seq: usize) -> CUdeviceptr {
        self.v_cache.ptr + seq as u64 * self.seq_kv_bytes
    }
    fn hist_k_at(&self, seq: usize) -> CUdeviceptr {
        self.hist_k + seq as u64 * self.seq_hist_bytes
    }
    fn hist_v_at(&self, seq: usize) -> CUdeviceptr {
        self.hist_v + seq as u64 * self.seq_hist_bytes
    }
    fn hist_attn_at(&self, seq: usize) -> CUdeviceptr {
        self.hist_attn + seq as u64 * self.seq_hist_bytes
    }
    fn hist_mlp_at(&self, seq: usize) -> CUdeviceptr {
        self.hist_mlp + seq as u64 * self.seq_hist_bytes
    }
}

struct Q8GroupKernel {
    function: CUfunction,
    columns_per_block: usize,
    threads: u32,
}

#[derive(Clone, Copy)]
struct ExpertKernels {
    nvfp4_up_situ: CUfunction,
    nvfp4_down: CUfunction,
    nvfp4_dp4a_up_situ: CUfunction,
    nvfp4_dp4a_up_half: Option<CUfunction>,
    nvfp4_dp4a_up_half_q8: Option<CUfunction>,
    nvfp4_staged_up: Option<(CUfunction, CUfunction)>,
    nvfp4_dp4a_down: CUfunction,
    /// Expert-parallel decode MoE down (bit-identical to `nvfp4_dp4a_down`).
    nvfp4_dp4a_down_xw: Option<CUfunction>,
    nvfp4_dp4a_down_xw_conv: Option<CUfunction>,
    mxfp8_up_situ: CUfunction,
    mxfp8_down: CUfunction,
    mxfp8_down_xw: Option<CUfunction>,
    mxfp8_down_xw_conv: Option<CUfunction>,
    nvfp4_dp4a_grouped_up_situ: CUfunction,
    nvfp4_dp4a_grouped_down: CUfunction,
    mxfp8_grouped_up_situ: CUfunction,
    mxfp8_grouped_down: CUfunction,
    nvfp4_dp4a_verify_up_half_q8: Option<CUfunction>,
    nvfp4_dp4a_verify_down: Option<CUfunction>,
    mxfp8_verify_up_situ: Option<CUfunction>,
    mxfp8_verify_down: Option<CUfunction>,
    nvfp4_tc_up_situ: Option<CUfunction>,
    nvfp4_tc_down: Option<CUfunction>,
    nvfp4_tc_up_situ_t160: Option<CUfunction>,
    nvfp4_tc_down_t160: Option<CUfunction>,
    mxfp8_tc_up_situ: Option<CUfunction>,
    mxfp8_tc_down: Option<CUfunction>,
    mxfp8_tc_up_situ_t160: Option<CUfunction>,
    mxfp8_tc_down_t160: Option<CUfunction>,
}

struct ExpertScalePlane {
    buffer: DeviceBuf,
    kernels: ExpertKernels,
}

#[derive(Clone, Copy)]
enum ExpertScaleSource<'a> {
    Raw(&'a [u8]),
    RangeV1(&'a [u8]),
    RansV1(&'a [u8]),
    NeighborV1(&'a [u8], usize),
    MeanV1(&'a [u8], usize),
}

impl ExpertScaleSource<'_> {
    fn tiles(
        self,
        count: usize,
        mut emit: impl FnMut(usize, &[u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        match self {
            Self::Raw(bytes) => {
                const TILE: usize = 1 << 20;
                for (i, tile) in bytes.chunks(TILE).enumerate() {
                    emit(i * TILE, tile)?;
                }
                Ok(())
            }
            Self::RangeV1(frame) => crate::inkling_scales::unpack_scales(frame, count, emit),
            Self::MeanV1(frame, width) => {
                crate::inkling_mxfp8::Matrix::parse_mean_scales(frame, count, width)?
                    .for_each_chunk(emit)
            }
            Self::NeighborV1(frame, width) => {
                crate::inkling_mxfp8::Matrix::parse_scales(frame, count, width)?
                    .for_each_chunk(emit)
            }
            Self::RansV1(frame) => {
                crate::inkling_mxfp8::Matrix::parse(frame, count)?.for_each_chunk(emit)
            }
        }
    }

    fn upload(
        self,
        ctx: &Arc<CudaContextHandle>,
        count: usize,
        nvfp4: bool,
        fns: &KernelFns,
    ) -> Result<ExpertScalePlane, String> {
        if let Some(kernels) = fns.packed_experts {
            let width = if nvfp4 { 6 } else { 3 };
            // The scale reader uses a pair of aligned u32 loads, including at the final group.
            let buffer = DeviceBuf::alloc_zeroed(ctx, count / 8 * width + 4)?;
            let mut scratch = Vec::new();
            let mut fits = true;
            self.tiles(count, |offset, bytes| {
                if fits {
                    fits = if nvfp4 {
                        crate::inkling_scales::pack_gpu_tile::<6, 64>(bytes, &mut scratch)
                    } else {
                        crate::inkling_scales::pack_gpu_tile::<3, 112>(bytes, &mut scratch)
                    };
                    if fits {
                        buffer.copy_from_host_at(offset / 8 * width, &scratch)?;
                    }
                }
                Ok(())
            })?;
            if fits {
                return Ok(ExpertScalePlane { buffer, kernels });
            }
        }
        let buffer = match self {
            Self::Raw(bytes) => DeviceBuf::from_host_bytes(ctx, bytes)?,
            Self::RangeV1(_) | Self::RansV1(_) | Self::NeighborV1(_, _) | Self::MeanV1(_, _) => {
                let buffer = DeviceBuf::alloc_zeroed(ctx, count)?;
                self.tiles(count, |offset, bytes| {
                    buffer.copy_from_host_at(offset, bytes)
                })?;
                buffer
            }
        };
        Ok(ExpertScalePlane {
            buffer,
            kernels: fns.experts,
        })
    }
}

struct KernelFns {
    _module: CUmodule,
    experts: ExpertKernels,
    packed_experts: Option<ExpertKernels>,
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
    q8_dp4a_gemv_c3072: Option<CUfunction>,
    q8_dp4a_gemv_c3072_r1: Option<CUfunction>,
    q8_dp4a_gemv_c2304: Option<CUfunction>,
    q8_dp4a_gemv_c2304_r1: Option<CUfunction>,
    short_conv4: CUfunction,
    prep_qkv_and_cache: CUfunction,
    gqa_rel_sink_attn: CUfunction,
    gqa_attn_split: Option<CUfunction>,
    gqa_attn_split_i8: Option<CUfunction>,
    gqa_attn_split_dec: Option<CUfunction>,
    gqa_attn_combine: Option<CUfunction>,
    router_topk8: CUfunction,
    router_gemv_topk8: Option<CUfunction>,
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
    ctrl_down_r8_t2_b: Option<CUfunction>,
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
    build_expert_groups_par_b: Option<CUfunction>,
    moe_combine_top8_b: CUfunction,
    inc_pos_by: CUfunction,
    /// Multi-sequence (#107) row-independent variants.
    prep_qkv_and_cache_ms: Option<CUfunction>,
    gqa_attn_split_dec_ms: Option<CUfunction>,
    gqa_attn_split_i8_ms: Option<CUfunction>,
    gqa_attn_combine_q8_dyn_ms: Option<CUfunction>,
    short_conv4_ms: Option<CUfunction>,
    short_conv4_save_ms: Option<CUfunction>,
    short_conv4_vms: Option<CUfunction>,
    short_conv4_save_vms: Option<CUfunction>,
    conv_commit_ms: Option<CUfunction>,
    argmax_b: CUfunction,
    conv_rewind: CUfunction,
    q8_dp4a_gemm_ktiled: CUfunction,
    mtp_tapmix: CUfunction,
    mtp_copy_anchor: CUfunction,
    mtp_ctx_store: CUfunction,
    mtp_init_draft: CUfunction,
    mtp_qk_norm_rope: CUfunction,
    mtp_attn: CUfunction,
    mtp_attn_split: Option<CUfunction>,
    mtp_attn_merge: Option<CUfunction>,
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
    gqa_rel_sink_attn_q8_b_i8: Option<CUfunction>,
    gqa_rel_sink_attn_q8_decode: Option<CUfunction>,
    gqa_rel_sink_attn_q8_decode_i8: Option<CUfunction>,
    gqa_attn_combine_q8: Option<CUfunction>,
    gqa_attn_combine_q8_dyn: Option<CUfunction>,
    short_conv4_save: Option<CUfunction>,
    short_conv4_save_b: Option<CUfunction>,
    short_conv4_tiled: Option<(CUfunction, CUfunction)>,
    loop_pass_setup_b: Option<CUfunction>,
    q8_dp4a_qkv3_gemv: Option<CUfunction>,
    q8_dp4a_qkv3_gemv_r1: Option<CUfunction>,
    short_conv4_kv2: Option<CUfunction>,
    // v0.7.0 SM120a Wide Tensor Core Prefill Suite (`Issue #10`)
    short_conv4_save_par_b: Option<CUfunction>,
    short_conv4_commit_hist_b: Option<CUfunction>,
    q8_tc_gemm_b: Option<CUfunction>,
    q8_tc_qkv3_gemm_b: Option<CUfunction>,
    quant_nvfp4_2term_b: Option<CUfunction>,
    gather_nvfp4_sorted_b: Option<CUfunction>,
    expand_mxfp8_indices: Option<CUfunction>,
    quant_mxfp8_b: Option<CUfunction>,
    flash_attn_tc_q8_b: Option<CUfunction>,
    flash_attn_tc_q8_b_i8: Option<CUfunction>,
    f32_to_bf16_b: Option<CUfunction>,
    ctrl_norm_bf16_b: Option<CUfunction>,
    bf16_tc_gemm_b: Option<CUfunction>,
    ctrl_apply_tc_b: Option<CUfunction>,
    moe_combine_top8_vec4_b: Option<CUfunction>,
    moe_combine_top8_f16_b: Option<CUfunction>,
    depth_mix_multi_b: Option<CUfunction>,
    // v0.8.0 V13 Prefill Suite (`Issue #79`)
    quant_nvfp4_2term_1p_b: Option<CUfunction>,
    flash_attn_tc_split_b: Option<CUfunction>,
    flash_attn_tc_split_b_i8: Option<CUfunction>,
    flash_attn_split_combine_q8_b: Option<CUfunction>,
    depth_mix_multi_ctrl_norm_bf16_b: Option<CUfunction>,
    // v0.8.0 Solo & Spec Decode PDL + Hoisted Kernels (`Issue #11` / `#104` / `#106`)
    depth_mix_inv: Option<CUfunction>,
    /// Present when every kernel in the cubin waits on `griddepcontrol.wait` before its first
    /// global access, so every launch may use programmatic dependent launch (#152 P0-B).
    pdl_all_marker: Option<CUfunction>,
    ctrl_down_r4_fast: Option<CUfunction>,
    q8_dp4a_gemm_b8_fast: Option<CUfunction>,
    q8_dp4a_gemm_b_ms4: Option<CUfunction>,
    q8_dp4a_gemm_b_ms8: Option<CUfunction>,
    kv_conv4_prep_cache: Option<CUfunction>,
    wo_conv4_save_c3072: Option<CUfunction>,
    build_expert_groups_active_b: Option<CUfunction>,
}

unsafe impl Send for KernelFns {}
unsafe impl Sync for KernelFns {}

struct StageEvents {
    ctx: Arc<CudaContextHandle>,
    h2d_ready: [CUevent; 2],
    comp_done: [CUevent; 2],
}

impl Drop for StageEvents {
    fn drop(&mut self) {
        unsafe {
            for ev in self.h2d_ready.into_iter().chain(self.comp_done) {
                let _ = (self.ctx.api.cuEventDestroy_v2)(ev);
            }
        }
    }
}

unsafe impl Send for StageEvents {}
unsafe impl Sync for StageEvents {}

struct SpilledPrefetchBuf {
    ctx: Arc<CudaContextHandle>,
    buf: DeviceBuf,
    pf_exps: usize,
    ready_ev: CUevent,
    consumed_ev: CUevent,
}

impl Drop for SpilledPrefetchBuf {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.ctx.api.cuEventDestroy_v2)(self.ready_ev);
            let _ = (self.ctx.api.cuEventDestroy_v2)(self.consumed_ev);
        }
    }
}

unsafe impl Send for SpilledPrefetchBuf {}
unsafe impl Sync for SpilledPrefetchBuf {}

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
        has_flash_tc: bool,
        has_drafter: bool,
    ) -> Result<Self, String> {
        let mut bytes = 0;
        let mut allocate = |size| {
            bytes += size;
            DeviceBuf::alloc_zeroed(ctx, size)
        };
        // Multi-sequence rounds (#107) run up to `MAX_MS_ROWS` decode-style rows per pass: every
        // row may take the split-KV path and precomputes all 48 control messages.
        let multi_seq = std::env::var("LEANSONIC_SEQS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .is_some_and(|n| n > 1);
        let msg_rows = if multi_seq {
            48 * MAX_MS_ROWS.min(capacity).max(MAX_SPEC_TOKENS)
        } else {
            48 * MAX_SPEC_TOKENS
        };
        let split_rows = if has_flash_tc && !multi_seq {
            capacity.min(32)
        } else {
            capacity
        };
        let part_ml_bytes = (split_rows * N_HEAD * splits * 2 * 4).max(4);
        let part_o_bytes = (split_rows * N_HEAD * splits * HD * 4).max(4);
        let fixed_part_bytes = if has_flash_tc && !multi_seq {
            part_ml_bytes + part_o_bytes
        } else {
            0
        };
        Ok(Self {
            capacity,
            d_taps: if has_drafter {
                Some(allocate(capacity * 4 * D * 4)?)
            } else {
                None
            },
            d_attn_part_ml: allocate(part_ml_bytes)?,
            d_attn_part_o: allocate(part_o_bytes)?,
            d_b_cur: allocate(capacity * D * 4)?,
            d_b_prelude_x: allocate(capacity * D * 4)?,
            d_b_blocks: allocate(4 * capacity * D * 4)?,
            d_b_prelude_block0: allocate(capacity * D * 4)?,
            d_b_bank: allocate(2 * capacity * D * 4)?,
            d_b_msgs: allocate(msg_rows.max(4 * capacity) * D * 4)?,
            d_b_ctrl_g: allocate(3 * capacity * RANK * 4)?,
            d_b_sel_ids: allocate(capacity * TOP_K * 4)?,
            d_b_sel_wts: allocate(capacity * TOP_K * 4)?,
            d_b_exp_toks: allocate(N_EXP * capacity * 4)?,
            d_b_exp_pairs: allocate(N_EXP * capacity * 4)?,
            d_b_activations: allocate(capacity * (TOP_K * FF).max(3 * D) * 4)?,
            layer_workspace: LayerWorkspace::new(allocate(capacity * TOP_K * D * 4)?, capacity),
            d_b_act_codes: allocate(capacity * (TOP_K * FF).max(3 * N_HEAD * HD))?,
            d_b_act_scales: allocate(capacity * (TOP_K * (FF / 16)).max(3 * N_HEAD * HD / 64) * 4)?,
            bytes_per_row: (bytes
                - fixed_part_bytes
                - msg_rows.saturating_sub(4 * capacity) * D * 4)
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
    device_codebook_scratch: Option<DeviceBuf>,
    slots: Vec<SlotStateGpu>,
    conv_history: DeviceBuf,
    /// Number of independent sequences with their own state (`LEANSONIC_SEQS`, default 1).
    pub n_seqs: usize,
    /// Sequence the single-sequence paths (prefill, decode, verify) operate on.
    cur_seq: usize,
    /// Host positions of the sequences that are not current (`seq_pos[cur_seq]` is stale;
    /// `host_pos` holds the current one).
    seq_pos: Vec<usize>,
    /// `[MAX_SPEC_TOKENS] x {pos, seq}` rows of the multi-sequence round in flight, and the
    /// device pointer the batched visit reads them from while it runs (`0` = single sequence).
    d_ms_rows: DeviceBuf,
    d_ms_tokens: DeviceBuf,
    ms_rows: std::cell::Cell<CUdeviceptr>,
    /// Verify rounds (#107 stage 3): row index inside its sequence, kept-prefix runs, and the
    /// recorded `short_conv4` inputs (`[N_SLOT][6656 * MAX_MS_ROWS]`, allocated on first use).
    d_ms_j: DeviceBuf,
    d_ms_runs: DeviceBuf,
    d_ms_xrec: Option<DeviceBuf>,
    ms_xrec: std::cell::Cell<CUdeviceptr>,
    ms_verify_runs: Vec<(usize, usize)>,
    /// Streams and start event for concurrent per-sequence drafts (`draft_for_seqs`).
    draft_streams: Vec<CUstream>,
    draft_ready: CUevent,
    /// Captured decode rounds per row count (captured on a row count's second use).
    ms_graphs: std::collections::HashMap<usize, CapturedGraphExec>,
    ms_round_hits: std::collections::HashMap<usize, u32>,
    /// `LEANSONIC_MS_GRAPH=0` launches every multi-sequence round directly (A/B).
    ms_graph: bool,
    pub resident_expert_layers: usize,
    pub spilled_expert_layers: usize,
    pub max_ctx: usize,
    use_moe_dp4a: bool,
    stage_host_up: bool,
    /// `LEANSONIC_MOE_DOWN_XW`: 0 = off, 1 = VRAM-resident layers (default), 2 = all layers.
    moe_down_xw: u8,
    /// Multi-sequence dense GEMMs use the token-tiled kernels (`LEANSONIC_MS_GEMM=0` keeps
    /// `ink2_q8_dp4a_gemm_b`).
    ms_gemm_tiles: bool,
    verify_moe: bool,
    solo_conv_fuse: bool,
    kv_conv_prep: bool,
    fuse_router: bool,
    c3072_r1: bool,
    tc_q8: bool,
    tc_attn: bool,
    tc_nvfp4: bool,
    tc_mxfp8: bool,
    tc_bf16: bool,
    par_conv: bool,
    pdl: bool,
    /// Launch every kernel with PDL, not only the single-sequence decode chain.
    pdl_all: bool,
    max_prefill_chunk: usize,
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
    d_ctrl_inv_cur: DeviceBuf,
    d_ctrl_inv_msgs: DeviceBuf,
    /// `[8600]` floats: `[0..6552)` = `q,r,gate,sink`, `[6552..7576)` = `k_swa`, `[7576..8600)` = `v_swa`
    d_qrgs: DeviceBuf,
    d_k_raw: DeviceBuf,
    d_v_raw: DeviceBuf,
    d_k_conv: DeviceBuf,
    d_v_conv: DeviceBuf,
    d_attn_out: DeviceBuf,
    /// Split-KV GQA attention (`LEANSONIC_ATTN_SPLIT=G`, `0` = legacy 24-CTA kernel).
    attn_split_g: usize,
    /// Split count for decode / DSpark verify rows (`b_len <= MAX_SPEC_TOKENS`,
    /// `LEANSONIC_ATTN_SPLIT_DECODE`, defaults to `attn_split_g`). Decode and verify always share
    /// it, so `--spec 0` and `--spec K` stay token-identical; prefill chunks keep `attn_split_g`.
    attn_split_g_decode: usize,
    /// Decode / verify split count of the 4 full-attention (global) layers
    /// (`LEANSONIC_ATTN_SPLIT_DECODE_GLOBAL`, default `max(attn_split_g_decode, 32)`): their
    /// single row scans the whole context, and 8 x 16 CTAs leave most SMs idle at 32K-128K.
    attn_split_g_decode_global: usize,
    d_dec_part_ml: DeviceBuf,
    d_dec_part_o: DeviceBuf,
    d_wo_out: DeviceBuf,
    d_mlp_h: DeviceBuf,
    d_router_logits: DeviceBuf,
    d_router_done: DeviceBuf,
    d_sel_ids: DeviceBuf,
    d_sel_wts: DeviceBuf,
    d_moe_act: DeviceBuf,
    d_moe_y: DeviceBuf,
    d_logits: DeviceBuf,
    // Quantized activation scratch buffers (`Q8_64` up to 3 vectors & `Q8_16` up to 8 vectors)
    d_act_q8_64: DeviceBuf,      // 3 * 3072 B
    d_act_sc_64: DeviceBuf,      // 3 * 48 * 4 B
    d_act_q8_16: DeviceBuf,      // max 8 * 1152 B
    d_act_sc_16: DeviceBuf,      // max 8 * 72 * 4 B
    d_moe_q8_16: DeviceBuf,      // 8 * 1152 B (disjoint from `d_act_q8_16` for fused `up_half_q8`)
    d_moe_sc_16: DeviceBuf,      // 8 * 72 * 4 B
    h_token_ring: PinnedHostBuf, // 65 * 4 B pinned token ring for zero-pageable GPU-chained decode
    // Multi-token batch scratch buffers (up to the selected prefill capacity) + Speculative verify buffers (`B <= 8`)
    prefill: PrefillBuffers,
    d_b_exp_counts: DeviceBuf,
    d_b_active_exps: DeviceBuf,
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
    stage_events: StageEvents,
    d_w13_prefetch: Option<SpilledPrefetchBuf>,
    pub enable_cuda_graph: bool,
}

unsafe impl Send for Inkling2GpuEngine {}

#[inline]
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn read_control_weight<'a>(
    st: &'a MappedSafeTensors,
    name: &str,
    shape: &[usize],
) -> Result<std::borrow::Cow<'a, [u8]>, String> {
    let count = shape.iter().product::<usize>();
    match (st.get(name), st.get(&format!("{name}.bf16planes_v1"))) {
        (Some(t), None) if t.dtype == "BF16" && t.shape == shape && t.data.len() == count * 2 => {
            Ok(std::borrow::Cow::Borrowed(t.data))
        }
        (None, Some(t)) if t.dtype == "U8" && t.shape == [t.data.len()] => {
            crate::inkling_bf16_planes::unpack(t.data, count)
                .map(std::borrow::Cow::Owned)
                .map_err(|e| format!("{name}: {e}"))
        }
        _ => Err(format!(
            "missing, ambiguous or invalid control weight {name}"
        )),
    }
}

fn concat_slices(slices: &[&[u8]]) -> Vec<u8> {
    let total: usize = slices.iter().map(|s| s.len()).sum();
    let mut out = Vec::with_capacity(total);
    for s in slices {
        out.extend_from_slice(s);
    }
    out
}

/// `LEANSONIC_PDL_ALL=0` (or `LEANSONIC_PDL=0`) keeps PDL on the single-sequence decode chain only.
/// Otherwise every launch uses PDL when the cubin carries `ink2_pdl_all_marker`, i.e. every
/// kernel executes `griddepcontrol.wait` before touching global memory (#152 P0-B).
pub(crate) fn pdl_all_enabled(ctx: &CudaContextHandle, marker: Option<CUfunction>) -> bool {
    let on = |k: &str| std::env::var(k).map(|v| v != "0").unwrap_or(true);
    on("LEANSONIC_PDL")
        && on("LEANSONIC_PDL_ALL")
        && ctx.api.cuLaunchKernelEx.is_some()
        && marker.is_some()
}

/// Programmatic dependent launch: the kernel may be scheduled while its stream predecessor drains.
pub(crate) fn launch_pdl_3d(
    ctx: &CudaContextHandle,
    stream: CUstream,
    func: CUfunction,
    grid: (u32, u32, u32),
    block: u32,
    shared_mem_bytes: u32,
    params: &mut [*mut c_void],
) -> Result<(), String> {
    let launch_ex = ctx
        .api
        .cuLaunchKernelEx
        .ok_or_else(|| "cuLaunchKernelEx unavailable".to_string())?;
    let mut attr = CUlaunchAttribute::programmatic_stream_serialization();
    let cfg = CUlaunchConfig {
        gridDimX: grid.0,
        gridDimY: grid.1,
        gridDimZ: grid.2,
        blockDimX: block,
        blockDimY: 1,
        blockDimZ: 1,
        sharedMemBytes: shared_mem_bytes,
        _pad0: 0,
        hStream: stream,
        attrs: &mut attr,
        numAttrs: 1,
        _pad1: 0,
    };
    unsafe {
        check_cu(
            launch_ex(&cfg, func, params.as_mut_ptr(), core::ptr::null_mut()),
            &format!("cuLaunchKernelEx(grid={grid:?}, block={block}, smem={shared_mem_bytes})"),
        )
    }
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
    let result = allocate_planar_q8(ctx, rows, cols)?;
    upload_bf16_rows(&result, bf16_bytes, 0, rows, cols)?;
    Ok(result)
}

fn allocate_planar_q8(
    ctx: &Arc<CudaContextHandle>,
    rows: usize,
    cols: usize,
) -> Result<PlanarQ8Gpu, String> {
    Ok(PlanarQ8Gpu {
        codes: DeviceBuf::alloc_zeroed(ctx, rows * cols)?,
        scales64: DeviceBuf::alloc_zeroed(ctx, rows * (cols / 64) * 4)?,
        cols: cols as i32,
    })
}

fn upload_bf16_rows(
    result: &PlanarQ8Gpu,
    bf16_bytes: &[u8],
    row_offset: usize,
    rows: usize,
    cols: usize,
) -> Result<(), String> {
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    let scale_stride = (cols / 64) * 4;
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
        result
            .codes
            .copy_from_host_at((row_offset + first) * cols, tile_codes)?;
        result
            .scales64
            .copy_from_host_at((row_offset + first) * scale_stride, tile_scales)?;
    }
    Ok(())
}

/// Pack complete BF16 rows into signed Q8 codes and little-endian FP32 scales.
pub fn pack_q8_rows(bf16_bytes: &[u8], cols: usize, codes: &mut [u8], scales_bytes: &mut [u8]) {
    assert!(cols > 0 && cols.is_multiple_of(64));
    assert!(codes.len().is_multiple_of(cols));
    assert_eq!(bf16_bytes.len(), codes.len() * 2);
    assert_eq!(scales_bytes.len(), codes.len() / 64 * 4);
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

#[derive(Clone, Copy)]
enum ExpertMatrix<'a> {
    Raw(&'a [u8]),
    Packed {
        bytes: &'a [u8],
        count: usize,
        contextual: bool,
    },
}

impl<'a> ExpertMatrix<'a> {
    fn read(
        st: &'a MappedSafeTensors,
        name: &str,
        dtype: &str,
        shape: &[usize],
    ) -> Result<Self, String> {
        let count = shape.iter().product();
        match (
            st.get(name),
            st.get(&format!("{name}.rans_v1")),
            st.get(&format!("{name}.nvctx_v1")),
        ) {
            (Some(t), None, None)
                if t.dtype == dtype && t.shape == shape && t.data.len() == count =>
            {
                Ok(Self::Raw(t.data))
            }
            (None, Some(t), None) if t.dtype == "U8" && t.shape == [t.data.len()] => {
                Ok(Self::Packed {
                    bytes: t.data,
                    count,
                    contextual: false,
                })
            }
            (None, None, Some(t))
                if dtype == "U8" && t.dtype == "U8" && t.shape == [t.data.len()] =>
            {
                Ok(Self::Packed {
                    bytes: t.data,
                    count,
                    contextual: true,
                })
            }
            _ => Err(format!(
                "missing, ambiguous or invalid {dtype} expert matrix {name}"
            )),
        }
    }
    fn len(&self) -> usize {
        match self {
            Self::Raw(bytes) => bytes.len(),
            Self::Packed { count, .. } => *count,
        }
    }
    // A standard byte-rANS frame lists exactly the symbols its decoder may emit.
    // Store the lookup table and the two indices per byte together, aligned to 16 bytes.
    fn device_codebook(self) -> Result<Vec<u8>, String> {
        let Self::Packed {
            bytes,
            count,
            contextual: false,
        } = self
        else {
            return Err("device codebooks require rANS MXFP8 frames".into());
        };
        let matrix = crate::inkling_mxfp8::Matrix::parse(bytes, count)?;
        let mut result = Vec::with_capacity(16 + count / 2);
        let mut index = [0u8; 256];
        for (symbol, &freq) in bytes[..512].as_chunks::<2>().0.iter().enumerate() {
            if freq != [0, 0] {
                if result.len() == 16 {
                    return Err("MXFP8 frame exceeds the 16-symbol device codebook limit".into());
                }
                index[symbol] = result.len() as u8;
                result.push(symbol as u8);
            }
        }
        result.resize(16, 0);
        matrix.for_each_chunk(|_, tile| {
            let (pairs, tail) = tile.as_chunks::<2>();
            if !tail.is_empty() {
                return Err("unaligned MXFP8 device codebook tile".into());
            }
            result.extend(
                pairs
                    .iter()
                    .map(|&[a, b]| index[a as usize] | (index[b as usize] << 4)),
            );
            Ok(())
        })?;
        Ok(result)
    }

    fn place(self, ctx: &Arc<CudaContextHandle>, on_device: bool) -> Result<TierBuf, String> {
        match self {
            Self::Raw(bytes) => {
                if on_device {
                    if let Ok(buf) = DeviceBuf::from_host_bytes(ctx, bytes) {
                        return Ok(TierBuf::Vram(buf));
                    }
                }
                Ok(TierBuf::HostPinned(PinnedHostBuf::from_bytes(ctx, bytes)?))
            }
            Self::Packed {
                bytes,
                count,
                contextual,
            } => {
                let matrix = if contextual {
                    crate::inkling_mxfp8::Matrix::parse_nvfp4(bytes, count)?
                } else {
                    crate::inkling_mxfp8::Matrix::parse(bytes, count)?
                };
                if on_device {
                    match DeviceBuf::alloc_zeroed(ctx, count) {
                        Ok(buf) => {
                            matrix.for_each_chunk(|offset, bytes| {
                                buf.copy_from_host_at(offset, bytes)
                            })?;
                            return Ok(TierBuf::Vram(buf));
                        }
                        Err(e) if e.starts_with("CUDA error 2 in `cuMemAlloc_v2`") => {}
                        Err(e) => return Err(e),
                    }
                }
                let pinned = PinnedHostBuf::alloc_uninit(ctx, count)?;
                // The pinned allocation is private until decode_at initializes every byte.
                let dst = unsafe { std::slice::from_raw_parts_mut(pinned.host_ptr.cast(), count) };
                matrix.decode_at(0, dst)?;
                Ok(TierBuf::HostPinned(pinned))
            }
        }
    }
}

// Keep the thrice-visited layers resident first. Among single-visit layers the
// grouped MXFP8 kernels incur the largest penalty when reading weights over PCIe.
fn plan_expert_placement(
    expert_views: &[[ExpertMatrix<'_>; 2]],
    budget: usize,
) -> [[bool; 2]; N_LAYER] {
    extend_expert_placement(expert_views, budget, [[false; 2]; N_LAYER])
}

fn extend_expert_placement(
    expert_views: &[[ExpertMatrix<'_>; 2]],
    mut remaining: usize,
    mut resident: [[bool; 2]; N_LAYER],
) -> [[bool; 2]; N_LAYER] {
    let mut place = |layer: usize, matrix: usize| {
        let bytes = expert_views[layer][matrix].len();
        if !resident[layer][matrix] && bytes <= remaining {
            resident[layer][matrix] = true;
            remaining -= bytes;
        }
    };
    for layer in (6..18).chain(NVFP4_LAYERS..N_LAYER) {
        place(layer, 0);
        place(layer, 1);
    }
    // The smaller NVFP4 down matrices fit independently of their gate/up matrices.
    for matrix in [1, 0] {
        for layer in (0..6).chain(18..NVFP4_LAYERS) {
            place(layer, matrix);
        }
    }
    resident
}

fn known_prompt_limit(tokens: usize) -> usize {
    tokens
        .clamp(EXPERT_PAGE, MAX_BATCH_TOKENS)
        .next_power_of_two()
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
        Self::load_model_path(
            model_dir.as_ref(),
            max_ctx,
            enable_cuda_graph,
            spec_k,
            MAX_BATCH_TOKENS,
        )
    }

    /// Size the prefill workspace for a prompt already tokenized by the caller.
    pub fn load_for_prompt(
        model_dir: impl AsRef<Path>,
        max_ctx: usize,
        enable_cuda_graph: bool,
        spec_k: usize,
        prompt_tokens: usize,
    ) -> Result<Self, String> {
        Self::load_model_path(
            model_dir.as_ref(),
            max_ctx,
            enable_cuda_graph,
            spec_k,
            known_prompt_limit(prompt_tokens),
        )
    }

    // Share the loader across callers passing different path types.
    fn load_model_path(
        model_dir: &Path,
        max_ctx: usize,
        enable_cuda_graph: bool,
        spec_k: usize,
        prompt_limit: usize,
    ) -> Result<Self, String> {
        let spec_k = spec_k.min(MAX_SPEC_TOKENS - 1);
        let user_reserve = match std::env::var("LEANSONIC_VRAM_RESERVE_MIB") {
            Ok(raw) => raw
                .parse::<usize>()
                .ok()
                .and_then(|mib| mib.checked_mul(1024 * 1024))
                .ok_or("LEANSONIC_VRAM_RESERVE_MIB must be a nonnegative whole number of MiB that fits in bytes")?,
            Err(std::env::VarError::NotPresent) => 0,
            Err(e) => return Err(e.to_string()),
        };
        let t_load = Instant::now();
        let ctx = CudaContextHandle::init(0)?;
        ctx.bind_current()?;
        // `LEANSONIC_ATTN_SPLIT=G` (`1..=64`) enables Split-KV + GQA-grouped attention across
        // BOTH Solo Decode (`B=1`) and `DSpark` Verify / Prefill (`B>=1`) for `0-ULP` spec parity!
        let attn_split_explicit = std::env::var("LEANSONIC_ATTN_SPLIT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        let use_moe_dp4a = std::env::var("LEANSONIC_MOE_DP4A")
            .map(|v| v != "0")
            .unwrap_or(true);
        let use_tc_prefill = std::env::var("LEANSONIC_TC_PREFILL")
            .map(|v| v != "0")
            .unwrap_or(true);
        let env_flag = |k: &str| std::env::var(k).map(|v| v != "0").unwrap_or(use_tc_prefill);
        let tc_q8 = env_flag("LEANSONIC_TC_Q8");
        let tc_attn = env_flag("LEANSONIC_TC_ATTN");
        let tc_nvfp4 = env_flag("LEANSONIC_TC_NVFP4");
        let tc_mxfp8 = env_flag("LEANSONIC_TC_MXFP8");
        let tc_bf16 = env_flag("LEANSONIC_TC_BF16");
        let par_conv = env_flag("LEANSONIC_PAR_CONV");

        let module_image = if let Ok(cubin_path) = std::env::var("LEANSONIC_CUBIN_PATH") {
            std::fs::read(&cubin_path).map_err(|e| format!("read {cubin_path}: {e}"))?
        } else {
            compile_cuda_to_ptx_nvrtc(CUDA_KERNEL_SRC, "compute_120")?
        };

        let pack_gpu_scales = match std::env::var("LEANSONIC_GPU_SCALES").as_deref() {
            Ok("packed") | Err(std::env::VarError::NotPresent) => true,
            Ok("raw") => false,
            _ => return Err("LEANSONIC_GPU_SCALES must be raw or packed".into()),
        };

        let fns = Self::load_module_functions(&ctx, &module_image, pack_gpu_scales)?;
        let has_split_attn = fns.gqa_attn_split.is_some() && fns.gqa_attn_combine.is_some();
        let attn_split_g = match attn_split_explicit {
            Some(g) => {
                if g > 0 && !has_split_attn {
                    return Err(
                        "LEANSONIC_ATTN_SPLIT>0 requested, but loaded cubin lacks ink2_gqa_attn_split/combine"
                            .into(),
                    );
                }
                g.min(64)
            }
            None => {
                if has_split_attn {
                    16
                } else {
                    0
                }
            }
        };
        // Decode-style rows per pass: DSpark verify, or a multi-sequence round (#107).
        let multi_seq = std::env::var("LEANSONIC_SEQS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .is_some_and(|n| n > 1);
        // Long contexts want more splits for the single decode row than prefill chunks do
        // (prefill already has `b_len` rows of parallelism). At short single-sequence contexts
        // (`max_ctx <= 4096`), unsplit `gqa_rel_sink_attn_q8_decode` (`G = 0`) eliminates all 48
        // `gqa_attn_combine_q8_dyn` launches and `part_o` round-trips while keeping `spec 0 == spec K`.
        let default_dec_split = if !multi_seq && max_ctx <= 4096 {
            0
        } else {
            attn_split_g
        };
        let attn_split_g_decode = if attn_split_g > 0 {
            std::env::var("LEANSONIC_ATTN_SPLIT_DECODE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(default_dec_split)
                .min(64)
        } else {
            0
        };
        let default_global_dec_split = if max_ctx > 16384 {
            attn_split_g_decode.max(32)
        } else {
            attn_split_g_decode
        };
        let attn_split_g_decode_global = if attn_split_g > 0 {
            std::env::var("LEANSONIC_ATTN_SPLIT_DECODE_GLOBAL")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(default_global_dec_split)
                .min(64)
        } else {
            0
        };
        let dec_split_max = attn_split_g_decode.max(attn_split_g_decode_global);
        let dec_rows = if multi_seq {
            MAX_MS_ROWS.max(MAX_SPEC_TOKENS)
        } else {
            MAX_SPEC_TOKENS
        };
        let st_path = model_dir.join("model-00001-of-00001.safetensors");
        let st = MappedSafeTensors::open(&st_path)
            .map_err(|e| format!("open {}: {e}", st_path.display()))?;

        let embed = match (
            st.get("model.llm.embed.weight"),
            st.get("model.llm.embed.weight.prefix_v1"),
        ) {
            (Some(raw), None)
                if raw.dtype == "BF16"
                    && raw.shape == [PADDED_VOCAB, D]
                    && raw.data.len() == PADDED_VOCAB * D * 2 =>
            {
                PinnedHostBuf::from_bytes(&ctx, raw.data)?
            }
            (None, Some(packed)) if packed.dtype == "U8" && packed.shape == [packed.data.len()] => {
                let buf = PinnedHostBuf::alloc_uninit(&ctx, PADDED_VOCAB * D * 2)?;
                // This allocation is exclusively owned until all tiles decode.
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(
                        buf.host_ptr.cast::<std::mem::MaybeUninit<u8>>(),
                        buf.byte_len,
                    )
                };
                crate::inkling_embedding::unpack_embedding(packed.data, dst)?;
                buf
            }
            _ => return Err("invalid or ambiguous Inkling input embedding".into()),
        };
        let embed = TierBuf::HostPinned(embed);
        st.discard_resident_pages();
        let embed_norm = DeviceBuf::from_host_bytes(
            &ctx,
            st.require("model.llm.embed_norm.weight", "BF16", &[D]).data,
        )?;
        let final_norm = DeviceBuf::from_host_bytes(
            &ctx,
            st.require("model.llm.norm.weight", "BF16", &[D]).data,
        )?;
        let unembed_q8 = load_output_head(&ctx, &st)?;
        let output_query = DeviceBuf::from_host_bytes(
            &ctx,
            st.require("model.llm.attn_res.output_query", "F32", &[D])
                .data,
        )?;

        // Three prior FP32 samples per channel for each of the four convolutions.
        let kv_history_bytes = 3 * N_KV * HD * size_of::<f32>();
        let residual_history_bytes = 3 * D * size_of::<f32>();
        let slot_history_bytes = 2 * kv_history_bytes + 2 * residual_history_bytes;
        // `LEANSONIC_SEQS=C` keeps per-sequence state (KV, rings, conv histories) for C
        // independent sequences (#12 / #107 multi-sequence decode). Default 1 is unchanged.
        let n_seqs = std::env::var("LEANSONIC_SEQS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(1);
        let seq_hist_bytes = N_SLOT * slot_history_bytes;
        let conv_history = DeviceBuf::alloc_zeroed(&ctx, n_seqs * seq_hist_bytes)?;
        // `LEANSONIC_KV=int8` stores Full-attention (global) KV as int8 + per-(row, head) FP32
        // absmax scales (#14). Default FP16 is unchanged bit-for-bit.
        // `LEANSONIC_KV=int8-all` also stores the sliding-window rings as INT8 (same kernels:
        // each slot picks the INT8 path from its own scale buffers).
        // #80: on 16 GB cards a long context's FP16 global KV (up to ~2 GiB at 128K) pushes
        // expert layers out of VRAM, so with `LEANSONIC_KV` unset and `max_ctx > 8,192` they
        // start with INT8 global KV. Standard contexts keep FP16 (bit-identical on all GPUs).
        let (_, total_vram_at_load) = ctx.mem_info()?;
        let kv_env = std::env::var("LEANSONIC_KV").ok();
        let kv_auto = kv_env.is_none()
            && total_vram_at_load <= AUTO_INT8_KV_MAX_VRAM
            && max_ctx > AUTO_INT8_KV_MIN_CTX;
        // #107: C sequences keep C copies of every KV cache and ring. When their FP16 bytes
        // would not fit next to the resident model (~17.5 GiB with workspaces), store all
        // layers as INT8 instead (e.g. PRO 4500: C = 64 x ctx 4096 needs ~19.8 GiB in FP16).
        let ms_fp16_kv_bytes = n_seqs
            * (8 * max_ctx + (N_SLOT - 8) * max_ctx.min(EXTENT + EXPERT_PAGE))
            * N_KV
            * HD
            * 2
            * 2;
        let kv_auto_ms = kv_env.is_none()
            && n_seqs > 1
            && ms_fp16_kv_bytes + (17_920usize << 20) > total_vram_at_load;
        let kv_mode = if kv_auto_ms {
            "int8-all".to_string()
        } else if kv_auto {
            "int8".to_string()
        } else {
            kv_env.unwrap_or_default().to_ascii_lowercase()
        };
        let kv_int8 = kv_mode == "int8" || kv_mode == "int8-all";
        let kv_int8_swa = kv_mode == "int8-all";
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
            // Opt-in INT8 KV (`1 + 4/128` B/value vs `2`): Full-attention layers, and with
            // `int8-all` the sliding-window rings too.
            let int8 = kv_int8 && (is_global || kv_int8_swa);
            let kv_bytes_per_value = if int8 { 1 } else { 2 };
            let seq_kv_bytes = kv_cap * N_KV * HD * kv_bytes_per_value;
            let seq_scale_bytes = kv_cap * N_KV * 4;
            slots.push(SlotStateGpu {
                kv_cap,
                is_global,
                k_cache: DeviceBuf::alloc_zeroed(&ctx, n_seqs * seq_kv_bytes)?,
                v_cache: DeviceBuf::alloc_zeroed(&ctx, n_seqs * seq_kv_bytes)?,
                k_scale: if int8 {
                    Some(DeviceBuf::alloc_zeroed(&ctx, n_seqs * seq_scale_bytes)?)
                } else {
                    None
                },
                v_scale: if int8 {
                    Some(DeviceBuf::alloc_zeroed(&ctx, n_seqs * seq_scale_bytes)?)
                } else {
                    None
                },
                hist_k,
                hist_v,
                hist_attn,
                hist_mlp,
                seq_kv_bytes: seq_kv_bytes as u64,
                seq_scale_bytes: seq_scale_bytes as u64,
                seq_hist_bytes: seq_hist_bytes as u64,
            });
        }

        // One `[N_SLOT, 4]` pointer table per sequence.
        let mut hist_ptr_bytes = Vec::with_capacity(n_seqs * N_SLOT * 4 * 8);
        for q in 0..n_seqs {
            for s in &slots {
                for p in [
                    s.hist_k_at(q),
                    s.hist_v_at(q),
                    s.hist_attn_at(q),
                    s.hist_mlp_at(q),
                ] {
                    hist_ptr_bytes.extend_from_slice(&p.to_le_bytes());
                }
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
                q8_dp4a_gemm_b8_fast: fns.q8_dp4a_gemm_b8_fast,
                q8_dp4a_gemv: fns.q8_dp4a_gemv,
                q8_dp4a_gemm_ktiled: fns.q8_dp4a_gemm_ktiled,
                tapmix: fns.mtp_tapmix,
                copy_anchor: fns.mtp_copy_anchor,
                ctx_store: fns.mtp_ctx_store,
                init_draft: fns.mtp_init_draft,
                qk_norm_rope: fns.mtp_qk_norm_rope,
                attn: fns.mtp_attn,
                attn_split: fns.mtp_attn_split.zip(fns.mtp_attn_merge),
                add_inplace: fns.mtp_add_inplace,
                silu_mul: fns.mtp_silu_mul,
                concat_frnn: fns.mtp_concat_frnn,
                add_bias_rows: fns.mtp_add_bias_rows,
                markov_raw: fns.mtp_markov_raw,
                markov_step_q8: fns.mtp_markov_step_q8,
                fold_mv_q8: fns.mtp_fold_mv_q8,
                fold_fin: fns.mtp_fold_fin,
                pdl_all: pdl_all_enabled(&ctx, fns.pdl_all_marker),
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

            let q = ("wq_du.weight", N_HEAD * HD);
            let r = ("wr_du.weight", N_REL * N_HEAD);
            let gate = ("gate_attn.weight", N_HEAD * HD);
            let sink = ("attn_sink_proj.weight", N_HEAD);
            let k = ("wk_dv.weight", N_KV * HD);
            let v = ("wv_dv.weight", N_KV * HD);
            let projection =
                |parts: &[(&str, usize)]| load_attention_matrix(&ctx, &st, &pfx, parts, D);
            let (wq_all_q8, wk_glob_q8, wv_glob_q8) = if global {
                (
                    projection(&[q, r, gate, sink])?,
                    Some(projection(&[k, ("k_sconv.weight_proj", 4 * N_KV)])?),
                    Some(projection(&[v, ("v_sconv.weight_proj", 4 * N_KV)])?),
                )
            } else {
                (projection(&[q, r, gate, sink, k, v])?, None, None)
            };
            let wo_q8 =
                load_attention_matrix(&ctx, &st, &pfx, &[("wo_ud.weight", D)], N_HEAD * HD)?;

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
                &read_control_weight(&st, &format!("{pfx}mlp.gate.weight"), &[N_EXP, D])?,
            )?;
            let router_bias = DeviceBuf::from_host_bytes(
                &ctx,
                st.require(&format!("{pfx}mlp.gate.bias"), "F32", &[N_EXP])
                    .data,
            )?;

            let ctrl_slices = |cname: &str| {
                let prefix = format!("{pfx}controllers.{cname}.");
                Ok::<_, String>([
                    read_control_weight(&st, &format!("{prefix}down_weight"), &[RANK, 2 * D])?,
                    read_control_weight(&st, &format!("{prefix}gain_weight"), &[D, RANK])?,
                    read_control_weight(&st, &format!("{prefix}forget_weight"), &[D, RANK])?,
                ])
            };

            let (ctrl3, ctrl_mlp) = if in_core {
                let [da, ga, fa] = ctrl_slices("attn")?;
                let [dk, gk, fk] = ctrl_slices("k")?;
                let [dv, gv, fv] = ctrl_slices("v")?;
                let [dm, gm, fm] = ctrl_slices("mlp")?;
                (
                    Some(CtrlGpu {
                        down: DeviceBuf::from_host_bytes(&ctx, &concat_slices(&[&da, &dk, &dv]))?,
                        gain: DeviceBuf::from_host_bytes(&ctx, &concat_slices(&[&ga, &gk, &gv]))?,
                        forget: DeviceBuf::from_host_bytes(&ctx, &concat_slices(&[&fa, &fk, &fv]))?,
                    }),
                    Some(CtrlGpu {
                        down: DeviceBuf::from_host_bytes(&ctx, &dm)?,
                        gain: DeviceBuf::from_host_bytes(&ctx, &gm)?,
                        forget: DeviceBuf::from_host_bytes(&ctx, &fm)?,
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

            let load_scale = |name: &str, dtype: &str, shape: &[usize]| {
                let count = shape.iter().product();
                let source = match (
                    st.get(name),
                    st.get(&format!("{name}.range_v1")),
                    st.get(&format!("{name}.rans_v1")),
                    st.get(&format!("{name}.neighbor_v1")),
                    st.get(&format!("{name}.mean8_v1")),
                ) {
                    (Some(t), None, None, None, None)
                        if t.dtype == dtype && t.shape == shape && t.data.len() == count =>
                    {
                        ExpertScaleSource::Raw(t.data)
                    }
                    (None, Some(t), None, None, None)
                        if t.dtype == "U8" && t.shape == [t.data.len()] =>
                    {
                        ExpertScaleSource::RangeV1(t.data)
                    }
                    (None, None, Some(t), None, None)
                        if t.dtype == "U8" && t.shape == [t.data.len()] =>
                    {
                        ExpertScaleSource::RansV1(t.data)
                    }
                    (None, None, None, Some(t), None)
                        if t.dtype == "U8" && t.shape == [t.data.len()] =>
                    {
                        ExpertScaleSource::NeighborV1(t.data, shape[2])
                    }
                    (None, None, None, None, Some(t))
                        if t.dtype == "U8" && t.shape == [t.data.len()] =>
                    {
                        ExpertScaleSource::MeanV1(t.data, shape[2])
                    }
                    _ => {
                        return Err(format!(
                            "missing, ambiguous or invalid Inkling expert scale {name}"
                        ))
                    }
                };
                source
                    .upload(&ctx, count, il < NVFP4_LAYERS, &fns)
                    .map_err(|e| format!("{name}: {e}"))
            };
            let (dtype, codes_per_byte) = if il < NVFP4_LAYERS {
                ("U8", 2)
            } else {
                ("F8_E4M3", 1)
            };
            expert_views.push([
                ExpertMatrix::read(
                    &st,
                    &format!("{pfx}mlp.experts.w13_weight"),
                    dtype,
                    &[N_EXP, 2 * FF, D / codes_per_byte],
                )?,
                ExpertMatrix::read(
                    &st,
                    &format!("{pfx}mlp.experts.w2_weight"),
                    dtype,
                    &[N_EXP, D, FF / codes_per_byte],
                )?,
            ]);
            let expert_scales = if il < NVFP4_LAYERS {
                ExpertScales::Nvfp4 {
                    s13: load_scale(
                        &format!("{pfx}mlp.experts.w13_weight_scale"),
                        "F8_E4M3",
                        &[N_EXP, 2 * FF, D / 16],
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
                    s2: load_scale(
                        &format!("{pfx}mlp.experts.w2_weight_scale"),
                        "F8_E4M3",
                        &[N_EXP, D, FF / 16],
                    )?,
                    a2: DeviceBuf::from_host_bytes(
                        &ctx,
                        st.require(&format!("{pfx}mlp.experts.w2_weight_amax"), "F32", &[N_EXP])
                            .data,
                    )?,
                }
            } else {
                ExpertScales::Mxfp8 {
                    s13: load_scale(
                        &format!("{pfx}mlp.experts.w13_weight_scale"),
                        "U8",
                        &[N_EXP, 2 * FF, D / 32],
                    )?,
                    s2: load_scale(
                        &format!("{pfx}mlp.experts.w2_weight_scale"),
                        "U8",
                        &[N_EXP, D, FF / 32],
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

        let saved_scale_bytes = if pack_gpu_scales {
            let mut packed = 0;
            let mut saved = 0;
            for layer in &layers {
                let (up, down, group) = match &layer.expert_scales {
                    ExpertScales::Nvfp4 { s13, s2, .. } => (s13, s2, 16),
                    ExpertScales::Mxfp8 { s13, s2 } => (s13, s2, 32),
                };
                for (plane, raw_bytes) in [
                    (up, 2 * N_EXP * D * FF / group),
                    (down, N_EXP * D * FF / group),
                ] {
                    if plane.buffer.byte_len < raw_bytes {
                        packed += 1;
                        saved += raw_bytes - plane.buffer.byte_len;
                    }
                }
            }
            eprintln!(
                "[LeanSonic GPU] Packed {packed} expert scale planes, saving {saved} device bytes"
            );
            saved
        } else {
            0
        };
        let make_ev = || -> Result<CUevent, String> {
            let mut ev: CUevent = core::ptr::null_mut();
            unsafe {
                check_cu(
                    (ctx.api.cuEventCreate)(&mut ev, CU_EVENT_DISABLE_TIMING),
                    "cuEventCreate(stage)",
                )?;
            }
            Ok(ev)
        };
        let stage_events = StageEvents {
            ctx: Arc::clone(&ctx),
            h2d_ready: [make_ev()?, make_ev()?],
            comp_done: [make_ev()?, make_ev()?],
        };
        let has_flash_tc = tc_attn
            && fns.flash_attn_tc_q8_b.is_some()
            && (!kv_int8 || fns.flash_attn_tc_q8_b_i8.is_some());
        let pdl = std::env::var("LEANSONIC_PDL")
            .map(|v| v != "0")
            .unwrap_or(true)
            && ctx.api.cuLaunchKernelEx.is_some()
            && fns.depth_mix_inv.is_some();
        let pdl_all = pdl && pdl_all_enabled(&ctx, fns.pdl_all_marker);
        let d_ms_rows = DeviceBuf::alloc_zeroed(&ctx, MAX_MS_ROWS * 8)?;
        let d_ms_j = DeviceBuf::alloc_zeroed(&ctx, MAX_MS_ROWS * 4)?;
        let d_ms_runs = DeviceBuf::alloc_zeroed(&ctx, MAX_MS_ROWS * 16)?;
        let d_ms_tokens = DeviceBuf::alloc_zeroed(&ctx, MAX_MS_ROWS * 4)?;
        let mut device_codebooks = Vec::new();
        let device_codebook_scratch = if std::env::var("LEANSONIC_MXFP8_DEVICE_U4").as_deref()
            == Ok("1")
        {
            if fns.expand_mxfp8_indices.is_none() {
                return Err("missing ink2_expand_mxfp8_indices in the selected CUDA module".into());
            }
            for &[up, down] in &expert_views[NVFP4_LAYERS..] {
                device_codebooks.push([up.device_codebook()?, down.device_codebook()?]);
            }
            let largest = expert_views[NVFP4_LAYERS..]
                .iter()
                .flatten()
                .map(ExpertMatrix::len)
                .max()
                .ok_or("missing MXFP8 layers")?;
            for (views, packed) in expert_views[NVFP4_LAYERS..]
                .iter_mut()
                .zip(&device_codebooks)
            {
                *views = [ExpertMatrix::Raw(&packed[0]), ExpertMatrix::Raw(&packed[1])];
            }
            Some(DeviceBuf::alloc_zeroed(&ctx, largest)?)
        } else {
            None
        };

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
            d_ctrl_inv_cur: DeviceBuf::alloc_zeroed(&ctx, 3 * 4)?,
            d_ctrl_inv_msgs: DeviceBuf::alloc_zeroed(&ctx, 48 * 4)?,
            d_qrgs: DeviceBuf::alloc_zeroed(&ctx, SWA_ALL_ROWS * 4)?,
            d_k_raw: DeviceBuf::alloc_zeroed(&ctx, 1056 * 4)?,
            d_v_raw: DeviceBuf::alloc_zeroed(&ctx, 1056 * 4)?,
            d_k_conv: DeviceBuf::alloc_zeroed(&ctx, 1024 * 4)?,
            d_v_conv: DeviceBuf::alloc_zeroed(&ctx, 1024 * 4)?,
            d_attn_out: DeviceBuf::alloc_zeroed(&ctx, 3072 * 4)?,
            attn_split_g,
            attn_split_g_decode,
            attn_split_g_decode_global,
            d_dec_part_ml: DeviceBuf::alloc_zeroed(
                &ctx,
                (dec_rows * N_HEAD * dec_split_max * 2 * 4).max(4),
            )?,
            d_dec_part_o: DeviceBuf::alloc_zeroed(
                &ctx,
                (dec_rows * N_HEAD * dec_split_max * HD * 4).max(4),
            )?,
            d_wo_out: DeviceBuf::alloc_zeroed(&ctx, D * 4)?,
            d_mlp_h: DeviceBuf::alloc_zeroed(&ctx, D * 4)?,
            d_router_logits: DeviceBuf::alloc_zeroed(&ctx, N_EXP * 4)?,
            d_sel_ids: DeviceBuf::alloc_zeroed(&ctx, TOP_K * 4)?,
            d_sel_wts: DeviceBuf::alloc_zeroed(&ctx, TOP_K * 4)?,
            d_moe_act: DeviceBuf::alloc_zeroed(&ctx, TOP_K * FF * 4)?,
            d_moe_y: DeviceBuf::alloc_zeroed(&ctx, D * 4)?,
            d_logits: DeviceBuf::alloc_zeroed(&ctx, VOCAB * 4)?,
            d_router_done: DeviceBuf::alloc_zeroed(&ctx, 4)?,
            d_act_q8_64: DeviceBuf::alloc_zeroed(&ctx, 3 * 3072)?,
            d_act_sc_64: DeviceBuf::alloc_zeroed(&ctx, 3 * 48 * 4)?,
            d_act_q8_16: DeviceBuf::alloc_zeroed(&ctx, TOP_K * FF)?,
            d_act_sc_16: DeviceBuf::alloc_zeroed(&ctx, TOP_K * (FF / 16) * 4)?,
            d_moe_q8_16: DeviceBuf::alloc_zeroed(&ctx, TOP_K * FF)?,
            d_moe_sc_16: DeviceBuf::alloc_zeroed(&ctx, TOP_K * (FF / 16) * 4)?,
            h_token_ring: PinnedHostBuf::alloc_zeroed(&ctx, 65 * 4)?,
            prefill: PrefillBuffers::new(
                &ctx,
                EXPERT_PAGE,
                attn_split_g,
                has_flash_tc,
                drafter.is_some(),
            )?,
            d_b_exp_counts: DeviceBuf::alloc_zeroed(&ctx, N_EXP * 4)?,
            d_b_active_exps: DeviceBuf::alloc_zeroed(&ctx, N_EXP * 4)?,
            d_spec_logits: DeviceBuf::alloc_zeroed(&ctx, dec_rows * VOCAB * 4)?,
            d_spec_tokens_out: DeviceBuf::alloc_zeroed(&ctx, dec_rows * 4)?,
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
            n_seqs,
            cur_seq: 0,
            seq_pos: vec![0; n_seqs],
            d_ms_rows,
            d_ms_tokens,
            ms_rows: std::cell::Cell::new(0),
            d_ms_j,
            d_ms_runs,
            d_ms_xrec: None,
            ms_xrec: std::cell::Cell::new(0),
            ms_verify_runs: Vec::new(),
            draft_streams: Vec::new(),
            draft_ready: core::ptr::null_mut(),
            ms_graphs: std::collections::HashMap::new(),
            ms_round_hits: std::collections::HashMap::new(),
            ms_graph: std::env::var("LEANSONIC_MS_GRAPH").map_or(true, |v| v != "0"),
            embed,
            embed_norm,
            final_norm,
            unembed_q8,
            output_query,
            d_all_msg_queries,
            layers,
            slots,
            expert_weights: Vec::with_capacity(N_LAYER),
            device_codebook_scratch,
            resident_expert_layers: 0,
            spilled_expert_layers: 0,
            max_ctx,
            use_moe_dp4a,
            stage_host_up: false,
            ms_gemm_tiles: std::env::var("LEANSONIC_MS_GEMM").map_or(true, |v| v != "0"),
            moe_down_xw: std::env::var("LEANSONIC_MOE_DOWN_XW")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1),
            verify_moe: std::env::var("LEANSONIC_VERIFY_MOE")
                .map(|v| v != "0")
                .unwrap_or(true),
            solo_conv_fuse: std::env::var("LEANSONIC_SOLO_CONV_FUSE")
                .map(|v| v == "1")
                .unwrap_or(false),
            kv_conv_prep: std::env::var("LEANSONIC_KV_CONV_PREP")
                .map(|v| v == "1")
                .unwrap_or(false),
            fuse_router: std::env::var("LEANSONIC_FUSE_ROUTER")
                .map(|v| v == "1")
                .unwrap_or(false),
            c3072_r1: std::env::var("LEANSONIC_C3072_R1")
                .map(|v| v != "0")
                .unwrap_or(true),
            tc_q8,
            tc_attn,
            tc_nvfp4,
            tc_mxfp8,
            tc_bf16,
            par_conv,
            pdl,
            pdl_all,
            max_prefill_chunk: EXPERT_PAGE,
            host_pos: 0,
            decode_graph: None,
            prefill_graph: None,
            stage_events,
            d_w13_prefetch: None,
            enable_cuda_graph,
        };

        // Scales, dense weights, KV, drafting state and scratch are already allocated.
        // The two graph executions consumed 8 MiB in the 16 GB Blackwell measurements.
        // The reserve only limits optional expert, workspace and prefetch allocations.
        // Model state and the mandatory minimum workspace have already been allocated.
        let (free, _) = eng.ctx.mem_info()?;
        let graph_reserve = if enable_cuda_graph {
            16 * 1024 * 1024
        } else {
            0
        };
        let placement_reserve = user_reserve.saturating_add(graph_reserve);
        let budget = free.saturating_sub(placement_reserve);
        let mut resident = plan_expert_placement(&expert_views, budget);
        let all_experts_resident = resident.iter().all(|pair| *pair == [true, true]);
        let explicit_chunk = match std::env::var("LEANSONIC_PREFILL_CHUNK") {
            Ok(s) => Some(
                s.parse::<usize>()
                    .ok()
                    .filter(|n| (EXPERT_PAGE..=MAX_BATCH_TOKENS).contains(n) && n.is_power_of_two())
                    .ok_or("LEANSONIC_PREFILL_CHUNK must be 64, 128, 256, 512, 1024 or 2048")?,
            ),
            Err(std::env::VarError::NotPresent) => None,
            Err(e) => return Err(e.to_string()),
        };
        let limit = explicit_chunk.unwrap_or(prompt_limit);
        let _ = saved_scale_bytes;
        if eng.fns.build_expert_groups_wide_b.is_some() {
            if all_experts_resident {
                let expert_bytes: usize = expert_views
                    .iter()
                    .zip(&resident)
                    .flat_map(|(views, selected)| views.iter().zip(selected))
                    .filter(|(_, selected)| **selected)
                    .map(|(view, _)| view.len())
                    .sum();
                let headroom = budget - expert_bytes;
                for width in [2048, 1024, 512, 256, 128] {
                    if width > limit || width > max_ctx {
                        continue;
                    }
                    let extra = (width - EXPERT_PAGE) * eng.prefill.bytes_per_row;
                    if extra > headroom {
                        continue;
                    }
                    // On full-VRAM cards (`all_experts_resident` with no explicit override), defer
                    // widening until a >64-token prompt actually arrives, saving 400–800 MiB VRAM on short runs!
                    // #107: a multi-sequence engine cannot widen later (that reallocates every
                    // sequence's rings), so it takes the same width now. Prompts then split into
                    // the same chunks as on a single-sequence engine, keeping prefill bit-identical.
                    if explicit_chunk.is_none() && eng.n_seqs == 1 {
                        eng.max_prefill_chunk = width;
                        break;
                    }
                    match eng.resize_prefill(width) {
                        Ok(()) => {}
                        Err(e) if e.starts_with("CUDA error 2 in `cuMemAlloc_v2`") => continue,
                        Err(e) => return Err(e),
                    }
                    let (now_free, _) = eng.ctx.mem_info()?;
                    if plan_expert_placement(
                        &expert_views,
                        now_free.saturating_sub(placement_reserve),
                    ) == resident
                    {
                        eng.max_prefill_chunk = width;
                        break;
                    }
                    eng.resize_prefill(EXPERT_PAGE)?;
                }
            } else {
                // Spilled 16 GB Mode: keep Tier-1 mandatory matrices (`12` thrice-visited Loop Core +
                // `4` MXFP8 Coda + all `8` remaining `w2` matrices) 100% resident FIRST, secure up to
                // `prompt_limit.min(1024)` so Copy-Engine HtoD transfers drop by `16x` (`64 -> 1024`),
                // and then extend placement with as many optional single-visit `w13` matrices as fit!
                let mut tier1_bytes = 0usize;
                for layer in (6..18).chain(NVFP4_LAYERS..N_LAYER) {
                    tier1_bytes += expert_views[layer][0].len() + expert_views[layer][1].len();
                }
                for layer in (0..6).chain(18..NVFP4_LAYERS) {
                    tier1_bytes += expert_views[layer][1].len();
                }
                if tier1_bytes <= budget {
                    let tier1 = plan_expert_placement(&expert_views, tier1_bytes);
                    let spilled_cap = explicit_chunk.unwrap_or_else(|| prompt_limit.min(1024));
                    let headroom = budget - tier1_bytes;
                    for width in [2048, 1024, 512, 256, 128] {
                        if width > spilled_cap || width > limit || width > max_ctx {
                            continue;
                        }
                        let extra = (width - EXPERT_PAGE) * eng.prefill.bytes_per_row;
                        if extra > headroom {
                            continue;
                        }
                        match eng.resize_prefill(width) {
                            Ok(()) => {}
                            Err(e) if e.starts_with("CUDA error 2 in `cuMemAlloc_v2`") => continue,
                            Err(e) => return Err(e),
                        }
                        let (now_free, _) = eng.ctx.mem_info()?;
                        let available = now_free.saturating_sub(placement_reserve);
                        if available >= tier1_bytes {
                            resident = extend_expert_placement(
                                &expert_views,
                                available - tier1_bytes,
                                tier1,
                            );
                            eng.max_prefill_chunk = width;
                            break;
                        }
                        eng.resize_prefill(EXPERT_PAGE)?;
                    }
                }
            }
        }
        // The dense tensors now live in their final buffers. Release their source pages
        // before copying the larger expert matrices into VRAM or pinned host memory.
        st.discard_resident_pages();
        let mut mixed_layers = 0;
        for ([up, down], [up_on_device, down_on_device]) in expert_views.into_iter().zip(resident) {
            let pair = [
                up.place(&eng.ctx, up_on_device)?,
                down.place(&eng.ctx, down_on_device)?,
            ];
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

        drop(device_codebooks);
        eng.stage_host_up = eng.select_host_up_staging()?;
        if eng.stage_host_up {
            eprintln!("[LeanSonic GPU] Staging selected host-backed NVFP4 gate/up weights");
        }
        if enable_cuda_graph {
            eng.recapture_cuda_graphs()?;
        }

        if eng.spilled_expert_layers > 0 {
            const EXP_W13_BYTES: usize = 2 * FF * (D / 2);
            let (free_pre, _) = eng.ctx.mem_info()?;
            let pf_exps = (free_pre.saturating_sub(user_reserve.saturating_add(48 * 1024 * 1024))
                / EXP_W13_BYTES)
                .min(N_EXP);
            if pf_exps >= 8 {
                if let Ok(buf) = DeviceBuf::alloc_zeroed(&eng.ctx, pf_exps * EXP_W13_BYTES) {
                    let mut ready_ev: CUevent = core::ptr::null_mut();
                    let mut consumed_ev: CUevent = core::ptr::null_mut();
                    unsafe {
                        check_cu(
                            (eng.ctx.api.cuEventCreate)(&mut ready_ev, CU_EVENT_DISABLE_TIMING),
                            "cuEventCreate(pf_ready)",
                        )?;
                        check_cu(
                            (eng.ctx.api.cuEventCreate)(&mut consumed_ev, CU_EVENT_DISABLE_TIMING),
                            "cuEventCreate(pf_consumed)",
                        )?;
                        check_cu(
                            (eng.ctx.api.cuEventRecord)(consumed_ev, eng.ctx.compute_stream),
                            "cuEventRecord(pf_init_consumed)",
                        )?;
                    }
                    eng.d_w13_prefetch = Some(SpilledPrefetchBuf {
                        ctx: Arc::clone(&eng.ctx),
                        buf,
                        pf_exps,
                        ready_ev,
                        consumed_ev,
                    });
                }
            }
        }

        let (free_after, total_vram) = eng.ctx.mem_info()?;
        let drafter_tag = if eng.drafter.is_some() {
            format!("DSpark(K={spec_k})")
        } else if spec_k > 0 {
            format!("Suffix(K={spec_k})")
        } else {
            "off".to_string()
        };
        let chunk_tag = if eng.max_prefill_chunk > eng.prefill.capacity {
            format!("{}..{}", eng.prefill.capacity, eng.max_prefill_chunk)
        } else {
            format!("{}", eng.prefill.capacity)
        };
        eprintln!(
            "[LeanSonic GPU] {} | VRAM used: {:.2} GiB / {:.2} GiB | Resident expert layers: {}/24 (Spilled: {}) | Mixed VRAM/host layers: {} | Spec: {} | KV: {} | Prefill chunk: {} | Loaded in {:.2}s",
            eng.ctx.device_name,
            (total_vram - free_after) as f64 / (1024.0 * 1024.0 * 1024.0),
            total_vram as f64 / (1024.0 * 1024.0 * 1024.0),
            eng.resident_expert_layers,
            eng.spilled_expert_layers,
            mixed_layers,
            drafter_tag,
            if eng.slots.iter().all(|st| st.k_scale.is_some()) {
                "int8 (all layers)"
            } else if kv_auto {
                "int8 (auto, <=16GB ctx>8192; set LEANSONIC_KV=fp16 to override)"
            } else if eng.slots.iter().any(|st| st.k_scale.is_some()) {
                "int8 (global layers)"
            } else {
                "fp16"
            },
            chunk_tag,
            t_load.elapsed().as_secs_f64()
        );
        Ok(eng)
    }

    /// Make `seq` the sequence that prefill / decode / verify operate on (`LEANSONIC_SEQS` > 1).
    /// The other sequences keep their KV, rings, conv histories and positions untouched.
    pub fn select_seq(&mut self, seq: usize) -> Result<(), String> {
        if seq >= self.n_seqs {
            return Err(format!("sequence {seq} >= LEANSONIC_SEQS={}", self.n_seqs));
        }
        if seq == self.cur_seq {
            return Ok(());
        }
        self.ctx.bind_current()?;
        self.ctx.sync_compute()?;
        self.seq_pos[self.cur_seq] = self.host_pos;
        self.cur_seq = seq;
        self.host_pos = self.seq_pos[seq];
        if let Some(d) = &mut self.drafter {
            d.select_seq(seq)?;
        }
        let pos = self.host_pos as i32;
        unsafe {
            check_cu(
                (self.ctx.api.cuMemcpyHtoD_v2)(self.d_pos.ptr, (&pos as *const i32).cast(), 4),
                "HtoD(pos)",
            )?;
        }
        // Graphs bake the state pointers of the sequence they were captured for. Multi-sequence
        // decoding goes through `decode_round`, so single steps of a switched sequence simply
        // run without the graph.
        self.decode_graph = None;
        self.prefill_graph = None;
        if self.enable_cuda_graph && std::env::var("LEANSONIC_SEQ_GRAPH").is_ok_and(|v| v == "1") {
            let ctx_clone = Arc::clone(&self.ctx);
            let d_graph = CapturedGraphExec::capture(&ctx_clone, |stream| {
                self.enqueue_step_impl(stream, false)
            })?;
            self.decode_graph = Some(d_graph);
        }
        Ok(())
    }

    fn recapture_cuda_graphs(&mut self) -> Result<(), String> {
        if !self.enable_cuda_graph {
            return Ok(());
        }
        self.decode_graph = None;
        self.prefill_graph = None;
        let ctx_clone = Arc::clone(&self.ctx);
        let d_graph =
            CapturedGraphExec::capture(&ctx_clone, |stream| self.enqueue_step_impl(stream, false))?;
        self.decode_graph = Some(d_graph);
        if std::env::var("LEANSONIC_SERIAL_PREFILL")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            let p_graph = CapturedGraphExec::capture(&ctx_clone, |stream| {
                self.enqueue_step_impl(stream, true)
            })?;
            self.prefill_graph = Some(p_graph);
        }
        self.reset()
    }

    fn ensure_prefill_capacity(&mut self, prompt_len: usize) -> Result<(), String> {
        if self.host_pos != 0
            || self.n_seqs > 1
            || prompt_len <= self.prefill.capacity
            || self.prefill.capacity >= self.max_prefill_chunk
        {
            return Ok(());
        }
        let target = prompt_len
            .next_power_of_two()
            .clamp(EXPERT_PAGE, self.max_prefill_chunk);
        if target <= self.prefill.capacity {
            return Ok(());
        }
        let mut width = target;
        while width > self.prefill.capacity {
            match self.resize_prefill(width) {
                Ok(()) => {
                    return self.recapture_cuda_graphs();
                }
                Err(e) if e.starts_with("CUDA error 2 in `cuMemAlloc_v2`") => {
                    width /= 2;
                }
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    fn resize_prefill(&mut self, capacity: usize) -> Result<(), String> {
        let kv_int8 = self.slots.iter().any(|st| st.k_scale.is_some());
        let has_flash_tc = self.tc_attn
            && self.fns.flash_attn_tc_q8_b.is_some()
            && (!kv_int8 || self.fns.flash_attn_tc_q8_b_i8.is_some());
        let buffers = PrefillBuffers::new(
            &self.ctx,
            capacity,
            self.attn_split_g,
            has_flash_tc,
            self.drafter.is_some(),
        )?;
        let mut replacements = Vec::new();
        let cap = self.max_ctx.min(EXTENT + capacity);
        for (index, slot) in self.slots.iter().enumerate() {
            // Global slots always retain the full context.
            if slot.is_global || slot.kv_cap == cap {
                continue;
            }
            let int8 = slot.k_scale.is_some();
            let bytes = cap * N_KV * HD * if int8 { 1 } else { 2 };
            let k = DeviceBuf::alloc_zeroed(&self.ctx, self.n_seqs * bytes)?;
            let v = DeviceBuf::alloc_zeroed(&self.ctx, self.n_seqs * bytes)?;
            let scales = if int8 {
                Some((
                    DeviceBuf::alloc_zeroed(&self.ctx, self.n_seqs * cap * N_KV * 4)?,
                    DeviceBuf::alloc_zeroed(&self.ctx, self.n_seqs * cap * N_KV * 4)?,
                ))
            } else {
                None
            };
            replacements.push((index, k, v, scales));
        }
        // No token has run yet. Commit only after every allocation succeeds.
        self.prefill = buffers;
        for (i, k, v, scales) in replacements {
            self.slots[i].kv_cap = cap;
            let int8 = scales.is_some();
            self.slots[i].seq_kv_bytes = (cap * N_KV * HD * if int8 { 1 } else { 2 }) as u64;
            self.slots[i].seq_scale_bytes = (cap * N_KV * 4) as u64;
            self.slots[i].k_cache = k;
            self.slots[i].v_cache = v;
            if let Some((ks, vs)) = scales {
                self.slots[i].k_scale = Some(ks);
                self.slots[i].v_scale = Some(vs);
            }
        }
        Ok(())
    }

    fn load_module_functions(
        ctx: &Arc<CudaContextHandle>,
        image: &[u8],
        pack_gpu_scales: bool,
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
        let set_dyn_smem =
            |opt_f: Option<CUfunction>, bytes: i32| -> Result<Option<CUfunction>, String> {
                if let Some(f) = opt_f {
                    unsafe {
                        // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
                        check_cu(
                            (ctx.api.cuFuncSetAttribute)(f, 8, bytes),
                            "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)",
                        )?;
                    }
                }
                Ok(opt_f)
            };
        let expert_kernels = |suffix: &str| -> Result<ExpertKernels, String> {
            let load = |name| get_fn(&format!("ink2_moe_{name}{suffix}"));
            Ok(ExpertKernels {
                nvfp4_up_situ: load("nvfp4_up_situ")?,
                nvfp4_down: load("nvfp4_down")?,
                nvfp4_dp4a_up_situ: load("nvfp4_dp4a_up_situ")?,
                nvfp4_dp4a_up_half: load("nvfp4_dp4a_up_half").ok(),
                nvfp4_dp4a_up_half_q8: load("nvfp4_dp4a_up_half_q8").ok(),
                nvfp4_staged_up: get_fn("ink2_gather_nvfp4_up_192")
                    .ok()
                    .zip(load("nvfp4_dp4a_up_half_staged").ok()),
                nvfp4_dp4a_down: load("nvfp4_dp4a_down")?,
                nvfp4_dp4a_down_xw: load("nvfp4_dp4a_down_xw").ok(),
                nvfp4_dp4a_down_xw_conv: load("nvfp4_dp4a_down_xw_conv").ok(),
                mxfp8_up_situ: load("mxfp8_up_situ")?,
                mxfp8_down: load("mxfp8_down")?,
                mxfp8_down_xw: load("mxfp8_down_xw").ok(),
                mxfp8_down_xw_conv: load("mxfp8_down_xw_conv").ok(),
                nvfp4_dp4a_grouped_up_situ: load("nvfp4_dp4a_grouped_up_situ")?,
                nvfp4_dp4a_grouped_down: load("nvfp4_dp4a_grouped_down")?,
                mxfp8_grouped_up_situ: load("mxfp8_grouped_up_situ")?,
                mxfp8_grouped_down: load("mxfp8_grouped_down")?,
                nvfp4_dp4a_verify_up_half_q8: load("nvfp4_dp4a_verify_up_half_q8").ok(),
                nvfp4_dp4a_verify_down: load("nvfp4_dp4a_verify_down").ok(),
                mxfp8_verify_up_situ: load("mxfp8_verify_up_situ").ok(),
                mxfp8_verify_down: load("mxfp8_verify_down").ok(),
                nvfp4_tc_up_situ: set_dyn_smem(load("nvfp4_tc_up_situ").ok(), 62_976)?,
                nvfp4_tc_down: set_dyn_smem(load("nvfp4_tc_down").ok(), 62_976)?,
                nvfp4_tc_up_situ_t160: set_dyn_smem(load("nvfp4_tc_up_situ_t160").ok(), 58_112)?,
                nvfp4_tc_down_t160: set_dyn_smem(load("nvfp4_tc_down_t160").ok(), 58_112)?,
                mxfp8_tc_up_situ: set_dyn_smem(load("mxfp8_tc_up_situ").ok(), 78_336)?,
                mxfp8_tc_down: set_dyn_smem(load("mxfp8_tc_down").ok(), 78_336)?,
                mxfp8_tc_up_situ_t160: set_dyn_smem(load("mxfp8_tc_up_situ_t160").ok(), 88_064)?,
                mxfp8_tc_down_t160: set_dyn_smem(load("mxfp8_tc_down_t160").ok(), 88_064)?,
            })
        };
        let packed_experts = if pack_gpu_scales {
            expert_kernels("_packed_scales").ok()
        } else {
            None
        };
        let q8_tc_gemm_b = set_dyn_smem(get_fn("ink2_q8_tc_gemm_b").ok(), 64_512)?;
        let q8_tc_qkv3_gemm_b = set_dyn_smem(get_fn("ink2_q8_tc_qkv3_gemm_b").ok(), 64_512)?;
        let flash_attn_tc_q8_b = set_dyn_smem(get_fn("ink2_flash_attn_tc_q8_b").ok(), 64_512)?;
        let flash_attn_tc_q8_b_i8 =
            set_dyn_smem(get_fn("ink2_flash_attn_tc_q8_b_i8").ok(), 64_512)?;
        let flash_attn_tc_split_b =
            set_dyn_smem(get_fn("ink2_flash_attn_tc_split_b").ok(), 64_512)?;
        let flash_attn_tc_split_b_i8 =
            set_dyn_smem(get_fn("ink2_flash_attn_tc_split_b_i8").ok(), 64_512)?;
        Ok(KernelFns {
            _module: module,
            experts: expert_kernels("")?,
            packed_experts,
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
            q8_dp4a_gemv_c3072: get_fn("ink2_q8_dp4a_gemv_c3072").ok(),
            q8_dp4a_gemv_c3072_r1: get_fn("ink2_q8_dp4a_gemv_c3072_r1").ok(),
            q8_dp4a_gemv_c2304: get_fn("ink2_q8_dp4a_gemv_c2304").ok(),
            q8_dp4a_gemv_c2304_r1: get_fn("ink2_q8_dp4a_gemv_c2304_r1").ok(),
            short_conv4: get_fn("ink2_short_conv4")?,
            prep_qkv_and_cache: get_fn("ink2_prep_qkv_and_cache")?,
            gqa_rel_sink_attn: get_fn("ink2_gqa_rel_sink_attn")?,
            gqa_attn_split: get_fn("ink2_gqa_attn_split").ok(),
            gqa_attn_split_i8: get_fn("ink2_gqa_attn_split_i8").ok(),
            gqa_attn_split_dec: get_fn("ink2_gqa_attn_split_dec").ok(),
            gqa_attn_combine: get_fn("ink2_gqa_attn_combine").ok(),
            router_topk8: get_fn("ink2_router_topk8")?,
            router_gemv_topk8: get_fn("ink2_router_gemv_topk8").ok(),
            copy_d: get_fn("ink2_copy_d")?,
            argmax_and_inc_pos: get_fn("ink2_argmax_and_inc_pos")?,
            inc_pos: get_fn("ink2_inc_pos")?,
            embed_rmsnorm_b: get_fn("ink2_embed_rmsnorm_b")?,
            rmsnorm_d_b: get_fn("ink2_rmsnorm_d_b")?,
            rmsnorm_quant_q8_64_b: get_fn("ink2_rmsnorm_quant_q8_64_b")?,
            quant_q8_64_b: get_fn("ink2_quant_q8_64_b")?,
            depth_mix_b: get_fn("ink2_depth_mix_b")?,
            ctrl_down_b: get_fn("ink2_ctrl_down_b")?,
            ctrl_down_r8_t2_b: get_fn("ink2_ctrl_down_r8_t2_b").ok(),
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
            build_expert_groups_par_b: get_fn("ink2_build_expert_groups_par_b").ok(),
            moe_combine_top8_b: get_fn("ink2_moe_combine_top8_b")?,
            inc_pos_by: get_fn("ink2_inc_pos_by")?,
            prep_qkv_and_cache_ms: get_fn("ink2_prep_qkv_and_cache_ms").ok(),
            gqa_attn_split_dec_ms: get_fn("ink2_gqa_attn_split_dec_ms").ok(),
            gqa_attn_split_i8_ms: get_fn("ink2_gqa_attn_split_i8_ms").ok(),
            gqa_attn_combine_q8_dyn_ms: get_fn("ink2_gqa_attn_combine_q8_dyn_ms").ok(),
            short_conv4_ms: get_fn("ink2_short_conv4_ms").ok(),
            short_conv4_save_ms: get_fn("ink2_short_conv4_save_ms").ok(),
            short_conv4_vms: get_fn("ink2_short_conv4_vms").ok(),
            short_conv4_save_vms: get_fn("ink2_short_conv4_save_vms").ok(),
            conv_commit_ms: get_fn("ink2_conv_commit_ms").ok(),
            argmax_b: get_fn("ink2_argmax_b")?,
            conv_rewind: get_fn("ink2_conv_rewind")?,
            q8_dp4a_gemm_ktiled: get_fn("ink2_q8_dp4a_gemm_ktiled")?,
            mtp_tapmix: get_fn("mtp_tapmix")?,
            mtp_copy_anchor: get_fn("mtp_copy_anchor")?,
            mtp_ctx_store: get_fn("mtp_ctx_store")?,
            mtp_init_draft: get_fn("mtp_init_draft")?,
            mtp_qk_norm_rope: get_fn("mtp_qk_norm_rope")?,
            mtp_attn: get_fn("mtp_attn")?,
            mtp_attn_split: get_fn("mtp_attn_split").ok(),
            mtp_attn_merge: get_fn("mtp_attn_merge").ok(),
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
            gqa_rel_sink_attn_q8_b_i8: get_fn("ink2_gqa_rel_sink_attn_q8_b_i8").ok(),
            gqa_rel_sink_attn_q8_decode: get_fn("ink2_gqa_rel_sink_attn_q8_decode").ok(),
            gqa_rel_sink_attn_q8_decode_i8: get_fn("ink2_gqa_rel_sink_attn_q8_decode_i8").ok(),
            gqa_attn_combine_q8: get_fn("ink2_gqa_attn_combine_q8").ok(),
            gqa_attn_combine_q8_dyn: get_fn("ink2_gqa_attn_combine_q8_dyn").ok(),
            short_conv4_save: get_fn("ink2_short_conv4_save").ok(),
            short_conv4_save_b: get_fn("ink2_short_conv4_save_b").ok(),
            short_conv4_tiled: get_fn("ink2_short_conv4_tiled_b")
                .ok()
                .zip(get_fn("ink2_short_conv4_tiled_save_b").ok()),
            loop_pass_setup_b: get_fn("ink2_loop_pass_setup_b").ok(),
            q8_dp4a_qkv3_gemv: get_fn("ink2_q8_dp4a_qkv3_gemv").ok(),
            q8_dp4a_qkv3_gemv_r1: get_fn("ink2_q8_dp4a_qkv3_gemv_r1").ok(),
            short_conv4_kv2: get_fn("ink2_short_conv4_kv2").ok(),
            short_conv4_save_par_b: get_fn("ink2_short_conv4_save_par_b").ok(),
            short_conv4_commit_hist_b: get_fn("ink2_short_conv4_commit_hist_b").ok(),
            q8_tc_gemm_b,
            q8_tc_qkv3_gemm_b,
            quant_nvfp4_2term_b: get_fn("ink2_quant_nvfp4_2term_b").ok(),
            gather_nvfp4_sorted_b: get_fn("ink2_gather_nvfp4_sorted_b").ok(),
            expand_mxfp8_indices: get_fn("ink2_expand_mxfp8_indices").ok(),
            quant_mxfp8_b: get_fn("ink2_quant_mxfp8_b").ok(),
            flash_attn_tc_q8_b,
            flash_attn_tc_q8_b_i8,
            f32_to_bf16_b: get_fn("ink2_f32_to_bf16_b").ok(),
            ctrl_norm_bf16_b: get_fn("ink2_ctrl_norm_bf16_b").ok(),
            bf16_tc_gemm_b: get_fn("ink2_bf16_tc_gemm_b").ok(),
            ctrl_apply_tc_b: get_fn("ink2_ctrl_apply_tc_b").ok(),
            moe_combine_top8_vec4_b: get_fn("ink2_moe_combine_top8_vec4_b").ok(),
            moe_combine_top8_f16_b: get_fn("ink2_moe_combine_top8_f16_b").ok(),
            depth_mix_multi_b: get_fn("ink2_depth_mix_multi_b").ok(),
            quant_nvfp4_2term_1p_b: get_fn("ink2_quant_nvfp4_2term_1p_b").ok(),
            flash_attn_tc_split_b,
            flash_attn_tc_split_b_i8,
            flash_attn_split_combine_q8_b: get_fn("ink2_flash_attn_split_combine_q8_b").ok(),
            depth_mix_multi_ctrl_norm_bf16_b: get_fn("ink2_depth_mix_multi_ctrl_norm_bf16_b").ok(),
            depth_mix_inv: get_fn("ink2_depth_mix_inv").ok(),
            pdl_all_marker: get_fn("ink2_pdl_all_marker").ok(),
            ctrl_down_r4_fast: get_fn("ink2_ctrl_down_r4_fast").ok(),
            q8_dp4a_gemm_b8_fast: get_fn("ink2_q8_dp4a_gemm_b8_fast").ok(),
            q8_dp4a_gemm_b_ms4: get_fn("ink2_q8_dp4a_gemm_b_ms4").ok(),
            q8_dp4a_gemm_b_ms8: get_fn("ink2_q8_dp4a_gemm_b_ms8").ok(),
            kv_conv4_prep_cache: get_fn("ink2_kv_conv4_prep_cache").ok(),
            wo_conv4_save_c3072: get_fn("ink2_wo_conv4_save_c3072").ok(),
            build_expert_groups_active_b: get_fn("ink2_build_expert_groups_active_b").ok(),
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
                (self.ctx.api.cuMemsetD8_v2)(
                    self.conv_history.ptr
                        + (self.cur_seq * self.conv_history.byte_len / self.n_seqs) as u64,
                    0,
                    self.conv_history.byte_len / self.n_seqs,
                ),
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
    fn launch_smem(
        &self,
        stream: CUstream,
        func: CUfunction,
        grid: (u32, u32),
        block: u32,
        shared_mem_bytes: u32,
        params: &mut [*mut c_void],
    ) -> Result<(), String> {
        self.launch_smem_3d(
            stream,
            func,
            (grid.0, grid.1, 1),
            block,
            shared_mem_bytes,
            params,
        )
    }

    fn launch_smem_3d(
        &self,
        stream: CUstream,
        func: CUfunction,
        grid: (u32, u32, u32),
        block: u32,
        shared_mem_bytes: u32,
        params: &mut [*mut c_void],
    ) -> Result<(), String> {
        if self.pdl_all {
            return launch_pdl_3d(
                &self.ctx,
                stream,
                func,
                grid,
                block,
                shared_mem_bytes,
                params,
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
                    shared_mem_bytes,
                    stream,
                    params.as_mut_ptr(),
                    core::ptr::null_mut(),
                ),
                &format!("cuLaunchKernel(grid={grid:?}, block={block}, smem={shared_mem_bytes})"),
            )
        }
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
        self.launch_smem(stream, func, grid, block, 0, params)
    }

    // Uses the ordinary stream ordering at both scratch reuse boundaries. The next
    // projection can overwrite the buffer after its preceding consumer completes.
    fn mxfp8_weight_ptr(
        &self,
        stream: CUstream,
        weight: &TierBuf,
        expert_bytes: usize,
        selected: bool,
    ) -> Result<CUdeviceptr, String> {
        let Some(scratch) = &self.device_codebook_scratch else {
            return Ok(weight.ptr());
        };
        let mut src = weight.ptr();
        let mut dst = scratch.ptr;
        let mut ids = if selected { self.d_sel_ids.ptr } else { 0 };
        let mut words = (expert_bytes / 8) as u32;
        let mut args = [
            (&mut src as *mut _) as *mut c_void,
            (&mut dst as *mut _) as *mut c_void,
            (&mut ids as *mut _) as *mut c_void,
            (&mut words as *mut _) as *mut c_void,
        ];
        self.launch(
            stream,
            self.fns
                .expand_mxfp8_indices
                .ok_or("missing device codebook kernel")?,
            (64, if selected { TOP_K } else { N_EXP } as u32),
            256,
            &mut args,
        )?;
        Ok(dst)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_pdl(
        &self,
        stream: CUstream,
        func: CUfunction,
        grid: (u32, u32),
        block: u32,
        params: &mut [*mut c_void],
    ) -> Result<(), String> {
        if self.pdl {
            if let Some(launch_ex) = self.ctx.api.cuLaunchKernelEx {
                let mut attr = CUlaunchAttribute::programmatic_stream_serialization();
                let cfg = CUlaunchConfig {
                    gridDimX: grid.0,
                    gridDimY: grid.1,
                    gridDimZ: 1,
                    blockDimX: block,
                    blockDimY: 1,
                    blockDimZ: 1,
                    sharedMemBytes: 0,
                    _pad0: 0,
                    hStream: stream,
                    attrs: &mut attr,
                    numAttrs: 1,
                    _pad1: 0,
                };
                return unsafe {
                    check_cu(
                        launch_ex(&cfg, func, params.as_mut_ptr(), core::ptr::null_mut()),
                        &format!("cuLaunchKernelEx(grid={grid:?}, block={block})"),
                    )
                };
            }
        }
        self.launch_smem(stream, func, grid, block, 0, params)
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
        self.launch_pdl(stream, self.fns.copy_d, (9, 1), 256, &mut args)
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_ctrl_n(
        &self,
        stream: CUstream,
        cur_ptr: CUdeviceptr,
        msg_ptr: CUdeviceptr,
        inv_cur_ptr: CUdeviceptr,
        inv_msg_ptr: CUdeviceptr,
        ctrl: &CtrlGpu,
        n_ctrls: u32,
    ) -> Result<(), String> {
        let mut p_cur = cur_ptr;
        let mut p_msg = msg_ptr;
        let mut p_down = ctrl.down.ptr;
        let mut p_g = self.d_ctrl_g.ptr;
        if let (true, Some(f_r4)) = (
            inv_cur_ptr != 0 && inv_msg_ptr != 0,
            self.fns.ctrl_down_r4_fast,
        ) {
            let mut p_ic = inv_cur_ptr;
            let mut p_im = inv_msg_ptr;
            let mut args_r4: [*mut c_void; 6] = [
                (&mut p_cur as *mut _) as *mut c_void,
                (&mut p_msg as *mut _) as *mut c_void,
                (&mut p_ic as *mut _) as *mut c_void,
                (&mut p_im as *mut _) as *mut c_void,
                (&mut p_down as *mut _) as *mut c_void,
                (&mut p_g as *mut _) as *mut c_void,
            ];
            self.launch_pdl(stream, f_r4, (16, n_ctrls), 256, &mut args_r4)?;
        } else {
            let mut args1: [*mut c_void; 4] = [
                (&mut p_cur as *mut _) as *mut c_void,
                (&mut p_msg as *mut _) as *mut c_void,
                (&mut p_down as *mut _) as *mut c_void,
                (&mut p_g as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.ctrl_down, (64, n_ctrls), 256, &mut args1)?;
        }

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
            self.launch_pdl(stream, f_coal, (36, n_ctrls), 64, &mut args2)
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
        self.launch_pdl(
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
        // One warp per row either way (bit-identical). Small outputs such as the 128-row router
        // use one warp per CTA so the rows land on different SMs.
        let (grid_x, block) = if rows <= 1024 {
            (rows as u32, 32)
        } else {
            ((rows as u32).div_ceil(8), 256)
        };
        self.launch_pdl(stream, self.fns.bf16_gemv, (grid_x, 1), block, &mut args)
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
        let grid = (
            cols.div_ceil(kernel.columns_per_block) as u32,
            n_vecs as u32,
        );
        if kernel.columns_per_block == 128 {
            self.launch_pdl(stream, kernel.function, grid, kernel.threads, &mut args)
        } else {
            self.launch(stream, kernel.function, grid, kernel.threads, &mut args)
        }
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
        let grid_x_16 = (rows as u32).div_ceil(16);
        let grid_x_8 = (rows as u32).div_ceil(8);
        // `wo` (cols 3072) and `wq_all`/`unembed` (cols 2304) have load-hoisted + PDL twins with the same arithmetic.
        let r1_fn_3072 = if self.c3072_r1 {
            self.fns.q8_dp4a_gemv_c3072_r1
        } else {
            None
        };
        let r1_fn_2304 = if self.c3072_r1 && rows <= 16_384 {
            self.fns.q8_dp4a_gemv_c2304_r1
        } else {
            None
        };
        let (f, grid_x, use_pdl) = match (
            cols,
            r1_fn_3072,
            self.fns.q8_dp4a_gemv_c3072,
            r1_fn_2304,
            self.fns.q8_dp4a_gemv_c2304,
        ) {
            (3072, Some(f), _, _, _) => (f, grid_x_8, true),
            (3072, None, Some(f), _, _) => (f, grid_x_16, true),
            (2304, _, _, Some(f), _) => (f, grid_x_8, true),
            (2304, _, _, None, Some(f)) => (f, grid_x_16, true),
            _ => (self.fns.q8_dp4a_gemv, grid_x_16, false),
        };
        if use_pdl {
            self.launch_pdl(stream, f, (grid_x, 1), 256, &mut args)
        } else {
            self.launch(stream, f, (grid_x, 1), 256, &mut args)
        }
    }

    /// Fused G=0 attention + Q8_64 epilogue for this slot's KV format (FP16 or opt-in INT8).
    /// `None` falls back to the unfused attention + separate quantization path.
    /// `staged`: the latency-staged variant for single-token decode and small verify batches
    /// (falls back to the plain kernel when the cubin lacks it).
    fn attn_q8_fn(&self, st: &SlotStateGpu, staged: bool) -> Option<CUfunction> {
        match (st.k_scale.is_some(), staged) {
            (false, false) => self.fns.gqa_rel_sink_attn_q8_b,
            (false, true) => self
                .fns
                .gqa_rel_sink_attn_q8_decode
                .or(self.fns.gqa_rel_sink_attn_q8_b),
            (true, false) => self.fns.gqa_rel_sink_attn_q8_b_i8,
            (true, true) => self
                .fns
                .gqa_rel_sink_attn_q8_decode_i8
                .or(self.fns.gqa_rel_sink_attn_q8_b_i8),
        }
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
        kv_scale_ptrs: (CUdeviceptr, CUdeviceptr),
        kv_cap: i32,
        is_global: i32,
        attn_out_ptr: CUdeviceptr,
        out_q8_ptr: CUdeviceptr,
        out_sc64_ptr: CUdeviceptr,
        b_len: usize,
        ms_seq: Option<(u64, u64)>,
    ) -> Result<(), String> {
        let mut pqrgs = qrgs_ptr;
        let mut qstride = qrgs_stride;
        let mut prp = rel_proj_ptr;
        let mut pkc = k_cache_ptr;
        let mut pvc = v_cache_ptr;
        let (mut pks, mut pvs) = kv_scale_ptrs;
        let mut ppos = self.d_pos.ptr;
        let mut cap = kv_cap;
        let mut ig = is_global;
        // Decode / verify rows may use their own split count (see `attn_split_g_decode` and
        // `attn_split_g_decode_global`); decode and verify always agree, so spec 0 == spec K.
        let g_dec = if is_global != 0 {
            self.attn_split_g_decode_global
        } else {
            self.attn_split_g_decode
        };
        let decode_rows = self.small_rows(b_len) && g_dec != self.attn_split_g;
        let g_split = if decode_rows {
            g_dec
        } else {
            self.attn_split_g
        };
        let (mut pml, mut po) = if decode_rows {
            (self.d_dec_part_ml.ptr, self.d_dec_part_o.ptr)
        } else {
            (
                self.prefill.d_attn_part_ml.ptr,
                self.prefill.d_attn_part_o.ptr,
            )
        };
        let mut split_args: [*mut c_void; 12] = [
            (&mut pqrgs as *mut _) as *mut c_void,
            (&mut qstride as *mut _) as *mut c_void,
            (&mut prp as *mut _) as *mut c_void,
            (&mut pkc as *mut _) as *mut c_void,
            (&mut pvc as *mut _) as *mut c_void,
            (&mut pks as *mut _) as *mut c_void,
            (&mut pvs as *mut _) as *mut c_void,
            (&mut ppos as *mut _) as *mut c_void,
            (&mut cap as *mut _) as *mut c_void,
            (&mut ig as *mut _) as *mut c_void,
            (&mut pml as *mut _) as *mut c_void,
            (&mut po as *mut _) as *mut c_void,
        ];
        let g_u32 = g_split as u32;
        let split_fn = if pks != 0 {
            self.fns
                .gqa_attn_split_i8
                .ok_or("LEANSONIC_KV=int8 requires a cubin with ink2_gqa_attn_split_i8")?
        } else if b_len <= MAX_SPEC_TOKENS {
            // Decode / verify rows: pipelined FP16 loop (same arithmetic as the prefill kernel).
            self.fns
                .gqa_attn_split_dec
                .unwrap_or(self.fns.gqa_attn_split.unwrap())
        } else {
            self.fns.gqa_attn_split.unwrap()
        };
        if let Some((seq_kv_bytes, seq_scale_bytes)) = ms_seq {
            // Multi-sequence round (#107): per-row `{pos, seq}` table, caches of sequence 0.
            ppos = self.ms_rows.get();
            let mut skb = seq_kv_bytes as i64;
            let mut ssf = (seq_scale_bytes / 4) as i32;
            let mut ms_args: [*mut c_void; 14] = [
                (&mut pqrgs as *mut _) as *mut c_void,
                (&mut qstride as *mut _) as *mut c_void,
                (&mut prp as *mut _) as *mut c_void,
                (&mut pkc as *mut _) as *mut c_void,
                (&mut pvc as *mut _) as *mut c_void,
                (&mut pks as *mut _) as *mut c_void,
                (&mut pvs as *mut _) as *mut c_void,
                (&mut ppos as *mut _) as *mut c_void,
                (&mut cap as *mut _) as *mut c_void,
                (&mut ig as *mut _) as *mut c_void,
                (&mut pml as *mut _) as *mut c_void,
                (&mut po as *mut _) as *mut c_void,
                (&mut skb as *mut _) as *mut c_void,
                (&mut ssf as *mut _) as *mut c_void,
            ];
            let f = if pks != 0 {
                self.fns.gqa_attn_split_i8_ms
            } else {
                self.fns.gqa_attn_split_dec_ms
            }
            .ok_or("cubin lacks the multi-sequence split attention kernels")?;
            self.launch(stream, f, ((8 * b_len) as u32, g_u32), 256, &mut ms_args)?;
            let f_comb = self
                .fns
                .gqa_attn_combine_q8_dyn_ms
                .ok_or("cubin lacks ink2_gqa_attn_combine_q8_dyn_ms")?;
            let mut gi = g_split as i32;
            let mut pq8 = out_q8_ptr;
            let mut psc = out_sc64_ptr;
            let mut comb_args: [*mut c_void; 9] = [
                (&mut pml as *mut _) as *mut c_void,
                (&mut po as *mut _) as *mut c_void,
                (&mut pqrgs as *mut _) as *mut c_void,
                (&mut qstride as *mut _) as *mut c_void,
                (&mut gi as *mut _) as *mut c_void,
                (&mut ppos as *mut _) as *mut c_void,
                (&mut ig as *mut _) as *mut c_void,
                (&mut pq8 as *mut _) as *mut c_void,
                (&mut psc as *mut _) as *mut c_void,
            ];
            return self.launch(stream, f_comb, (24, b_len as u32), 128, &mut comb_args);
        }
        if b_len == 1 {
            self.launch_pdl(stream, split_fn, (8, g_u32), 256, &mut split_args)?;
        } else {
            self.launch(
                stream,
                split_fn,
                ((8 * b_len) as u32, g_u32),
                256,
                &mut split_args,
            )?;
        }
        let mut gi = g_split as i32;
        if let Some(f_comb_dyn) = self.fns.gqa_attn_combine_q8_dyn {
            let mut pq8 = out_q8_ptr;
            let mut psc = out_sc64_ptr;
            let mut comb_args: [*mut c_void; 9] = [
                (&mut pml as *mut _) as *mut c_void,
                (&mut po as *mut _) as *mut c_void,
                (&mut pqrgs as *mut _) as *mut c_void,
                (&mut qstride as *mut _) as *mut c_void,
                (&mut gi as *mut _) as *mut c_void,
                (&mut ppos as *mut _) as *mut c_void,
                (&mut ig as *mut _) as *mut c_void,
                (&mut pq8 as *mut _) as *mut c_void,
                (&mut psc as *mut _) as *mut c_void,
            ];
            if b_len == 1 {
                self.launch_pdl(stream, f_comb_dyn, (24, 1), 128, &mut comb_args)
            } else {
                self.launch(stream, f_comb_dyn, (24, b_len as u32), 128, &mut comb_args)
            }
        } else if let Some(f_comb_q8) = self.fns.gqa_attn_combine_q8 {
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
            if b_len == 1 {
                self.launch_pdl(stream, f_comb_q8, (24, 1), 128, &mut comb_args)
            } else {
                self.launch(stream, f_comb_q8, (24, b_len as u32), 128, &mut comb_args)
            }
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

    fn enqueue_nvfp4_up(
        &self,
        stream: CUstream,
        w13: &TierBuf,
        s13: &ExpertScalePlane,
        a13: &DeviceBuf,
        use_staging: bool,
    ) -> Result<bool, String> {
        let mut pw13 = w13.ptr();
        let mut ps13 = s13.buffer.ptr;
        let mut pa13 = a13.ptr;
        let mut pxq = self.d_act_q8_16.ptr;
        let mut pxs = self.d_act_sc_16.ptr;
        let mut pids = self.d_sel_ids.ptr;
        if let (TierBuf::Vram(_), Some(f_q8)) = (w13, s13.kernels.nvfp4_dp4a_up_half_q8) {
            let mut poq = self.d_moe_q8_16.ptr;
            let mut pos = self.d_moe_sc_16.ptr;
            let mut q8_args: [*mut c_void; 8] = [
                (&mut pw13 as *mut _) as *mut c_void,
                (&mut ps13 as *mut _) as *mut c_void,
                (&mut pa13 as *mut _) as *mut c_void,
                (&mut pxq as *mut _) as *mut c_void,
                (&mut pxs as *mut _) as *mut c_void,
                (&mut pids as *mut _) as *mut c_void,
                (&mut poq as *mut _) as *mut c_void,
                (&mut pos as *mut _) as *mut c_void,
            ];
            self.launch_pdl(stream, f_q8, (72, 8), 512, &mut q8_args)?;
            return Ok(true);
        }
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
        const TILE: usize = 192;
        const { assert!(FF.is_multiple_of(TILE)) };
        let scratch = &self.prefill.layer_workspace.pair_output;
        let staged_up = s13
            .kernels
            .nvfp4_staged_up
            .filter(|_| use_staging && scratch.byte_len >= TOP_K * TILE * D);
        if let (TierBuf::HostPinned(_), Some((gather, up))) = (w13, staged_up) {
            // The singleton visit uses dedicated activation buffers. Its prefill
            // workspace is free, and both kernels run on the same compute stream.
            let mut staged = scratch.ptr;
            for channel in (0..FF).step_by(TILE) {
                let mut start = channel as i32;
                let mut copy_args = [
                    (&mut pw13 as *mut _) as *mut c_void,
                    (&mut pids as *mut _) as *mut c_void,
                    (&mut staged as *mut _) as *mut c_void,
                    (&mut start as *mut _) as *mut c_void,
                ];
                self.launch(stream, gather, (32, TOP_K as u32), 256, &mut copy_args)?;
                let mut tile_args = [
                    (&mut staged as *mut _) as *mut c_void,
                    up_args[1],
                    up_args[2],
                    up_args[3],
                    up_args[4],
                    up_args[5],
                    up_args[6],
                    (&mut start as *mut _) as *mut c_void,
                ];
                self.launch(
                    stream,
                    up,
                    ((TILE / 4) as u32, TOP_K as u32),
                    128,
                    &mut tile_args,
                )?;
            }
        } else if let (TierBuf::Vram(_), Some(f_half)) = (w13, s13.kernels.nvfp4_dp4a_up_half) {
            // Reading mapped host weights with half warps regresses PCIe decode.
            self.launch_pdl(stream, f_half, (288, 8), 128, &mut up_args)?;
        } else {
            self.launch(
                stream,
                s13.kernels.nvfp4_dp4a_up_situ,
                (288, 8),
                256,
                &mut up_args,
            )?;
        }
        Ok(false)
    }

    /// Host-mapped loads depend on the PCIe link as well as the GPU. Time both
    /// paths on an actual spilled matrix before capturing the decode graph.
    fn select_host_up_staging(&self) -> Result<bool, String> {
        let target =
            self.layers
                .iter()
                .zip(&self.expert_weights)
                .find_map(
                    |(layer, weights)| match (&weights[0], &layer.expert_scales) {
                        (w @ TierBuf::HostPinned(_), ExpertScales::Nvfp4 { s13, a13, .. })
                            if self.use_moe_dp4a
                                && s13.kernels.nvfp4_staged_up.is_some()
                                && self.prefill.layer_workspace.pair_output.byte_len
                                    >= TOP_K * 192 * D =>
                        {
                            Some((w, s13, a13))
                        }
                        _ => None,
                    },
                );
        let Some((w, scales, alpha)) = target else {
            return Ok(false);
        };
        match std::env::var("LEANSONIC_STAGE_HOST_UP").as_deref() {
            Ok("0") => return Ok(false),
            Ok("1") => return Ok(true),
            _ => {}
        }
        let stream = self.ctx.compute_stream;
        let mut times = [std::time::Duration::ZERO; 2];
        // Select distinct experts on both passes. Swapping the order balances warm-up bias.
        for (pass, order) in [[false, true], [true, false]].into_iter().enumerate() {
            let ids: Vec<u8> = (0..TOP_K)
                .flat_map(|k| ((pass * 7 + k * 13) as i32).to_le_bytes())
                .collect();
            self.d_sel_ids.copy_from_host_at(0, &ids)?;
            for mode in order {
                let _ = self.enqueue_nvfp4_up(stream, w, scales, alpha, mode)?;
                self.ctx.sync_compute()?;
                let start = Instant::now();
                for _ in 0..2 {
                    let _ = self.enqueue_nvfp4_up(stream, w, scales, alpha, mode)?;
                }
                self.ctx.sync_compute()?;
                times[mode as usize] += start.elapsed();
            }
        }
        // The first real router overwrites the probe IDs before any expert reads them.
        Ok(times[1].as_secs_f64() < 0.97 * times[0].as_secs_f64())
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_visit(
        &self,
        stream: CUstream,
        il: usize,
        slot_idx: usize,
        n_blocks: &mut usize,
        msg_base_ptr: CUdeviceptr,
        msg_inv_base_ptr: CUdeviceptr,
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
                    self.launch_pdl(stream, f_fused, (n_q, 1), 256, &mut args)?;
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
                    self.launch_pdl(stream, self.fns.depth_mix, (n_q, 1), 256, &mut args)?;
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
            let inv_cur_ptr = if let Some(f_dmi) = self.fns.depth_mix_inv {
                let mut pinv = self.d_ctrl_inv_cur.ptr;
                let mut args: [*mut c_void; 8] = [
                    (&mut pb as *mut _) as *mut c_void,
                    (&mut nb as *mut _) as *mut c_void,
                    (&mut pc as *mut _) as *mut c_void,
                    (&mut ac as *mut _) as *mut c_void,
                    (&mut pq as *mut _) as *mut c_void,
                    (&mut pout as *mut _) as *mut c_void,
                    (&mut b3 as *mut _) as *mut c_void,
                    (&mut pinv as *mut _) as *mut c_void,
                ];
                self.launch_pdl(stream, f_dmi, (n_q, 1), 256, &mut args)?;
                self.d_ctrl_inv_cur.ptr
            } else {
                let mut args: [*mut c_void; 7] = [
                    (&mut pb as *mut _) as *mut c_void,
                    (&mut nb as *mut _) as *mut c_void,
                    (&mut pc as *mut _) as *mut c_void,
                    (&mut ac as *mut _) as *mut c_void,
                    (&mut pq as *mut _) as *mut c_void,
                    (&mut pout as *mut _) as *mut c_void,
                    (&mut b3 as *mut _) as *mut c_void,
                ];
                self.launch_pdl(stream, self.fns.depth_mix, (n_q, 1), 256, &mut args)?;
                0
            };
            self.enqueue_ctrl_n(
                stream,
                self.d_inputs.ptr,
                msg_base_ptr,
                inv_cur_ptr,
                msg_inv_base_ptr,
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
            let (f_use, rows_per_cta) = match (self.c3072_r1, self.fns.q8_dp4a_qkv3_gemv_r1) {
                (true, Some(f_r1)) => (f_r1, 8u32),
                _ => (f_qkv3, 16u32),
            };
            let grid_x = (QRGS_ROWS as u32).div_ceil(rows_per_cta)
                + (rk as u32).div_ceil(rows_per_cta)
                + (rv as u32).div_ceil(rows_per_cta);
            self.launch_pdl(stream, f_use, (grid_x, 1), 256, &mut args)?;
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

        // 5 + 6. K/V Short-Conv4 + Q/R/K Clip + V Head-RMSNorm + Log-Scaling + KV Cache Write
        let q_ptr = self.d_qrgs.ptr;
        let r_ptr = q_ptr + 3072 * 4;
        let gate_ptr = r_ptr + 384 * 4;
        let sink_ptr = gate_ptr + 3072 * 4;
        if let (true, Some(f_kv_prep)) = (self.kv_conv_prep, self.fns.kv_conv4_prep_cache) {
            let mut pq = q_ptr;
            let mut pr = r_ptr;
            let mut pkr = k_raw_ptr;
            let mut pwk = l.k_conv4.ptr;
            let mut phk = st.hist_k_at(self.cur_seq);
            let mut pvr = v_raw_ptr;
            let mut pwv = l.v_conv4.ptr;
            let mut phv = st.hist_v_at(self.cur_seq);
            let mut pqc = l.q_clip.ptr;
            let mut prc = l.rel_clip.ptr;
            let mut pkc = st.k_at(self.cur_seq);
            let mut pvc = st.v_at(self.cur_seq);
            let (mut pks, mut pvs) = st.kv_scale_ptrs_at(self.cur_seq);
            let mut ppos = self.d_pos.ptr;
            let mut cap = st.kv_cap as i32;
            let mut ig = i32::from(l.global);
            let mut args: [*mut c_void; 17] = [
                (&mut pq as *mut _) as *mut c_void,
                (&mut pr as *mut _) as *mut c_void,
                (&mut pkr as *mut _) as *mut c_void,
                (&mut pwk as *mut _) as *mut c_void,
                (&mut phk as *mut _) as *mut c_void,
                (&mut pvr as *mut _) as *mut c_void,
                (&mut pwv as *mut _) as *mut c_void,
                (&mut phv as *mut _) as *mut c_void,
                (&mut pqc as *mut _) as *mut c_void,
                (&mut prc as *mut _) as *mut c_void,
                (&mut pkc as *mut _) as *mut c_void,
                (&mut pvc as *mut _) as *mut c_void,
                (&mut pks as *mut _) as *mut c_void,
                (&mut pvs as *mut _) as *mut c_void,
                (&mut ppos as *mut _) as *mut c_void,
                (&mut cap as *mut _) as *mut c_void,
                (&mut ig as *mut _) as *mut c_void,
            ];
            self.launch_pdl(stream, f_kv_prep, (5, 1), 256, &mut args)?;
        } else {
            if let Some(f_kv2) = self.fns.short_conv4_kv2 {
                let mut pk = k_raw_ptr;
                let mut pwk = l.k_conv4.ptr;
                let mut pdk = if l.global { k_raw_ptr + 1024 * 4 } else { 0 };
                let mut phk = st.hist_k_at(self.cur_seq);
                let mut pok = self.d_k_conv.ptr;
                let mut pv = v_raw_ptr;
                let mut pwv = l.v_conv4.ptr;
                let mut pdv = if l.global { v_raw_ptr + 1024 * 4 } else { 0 };
                let mut phv = st.hist_v_at(self.cur_seq);
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
                self.launch_pdl(stream, f_kv2, (4, 2), 256, &mut args)?;
            } else {
                for (raw_ptr, conv_w, hist_ptr, dst_ptr) in [
                    (
                        k_raw_ptr,
                        l.k_conv4.ptr,
                        st.hist_k_at(self.cur_seq),
                        self.d_k_conv.ptr,
                    ),
                    (
                        v_raw_ptr,
                        l.v_conv4.ptr,
                        st.hist_v_at(self.cur_seq),
                        self.d_v_conv.ptr,
                    ),
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
                    self.launch_pdl(stream, self.fns.short_conv4, (4, 1), 256, &mut args)?;
                }
            }

            let mut pq = q_ptr;
            let mut pr = r_ptr;
            let mut pk = self.d_k_conv.ptr;
            let mut pv = self.d_v_conv.ptr;
            let mut pqc = l.q_clip.ptr;
            let mut prc = l.rel_clip.ptr;
            let mut pkc = st.k_at(self.cur_seq);
            let mut pvc = st.v_at(self.cur_seq);
            let (mut pks, mut pvs) = st.kv_scale_ptrs_at(self.cur_seq);
            let mut ppos = self.d_pos.ptr;
            let mut cap = st.kv_cap as i32;
            let mut ig = i32::from(l.global);
            let mut args: [*mut c_void; 13] = [
                (&mut pq as *mut _) as *mut c_void,
                (&mut pr as *mut _) as *mut c_void,
                (&mut pk as *mut _) as *mut c_void,
                (&mut pv as *mut _) as *mut c_void,
                (&mut pqc as *mut _) as *mut c_void,
                (&mut prc as *mut _) as *mut c_void,
                (&mut pkc as *mut _) as *mut c_void,
                (&mut pvc as *mut _) as *mut c_void,
                (&mut pks as *mut _) as *mut c_void,
                (&mut pvs as *mut _) as *mut c_void,
                (&mut ppos as *mut _) as *mut c_void,
                (&mut cap as *mut _) as *mut c_void,
                (&mut ig as *mut _) as *mut c_void,
            ];
            self.launch_pdl(stream, self.fns.prep_qkv_and_cache, (5, 1), 256, &mut args)?;
        }

        // 7. GQA Relative-Bias + Sink Attention + In-Register `Q8_64` Epilogue!
        let g_dec = if l.global {
            self.attn_split_g_decode_global
        } else {
            self.attn_split_g_decode
        };
        if g_dec == 0 {
            if let Some(f_attn_q8) = self.attn_q8_fn(st, true) {
                let mut pqrgs = self.d_qrgs.ptr;
                let mut qstride = SWA_ALL_ROWS as i32;
                let mut prp = l.rel_proj.ptr;
                let mut pkc = st.k_at(self.cur_seq);
                let mut pvc = st.v_at(self.cur_seq);
                let (mut pks, mut pvs) = st.kv_scale_ptrs_at(self.cur_seq);
                let mut ppos = self.d_pos.ptr;
                let mut cap = st.kv_cap as i32;
                let mut ig = i32::from(l.global);
                let mut pq8 = self.d_act_q8_64.ptr;
                let mut psc = self.d_act_sc_64.ptr;
                let mut args: [*mut c_void; 12] = [
                    (&mut pqrgs as *mut _) as *mut c_void,
                    (&mut qstride as *mut _) as *mut c_void,
                    (&mut prp as *mut _) as *mut c_void,
                    (&mut pkc as *mut _) as *mut c_void,
                    (&mut pvc as *mut _) as *mut c_void,
                    (&mut pks as *mut _) as *mut c_void,
                    (&mut pvs as *mut _) as *mut c_void,
                    (&mut ppos as *mut _) as *mut c_void,
                    (&mut cap as *mut _) as *mut c_void,
                    (&mut ig as *mut _) as *mut c_void,
                    (&mut pq8 as *mut _) as *mut c_void,
                    (&mut psc as *mut _) as *mut c_void,
                ];
                self.launch_pdl(stream, f_attn_q8, (24, 1), 256, &mut args)?;
            } else {
                let mut pq = q_ptr;
                let mut pr = r_ptr;
                let mut pg = gate_ptr;
                let mut ps = sink_ptr;
                let mut prp = l.rel_proj.ptr;
                let mut pkc = st.k_at(self.cur_seq);
                let mut pvc = st.v_at(self.cur_seq);
                let (mut pks, mut pvs) = st.kv_scale_ptrs_at(self.cur_seq);
                let mut ppos = self.d_pos.ptr;
                let mut cap = st.kv_cap as i32;
                let mut ig = i32::from(l.global);
                let mut pout = self.d_attn_out.ptr;
                let mut args: [*mut c_void; 13] = [
                    (&mut pq as *mut _) as *mut c_void,
                    (&mut pr as *mut _) as *mut c_void,
                    (&mut pg as *mut _) as *mut c_void,
                    (&mut ps as *mut _) as *mut c_void,
                    (&mut prp as *mut _) as *mut c_void,
                    (&mut pkc as *mut _) as *mut c_void,
                    (&mut pvc as *mut _) as *mut c_void,
                    (&mut pks as *mut _) as *mut c_void,
                    (&mut pvs as *mut _) as *mut c_void,
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
                st.k_at(self.cur_seq),
                st.v_at(self.cur_seq),
                st.kv_scale_ptrs_at(self.cur_seq),
                st.kv_cap as i32,
                i32::from(l.global),
                self.d_attn_out.ptr,
                self.d_act_q8_64.ptr,
                self.d_act_sc_64.ptr,
                1,
                None,
            )?;
        }

        // 8. `wo` Planar `Q8_64` `__dp4a` Projection + `attn_sconv` (+ fused block-start save!)
        let is_block_start = il.is_multiple_of(6);
        let save_blk_ptr = if is_block_start {
            let blk_dst = self.d_blocks.ptr + (*n_blocks as u64) * d_bytes;
            *n_blocks += 1;
            blk_dst
        } else {
            0
        };
        if let (true, Some(f_wo_conv)) = (self.solo_conv_fuse, self.fns.wo_conv4_save_c3072) {
            let mut pw = l.wo_q8.codes.ptr;
            let mut pws = l.wo_q8.scales64.ptr;
            let mut px = self.d_act_q8_64.ptr;
            let mut pxs = self.d_act_sc_64.ptr;
            let mut pwc = l.attn_conv4.ptr;
            let mut phist = st.hist_attn_at(self.cur_seq);
            let mut pcur = self.d_cur.ptr;
            let mut psave = save_blk_ptr;
            let mut dr = i32::from(!is_block_start);
            let mut args: [*mut c_void; 9] = [
                (&mut pw as *mut _) as *mut c_void,
                (&mut pws as *mut _) as *mut c_void,
                (&mut px as *mut _) as *mut c_void,
                (&mut pxs as *mut _) as *mut c_void,
                (&mut pwc as *mut _) as *mut c_void,
                (&mut phist as *mut _) as *mut c_void,
                (&mut pcur as *mut _) as *mut c_void,
                (&mut psave as *mut _) as *mut c_void,
                (&mut dr as *mut _) as *mut c_void,
            ];
            self.launch_pdl(stream, f_wo_conv, (144, 1), 256, &mut args)?;
        } else {
            self.enqueue_q8_gemv_slice_at(
                stream,
                &l.wo_q8,
                0,
                D as i32,
                0,
                self.d_wo_out.ptr,
                0.0,
            )?;
            if let Some(f_sc_save) = self.fns.short_conv4_save {
                let mut p_cur = self.d_wo_out.ptr;
                let mut p_w = l.attn_conv4.ptr;
                let mut p_dyn: CUdeviceptr = 0;
                let mut p_hist = st.hist_attn_at(self.cur_seq);
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
                self.launch_pdl(stream, f_sc_save, (9, 1), 256, &mut args)?;
            } else {
                if save_blk_ptr != 0 {
                    self.enqueue_copy_d(stream, self.d_cur.ptr, save_blk_ptr)?;
                }
                let mut p_cur = self.d_wo_out.ptr;
                let mut p_w = l.attn_conv4.ptr;
                let mut p_dyn: CUdeviceptr = 0;
                let mut p_hist = st.hist_attn_at(self.cur_seq);
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
                self.launch_pdl(stream, self.fns.short_conv4, (9, 1), 256, &mut args)?;
            }
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
                self.launch_pdl(stream, f_mlp_fused, (1, 1), 256, &mut args)?;
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
            let inv_cur_ptr = if has_msgs {
                if let Some(f_dmi) = self.fns.depth_mix_inv {
                    let mut pinv = self.d_ctrl_inv_cur.ptr;
                    let mut args: [*mut c_void; 8] = [
                        (&mut pb as *mut _) as *mut c_void,
                        (&mut nb as *mut _) as *mut c_void,
                        (&mut pc as *mut _) as *mut c_void,
                        (&mut ac as *mut _) as *mut c_void,
                        (&mut pq as *mut _) as *mut c_void,
                        (&mut pout as *mut _) as *mut c_void,
                        (&mut b3 as *mut _) as *mut c_void,
                        (&mut pinv as *mut _) as *mut c_void,
                    ];
                    self.launch_pdl(stream, f_dmi, (1, 1), 256, &mut args)?;
                    self.d_ctrl_inv_cur.ptr
                } else {
                    let mut args: [*mut c_void; 7] = [
                        (&mut pb as *mut _) as *mut c_void,
                        (&mut nb as *mut _) as *mut c_void,
                        (&mut pc as *mut _) as *mut c_void,
                        (&mut ac as *mut _) as *mut c_void,
                        (&mut pq as *mut _) as *mut c_void,
                        (&mut pout as *mut _) as *mut c_void,
                        (&mut b3 as *mut _) as *mut c_void,
                    ];
                    self.launch_pdl(stream, self.fns.depth_mix, (1, 1), 256, &mut args)?;
                    0
                }
            } else {
                let mut args: [*mut c_void; 7] = [
                    (&mut pb as *mut _) as *mut c_void,
                    (&mut nb as *mut _) as *mut c_void,
                    (&mut pc as *mut _) as *mut c_void,
                    (&mut ac as *mut _) as *mut c_void,
                    (&mut pq as *mut _) as *mut c_void,
                    (&mut pout as *mut _) as *mut c_void,
                    (&mut b3 as *mut _) as *mut c_void,
                ];
                self.launch_pdl(stream, self.fns.depth_mix, (1, 1), 256, &mut args)?;
                0
            };
            if has_msgs {
                let mlp_inv_msg = if msg_inv_base_ptr != 0 {
                    msg_inv_base_ptr + 3 * 4
                } else {
                    0
                };
                self.enqueue_ctrl_n(
                    stream,
                    self.d_mlp_h.ptr,
                    msg_base_ptr + 3 * d_bytes,
                    inv_cur_ptr,
                    mlp_inv_msg,
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
                    self.launch_pdl(stream, f_rn_q8, (1, 1), 256, &mut args)?;
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
                self.launch_pdl(stream, self.fns.rmsnorm_d, (1, 1), 256, &mut args)?;
            }
        }

        // 10. Exact BF16 Router + Warp-Butterfly TopK8 + Wide `uint4` Full-GPU MoE!
        if let (true, Some(f_rg)) = (self.fuse_router, self.fns.router_gemv_topk8) {
            let mut pw = l.router_w.ptr;
            let mut px = self.d_mlp_h.ptr;
            let mut pl = self.d_router_logits.ptr;
            let mut pb = l.router_bias.ptr;
            let mut pids = self.d_sel_ids.ptr;
            let mut pwts = self.d_sel_wts.ptr;
            let mut pdone = self.d_router_done.ptr;
            let mut args: [*mut c_void; 7] = [
                (&mut pw as *mut _) as *mut c_void,
                (&mut px as *mut _) as *mut c_void,
                (&mut pl as *mut _) as *mut c_void,
                (&mut pb as *mut _) as *mut c_void,
                (&mut pids as *mut _) as *mut c_void,
                (&mut pwts as *mut _) as *mut c_void,
                (&mut pdone as *mut _) as *mut c_void,
            ];
            self.launch_pdl(stream, f_rg, (N_EXP as u32, 1), 32, &mut args)?;
        } else {
            self.enqueue_bf16_gemv(
                stream,
                l.router_w.ptr,
                self.d_mlp_h.ptr,
                self.d_router_logits.ptr,
                N_EXP as i32,
                D as i32,
                0.0,
            )?;
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
            self.launch_pdl(stream, self.fns.router_topk8, (1, 1), 32, &mut args)?;
        }

        let [w13, w2] = &self.expert_weights[il];
        let mut mlp_conv_done = false;
        match &l.expert_scales {
            ExpertScales::Nvfp4 { s13, a13, s2, a2 } => {
                if self.use_moe_dp4a {
                    if !mlp_q8_16_ready {
                        self.enqueue_quant_q8_16(stream, self.d_mlp_h.ptr, 1, D)?;
                    }
                    let q8_fused =
                        self.enqueue_nvfp4_up(stream, w13, s13, a13, self.stage_host_up)?;
                    let (mut pxq, mut pxs) = if q8_fused {
                        (self.d_moe_q8_16.ptr, self.d_moe_sc_16.ptr)
                    } else {
                        self.enqueue_quant_q8_16(stream, self.d_moe_act.ptr, TOP_K, FF)?;
                        (self.d_act_q8_16.ptr, self.d_act_sc_16.ptr)
                    };
                    let mut pids = self.d_sel_ids.ptr;
                    let mut pw2 = w2.ptr();
                    let mut ps2 = s2.buffer.ptr;
                    let mut pa2 = a2.ptr;
                    let mut pwts = self.d_sel_wts.ptr;
                    let allow_xw = self.moe_down_xw == 2
                        || (self.moe_down_xw == 1 && matches!(w2, TierBuf::Vram(_)));
                    if let (true, true, Some(f_conv)) = (
                        allow_xw,
                        self.solo_conv_fuse,
                        s2.kernels.nvfp4_dp4a_down_xw_conv,
                    ) {
                        let mut pwc = l.mlp_conv4.ptr;
                        let mut phist = st.hist_mlp_at(self.cur_seq);
                        let mut pcur = self.d_cur.ptr;
                        let mut dn_conv_args: [*mut c_void; 10] = [
                            (&mut pw2 as *mut _) as *mut c_void,
                            (&mut ps2 as *mut _) as *mut c_void,
                            (&mut pa2 as *mut _) as *mut c_void,
                            (&mut pxq as *mut _) as *mut c_void,
                            (&mut pxs as *mut _) as *mut c_void,
                            (&mut pids as *mut _) as *mut c_void,
                            (&mut pwts as *mut _) as *mut c_void,
                            (&mut pwc as *mut _) as *mut c_void,
                            (&mut phist as *mut _) as *mut c_void,
                            (&mut pcur as *mut _) as *mut c_void,
                        ];
                        self.launch_pdl(
                            stream,
                            f_conv,
                            ((D / 2) as u32, 1),
                            256,
                            &mut dn_conv_args,
                        )?;
                        mlp_conv_done = true;
                    } else {
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
                        if let (true, Some(f)) = (allow_xw, s2.kernels.nvfp4_dp4a_down_xw) {
                            self.launch_pdl(stream, f, ((D / 2) as u32, 1), 256, &mut dn_args)?;
                        } else {
                            self.launch(
                                stream,
                                s2.kernels.nvfp4_dp4a_down,
                                (288, 1),
                                256,
                                &mut dn_args,
                            )?;
                        }
                    }
                } else {
                    let mut pw13 = w13.ptr();
                    let mut ps13 = s13.buffer.ptr;
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
                        s13.kernels.nvfp4_up_situ,
                        (288, 8),
                        256,
                        &mut up_args,
                    )?;

                    let mut pw2 = w2.ptr();
                    let mut ps2 = s2.buffer.ptr;
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
                    self.launch(stream, s2.kernels.nvfp4_down, (288, 1), 256, &mut dn_args)?;
                }
            }
            ExpertScales::Mxfp8 { s13, s2 } => {
                let mut pw13 = self.mxfp8_weight_ptr(stream, w13, 2 * FF * D, true)?;
                let mut ps13 = s13.buffer.ptr;
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
                self.launch_pdl(
                    stream,
                    s13.kernels.mxfp8_up_situ,
                    (288, 8),
                    256,
                    &mut up_args,
                )?;

                let mut pw2 = self.mxfp8_weight_ptr(stream, w2, FF * D, true)?;
                let mut ps2 = s2.buffer.ptr;
                let mut pwts = self.d_sel_wts.ptr;
                let allow_xw = self.moe_down_xw == 2
                    || (self.moe_down_xw == 1 && matches!(w2, TierBuf::Vram(_)));
                if let (true, true, Some(f_conv)) =
                    (allow_xw, self.solo_conv_fuse, s2.kernels.mxfp8_down_xw_conv)
                {
                    let mut pwc = l.mlp_conv4.ptr;
                    let mut phist = st.hist_mlp_at(self.cur_seq);
                    let mut pcur = self.d_cur.ptr;
                    let mut dn_conv_args: [*mut c_void; 8] = [
                        (&mut pw2 as *mut _) as *mut c_void,
                        (&mut ps2 as *mut _) as *mut c_void,
                        (&mut pact as *mut _) as *mut c_void,
                        (&mut pids as *mut _) as *mut c_void,
                        (&mut pwts as *mut _) as *mut c_void,
                        (&mut pwc as *mut _) as *mut c_void,
                        (&mut phist as *mut _) as *mut c_void,
                        (&mut pcur as *mut _) as *mut c_void,
                    ];
                    self.launch_pdl(stream, f_conv, ((D / 2) as u32, 1), 256, &mut dn_conv_args)?;
                    mlp_conv_done = true;
                } else {
                    let mut py = self.d_moe_y.ptr;
                    let mut dn_args: [*mut c_void; 6] = [
                        (&mut pw2 as *mut _) as *mut c_void,
                        (&mut ps2 as *mut _) as *mut c_void,
                        (&mut pact as *mut _) as *mut c_void,
                        (&mut pids as *mut _) as *mut c_void,
                        (&mut pwts as *mut _) as *mut c_void,
                        (&mut py as *mut _) as *mut c_void,
                    ];
                    if let (true, Some(f)) = (allow_xw, s2.kernels.mxfp8_down_xw) {
                        self.launch_pdl(stream, f, ((D / 2) as u32, 1), 256, &mut dn_args)?;
                    } else {
                        self.launch(stream, s2.kernels.mxfp8_down, (288, 1), 256, &mut dn_args)?;
                    }
                }
            }
        }

        // 11. `mlp_sconv` + Residual Add in-place into `d_cur`!
        if !mlp_conv_done {
            let mut p_cur = self.d_moe_y.ptr;
            let mut p_w = l.mlp_conv4.ptr;
            let mut p_dyn: CUdeviceptr = 0;
            let mut p_hist = st.hist_mlp_at(self.cur_seq);
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
            self.launch_pdl(stream, self.fns.short_conv4, (9, 1), 256, &mut args)?;
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
            self.enqueue_visit(stream, il, il, &mut n_blocks, 0, 0)?;
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
            self.launch_pdl(stream, f_setup, (9, 1), 256, &mut args)?;
        } else {
            self.enqueue_copy_d(stream, self.d_cur.ptr, self.d_prelude_x.ptr)?;
            self.enqueue_copy_d(stream, self.d_blocks.ptr, self.d_prelude_block0.ptr)?;
        }

        for pass in 0..3 {
            let mut have_msg_invs = false;
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
                    self.launch_pdl(stream, f_setup, (9, 1), 256, &mut args)?;
                } else {
                    self.enqueue_copy_d(stream, self.d_blocks.ptr + 2 * d_bytes, self.d_bank.ptr)?;
                    self.enqueue_copy_d(stream, self.d_cur.ptr, self.d_bank.ptr + d_bytes)?;
                    self.enqueue_copy_d(stream, self.d_prelude_x.ptr, self.d_cur.ptr)?;
                    self.enqueue_copy_d(stream, self.d_prelude_block0.ptr, self.d_blocks.ptr)?;
                }
                n_blocks = 1;
                // Precompute ALL `12 * 4 = 48` cross-pass messages (+ RMS inverses!) in ONE `48`-CTA launch!
                let mut pb = self.d_bank.ptr;
                let mut nb = 2i32;
                let mut pc: CUdeviceptr = 0;
                let mut ac = 0i32;
                let mut pq = self.d_all_msg_queries.ptr;
                let mut pout = self.d_msgs.ptr;
                let mut b3 = 0i32;
                if let Some(f_dmi) = self.fns.depth_mix_inv {
                    let mut pinv = self.d_ctrl_inv_msgs.ptr;
                    let mut args: [*mut c_void; 8] = [
                        (&mut pb as *mut _) as *mut c_void,
                        (&mut nb as *mut _) as *mut c_void,
                        (&mut pc as *mut _) as *mut c_void,
                        (&mut ac as *mut _) as *mut c_void,
                        (&mut pq as *mut _) as *mut c_void,
                        (&mut pout as *mut _) as *mut c_void,
                        (&mut b3 as *mut _) as *mut c_void,
                        (&mut pinv as *mut _) as *mut c_void,
                    ];
                    self.launch_pdl(stream, f_dmi, (48, 1), 256, &mut args)?;
                    have_msg_invs = true;
                } else {
                    let mut args: [*mut c_void; 7] = [
                        (&mut pb as *mut _) as *mut c_void,
                        (&mut nb as *mut _) as *mut c_void,
                        (&mut pc as *mut _) as *mut c_void,
                        (&mut ac as *mut _) as *mut c_void,
                        (&mut pq as *mut _) as *mut c_void,
                        (&mut pout as *mut _) as *mut c_void,
                        (&mut b3 as *mut _) as *mut c_void,
                    ];
                    self.launch_pdl(stream, self.fns.depth_mix, (48, 1), 256, &mut args)?;
                }
            }
            for ip in 6..18 {
                let slot = if pass == 0 {
                    ip
                } else {
                    N_LAYER + (pass - 1) * 12 + (ip - 6)
                };
                let (msg_ptr, msg_inv_ptr) = if pass > 0 {
                    let m_off = (ip - 6) * 4;
                    (
                        self.d_msgs.ptr + (m_off as u64) * d_bytes,
                        if have_msg_invs {
                            self.d_ctrl_inv_msgs.ptr + (m_off as u64) * 4
                        } else {
                            0
                        },
                    )
                } else {
                    (0, 0)
                };
                self.enqueue_visit(stream, ip, slot, &mut n_blocks, msg_ptr, msg_inv_ptr)?;
            }
        }

        for il in 18..24 {
            self.enqueue_visit(stream, il, il, &mut n_blocks, 0, 0)?;
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
                    self.launch_pdl(stream, f_mlp_fused, (1, 1), 256, &mut args)?;
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
                    self.launch_pdl(stream, self.fns.depth_mix, (1, 1), 256, &mut args)?;
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
                    self.launch_pdl(stream, self.fns.rmsnorm_d, (1, 1), 256, &mut rn_args)?;
                }
            }
            let mut pp = self.d_pos.ptr;
            let mut args: [*mut c_void; 1] = [(&mut pp as *mut _) as *mut c_void];
            self.launch_pdl(stream, self.fns.inc_pos, (1, 1), 32, &mut args)?;
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
            self.launch_pdl(stream, f_fused, (1, 1), 256, &mut args)?;
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
            self.launch_pdl(stream, self.fns.depth_mix, (1, 1), 256, &mut args)?;
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
                self.launch_pdl(stream, self.fns.rmsnorm_d, (1, 1), 256, &mut rn_args)?;
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
            let mut pt = self.d_token_in.ptr;
            let mut pp = self.d_pos.ptr;
            let mut args: [*mut c_void; 3] = [
                (&mut pl as *mut _) as *mut c_void,
                (&mut pt as *mut _) as *mut c_void,
                (&mut pp as *mut _) as *mut c_void,
            ];
            self.launch_pdl(stream, self.fns.argmax_and_inc_pos, (1, 1), 256, &mut args)?;
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
        unsafe {
            let ring = self.h_token_ring.host_ptr.cast::<i32>();
            core::ptr::write_volatile(ring, token as i32);
            check_cu(
                (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                    self.d_token_in.ptr,
                    ring.cast(),
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
        unsafe {
            check_cu(
                (self.ctx.api.cuMemcpyDtoDAsync_v2)(
                    self.h_token_ring.dev_ptr + 4,
                    self.d_token_in.ptr,
                    4,
                    self.ctx.compute_stream,
                ),
                "DtoHAsync(token_out)",
            )?;
        }
        self.ctx.sync_compute()?;
        let out_tok =
            unsafe { core::ptr::read_volatile(self.h_token_ring.host_ptr.cast::<i32>().add(1)) };
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

        self.ensure_prefill_capacity(tokens.len())?;

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

    /// Tensor Core batch paths (>= 32 rows) are off in multi-sequence rounds, whose rows must
    /// match B = 1 decode arithmetic for any batch size.
    fn tc_rows(&self, b_len: usize) -> bool {
        b_len >= 32 && self.ms_rows.get() == 0
    }
    /// Decode / verify-style rows: `<= MAX_SPEC_TOKENS`, or any multi-sequence round.
    fn small_rows(&self, b_len: usize) -> bool {
        b_len <= MAX_SPEC_TOKENS || self.ms_rows.get() != 0
    }

    /// One decode step for up to `MAX_SPEC_TOKENS` independent sequences in a single batched
    /// pass (#107): row `i` feeds `tokens[i]` to sequence `seqs[i]` at that sequence's position
    /// and returns its greedy next token. Each row uses the same arithmetic as a B = 1 decode
    /// step of that sequence, so the tokens equal `select_seq(s)` + `forward_greedy_token`.
    pub fn decode_round(&mut self, seqs: &[usize], tokens: &[u32]) -> Result<Vec<u32>, String> {
        let b_len = seqs.len();
        let max_rows = if self.n_seqs > 1 {
            MAX_MS_ROWS.min(self.prefill.capacity)
        } else {
            MAX_SPEC_TOKENS
        };
        if b_len == 0 || b_len > max_rows || tokens.len() != b_len {
            return Err(format!(
                "decode_round: 1..={max_rows} rows with one token each"
            ));
        }
        if self.attn_split_g == 0 || self.drafter.is_some() {
            return Err("decode_round needs split-KV attention and --spec 0".into());
        }
        let mut rows = Vec::with_capacity(b_len * 2);
        for (i, &s) in seqs.iter().enumerate() {
            if s >= self.n_seqs || seqs[..i].contains(&s) {
                return Err(format!("decode_round: bad or repeated sequence {s}"));
            }
            let pos = if s == self.cur_seq {
                self.host_pos
            } else {
                self.seq_pos[s]
            };
            if pos + 1 > self.max_ctx {
                return Err(format!("decode_round: sequence {s} reached max_ctx"));
            }
            rows.push(pos as i32);
            rows.push(s as i32);
        }
        self.ctx.bind_current()?;
        unsafe {
            check_cu(
                (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                    self.d_ms_rows.ptr,
                    rows.as_ptr().cast(),
                    b_len * 8,
                    self.ctx.compute_stream,
                ),
                "HtoD(ms_rows)",
            )?;
            check_cu(
                (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                    self.d_ms_tokens.ptr,
                    tokens.as_ptr().cast(),
                    b_len * 4,
                    self.ctx.compute_stream,
                ),
                "HtoD(ms_tokens)",
            )?;
        }
        self.ms_rows.set(self.d_ms_rows.ptr);
        // Rows, positions and tokens live in device tables, so a round of `b_len` rows replays
        // the same launch sequence every step: capture it once the row count repeats.
        let hits = {
            let h = self.ms_round_hits.entry(b_len).or_insert(0);
            *h += 1;
            *h
        };
        let r = if !self.enable_cuda_graph || !self.ms_graph {
            self.enqueue_batch_step_impl(
                self.ctx.compute_stream,
                self.d_ms_tokens.ptr,
                b_len,
                BatchHeadMode::AllHeads,
                false,
            )
        } else if let Some(g) = self.ms_graphs.get(&b_len) {
            g.replay()
        } else if hits >= 2 {
            let ctx_clone = Arc::clone(&self.ctx);
            let tok_ptr = self.d_ms_tokens.ptr;
            match CapturedGraphExec::capture(&ctx_clone, |stream| {
                self.enqueue_batch_step_impl(stream, tok_ptr, b_len, BatchHeadMode::AllHeads, false)
            }) {
                Ok(g) => {
                    let r = g.replay();
                    self.ms_graphs.insert(b_len, g);
                    r
                }
                Err(e) => Err(e),
            }
        } else {
            self.enqueue_batch_step_impl(
                self.ctx.compute_stream,
                self.d_ms_tokens.ptr,
                b_len,
                BatchHeadMode::AllHeads,
                false,
            )
        };
        self.ms_rows.set(0);
        r?;
        self.ctx.sync_compute()?;
        for &s in seqs {
            if s == self.cur_seq {
                self.host_pos += 1;
            } else {
                self.seq_pos[s] += 1;
            }
        }
        let pos = self.host_pos as i32;
        let mut out_i32 = vec![0i32; b_len];
        unsafe {
            check_cu(
                (self.ctx.api.cuMemcpyHtoD_v2)(self.d_pos.ptr, (&pos as *const i32).cast(), 4),
                "HtoD(pos)",
            )?;
            check_cu(
                (self.ctx.api.cuMemcpyDtoH_v2)(
                    out_i32.as_mut_ptr().cast(),
                    self.d_spec_tokens_out.ptr,
                    b_len * 4,
                ),
                "DtoH(ms_tokens_out)",
            )?;
        }
        Ok(out_i32.into_iter().map(|v| v as u32).collect())
    }

    /// Speculative verify round over several sequences (#107 stage 3). `runs[r] = (seq, tokens)`
    /// feeds `tokens` (the last accepted token, then drafts) to `seq` at consecutive positions;
    /// all runs share one batched pass of at most `MAX_MS_ROWS` rows. Returns every row's greedy
    /// token, which equals what B = 1 decode steps of that sequence would produce. Positions and
    /// `short_conv4` histories do not move until `commit_verify_round` keeps each run's
    /// accepted prefix.
    pub fn verify_round(&mut self, runs: &[(usize, &[u32])]) -> Result<Vec<Vec<u32>>, String> {
        let b_len: usize = runs.iter().map(|r| r.1.len()).sum();
        let max_rows = MAX_MS_ROWS.min(self.prefill.capacity);
        if self.n_seqs < 2 || runs.is_empty() || b_len > max_rows {
            return Err(format!(
                "verify_round: LEANSONIC_SEQS > 1 and 1..={max_rows} rows"
            ));
        }
        if self.attn_split_g == 0 {
            return Err("verify_round needs split-KV attention".into());
        }
        if self.fns.short_conv4_vms.is_none() || self.fns.conv_commit_ms.is_none() {
            return Err("cubin lacks ink2_short_conv4_vms / ink2_conv_commit_ms".into());
        }
        if self.d_ms_xrec.is_none() {
            self.d_ms_xrec = Some(DeviceBuf::alloc_zeroed(
                &self.ctx,
                N_SLOT * CONV_XREC_SLOT_FLOATS * 4,
            )?);
        }
        let mut rows = Vec::with_capacity(b_len * 2);
        let mut js = Vec::with_capacity(b_len);
        let mut tokens = Vec::with_capacity(b_len);
        for (i, &(s, toks)) in runs.iter().enumerate() {
            if s >= self.n_seqs || runs[..i].iter().any(|r| r.0 == s) || toks.is_empty() {
                return Err(format!("verify_round: bad, repeated or empty sequence {s}"));
            }
            let pos = if s == self.cur_seq {
                self.host_pos
            } else {
                self.seq_pos[s]
            };
            if pos + toks.len() > self.max_ctx {
                return Err(format!("verify_round: sequence {s} would pass max_ctx"));
            }
            for (j, &t) in toks.iter().enumerate() {
                rows.push((pos + j) as i32);
                rows.push(s as i32);
                js.push(j as i32);
                tokens.push(t);
            }
        }
        self.ctx.bind_current()?;
        unsafe {
            for (dst, src, bytes, what) in [
                (
                    self.d_ms_rows.ptr,
                    rows.as_ptr().cast::<c_void>(),
                    b_len * 8,
                    "HtoD(ms_rows)",
                ),
                (self.d_ms_j.ptr, js.as_ptr().cast(), b_len * 4, "HtoD(ms_j)"),
                (
                    self.d_ms_tokens.ptr,
                    tokens.as_ptr().cast(),
                    b_len * 4,
                    "HtoD(ms_tokens)",
                ),
            ] {
                check_cu(
                    (self.ctx.api.cuMemcpyHtoDAsync_v2)(dst, src, bytes, self.ctx.compute_stream),
                    what,
                )?;
            }
        }
        self.ms_rows.set(self.d_ms_rows.ptr);
        self.ms_xrec
            .set(self.d_ms_xrec.as_ref().map_or(0, |b| b.ptr));
        let r = self.enqueue_batch_step_impl(
            self.ctx.compute_stream,
            self.d_ms_tokens.ptr,
            b_len,
            BatchHeadMode::AllHeads,
            false,
        );
        self.ms_rows.set(0);
        self.ms_xrec.set(0);
        r?;
        self.ctx.sync_compute()?;
        let mut out_i32 = vec![0i32; b_len];
        unsafe {
            check_cu(
                (self.ctx.api.cuMemcpyDtoH_v2)(
                    out_i32.as_mut_ptr().cast(),
                    self.d_spec_tokens_out.ptr,
                    b_len * 4,
                ),
                "DtoH(ms_tokens_out)",
            )?;
        }
        self.ms_verify_runs = runs.iter().map(|r| (r.0, r.1.len())).collect();
        let mut out = Vec::with_capacity(runs.len());
        let mut b = 0;
        for &(_, toks) in runs {
            out.push(
                out_i32[b..b + toks.len()]
                    .iter()
                    .map(|&v| v as u32)
                    .collect(),
            );
            b += toks.len();
        }
        Ok(out)
    }

    /// Keeps the first `keep[r]` rows (`1..=len`) of each run of the last `verify_round`:
    /// advances those sequences' positions and sets their `short_conv4` histories to what
    /// `keep[r]` B = 1 decode steps would have left. Later rows' KV entries are overwritten
    /// by the next rounds, as in single-sequence speculative decoding.
    pub fn commit_verify_round(&mut self, keep: &[usize]) -> Result<(), String> {
        let runs = std::mem::take(&mut self.ms_verify_runs);
        if keep.len() != runs.len() {
            return Err("commit_verify_round: one keep count per run".into());
        }
        let mut table = Vec::with_capacity(runs.len() * 4);
        let mut first = 0usize;
        for (&(s, len), &k) in runs.iter().zip(keep) {
            if k == 0 || k > len {
                return Err(format!("commit_verify_round: keep {k} outside 1..={len}"));
            }
            table.extend_from_slice(&[first as i32, k as i32, s as i32, 0]);
            first += len;
        }
        let f = self
            .fns
            .conv_commit_ms
            .ok_or("cubin lacks ink2_conv_commit_ms")?;
        let xrec = self
            .d_ms_xrec
            .as_ref()
            .ok_or("commit_verify_round without a round")?;
        unsafe {
            check_cu(
                (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                    self.d_ms_runs.ptr,
                    table.as_ptr().cast(),
                    table.len() * 4,
                    self.ctx.compute_stream,
                ),
                "HtoD(ms_runs)",
            )?;
        }
        let mut p_x = xrec.ptr;
        let mut p_h = self.d_conv_hist_ptrs.ptr;
        let mut p_r = self.d_ms_runs.ptr;
        let mut stride = MAX_MS_ROWS as i32;
        let mut args: [*mut c_void; 4] = [
            (&mut p_x as *mut _) as *mut c_void,
            (&mut p_h as *mut _) as *mut c_void,
            (&mut p_r as *mut _) as *mut c_void,
            (&mut stride as *mut _) as *mut c_void,
        ];
        self.launch_smem_3d(
            self.ctx.compute_stream,
            f,
            (9, (N_SLOT * 4) as u32, runs.len() as u32),
            256,
            0,
            &mut args,
        )?;
        self.ctx.sync_compute()?;
        for (&(s, _), &k) in runs.iter().zip(keep) {
            if s == self.cur_seq {
                self.host_pos += k;
            } else {
                self.seq_pos[s] += k;
            }
        }
        // DSpark: each sequence's drafter context takes its kept rows' taps.
        if let Some(d) = &mut self.drafter {
            let taps = self
                .prefill
                .d_taps
                .as_ref()
                .ok_or("missing DSpark tap storage")?;
            let mut first = 0usize;
            for (&(s, len), &k) in runs.iter().zip(keep) {
                d.select_seq(s)?;
                d.enqueue_add_ctx(
                    self.ctx.compute_stream,
                    taps.ptr + (first * D * 4) as u64,
                    self.prefill.capacity,
                    k,
                )?;
                first += len;
            }
            d.select_seq(self.cur_seq)?;
        }
        let pos = self.host_pos as i32;
        unsafe {
            check_cu(
                (self.ctx.api.cuMemcpyHtoD_v2)(self.d_pos.ptr, (&pos as *const i32).cast(), 4),
                "HtoD(pos)",
            )?;
        }
        Ok(())
    }

    /// DSpark drafts for several sequences at once: `reqs[i] = (seq, next_tok, max_k)`. Each
    /// sequence drafts from its own context and scratch on its own stream, so the drafts run
    /// concurrently; the tokens equal `draft_for_seq` one sequence at a time.
    pub fn draft_for_seqs(
        &mut self,
        reqs: &[(usize, u32, usize)],
    ) -> Result<Vec<Vec<u32>>, String> {
        if self.drafter.is_none() {
            return Ok(vec![Vec::new(); reqs.len()]);
        }
        while self.draft_streams.len() < reqs.len() {
            let mut st: CUstream = core::ptr::null_mut();
            unsafe {
                check_cu(
                    (self.ctx.api.cuStreamCreate)(&mut st, 1 /* CU_STREAM_NON_BLOCKING */),
                    "cuStreamCreate(draft)",
                )?;
            }
            self.draft_streams.push(st);
        }
        if self.draft_ready.is_null() {
            unsafe {
                check_cu(
                    (self.ctx.api.cuEventCreate)(&mut self.draft_ready, CU_EVENT_DISABLE_TIMING),
                    "cuEventCreate(draft_ready)",
                )?;
            }
        }
        // Drafts read the contexts the compute stream last appended to.
        unsafe {
            check_cu(
                (self.ctx.api.cuEventRecord)(self.draft_ready, self.ctx.compute_stream),
                "cuEventRecord(draft_ready)",
            )?;
        }
        let mut pending = Vec::with_capacity(reqs.len());
        for (i, &(seq, tok, max_k)) in reqs.iter().enumerate() {
            let pos = if seq == self.cur_seq {
                self.host_pos
            } else {
                self.seq_pos[seq]
            };
            let max_k = max_k.min(self.max_ctx.saturating_sub(pos + 1));
            let stream = self.draft_streams[i];
            unsafe {
                check_cu(
                    (self.ctx.api.cuStreamWaitEvent)(stream, self.draft_ready, 0),
                    "cuStreamWaitEvent(draft)",
                )?;
            }
            let d = self.drafter.as_mut().unwrap();
            d.select_seq(seq)?;
            pending.push(d.enqueue_draft(
                stream,
                self.embed.ptr(),
                self.embed_norm.ptr,
                &self.unembed_q8,
                tok,
                max_k,
            )?);
        }
        let mut out = Vec::with_capacity(reqs.len());
        for (i, &(seq, _, _)) in reqs.iter().enumerate() {
            self.ctx.sync_stream(self.draft_streams[i])?;
            let (k, tau) = pending[i];
            let d = self.drafter.as_mut().unwrap();
            d.select_seq(seq)?;
            out.push(if k == 0 {
                Vec::new()
            } else {
                d.finish_draft(k, tau)?
            });
        }
        let cur = self.cur_seq;
        self.drafter.as_mut().unwrap().select_seq(cur)?;
        Ok(out)
    }

    /// DSpark drafts (up to `max_k`) for sequence `seq` after its pending token `next_tok`,
    /// from that sequence's own drafter context. Empty without a drafter (`--spec 0`).
    pub fn draft_for_seq(
        &mut self,
        seq: usize,
        next_tok: u32,
        max_k: usize,
    ) -> Result<Vec<u32>, String> {
        let Some(d) = self.drafter.as_mut() else {
            return Ok(Vec::new());
        };
        let pos = if seq == self.cur_seq {
            self.host_pos
        } else {
            self.seq_pos[seq]
        };
        let max_k = max_k.min(self.max_ctx.saturating_sub(pos + 1));
        d.select_seq(seq)?;
        let r = d.draft_tokens(
            self.ctx.compute_stream,
            self.embed.ptr(),
            self.embed_norm.ptr,
            &self.unembed_q8,
            next_tok,
            max_k,
        );
        d.select_seq(self.cur_seq)?;
        r
    }

    /// Benchmark-only: move the decode position to `pos` without prefilling the rows in between.
    /// Those KV rows hold whatever the cache contains (zeros after load), so the logits are not
    /// meaningful, but each decode step reads exactly as many KV rows as a real `pos`-token
    /// context does. Used by `bench` (`LEANSONIC_BENCH_DECODE_AT`) to time long-context decode
    /// without a multi-minute prefill.
    pub fn bench_jump_position(&mut self, pos: usize) -> Result<(), String> {
        if pos < self.host_pos || pos >= self.max_ctx {
            return Err(format!(
                "bench_jump_position: {pos} must be in [{}, {})",
                self.host_pos, self.max_ctx
            ));
        }
        self.ctx.bind_current()?;
        self.ctx.sync_compute()?;
        let p = pos as i32;
        unsafe {
            check_cu(
                (self.ctx.api.cuMemcpyHtoD_v2)(self.d_pos.ptr, (&p as *const i32).cast(), 4),
                "HtoD(bench pos)",
            )?;
        }
        self.host_pos = pos;
        Ok(())
    }

    /// `O(1)` Rewind of all 48 slots' `short_conv4` histories + `*d_pos` / `host_pos` (`keep < b_len`).
    pub fn rewind_speculative(&mut self, keep: usize, b_len: usize) -> Result<(), String> {
        if keep < b_len {
            let mut pseq = self.d_conv_seq.ptr;
            let mut phist = self.d_conv_hist_ptrs.ptr + (self.cur_seq * N_SLOT * 4 * 8) as u64;
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
        if self.spec_k == 0 && self.drafter.is_none() && stop_tok.is_none() && n_gen > 0 {
            if self.host_pos + n_gen > self.max_ctx {
                return Err(format!("sequence exceeded max_ctx ({})", self.max_ctx));
            }
            unsafe {
                let ring = self.h_token_ring.host_ptr.cast::<i32>();
                core::ptr::write_volatile(ring, cur_tok as i32);
                check_cu(
                    (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                        self.d_token_in.ptr,
                        ring.cast(),
                        4,
                        self.ctx.compute_stream,
                    ),
                    "HtoD(token_in_chain)",
                )?;
            }
            while steps_done < n_gen {
                let chunk = (n_gen - steps_done).min(64);
                for s in 0..chunk {
                    if let Some(ref graph) = self.decode_graph {
                        graph.replay()?;
                    } else {
                        self.enqueue_step_impl(self.ctx.compute_stream, false)?;
                    }
                    unsafe {
                        check_cu(
                            (self.ctx.api.cuMemcpyDtoDAsync_v2)(
                                self.h_token_ring.dev_ptr + ((1 + s) * 4) as u64,
                                self.d_token_in.ptr,
                                4,
                                self.ctx.compute_stream,
                            ),
                            "DtoHAsync(token_chain)",
                        )?;
                    }
                }
                self.ctx.sync_compute()?;
                let ring = self.h_token_ring.host_ptr.cast::<i32>();
                out_ids.push(cur_tok);
                on_token(cur_tok);
                for s in 0..(chunk - 1) {
                    let t = unsafe { core::ptr::read_volatile(ring.add(1 + s)) } as u32;
                    out_ids.push(t);
                    on_token(t);
                }
                cur_tok = unsafe { core::ptr::read_volatile(ring.add(chunk)) } as u32;
                self.host_pos += chunk;
                steps_done += chunk;
            }
            return Ok((out_ids, cur_tok));
        }

        // Draft pause: after `PAUSE_AFTER` steps in a row without an accepted draft, decode
        // the next `pause_len` tokens without drafting (8, doubling to 64 while drafts keep
        // failing). Drafting a context that rejects everything costs a full drafter pass per
        // token (e.g. 128K: drafter attention over the whole context). The emitted tokens are
        // the greedy tokens either way. `LEANSONIC_SPEC_PAUSE=0` disables the pause.
        const PAUSE_AFTER: usize = 4;
        let pause_enabled = std::env::var("LEANSONIC_SPEC_PAUSE").map_or(true, |v| v != "0");
        let (mut zero_streak, mut pause_left, mut pause_len) = (0usize, 0usize, 8usize);
        while steps_done < n_gen {
            if Some(cur_tok) == stop_tok {
                break;
            }
            let rem_steps = n_gen - steps_done;
            let max_draft = self
                .spec_k
                .min(rem_steps.saturating_sub(1))
                .min(self.max_ctx.saturating_sub(self.host_pos + 1));

            let paused = pause_left > 0;
            pause_left = pause_left.saturating_sub(1);
            let drafts = if max_draft == 0 || paused {
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
                if pause_enabled && !paused && max_draft > 0 {
                    zero_streak += 1;
                    if zero_streak >= PAUSE_AFTER {
                        pause_left = pause_len;
                        pause_len = (pause_len * 2).min(64);
                        zero_streak = 0;
                    }
                }
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
            if accepted > 0 {
                zero_streak = 0;
                pause_len = 8;
            } else if pause_enabled {
                zero_streak += 1;
                if zero_streak >= PAUSE_AFTER {
                    pause_left = pause_len;
                    pause_len = (pause_len * 2).min(64);
                    zero_streak = 0;
                }
            }
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

    #[allow(clippy::too_many_arguments)]
    fn enqueue_ctrl_n_b_inner(
        &self,
        stream: CUstream,
        cur_ptr: CUdeviceptr,
        msg_ptr: CUdeviceptr,
        ctrl: &CtrlGpu,
        n_ctrls: u32,
        b_len: usize,
        norm_bf16_ready: bool,
    ) -> Result<(), String> {
        let b_u32 = b_len as u32;
        let mut b_i32 = b_len as i32;
        let mut p_cur = cur_ptr;
        let mut p_msg = msg_ptr;
        let mut p_down = ctrl.down.ptr;
        let mut p_g = self.prefill.d_b_ctrl_g.ptr;
        if let (true, Some(f_norm), Some(f_tc)) = (
            self.tc_bf16 && self.tc_rows(b_len),
            self.fns.ctrl_norm_bf16_b,
            self.fns.bf16_tc_gemm_b,
        ) {
            // Offset past `mlp_hidden` + `router_logits` (`capacity * (D + N_EXP) * 4`) inside `pair_output`
            // so `ctrl_norm_bf16_b` is 100% disjoint even when `cur_ptr == layer_workspace.mlp_hidden`!
            let mut p_bf16 = self.prefill.layer_workspace.pair_output.ptr
                + (self.prefill.capacity * (D + N_EXP) * 4) as u64;
            if !norm_bf16_ready {
                let mut norm_args: [*mut c_void; 3] = [
                    (&mut p_cur as *mut _) as *mut c_void,
                    (&mut p_msg as *mut _) as *mut c_void,
                    (&mut p_bf16 as *mut _) as *mut c_void,
                ];
                self.launch(stream, f_norm, (n_ctrls * b_u32, 1), 256, &mut norm_args)?;
            }
            let mut rows = 64i32;
            let mut cols = (2 * D) as i32;
            let t_blocks = b_u32.div_ceil(64);
            let mut t_blocks_i32 = t_blocks as i32;
            let mut gemm_args: [*mut c_void; 7] = [
                (&mut p_down as *mut _) as *mut c_void,
                (&mut p_bf16 as *mut _) as *mut c_void,
                (&mut p_g as *mut _) as *mut c_void,
                (&mut rows as *mut _) as *mut c_void,
                (&mut cols as *mut _) as *mut c_void,
                (&mut b_i32 as *mut _) as *mut c_void,
                (&mut t_blocks_i32 as *mut _) as *mut c_void,
            ];
            self.launch(stream, f_tc, (1, t_blocks * n_ctrls), 256, &mut gemm_args)?;
        } else {
            let mut args1: [*mut c_void; 5] = [
                (&mut p_cur as *mut _) as *mut c_void,
                (&mut p_msg as *mut _) as *mut c_void,
                (&mut p_down as *mut _) as *mut c_void,
                (&mut p_g as *mut _) as *mut c_void,
                (&mut b_i32 as *mut _) as *mut c_void,
            ];
            // A short batch has too few blocks to amortize an eight-wide tile.
            // Optional symbols also support previously built external cubins.
            let (down, grid) = match (self.fns.ctrl_down_r8_t2_b, self.fns.ctrl_down_r8_b) {
                (Some(f), _) if b_len >= 64 && self.ms_rows.get() == 0 => {
                    (f, (8, n_ctrls * b_u32.div_ceil(2)))
                }
                (_, Some(f)) if n_ctrls * b_u32 >= 2 => (f, (8, n_ctrls * b_u32)),
                _ => (self.fns.ctrl_down_b, (64, n_ctrls * b_u32)),
            };
            self.launch(stream, down, grid, 256, &mut args1)?;
        }

        let mut p_gain = ctrl.gain.ptr;
        let mut p_forget = ctrl.forget.ptr;
        if self.tc_bf16 && self.tc_rows(b_len) {
            if let Some(f_apply_tc) = self.fns.ctrl_apply_tc_b {
                let t_blocks = b_u32.div_ceil(64);
                let mut tb = t_blocks as i32;
                let mut args_tc: [*mut c_void; 7] = [
                    (&mut p_cur as *mut _) as *mut c_void,
                    (&mut p_msg as *mut _) as *mut c_void,
                    (&mut p_g as *mut _) as *mut c_void,
                    (&mut p_gain as *mut _) as *mut c_void,
                    (&mut p_forget as *mut _) as *mut c_void,
                    (&mut b_i32 as *mut _) as *mut c_void,
                    (&mut tb as *mut _) as *mut c_void,
                ];
                return self.launch(
                    stream,
                    f_apply_tc,
                    (72, t_blocks * n_ctrls),
                    256,
                    &mut args_tc,
                );
            }
        }
        let mut args2: [*mut c_void; 6] = [
            (&mut p_cur as *mut _) as *mut c_void,
            (&mut p_msg as *mut _) as *mut c_void,
            (&mut p_g as *mut _) as *mut c_void,
            (&mut p_gain as *mut _) as *mut c_void,
            (&mut p_forget as *mut _) as *mut c_void,
            (&mut b_i32 as *mut _) as *mut c_void,
        ];
        if b_len >= 16 && self.ms_rows.get() == 0 {
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
        if self.tc_q8 && self.tc_rows(b_len) {
            if let Some(f_tc) = self.fns.q8_tc_gemm_b {
                let grid_x = (rows as u32).div_ceil(128);
                let grid_y = (b_len as u32).div_ceil(128);
                return self.launch_smem(stream, f_tc, (grid_x, grid_y), 512, 64_512, &mut args);
            }
        }
        let grid_x = (rows as u32).div_ceil(16);
        if b_len <= 8 && cols <= 3072 {
            if let Some(f_b8) = self.fns.q8_dp4a_gemm_b8_fast {
                return self.launch(stream, f_b8, (grid_x, 1), 256, &mut args);
            }
        }
        let grid_y = (b_len as u32).div_ceil(8);
        // #107: multi-sequence rounds use `ink2_q8_dp4a_gemm_b` arithmetic with 4 or 8 token
        // tiles per block, so each weight fragment is loaded once for up to 64 tokens.
        if self.ms_rows.get() != 0 && self.ms_gemm_tiles {
            let wide = grid_y > 4;
            let f = if wide {
                self.fns.q8_dp4a_gemm_b_ms8
            } else {
                self.fns.q8_dp4a_gemm_b_ms4
            };
            if let Some(f) = f {
                let gy = grid_y.div_ceil(if wide { 8 } else { 4 });
                return self.launch(stream, f, (grid_x, gy), 256, &mut args);
            }
        }
        self.launch(
            stream,
            self.fns.q8_dp4a_gemm_b,
            (grid_x, grid_y),
            256,
            &mut args,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_short_conv4_par_b(
        &self,
        stream: CUstream,
        cur_ptr: CUdeviceptr,
        in_stride: i32,
        w_ptr: CUdeviceptr,
        dyn_ptr: CUdeviceptr,
        dyn_f4_stride: i32,
        hist_ptr: CUdeviceptr,
        out_ptr: CUdeviceptr,
        out_stride: i32,
        res_ptr: CUdeviceptr,
        res_stride: i32,
        save_blk_ptr: CUdeviceptr,
        channels: i32,
        b_len: usize,
        has_dyn: i32,
        do_res_add: i32,
    ) -> Result<bool, String> {
        // Multi-sequence rounds keep per-row histories (`ink2_short_conv4_ms`).
        if !self.par_conv || b_len < 32 || self.ms_rows.get() != 0 {
            return Ok(false);
        }
        let (Some(f_par), Some(f_commit)) = (
            self.fns.short_conv4_save_par_b,
            self.fns.short_conv4_commit_hist_b,
        ) else {
            return Ok(false);
        };
        let mut p_cur = cur_ptr;
        let mut ins = in_stride;
        let mut p_w = w_ptr;
        let mut p_dyn = dyn_ptr;
        let mut df4 = dyn_f4_stride;
        let mut p_hist = hist_ptr;
        let mut p_out = out_ptr;
        let mut outs = out_stride;
        let mut p_res = res_ptr;
        let mut ress = res_stride;
        let mut p_save = save_blk_ptr;
        let mut ch = channels;
        let mut bi = b_len as i32;
        let mut hd = has_dyn;
        let mut dr = do_res_add;
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
            (&mut p_save as *mut _) as *mut c_void,
            (&mut ch as *mut _) as *mut c_void,
            (&mut bi as *mut _) as *mut c_void,
            (&mut hd as *mut _) as *mut c_void,
            (&mut dr as *mut _) as *mut c_void,
        ];
        let gx = (channels as u32).div_ceil(256);
        let gy = (b_len as u32).div_ceil(32);
        self.launch(stream, f_par, (gx, gy), 256, &mut args)?;

        let mut c_args: [*mut c_void; 5] = [
            (&mut p_cur as *mut _) as *mut c_void,
            (&mut ins as *mut _) as *mut c_void,
            (&mut p_hist as *mut _) as *mut c_void,
            (&mut ch as *mut _) as *mut c_void,
            (&mut bi as *mut _) as *mut c_void,
        ];
        self.launch(stream, f_commit, (gx, 1), 256, &mut c_args)?;
        Ok(true)
    }

    /// Row-independent short convolution of a multi-sequence round (#107): row `b` steps the
    /// history of sequence `ms_rows[b].y` (`hist_seq0` + `seq * hist_seq_bytes`).
    #[allow(clippy::too_many_arguments)]
    fn enqueue_short_conv4_ms(
        &self,
        stream: CUstream,
        cur: CUdeviceptr,
        in_stride: i32,
        w: CUdeviceptr,
        dyn_ptr: CUdeviceptr,
        dyn_f4_stride: i32,
        hist_seq0: CUdeviceptr,
        hist_seq_bytes: u64,
        out: CUdeviceptr,
        out_stride: i32,
        res: CUdeviceptr,
        res_stride: i32,
        save_prev: CUdeviceptr,
        channels: i32,
        b_len: usize,
        has_dyn: i32,
        do_res_add: i32,
        slot_idx: usize,
        kind_off: usize,
    ) -> Result<(), String> {
        let (mut p_cur, mut ins, mut p_w, mut p_dyn, mut df4, mut p_hist) =
            (cur, in_stride, w, dyn_ptr, dyn_f4_stride, hist_seq0);
        let (mut p_out, mut outs, mut p_res, mut ress, mut p_save) =
            (out, out_stride, res, res_stride, save_prev);
        let mut p_rows = self.ms_rows.get();
        let (mut ch, mut hd, mut dr) = (channels, has_dyn, do_res_add);
        let mut hsf = (hist_seq_bytes / 4) as i32;
        let grid = ((channels as u32).div_ceil(256), b_len as u32);
        let xrec_base = self.ms_xrec.get();
        if xrec_base != 0 {
            // Verify round: rows read earlier rows of their sequence; histories are committed
            // afterwards (`commit_verify_round`).
            let mut p_j = self.d_ms_j.ptr;
            let mut p_x = xrec_base
                + ((slot_idx * CONV_XREC_SLOT_FLOATS + kind_off * MAX_MS_ROWS) * 4) as u64;
            if save_prev != 0 {
                let f = self
                    .fns
                    .short_conv4_save_vms
                    .ok_or("cubin lacks ink2_short_conv4_save_vms")?;
                let mut args: [*mut c_void; 18] = [
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
                    (&mut p_rows as *mut _) as *mut c_void,
                    (&mut p_j as *mut _) as *mut c_void,
                    (&mut p_x as *mut _) as *mut c_void,
                    (&mut ch as *mut _) as *mut c_void,
                    (&mut hd as *mut _) as *mut c_void,
                    (&mut dr as *mut _) as *mut c_void,
                    (&mut hsf as *mut _) as *mut c_void,
                ];
                return self.launch(stream, f, grid, 256, &mut args);
            }
            let f = self
                .fns
                .short_conv4_vms
                .ok_or("cubin lacks ink2_short_conv4_vms")?;
            let mut args: [*mut c_void; 17] = [
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
                (&mut p_rows as *mut _) as *mut c_void,
                (&mut p_j as *mut _) as *mut c_void,
                (&mut p_x as *mut _) as *mut c_void,
                (&mut ch as *mut _) as *mut c_void,
                (&mut hd as *mut _) as *mut c_void,
                (&mut dr as *mut _) as *mut c_void,
                (&mut hsf as *mut _) as *mut c_void,
            ];
            return self.launch(stream, f, grid, 256, &mut args);
        }
        if save_prev != 0 {
            let f = self
                .fns
                .short_conv4_save_ms
                .ok_or("cubin lacks ink2_short_conv4_save_ms")?;
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
                (&mut p_rows as *mut _) as *mut c_void,
                (&mut ch as *mut _) as *mut c_void,
                (&mut hd as *mut _) as *mut c_void,
                (&mut dr as *mut _) as *mut c_void,
                (&mut hsf as *mut _) as *mut c_void,
            ];
            self.launch(stream, f, grid, 256, &mut args)
        } else {
            let f = self
                .fns
                .short_conv4_ms
                .ok_or("cubin lacks ink2_short_conv4_ms")?;
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
                (&mut p_rows as *mut _) as *mut c_void,
                (&mut ch as *mut _) as *mut c_void,
                (&mut hd as *mut _) as *mut c_void,
                (&mut dr as *mut _) as *mut c_void,
                (&mut hsf as *mut _) as *mut c_void,
            ];
            self.launch(stream, f, grid, 256, &mut args)
        }
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
        spilled_schedule: &[usize],
        spilled_cursor: &mut usize,
    ) -> Result<(), String> {
        let l = &self.layers[il];
        let st = &self.slots[slot_idx];
        let b_u32 = b_len as u32;
        let (conv, conv_save, channels_per_block) =
            match self.fns.short_conv4_tiled.filter(|_| b_len >= 64) {
                Some((plain, save)) => (plain, Some(save), 32),
                None => (self.fns.short_conv4_b, self.fns.short_conv4_save_b, 256),
            };
        let kv_conv_grid = (1024 / channels_per_block, 1);
        let hidden_conv_grid = (D as u32 / channels_per_block, 1);
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
                    if n_q > 1 {
                        if let Some(f_dm_m) = self.fns.depth_mix_multi_b {
                            let mut nqi = n_q as i32;
                            let mut m_args: [*mut c_void; 11] = [
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
                                (&mut nqi as *mut _) as *mut c_void,
                            ];
                            self.launch(stream, f_dm_m, (1, b_u32), 256, &mut m_args)?;
                        } else {
                            self.launch(
                                stream,
                                self.fns.depth_mix_b,
                                (n_q, b_u32),
                                256,
                                &mut args,
                            )?;
                        }
                    } else {
                        self.launch(stream, self.fns.depth_mix_b, (n_q, b_u32), 256, &mut args)?;
                    }
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
            let mut norm_bf16_ready = false;
            if let (true, Some(f_fused)) = (
                !record_seq
                    && self.tc_bf16
                    && self.tc_rows(b_len)
                    && self.fns.bf16_tc_gemm_b.is_some(),
                self.fns.depth_mix_multi_ctrl_norm_bf16_b,
            ) {
                let mut nqi = n_q as i32;
                let mut pmsg = msg_base_ptr;
                let mut pbf16 = self.prefill.layer_workspace.pair_output.ptr
                    + (self.prefill.capacity * (D + N_EXP) * 4) as u64;
                let mut f_args: [*mut c_void; 13] = [
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
                    (&mut nqi as *mut _) as *mut c_void,
                    (&mut pmsg as *mut _) as *mut c_void,
                    (&mut pbf16 as *mut _) as *mut c_void,
                ];
                self.launch(stream, f_fused, (1, b_u32), 256, &mut f_args)?;
                norm_bf16_ready = true;
            } else if n_q > 1 || !record_seq {
                if let Some(f_dm_m) = self.fns.depth_mix_multi_b {
                    let mut nqi = n_q as i32;
                    let mut m_args: [*mut c_void; 11] = [
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
                        (&mut nqi as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, f_dm_m, (1, b_u32), 256, &mut m_args)?;
                } else {
                    self.launch(stream, self.fns.depth_mix_b, (n_q, b_u32), 256, &mut args)?;
                }
            } else {
                self.launch(stream, self.fns.depth_mix_b, (n_q, b_u32), 256, &mut args)?;
            }
            self.enqueue_ctrl_n_b_inner(
                stream,
                self.prefill.d_b_activations.ptr,
                msg_base_ptr,
                l.ctrl3.as_ref().unwrap(),
                3,
                b_len,
                norm_bf16_ready,
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
        } else {
            let (wk_ref, wk_off, wv_ref, wv_off, kv_rows) = if l.global {
                (
                    l.wk_glob_q8.as_ref().unwrap(),
                    0usize,
                    l.wv_glob_q8.as_ref().unwrap(),
                    0usize,
                    1056i32,
                )
            } else {
                (
                    &l.wq_all_q8,
                    SWA_K_OFFSET,
                    &l.wq_all_q8,
                    SWA_V_OFFSET,
                    1024i32,
                )
            };
            if let (true, Some(f_qkv3_tc)) = (
                self.tc_q8 && self.tc_rows(b_len),
                self.fns.q8_tc_qkv3_gemm_b,
            ) {
                let mut pwq = l.wq_all_q8.codes.ptr;
                let mut pwqs = l.wq_all_q8.scales64.ptr;
                let mut pwk = wk_ref.codes.ptr + (wk_off * D) as u64;
                let mut pwks = wk_ref.scales64.ptr + (wk_off * (D / 64) * 4) as u64;
                let mut pwv = wv_ref.codes.ptr + (wv_off * D) as u64;
                let mut pwvs = wv_ref.scales64.ptr + (wv_off * (D / 64) * 4) as u64;
                let mut px = self.prefill.d_b_act_codes.ptr;
                let mut pxs = self.prefill.d_b_act_scales.ptr;
                let mut pyq = self.prefill.layer_workspace.qrgs;
                let mut pyk = self.prefill.layer_workspace.k_raw;
                let mut pyv = self.prefill.layer_workspace.v_raw;
                let mut rq = QRGS_ROWS as i32;
                let mut rkv = kv_rows;
                let mut bi = b_i32;
                let mut args: [*mut c_void; 14] = [
                    (&mut pwq as *mut _) as *mut c_void,
                    (&mut pwqs as *mut _) as *mut c_void,
                    (&mut pwk as *mut _) as *mut c_void,
                    (&mut pwks as *mut _) as *mut c_void,
                    (&mut pwv as *mut _) as *mut c_void,
                    (&mut pwvs as *mut _) as *mut c_void,
                    (&mut px as *mut _) as *mut c_void,
                    (&mut pxs as *mut _) as *mut c_void,
                    (&mut pyq as *mut _) as *mut c_void,
                    (&mut pyk as *mut _) as *mut c_void,
                    (&mut pyv as *mut _) as *mut c_void,
                    (&mut rq as *mut _) as *mut c_void,
                    (&mut rkv as *mut _) as *mut c_void,
                    (&mut bi as *mut _) as *mut c_void,
                ];
                let nb_q = (QRGS_ROWS as u32).div_ceil(128);
                let nb_kv = (kv_rows as u32).div_ceil(128);
                self.launch_smem(
                    stream,
                    f_qkv3_tc,
                    (nb_q + 2 * nb_kv, b_u32.div_ceil(128)),
                    512,
                    64_512,
                    &mut args,
                )?;
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
                    wk_ref,
                    wk_off,
                    kv_rows,
                    1,
                    b_len,
                    self.prefill.layer_workspace.k_raw,
                    kv_rows,
                    0.0,
                )?;
                self.enqueue_q8_gemm_slice_at(
                    stream,
                    wv_ref,
                    wv_off,
                    kv_rows,
                    2,
                    b_len,
                    self.prefill.layer_workspace.v_raw,
                    kv_rows,
                    0.0,
                )?;
            }
            k_raw_ptr = self.prefill.layer_workspace.k_raw;
            k_in_stride = kv_rows;
            v_raw_ptr = self.prefill.layer_workspace.v_raw;
            v_in_stride = kv_rows;
        }

        // 4. Register-Persistent / Token-Parallel Causal `short_conv4` for K & V
        for (raw_ptr, raw_stride, conv_w, hist_ptr, dst_ptr, seq_off_floats) in [
            (
                k_raw_ptr,
                k_in_stride,
                l.k_conv4.ptr,
                st.hist_k_at(self.cur_seq),
                self.prefill.layer_workspace.k_conv,
                0usize,
            ),
            (
                v_raw_ptr,
                v_in_stride,
                l.v_conv4.ptr,
                st.hist_v_at(self.cur_seq),
                self.prefill.layer_workspace.v_conv,
                1024 * CONV_SEQ_STEPS,
            ),
        ] {
            let dyn_ptr = if l.global { raw_ptr + 1024 * 4 } else { 0 };
            if self.ms_rows.get() != 0 {
                let hist0 = hist_ptr - self.cur_seq as u64 * st.seq_hist_bytes;
                self.enqueue_short_conv4_ms(
                    stream,
                    raw_ptr,
                    raw_stride,
                    conv_w,
                    dyn_ptr,
                    264,
                    hist0,
                    st.seq_hist_bytes,
                    dst_ptr,
                    1024,
                    0,
                    0,
                    0,
                    1024,
                    b_len,
                    i32::from(l.global),
                    0,
                    slot_idx,
                    seq_off_floats / CONV_SEQ_STEPS,
                )?;
                continue;
            }
            if !record_seq
                && self.enqueue_short_conv4_par_b(
                    stream,
                    raw_ptr,
                    raw_stride,
                    conv_w,
                    dyn_ptr,
                    264,
                    hist_ptr,
                    dst_ptr,
                    1024,
                    0,
                    0,
                    0,
                    1024,
                    b_len,
                    i32::from(l.global),
                    0,
                )?
            {
                continue;
            }
            let mut p_cur = raw_ptr;
            let mut ins = raw_stride;
            let mut p_w = conv_w;
            let mut p_dyn = dyn_ptr;
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
            self.launch(stream, conv, kv_conv_grid, 256, &mut args)?;
        }

        // 5. Batched Q/R/K Clip + V RMSNorm + KV Cache Write (Ring-Hazard-Free via `1088 >= 1024 + 64`!)
        {
            let mut pqrgs = self.prefill.layer_workspace.qrgs;
            let mut qstride = SWA_ALL_ROWS as i32;
            let mut pk = self.prefill.layer_workspace.k_conv;
            let mut pv = self.prefill.layer_workspace.v_conv;
            let mut pqc = l.q_clip.ptr;
            let mut prc = l.rel_clip.ptr;
            let mut pkc = st.k_at(self.cur_seq);
            let mut pvc = st.v_at(self.cur_seq);
            let (mut pks, mut pvs) = st.kv_scale_ptrs_at(self.cur_seq);
            let mut ppos = self.d_pos.ptr;
            let mut cap = st.kv_cap as i32;
            let mut ig = i32::from(l.global);
            if self.ms_rows.get() != 0 {
                // Multi-sequence round: sequence 0's buffers + per-row `{pos, seq}`.
                pkc = st.k_at(0);
                pvc = st.v_at(0);
                (pks, pvs) = st.kv_scale_ptrs_at(0);
                ppos = self.ms_rows.get();
                let mut skb = st.seq_kv_bytes as i64;
                let mut ssf = (st.seq_scale_bytes / 4) as i32;
                let mut args: [*mut c_void; 15] = [
                    (&mut pqrgs as *mut _) as *mut c_void,
                    (&mut qstride as *mut _) as *mut c_void,
                    (&mut pk as *mut _) as *mut c_void,
                    (&mut pv as *mut _) as *mut c_void,
                    (&mut pqc as *mut _) as *mut c_void,
                    (&mut prc as *mut _) as *mut c_void,
                    (&mut pkc as *mut _) as *mut c_void,
                    (&mut pvc as *mut _) as *mut c_void,
                    (&mut pks as *mut _) as *mut c_void,
                    (&mut pvs as *mut _) as *mut c_void,
                    (&mut ppos as *mut _) as *mut c_void,
                    (&mut cap as *mut _) as *mut c_void,
                    (&mut ig as *mut _) as *mut c_void,
                    (&mut skb as *mut _) as *mut c_void,
                    (&mut ssf as *mut _) as *mut c_void,
                ];
                let f = self
                    .fns
                    .prep_qkv_and_cache_ms
                    .ok_or("cubin lacks ink2_prep_qkv_and_cache_ms")?;
                self.launch(stream, f, (5, b_u32), 256, &mut args)?;
            } else {
                let mut args: [*mut c_void; 13] = [
                    (&mut pqrgs as *mut _) as *mut c_void,
                    (&mut qstride as *mut _) as *mut c_void,
                    (&mut pk as *mut _) as *mut c_void,
                    (&mut pv as *mut _) as *mut c_void,
                    (&mut pqc as *mut _) as *mut c_void,
                    (&mut prc as *mut _) as *mut c_void,
                    (&mut pkc as *mut _) as *mut c_void,
                    (&mut pvc as *mut _) as *mut c_void,
                    (&mut pks as *mut _) as *mut c_void,
                    (&mut pvs as *mut _) as *mut c_void,
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
        }

        // 6. Batched GQA Relative-Bias + Sink Attention + In-Register `Q8_64` Epilogue!
        let mut attn_done = false;
        if !record_seq && self.tc_attn && self.tc_rows(b_len) {
            let total_kv = self.host_pos + b_len;
            let use_long_split = l.global
                && total_kv >= 8192
                && self.fns.flash_attn_split_combine_q8_b.is_some()
                && if st.k_scale.is_some() {
                    self.fns.flash_attn_tc_split_b_i8.is_some()
                } else {
                    self.fns.flash_attn_tc_split_b.is_some()
                };
            if use_long_split {
                // `n_splits = 6` (`1,536` CTAs = `99.7%` wave eff on 110-SM PRO 5000) or `5` (`1,280` CTAs = `100%` on 80-SM PRO 4500),
                // reusing dead `k_raw .. pair_output_end` (`<= 9,504 * b_len <= 9,832 * capacity` dwords, `0` extra VRAM)!
                let long_splits: u32 = std::env::var("LEANSONIC_LONG_SPLITS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(if self.ctx.device_name.contains("4500") {
                        5
                    } else {
                        6
                    });
                let mut part_o_ptr = self.prefill.layer_workspace.k_raw;
                let mut part_ml_ptr =
                    part_o_ptr + (long_splits as usize * b_len * N_HEAD * 64 * 4) as u64;
                let mut pqrgs = self.prefill.layer_workspace.qrgs;
                let mut qstride = SWA_ALL_ROWS as i32;
                let mut prp = l.rel_proj.ptr;
                let mut pkc = st.k_at(self.cur_seq);
                let mut pvc = st.v_at(self.cur_seq);
                let mut ppos = self.d_pos.ptr;
                let mut cap = st.kv_cap as i32;
                let mut ig = i32::from(l.global);
                let mut bi = b_i32;
                if st.k_scale.is_some() {
                    let f_sp_i8 = self.fns.flash_attn_tc_split_b_i8.unwrap();
                    let (mut pks, mut pvs) = st.kv_scale_ptrs_at(self.cur_seq);
                    let mut s_args: [*mut c_void; 13] = [
                        (&mut pqrgs as *mut _) as *mut c_void,
                        (&mut qstride as *mut _) as *mut c_void,
                        (&mut prp as *mut _) as *mut c_void,
                        (&mut pkc as *mut _) as *mut c_void,
                        (&mut pvc as *mut _) as *mut c_void,
                        (&mut pks as *mut _) as *mut c_void,
                        (&mut pvs as *mut _) as *mut c_void,
                        (&mut ppos as *mut _) as *mut c_void,
                        (&mut cap as *mut _) as *mut c_void,
                        (&mut ig as *mut _) as *mut c_void,
                        (&mut bi as *mut _) as *mut c_void,
                        (&mut part_o_ptr as *mut _) as *mut c_void,
                        (&mut part_ml_ptr as *mut _) as *mut c_void,
                    ];
                    self.launch_smem_3d(
                        stream,
                        f_sp_i8,
                        (8, b_u32.div_ceil(64), long_splits),
                        384,
                        63_872,
                        &mut s_args,
                    )?;
                } else {
                    let f_sp = self.fns.flash_attn_tc_split_b.unwrap();
                    let mut s_args: [*mut c_void; 11] = [
                        (&mut pqrgs as *mut _) as *mut c_void,
                        (&mut qstride as *mut _) as *mut c_void,
                        (&mut prp as *mut _) as *mut c_void,
                        (&mut pkc as *mut _) as *mut c_void,
                        (&mut pvc as *mut _) as *mut c_void,
                        (&mut ppos as *mut _) as *mut c_void,
                        (&mut cap as *mut _) as *mut c_void,
                        (&mut ig as *mut _) as *mut c_void,
                        (&mut bi as *mut _) as *mut c_void,
                        (&mut part_o_ptr as *mut _) as *mut c_void,
                        (&mut part_ml_ptr as *mut _) as *mut c_void,
                    ];
                    self.launch_smem_3d(
                        stream,
                        f_sp,
                        (8, b_u32.div_ceil(64), long_splits),
                        384,
                        64_384,
                        &mut s_args,
                    )?;
                }
                let f_comb = self.fns.flash_attn_split_combine_q8_b.unwrap();
                let mut n_sp = long_splits as i32;
                let mut pq8 = self.prefill.d_b_act_codes.ptr;
                let mut psc = self.prefill.d_b_act_scales.ptr;
                let mut c_args: [*mut c_void; 8] = [
                    (&mut part_o_ptr as *mut _) as *mut c_void,
                    (&mut part_ml_ptr as *mut _) as *mut c_void,
                    (&mut pqrgs as *mut _) as *mut c_void,
                    (&mut qstride as *mut _) as *mut c_void,
                    (&mut bi as *mut _) as *mut c_void,
                    (&mut n_sp as *mut _) as *mut c_void,
                    (&mut pq8 as *mut _) as *mut c_void,
                    (&mut psc as *mut _) as *mut c_void,
                ];
                self.launch(stream, f_comb, (N_HEAD as u32, b_u32), 64, &mut c_args)?;
                attn_done = true;
            } else if st.k_scale.is_some() {
                if let Some(f_flash_i8) = self.fns.flash_attn_tc_q8_b_i8 {
                    let mut pqrgs = self.prefill.layer_workspace.qrgs;
                    let mut qstride = SWA_ALL_ROWS as i32;
                    let mut prp = l.rel_proj.ptr;
                    let mut pkc = st.k_at(self.cur_seq);
                    let mut pvc = st.v_at(self.cur_seq);
                    let (mut pks, mut pvs) = st.kv_scale_ptrs_at(self.cur_seq);
                    let mut ppos = self.d_pos.ptr;
                    let mut cap = st.kv_cap as i32;
                    let mut ig = i32::from(l.global);
                    let mut bi = b_i32;
                    let mut pq8 = self.prefill.d_b_act_codes.ptr;
                    let mut psc = self.prefill.d_b_act_scales.ptr;
                    let mut args: [*mut c_void; 13] = [
                        (&mut pqrgs as *mut _) as *mut c_void,
                        (&mut qstride as *mut _) as *mut c_void,
                        (&mut prp as *mut _) as *mut c_void,
                        (&mut pkc as *mut _) as *mut c_void,
                        (&mut pvc as *mut _) as *mut c_void,
                        (&mut pks as *mut _) as *mut c_void,
                        (&mut pvs as *mut _) as *mut c_void,
                        (&mut ppos as *mut _) as *mut c_void,
                        (&mut cap as *mut _) as *mut c_void,
                        (&mut ig as *mut _) as *mut c_void,
                        (&mut bi as *mut _) as *mut c_void,
                        (&mut pq8 as *mut _) as *mut c_void,
                        (&mut psc as *mut _) as *mut c_void,
                    ];
                    self.launch_smem(
                        stream,
                        f_flash_i8,
                        (8, b_u32.div_ceil(64)),
                        384,
                        63_872,
                        &mut args,
                    )?;
                    attn_done = true;
                }
            } else if let Some(f_flash_tc) = self.fns.flash_attn_tc_q8_b {
                let mut pqrgs = self.prefill.layer_workspace.qrgs;
                let mut qstride = SWA_ALL_ROWS as i32;
                let mut prp = l.rel_proj.ptr;
                let mut pkc = st.k_at(self.cur_seq);
                let mut pvc = st.v_at(self.cur_seq);
                let mut ppos = self.d_pos.ptr;
                let mut cap = st.kv_cap as i32;
                let mut ig = i32::from(l.global);
                let mut bi = b_i32;
                let mut pq8 = self.prefill.d_b_act_codes.ptr;
                let mut psc = self.prefill.d_b_act_scales.ptr;
                let mut args: [*mut c_void; 11] = [
                    (&mut pqrgs as *mut _) as *mut c_void,
                    (&mut qstride as *mut _) as *mut c_void,
                    (&mut prp as *mut _) as *mut c_void,
                    (&mut pkc as *mut _) as *mut c_void,
                    (&mut pvc as *mut _) as *mut c_void,
                    (&mut ppos as *mut _) as *mut c_void,
                    (&mut cap as *mut _) as *mut c_void,
                    (&mut ig as *mut _) as *mut c_void,
                    (&mut bi as *mut _) as *mut c_void,
                    (&mut pq8 as *mut _) as *mut c_void,
                    (&mut psc as *mut _) as *mut c_void,
                ];
                self.launch_smem(
                    stream,
                    f_flash_tc,
                    (8, b_u32.div_ceil(64)),
                    384,
                    64_384,
                    &mut args,
                )?;
                attn_done = true;
            }
        }
        let g_batch = if self.ms_rows.get() == 0 && b_len <= MAX_SPEC_TOKENS {
            if l.global {
                self.attn_split_g_decode_global
            } else {
                self.attn_split_g_decode
            }
        } else {
            self.attn_split_g
        };
        if !attn_done && g_batch == 0 {
            // Small speculative verification batches also underfill the GPU.
            let kernel = self.attn_q8_fn(st, record_seq && b_len <= MAX_SPEC_TOKENS);
            if let Some(f_attn_q8) = kernel {
                let mut pqrgs = self.prefill.layer_workspace.qrgs;
                let mut qstride = SWA_ALL_ROWS as i32;
                let mut prp = l.rel_proj.ptr;
                let mut pkc = st.k_at(self.cur_seq);
                let mut pvc = st.v_at(self.cur_seq);
                let (mut pks, mut pvs) = st.kv_scale_ptrs_at(self.cur_seq);
                let mut ppos = self.d_pos.ptr;
                let mut cap = st.kv_cap as i32;
                let mut ig = i32::from(l.global);
                let mut pq8 = self.prefill.d_b_act_codes.ptr;
                let mut psc = self.prefill.d_b_act_scales.ptr;
                let mut args: [*mut c_void; 12] = [
                    (&mut pqrgs as *mut _) as *mut c_void,
                    (&mut qstride as *mut _) as *mut c_void,
                    (&mut prp as *mut _) as *mut c_void,
                    (&mut pkc as *mut _) as *mut c_void,
                    (&mut pvc as *mut _) as *mut c_void,
                    (&mut pks as *mut _) as *mut c_void,
                    (&mut pvs as *mut _) as *mut c_void,
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
                let mut pkc = st.k_at(self.cur_seq);
                let mut pvc = st.v_at(self.cur_seq);
                let (mut pks, mut pvs) = st.kv_scale_ptrs_at(self.cur_seq);
                let mut ppos = self.d_pos.ptr;
                let mut cap = st.kv_cap as i32;
                let mut ig = i32::from(l.global);
                let mut pout = self.prefill.layer_workspace.attn_out;
                let mut args: [*mut c_void; 11] = [
                    (&mut pqrgs as *mut _) as *mut c_void,
                    (&mut qstride as *mut _) as *mut c_void,
                    (&mut prp as *mut _) as *mut c_void,
                    (&mut pkc as *mut _) as *mut c_void,
                    (&mut pvc as *mut _) as *mut c_void,
                    (&mut pks as *mut _) as *mut c_void,
                    (&mut pvs as *mut _) as *mut c_void,
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
        } else if !attn_done {
            // A multi-sequence round passes sequence 0's caches (+ per-row sequence offsets).
            let kv_seq = if self.ms_rows.get() != 0 {
                0
            } else {
                self.cur_seq
            };
            self.enqueue_gqa_attn_split_impl(
                stream,
                self.prefill.layer_workspace.qrgs,
                SWA_ALL_ROWS as i32,
                l.rel_proj.ptr,
                st.k_at(kv_seq),
                st.v_at(kv_seq),
                st.kv_scale_ptrs_at(kv_seq),
                st.kv_cap as i32,
                i32::from(l.global),
                self.prefill.layer_workspace.attn_out,
                self.prefill.d_b_act_codes.ptr,
                self.prefill.d_b_act_scales.ptr,
                b_len,
                (self.ms_rows.get() != 0).then_some((st.seq_kv_bytes, st.seq_scale_bytes)),
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
        let attn_res_ptr = if is_block_start {
            0
        } else {
            self.prefill.d_b_cur.ptr
        };
        let attn_conv_done = if self.ms_rows.get() != 0 {
            self.enqueue_short_conv4_ms(
                stream,
                self.prefill.layer_workspace.wo_out,
                D as i32,
                l.attn_conv4.ptr,
                0,
                0,
                st.hist_attn_at(0),
                st.seq_hist_bytes,
                self.prefill.d_b_cur.ptr,
                D as i32,
                attn_res_ptr,
                D as i32,
                save_blk_ptr,
                D as i32,
                b_len,
                0,
                i32::from(!is_block_start),
                slot_idx,
                2048,
            )?;
            true
        } else {
            false
        };
        let attn_conv_done = attn_conv_done
            || !record_seq
                && self.enqueue_short_conv4_par_b(
                    stream,
                    self.prefill.layer_workspace.wo_out,
                    D as i32,
                    l.attn_conv4.ptr,
                    0,
                    0,
                    st.hist_attn_at(self.cur_seq),
                    self.prefill.d_b_cur.ptr,
                    D as i32,
                    attn_res_ptr,
                    D as i32,
                    save_blk_ptr,
                    D as i32,
                    b_len,
                    0,
                    i32::from(!is_block_start),
                )?;
        if !attn_conv_done {
            if let Some(f_sc_save_b) = conv_save {
                let mut p_cur = self.prefill.layer_workspace.wo_out;
                let mut ins = D as i32;
                let mut p_w = l.attn_conv4.ptr;
                let mut p_dyn: CUdeviceptr = 0;
                let mut df4 = 0i32;
                let mut p_hist = st.hist_attn_at(self.cur_seq);
                let mut p_out = self.prefill.d_b_cur.ptr;
                let mut outs = D as i32;
                let mut p_res = attn_res_ptr;
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
                self.launch(stream, f_sc_save_b, hidden_conv_grid, 256, &mut args)?;
            } else {
                if save_blk_ptr != 0 {
                    self.enqueue_copy_n(stream, self.prefill.d_b_cur.ptr, save_blk_ptr, bd_floats)?;
                }
                let mut p_cur = self.prefill.layer_workspace.wo_out;
                let mut ins = D as i32;
                let mut p_w = l.attn_conv4.ptr;
                let mut p_dyn: CUdeviceptr = 0;
                let mut df4 = 0i32;
                let mut p_hist = st.hist_attn_at(self.cur_seq);
                let mut p_out = self.prefill.d_b_cur.ptr;
                let mut outs = D as i32;
                let mut p_res = attn_res_ptr;
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
                self.launch(stream, conv, hidden_conv_grid, 256, &mut args)?;
            }
        }

        // 8. Fused Batched MLP Prologue (`depth_mix + rmsnorm_d + quant_q8_16` when `!has_msgs`)
        let has_nvfp4_tc = self.tc_nvfp4
            && self.tc_rows(b_len)
            && self.fns.quant_nvfp4_2term_b.is_some()
            && self.fns.gather_nvfp4_sorted_b.is_some()
            && match &l.expert_scales {
                ExpertScales::Nvfp4 { s13, s2, .. } => {
                    s13.kernels.nvfp4_tc_up_situ.is_some() && s2.kernels.nvfp4_tc_down.is_some()
                }
                ExpertScales::Mxfp8 { .. } => false,
            };
        let [w13, w2] = &self.expert_weights[il];
        const EXP_W13_BYTES: usize = 2 * FF * (D / 2);
        const EXP_W2_BYTES: usize = D * (FF / 2);
        let pair_rows = b_len * TOP_K;
        let pair_end = self.prefill.layer_workspace.pair_output.ptr
            + self.prefill.layer_workspace.pair_output.byte_len as u64;
        let stage_reserved =
            (pair_rows * (D + D / 8 + 4)).max(self.prefill.capacity * ((D + N_EXP) * 4 + D * 2));
        let stage_ptr =
            (self.prefill.layer_workspace.pair_output.ptr + stage_reserved as u64 + 4095) & !4095;
        let avail_stage = pair_end.saturating_sub(stage_ptr) as usize;
        let half_stage_bytes = (avail_stage / 2) & !4095;
        let pingpong_exps = (half_stage_bytes / EXP_W13_BYTES).min(N_EXP);
        let use_async_w13_pingpong = has_nvfp4_tc
            && !record_seq
            && stream == self.ctx.compute_stream
            && pingpong_exps >= 3
            && matches!(l.expert_scales, ExpertScales::Nvfp4 { .. })
            && matches!(w13, TierBuf::HostPinned(_));
        let pf_opt = if use_async_w13_pingpong && !spilled_schedule.is_empty() {
            self.d_w13_prefetch.as_ref()
        } else {
            None
        };
        let pf_n = pf_opt.map(|pf| pf.pf_exps.min(N_EXP)).unwrap_or(0);
        let next_spilled_il = if pf_opt.is_some() {
            let idx = *spilled_cursor;
            *spilled_cursor += 1;
            spilled_schedule.get(idx + 1).copied()
        } else {
            None
        };
        if use_async_w13_pingpong && pf_n < N_EXP {
            // Attention finished reading `wo_out` in `pair_output` tail: start Copy-Engine HtoD prefetch
            // of Batch 0 & Batch 1 (`[pf_n .. )`) on `h2d_stream` NOW so MLP Prologue + Router + Quant + Gather overlap PCIe!
            if let TierBuf::HostPinned(hp) = w13 {
                unsafe {
                    check_cu(
                        (self.ctx.api.cuEventRecord)(self.stage_events.comp_done[0], stream),
                        "cuEventRecord(attn_tail_done)",
                    )?;
                    check_cu(
                        (self.ctx.api.cuStreamWaitEvent)(
                            self.ctx.h2d_stream,
                            self.stage_events.comp_done[0],
                            0,
                        ),
                        "cuStreamWaitEvent(h2d_wait_attn_tail)",
                    )?;
                    let b0_end = (pf_n + pingpong_exps).min(N_EXP);
                    check_cu(
                        (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                            stage_ptr,
                            hp.host_ptr.add(pf_n * EXP_W13_BYTES).cast(),
                            (b0_end - pf_n) * EXP_W13_BYTES,
                            self.ctx.h2d_stream,
                        ),
                        "HtoD(w13_pingpong_b0)",
                    )?;
                    check_cu(
                        (self.ctx.api.cuEventRecord)(
                            self.stage_events.h2d_ready[0],
                            self.ctx.h2d_stream,
                        ),
                        "cuEventRecord(w13_b0_ready)",
                    )?;
                    if b0_end < N_EXP {
                        let b1_end = (b0_end + pingpong_exps).min(N_EXP);
                        check_cu(
                            (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                                stage_ptr + half_stage_bytes as u64,
                                hp.host_ptr.add(b0_end * EXP_W13_BYTES).cast(),
                                (b1_end - b0_end) * EXP_W13_BYTES,
                                self.ctx.h2d_stream,
                            ),
                            "HtoD(w13_pingpong_b1)",
                        )?;
                        check_cu(
                            (self.ctx.api.cuEventRecord)(
                                self.stage_events.h2d_ready[1],
                                self.ctx.h2d_stream,
                            ),
                            "cuEventRecord(w13_b1_ready)",
                        )?;
                    }
                }
            }
        }

        let wants_mlp_q8_16 =
            matches!(l.expert_scales, ExpertScales::Nvfp4 { .. }) && !has_nvfp4_tc;
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
            let mut mlp_norm_bf16_ready = false;
            if let (true, Some(f_fused)) = (
                has_msgs
                    && !record_seq
                    && self.tc_bf16
                    && self.tc_rows(b_len)
                    && self.fns.bf16_tc_gemm_b.is_some(),
                self.fns.depth_mix_multi_ctrl_norm_bf16_b,
            ) {
                let mut nqi = 1i32;
                let mut pmsg = msg_base_ptr + 3 * bd_bytes;
                let mut pbf16 = self.prefill.layer_workspace.pair_output.ptr
                    + (self.prefill.capacity * (D + N_EXP) * 4) as u64;
                let mut f_args: [*mut c_void; 13] = [
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
                    (&mut nqi as *mut _) as *mut c_void,
                    (&mut pmsg as *mut _) as *mut c_void,
                    (&mut pbf16 as *mut _) as *mut c_void,
                ];
                self.launch(stream, f_fused, (1, b_u32), 256, &mut f_args)?;
                mlp_norm_bf16_ready = true;
            } else if let (false, Some(f_dm_m)) = (record_seq, self.fns.depth_mix_multi_b) {
                let mut nqi = 1i32;
                let mut m_args: [*mut c_void; 11] = [
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
                    (&mut nqi as *mut _) as *mut c_void,
                ];
                self.launch(stream, f_dm_m, (1, b_u32), 256, &mut m_args)?;
            } else {
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
            }
            if has_msgs {
                self.enqueue_ctrl_n_b_inner(
                    stream,
                    self.prefill.layer_workspace.mlp_hidden,
                    msg_base_ptr + 3 * bd_bytes,
                    l.ctrl_mlp.as_ref().unwrap(),
                    1,
                    b_len,
                    mlp_norm_bf16_ready,
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
        if let (true, Some(f_pack), Some(f_tc)) = (
            self.tc_bf16 && self.tc_rows(b_len),
            self.fns.f32_to_bf16_b,
            self.fns.bf16_tc_gemm_b,
        ) {
            let mut px = self.prefill.layer_workspace.mlp_hidden;
            // Disjoint from `mlp_hidden` and `router_logits` inside `pair_output`!
            let mut pbf = self.prefill.layer_workspace.pair_output.ptr
                + (self.prefill.capacity * (D + N_EXP) * 4) as u64;
            let n_vec8 = ((b_len * D) / 8) as i32;
            let mut nv = n_vec8;
            let mut pack_args: [*mut c_void; 3] = [
                (&mut px as *mut _) as *mut c_void,
                (&mut pbf as *mut _) as *mut c_void,
                (&mut nv as *mut _) as *mut c_void,
            ];
            self.launch(
                stream,
                f_pack,
                ((n_vec8 as u32).div_ceil(256), 1),
                256,
                &mut pack_args,
            )?;

            let mut pw = l.router_w.ptr;
            let mut py = self.prefill.layer_workspace.router_logits;
            let mut rows = N_EXP as i32;
            let mut cols = D as i32;
            let mut bi = b_i32;
            let t_blocks = b_u32.div_ceil(64);
            let mut tb = t_blocks as i32;
            let mut gemm_args: [*mut c_void; 7] = [
                (&mut pw as *mut _) as *mut c_void,
                (&mut pbf as *mut _) as *mut c_void,
                (&mut py as *mut _) as *mut c_void,
                (&mut rows as *mut _) as *mut c_void,
                (&mut cols as *mut _) as *mut c_void,
                (&mut bi as *mut _) as *mut c_void,
                (&mut tb as *mut _) as *mut c_void,
            ];
            self.launch(stream, f_tc, (2, t_blocks), 256, &mut gemm_args)?;
        } else {
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
        let mut active_groups_ready = false;
        if let (true, Some(f_act)) = (
            self.verify_moe && b_len <= MAX_SPEC_TOKENS,
            self.fns.build_expert_groups_active_b,
        ) {
            let mut pids = self.prefill.d_b_sel_ids.ptr;
            let mut pcnt = self.d_b_exp_counts.ptr;
            let mut ptok = self.prefill.d_b_exp_toks.ptr;
            let mut ppair = self.prefill.d_b_exp_pairs.ptr;
            let mut pact_exps = self.d_b_active_exps.ptr;
            let mut bi = b_i32;
            let mut args: [*mut c_void; 6] = [
                (&mut pids as *mut _) as *mut c_void,
                (&mut pcnt as *mut _) as *mut c_void,
                (&mut ptok as *mut _) as *mut c_void,
                (&mut ppair as *mut _) as *mut c_void,
                (&mut pact_exps as *mut _) as *mut c_void,
                (&mut bi as *mut _) as *mut c_void,
            ];
            self.launch(stream, f_act, (1, 1), 128, &mut args)?;
            active_groups_ready = true;
        } else {
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
            let (kernel, grid_x, block_x) = if b_len > EXPERT_PAGE {
                if let Some(f_par) = self.fns.build_expert_groups_par_b {
                    (f_par, N_EXP as u32, 256)
                } else {
                    (
                        self.fns
                            .build_expert_groups_wide_b
                            .ok_or("wide expert grouping unavailable")?,
                        N_EXP as u32 / 4,
                        128,
                    )
                }
            } else {
                (self.fns.build_expert_groups_b, 1, 128)
            };
            self.launch(stream, kernel, (grid_x, 1), block_x, &mut args)?;
        }

        let mut moe_wrote_f16 = false;
        match &l.expert_scales {
            ExpertScales::Nvfp4 { s13, a13, s2, a2 } => {
                let (opt_up, opt_dn, nvfp4_grid_x, nvfp4_block, nvfp4_tile_toks, nvfp4_smem) =
                    if b_len > 2048
                        && s13.kernels.nvfp4_tc_up_situ_t160.is_some()
                        && s2.kernels.nvfp4_tc_down_t160.is_some()
                    {
                        (
                            s13.kernels.nvfp4_tc_up_situ_t160,
                            s2.kernels.nvfp4_tc_down_t160,
                            12u32,
                            384u32,
                            160usize,
                            57_616u32,
                        )
                    } else {
                        (
                            s13.kernels.nvfp4_tc_up_situ,
                            s2.kernels.nvfp4_tc_down,
                            9u32,
                            512u32,
                            128usize,
                            62_480u32,
                        )
                    };
                if let (true, Some(f_q2), Some(f_gather), Some(f_up), Some(f_dn)) = (
                    has_nvfp4_tc,
                    self.fns.quant_nvfp4_2term_b,
                    self.fns.gather_nvfp4_sorted_b,
                    opt_up,
                    opt_dn,
                ) {
                    // Quantize `mlp_hidden` (`[b_len, D]`) into 2-term `(codes, ue4m3_scales, gs)` inside `(d_b_act_codes, d_b_act_scales)`
                    let gs_up_ptr = self.prefill.d_b_act_scales.ptr + (b_len * (D / 8)) as u64;
                    {
                        let mut px = self.prefill.layer_workspace.mlp_hidden;
                        let mut pcodes = self.prefill.d_b_act_codes.ptr;
                        let mut psc = self.prefill.d_b_act_scales.ptr;
                        let mut pgs = gs_up_ptr;
                        let mut nr = b_i32;
                        let mut cols = D as i32;
                        let mut q_args: [*mut c_void; 6] = [
                            (&mut px as *mut _) as *mut c_void,
                            (&mut pcodes as *mut _) as *mut c_void,
                            (&mut psc as *mut _) as *mut c_void,
                            (&mut pgs as *mut _) as *mut c_void,
                            (&mut nr as *mut _) as *mut c_void,
                            (&mut cols as *mut _) as *mut c_void,
                        ];
                        if let Some(f_q2_1p) = self.fns.quant_nvfp4_2term_1p_b {
                            self.launch(stream, f_q2_1p, (b_u32, 1), 160, &mut q_args)?;
                        } else {
                            self.launch(stream, f_q2, (b_u32.div_ceil(4), 1), 128, &mut q_args)?;
                        }
                    }

                    // Pre-gather `[b_len, D]` 2-term NVFP4 activations ONCE into contiguous `sorted_base` order inside `pair_output`!
                    let gather_tiles = (pair_rows.div_ceil(128) + N_EXP) as u32;
                    let nvfp4_tiles = (pair_rows.div_ceil(nvfp4_tile_toks) + N_EXP) as u32;
                    let sorted_codes_ptr = self.prefill.layer_workspace.pair_output.ptr;
                    let sorted_sc_ptr = sorted_codes_ptr + (pair_rows * D) as u64;
                    let sorted_gs_ptr = sorted_sc_ptr + (pair_rows * (D / 8)) as u64;
                    {
                        let mut psrc_c = self.prefill.d_b_act_codes.ptr;
                        let mut psrc_s = self.prefill.d_b_act_scales.ptr;
                        let mut psrc_g = gs_up_ptr;
                        let mut pcnt = self.d_b_exp_counts.ptr;
                        let mut ptok = self.prefill.d_b_exp_toks.ptr;
                        let mut pdst_c = sorted_codes_ptr;
                        let mut pdst_s = sorted_sc_ptr;
                        let mut pdst_g = sorted_gs_ptr;
                        let mut g_args: [*mut c_void; 8] = [
                            (&mut psrc_c as *mut _) as *mut c_void,
                            (&mut psrc_s as *mut _) as *mut c_void,
                            (&mut psrc_g as *mut _) as *mut c_void,
                            (&mut pcnt as *mut _) as *mut c_void,
                            (&mut ptok as *mut _) as *mut c_void,
                            (&mut pdst_c as *mut _) as *mut c_void,
                            (&mut pdst_s as *mut _) as *mut c_void,
                            (&mut pdst_g as *mut _) as *mut c_void,
                        ];
                        self.launch(stream, f_gather, (1, gather_tiles), 256, &mut g_args)?;
                    }

                    let mut psorted_c = sorted_codes_ptr;
                    let mut psorted_s = sorted_sc_ptr;
                    let mut psorted_g = sorted_gs_ptr;
                    let mut pcnt = self.d_b_exp_counts.ptr;
                    let mut ptok = self.prefill.d_b_exp_toks.ptr;
                    let mut ppair = self.prefill.d_b_exp_pairs.ptr;
                    let mut pact = self.prefill.d_b_activations.ptr;
                    let mut ps13 = s13.buffer.ptr;
                    let mut pa13 = a13.ptr;
                    let stage_exps = (avail_stage / EXP_W13_BYTES).min(N_EXP);
                    if let (TierBuf::HostPinned(hp), true) = (w13, use_async_w13_pingpong) {
                        if let Some(pf) = pf_opt {
                            unsafe {
                                check_cu(
                                    (self.ctx.api.cuStreamWaitEvent)(stream, pf.ready_ev, 0),
                                    "cuStreamWaitEvent(comp_wait_w13_pf)",
                                )?;
                            }
                            let mut pw13 = pf.buf.ptr;
                            let mut eb = 0i32;
                            let mut ee = pf_n as i32;
                            let mut up_args: [*mut c_void; 12] = [
                                (&mut pw13 as *mut _) as *mut c_void,
                                (&mut ps13 as *mut _) as *mut c_void,
                                (&mut pa13 as *mut _) as *mut c_void,
                                (&mut psorted_c as *mut _) as *mut c_void,
                                (&mut psorted_s as *mut _) as *mut c_void,
                                (&mut psorted_g as *mut _) as *mut c_void,
                                (&mut pcnt as *mut _) as *mut c_void,
                                (&mut ptok as *mut _) as *mut c_void,
                                (&mut ppair as *mut _) as *mut c_void,
                                (&mut eb as *mut _) as *mut c_void,
                                (&mut ee as *mut _) as *mut c_void,
                                (&mut pact as *mut _) as *mut c_void,
                            ];
                            self.launch_smem(
                                stream,
                                f_up,
                                (nvfp4_grid_x, nvfp4_tiles),
                                nvfp4_block,
                                nvfp4_smem,
                                &mut up_args,
                            )?;
                            unsafe {
                                check_cu(
                                    (self.ctx.api.cuEventRecord)(pf.consumed_ev, stream),
                                    "cuEventRecord(w13_pf_consumed)",
                                )?;
                            }
                        }
                        // Double-buffered `h2d_stream` + `compute_stream` ping-pong overlap for `[pf_n .. N_EXP)`!
                        let mut batch_idx = 0usize;
                        let mut exp_begin = pf_n;
                        while exp_begin < N_EXP {
                            let slot = batch_idx & 1;
                            let exp_end = (exp_begin + pingpong_exps).min(N_EXP);
                            let cur_stage_ptr = stage_ptr + (slot * half_stage_bytes) as u64;
                            unsafe {
                                check_cu(
                                    (self.ctx.api.cuStreamWaitEvent)(
                                        stream,
                                        self.stage_events.h2d_ready[slot],
                                        0,
                                    ),
                                    "cuStreamWaitEvent(comp_wait_w13_h2d)",
                                )?;
                            }
                            let mut pw13 = cur_stage_ptr;
                            let mut eb = exp_begin as i32;
                            let mut ee = exp_end as i32;
                            let mut up_args: [*mut c_void; 12] = [
                                (&mut pw13 as *mut _) as *mut c_void,
                                (&mut ps13 as *mut _) as *mut c_void,
                                (&mut pa13 as *mut _) as *mut c_void,
                                (&mut psorted_c as *mut _) as *mut c_void,
                                (&mut psorted_s as *mut _) as *mut c_void,
                                (&mut psorted_g as *mut _) as *mut c_void,
                                (&mut pcnt as *mut _) as *mut c_void,
                                (&mut ptok as *mut _) as *mut c_void,
                                (&mut ppair as *mut _) as *mut c_void,
                                (&mut eb as *mut _) as *mut c_void,
                                (&mut ee as *mut _) as *mut c_void,
                                (&mut pact as *mut _) as *mut c_void,
                            ];
                            self.launch_smem(
                                stream,
                                f_up,
                                (nvfp4_grid_x, nvfp4_tiles),
                                nvfp4_block,
                                nvfp4_smem,
                                &mut up_args,
                            )?;
                            let next2_begin = exp_begin + 2 * pingpong_exps;
                            if next2_begin < N_EXP {
                                let next2_end = (next2_begin + pingpong_exps).min(N_EXP);
                                unsafe {
                                    check_cu(
                                        (self.ctx.api.cuEventRecord)(
                                            self.stage_events.comp_done[slot],
                                            stream,
                                        ),
                                        "cuEventRecord(w13_comp_done)",
                                    )?;
                                    check_cu(
                                        (self.ctx.api.cuStreamWaitEvent)(
                                            self.ctx.h2d_stream,
                                            self.stage_events.comp_done[slot],
                                            0,
                                        ),
                                        "cuStreamWaitEvent(h2d_wait_comp_done)",
                                    )?;
                                    check_cu(
                                        (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                                            cur_stage_ptr,
                                            hp.host_ptr.add(next2_begin * EXP_W13_BYTES).cast(),
                                            (next2_end - next2_begin) * EXP_W13_BYTES,
                                            self.ctx.h2d_stream,
                                        ),
                                        "HtoD(w13_pingpong_next)",
                                    )?;
                                    check_cu(
                                        (self.ctx.api.cuEventRecord)(
                                            self.stage_events.h2d_ready[slot],
                                            self.ctx.h2d_stream,
                                        ),
                                        "cuEventRecord(w13_next_ready)",
                                    )?;
                                }
                            }
                            exp_begin = exp_end;
                            batch_idx += 1;
                        }
                        if let (Some(pf), Some(next_il)) = (pf_opt, next_spilled_il) {
                            if let TierBuf::HostPinned(hp_next) = &self.expert_weights[next_il][0] {
                                unsafe {
                                    check_cu(
                                        (self.ctx.api.cuStreamWaitEvent)(
                                            self.ctx.h2d_stream,
                                            pf.consumed_ev,
                                            0,
                                        ),
                                        "cuStreamWaitEvent(h2d_wait_pf_consumed)",
                                    )?;
                                    check_cu(
                                        (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                                            pf.buf.ptr,
                                            hp_next.host_ptr.cast(),
                                            pf_n * EXP_W13_BYTES,
                                            self.ctx.h2d_stream,
                                        ),
                                        "HtoD(w13_prefetch_next)",
                                    )?;
                                    check_cu(
                                        (self.ctx.api.cuEventRecord)(
                                            pf.ready_ev,
                                            self.ctx.h2d_stream,
                                        ),
                                        "cuEventRecord(w13_pf_ready_next)",
                                    )?;
                                }
                            }
                        }
                    } else if let (TierBuf::HostPinned(hp), true) = (w13, stage_exps >= 4) {
                        let mut exp_begin = 0usize;
                        while exp_begin < N_EXP {
                            let exp_end = (exp_begin + stage_exps).min(N_EXP);
                            unsafe {
                                check_cu(
                                    (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                                        stage_ptr,
                                        hp.host_ptr.add(exp_begin * EXP_W13_BYTES).cast(),
                                        (exp_end - exp_begin) * EXP_W13_BYTES,
                                        stream,
                                    ),
                                    "HtoD(spilled_w13_stage)",
                                )?;
                            }
                            let mut pw13 = stage_ptr;
                            let mut eb = exp_begin as i32;
                            let mut ee = exp_end as i32;
                            let mut up_args: [*mut c_void; 12] = [
                                (&mut pw13 as *mut _) as *mut c_void,
                                (&mut ps13 as *mut _) as *mut c_void,
                                (&mut pa13 as *mut _) as *mut c_void,
                                (&mut psorted_c as *mut _) as *mut c_void,
                                (&mut psorted_s as *mut _) as *mut c_void,
                                (&mut psorted_g as *mut _) as *mut c_void,
                                (&mut pcnt as *mut _) as *mut c_void,
                                (&mut ptok as *mut _) as *mut c_void,
                                (&mut ppair as *mut _) as *mut c_void,
                                (&mut eb as *mut _) as *mut c_void,
                                (&mut ee as *mut _) as *mut c_void,
                                (&mut pact as *mut _) as *mut c_void,
                            ];
                            self.launch_smem(
                                stream,
                                f_up,
                                (nvfp4_grid_x, nvfp4_tiles),
                                nvfp4_block,
                                nvfp4_smem,
                                &mut up_args,
                            )?;
                            exp_begin = exp_end;
                        }
                    } else {
                        let mut pw13 = w13.ptr();
                        let mut eb = 0i32;
                        let mut ee = N_EXP as i32;
                        let mut up_args: [*mut c_void; 12] = [
                            (&mut pw13 as *mut _) as *mut c_void,
                            (&mut ps13 as *mut _) as *mut c_void,
                            (&mut pa13 as *mut _) as *mut c_void,
                            (&mut psorted_c as *mut _) as *mut c_void,
                            (&mut psorted_s as *mut _) as *mut c_void,
                            (&mut psorted_g as *mut _) as *mut c_void,
                            (&mut pcnt as *mut _) as *mut c_void,
                            (&mut ptok as *mut _) as *mut c_void,
                            (&mut ppair as *mut _) as *mut c_void,
                            (&mut eb as *mut _) as *mut c_void,
                            (&mut ee as *mut _) as *mut c_void,
                            (&mut pact as *mut _) as *mut c_void,
                        ];
                        self.launch_smem(
                            stream,
                            f_up,
                            (nvfp4_grid_x, nvfp4_tiles),
                            nvfp4_block,
                            nvfp4_smem,
                            &mut up_args,
                        )?;
                    }

                    // Quantize `d_b_activations` (`[b_len * TOP_K, FF]`) into 2-term `(codes, ue4m3_scales, gs)`
                    let gs_dn_ptr = self.prefill.d_b_act_scales.ptr + (pair_rows * (FF / 8)) as u64;
                    let mut pxq = self.prefill.d_b_act_codes.ptr;
                    let mut pxs = self.prefill.d_b_act_scales.ptr;
                    {
                        let mut px = self.prefill.d_b_activations.ptr;
                        let mut pcodes = pxq;
                        let mut psc = pxs;
                        let mut pgs = gs_dn_ptr;
                        let mut nr = pair_rows as i32;
                        let mut cols = FF as i32;
                        let mut q_args: [*mut c_void; 6] = [
                            (&mut px as *mut _) as *mut c_void,
                            (&mut pcodes as *mut _) as *mut c_void,
                            (&mut psc as *mut _) as *mut c_void,
                            (&mut pgs as *mut _) as *mut c_void,
                            (&mut nr as *mut _) as *mut c_void,
                            (&mut cols as *mut _) as *mut c_void,
                        ];
                        if let Some(f_q2_1p) = self.fns.quant_nvfp4_2term_1p_b {
                            self.launch(stream, f_q2_1p, (pair_rows as u32, 1), 160, &mut q_args)?;
                        } else {
                            self.launch(
                                stream,
                                f_q2,
                                ((pair_rows as u32).div_ceil(4), 1),
                                128,
                                &mut q_args,
                            )?;
                        }
                    }
                    let mut pxgs_dn = gs_dn_ptr;
                    let mut ppy = self.prefill.layer_workspace.pair_output.ptr;
                    let mut ps2 = s2.buffer.ptr;
                    let mut pa2 = a2.ptr;
                    let w2_stage_ptr = (ppy + (pair_rows * D * 2) as u64 + 4095) & !4095;
                    let w2_stage_exps =
                        (pair_end.saturating_sub(w2_stage_ptr) as usize / EXP_W2_BYTES).min(N_EXP);
                    if let (TierBuf::HostPinned(hp2), true) = (w2, w2_stage_exps >= 4) {
                        let mut exp_begin = 0usize;
                        while exp_begin < N_EXP {
                            let exp_end = (exp_begin + w2_stage_exps).min(N_EXP);
                            unsafe {
                                check_cu(
                                    (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                                        w2_stage_ptr,
                                        hp2.host_ptr.add(exp_begin * EXP_W2_BYTES).cast(),
                                        (exp_end - exp_begin) * EXP_W2_BYTES,
                                        stream,
                                    ),
                                    "HtoD(spilled_w2_stage)",
                                )?;
                            }
                            let mut pw2 = w2_stage_ptr;
                            let mut eb = exp_begin as i32;
                            let mut ee = exp_end as i32;
                            let mut dn_args: [*mut c_void; 11] = [
                                (&mut pw2 as *mut _) as *mut c_void,
                                (&mut ps2 as *mut _) as *mut c_void,
                                (&mut pa2 as *mut _) as *mut c_void,
                                (&mut pxq as *mut _) as *mut c_void,
                                (&mut pxs as *mut _) as *mut c_void,
                                (&mut pxgs_dn as *mut _) as *mut c_void,
                                (&mut pcnt as *mut _) as *mut c_void,
                                (&mut ppair as *mut _) as *mut c_void,
                                (&mut eb as *mut _) as *mut c_void,
                                (&mut ee as *mut _) as *mut c_void,
                                (&mut ppy as *mut _) as *mut c_void,
                            ];
                            self.launch_smem(
                                stream,
                                f_dn,
                                (nvfp4_grid_x, nvfp4_tiles),
                                nvfp4_block,
                                nvfp4_smem,
                                &mut dn_args,
                            )?;
                            exp_begin = exp_end;
                        }
                    } else {
                        let mut pw2 = w2.ptr();
                        let mut eb = 0i32;
                        let mut ee = N_EXP as i32;
                        let mut dn_args: [*mut c_void; 11] = [
                            (&mut pw2 as *mut _) as *mut c_void,
                            (&mut ps2 as *mut _) as *mut c_void,
                            (&mut pa2 as *mut _) as *mut c_void,
                            (&mut pxq as *mut _) as *mut c_void,
                            (&mut pxs as *mut _) as *mut c_void,
                            (&mut pxgs_dn as *mut _) as *mut c_void,
                            (&mut pcnt as *mut _) as *mut c_void,
                            (&mut ppair as *mut _) as *mut c_void,
                            (&mut eb as *mut _) as *mut c_void,
                            (&mut ee as *mut _) as *mut c_void,
                            (&mut ppy as *mut _) as *mut c_void,
                        ];
                        self.launch_smem(
                            stream,
                            f_dn,
                            (nvfp4_grid_x, nvfp4_tiles),
                            nvfp4_block,
                            nvfp4_smem,
                            &mut dn_args,
                        )?;
                    }
                    moe_wrote_f16 = true;
                } else if let (true, Some(f_vup), Some(f_vdn)) = (
                    active_groups_ready && b_len <= MAX_SPEC_TOKENS,
                    s13.kernels.nvfp4_dp4a_verify_up_half_q8,
                    s2.kernels.nvfp4_dp4a_verify_down,
                ) {
                    if !mlp_q8_16_ready {
                        self.enqueue_quant_q8_16_b(
                            stream,
                            self.prefill.layer_workspace.mlp_hidden,
                            b_len,
                            D,
                        )?;
                    }
                    let max_active = (b_len * TOP_K).min(N_EXP) as u32;
                    let mut pw13 = w13.ptr();
                    let mut ps13 = s13.buffer.ptr;
                    let mut pa13 = a13.ptr;
                    let mut pxq = self.prefill.d_b_act_codes.ptr;
                    let mut pxs = self.prefill.d_b_act_scales.ptr;
                    let mut pact_exps = self.d_b_active_exps.ptr;
                    let mut pcnt = self.d_b_exp_counts.ptr;
                    let mut ptok = self.prefill.d_b_exp_toks.ptr;
                    let mut ppair = self.prefill.d_b_exp_pairs.ptr;
                    // Stage quantized intermediate `[B * 8, FF]` inside disjoint `d_b_activations` (`>= 2.25 MiB`)!
                    let mut pout_q8 = self.prefill.d_b_activations.ptr;
                    let mut pout_sc = pout_q8 + (MAX_SPEC_TOKENS * TOP_K * FF) as u64;
                    let mut up_args: [*mut c_void; 11] = [
                        (&mut pw13 as *mut _) as *mut c_void,
                        (&mut ps13 as *mut _) as *mut c_void,
                        (&mut pa13 as *mut _) as *mut c_void,
                        (&mut pxq as *mut _) as *mut c_void,
                        (&mut pxs as *mut _) as *mut c_void,
                        (&mut pact_exps as *mut _) as *mut c_void,
                        (&mut pcnt as *mut _) as *mut c_void,
                        (&mut ptok as *mut _) as *mut c_void,
                        (&mut ppair as *mut _) as *mut c_void,
                        (&mut pout_q8 as *mut _) as *mut c_void,
                        (&mut pout_sc as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, f_vup, (72, max_active), 512, &mut up_args)?;

                    let mut pw2 = w2.ptr();
                    let mut ps2 = s2.buffer.ptr;
                    let mut pa2 = a2.ptr;
                    let mut ppy = self.prefill.layer_workspace.pair_output.ptr;
                    let mut dn_args: [*mut c_void; 9] = [
                        (&mut pw2 as *mut _) as *mut c_void,
                        (&mut ps2 as *mut _) as *mut c_void,
                        (&mut pa2 as *mut _) as *mut c_void,
                        (&mut pout_q8 as *mut _) as *mut c_void,
                        (&mut pout_sc as *mut _) as *mut c_void,
                        (&mut pact_exps as *mut _) as *mut c_void,
                        (&mut pcnt as *mut _) as *mut c_void,
                        (&mut ppair as *mut _) as *mut c_void,
                        (&mut ppy as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, f_vdn, (288, max_active), 256, &mut dn_args)?;
                } else {
                    if !mlp_q8_16_ready {
                        self.enqueue_quant_q8_16_b(
                            stream,
                            self.prefill.layer_workspace.mlp_hidden,
                            b_len,
                            D,
                        )?;
                    }
                    let mut pw13 = w13.ptr();
                    let mut ps13 = s13.buffer.ptr;
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
                        s13.kernels.nvfp4_dp4a_grouped_up_situ,
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
                    let mut ps2 = s2.buffer.ptr;
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
                        s2.kernels.nvfp4_dp4a_grouped_down,
                        (72, 128),
                        256,
                        &mut dn_args,
                    )?;
                }
            }
            ExpertScales::Mxfp8 { s13, s2 } => {
                let (opt_mup, opt_mdn, mxfp8_block, mxfp8_tile_toks, mxfp8_smem) = if b_len >= 1792
                    && s13.kernels.mxfp8_tc_up_situ_t160.is_some()
                    && s2.kernels.mxfp8_tc_down_t160.is_some()
                {
                    (
                        s13.kernels.mxfp8_tc_up_situ_t160,
                        s2.kernels.mxfp8_tc_down_t160,
                        256u32,
                        160usize,
                        87_568u32,
                    )
                } else {
                    (
                        s13.kernels.mxfp8_tc_up_situ,
                        s2.kernels.mxfp8_tc_down,
                        512u32,
                        128usize,
                        77_840u32,
                    )
                };
                if let (true, Some(f_qm), Some(f_up), Some(f_dn)) = (
                    self.tc_mxfp8 && self.tc_rows(b_len),
                    self.fns.quant_mxfp8_b,
                    opt_mup,
                    opt_mdn,
                ) {
                    // Quantize `mlp_hidden` (`[b_len, D]`) -> `(d_b_act_codes, d_b_act_scales)` as block-32 MXFP8
                    {
                        let mut px = self.prefill.layer_workspace.mlp_hidden;
                        let mut pq8 = self.prefill.d_b_act_codes.ptr;
                        let mut psc = self.prefill.d_b_act_scales.ptr;
                        let mut cols = D as i32;
                        let mut q_args: [*mut c_void; 4] = [
                            (&mut px as *mut _) as *mut c_void,
                            (&mut pq8 as *mut _) as *mut c_void,
                            (&mut psc as *mut _) as *mut c_void,
                            (&mut cols as *mut _) as *mut c_void,
                        ];
                        self.launch(stream, f_qm, ((D / 128) as u32, b_u32), 32, &mut q_args)?;
                    }
                    let mut pw13 = self.mxfp8_weight_ptr(stream, w13, 2 * FF * D, false)?;
                    let mut ps13 = s13.buffer.ptr;
                    let mut pxq = self.prefill.d_b_act_codes.ptr;
                    let mut pxs = self.prefill.d_b_act_scales.ptr;
                    let mut pcnt = self.d_b_exp_counts.ptr;
                    let mut ptok = self.prefill.d_b_exp_toks.ptr;
                    let mut ppair = self.prefill.d_b_exp_pairs.ptr;
                    let mut pact = self.prefill.d_b_activations.ptr;
                    let mut up_args: [*mut c_void; 8] = [
                        (&mut pw13 as *mut _) as *mut c_void,
                        (&mut ps13 as *mut _) as *mut c_void,
                        (&mut pxq as *mut _) as *mut c_void,
                        (&mut pxs as *mut _) as *mut c_void,
                        (&mut pcnt as *mut _) as *mut c_void,
                        (&mut ptok as *mut _) as *mut c_void,
                        (&mut ppair as *mut _) as *mut c_void,
                        (&mut pact as *mut _) as *mut c_void,
                    ];
                    let mxfp8_tiles = ((b_len * TOP_K).div_ceil(mxfp8_tile_toks) + N_EXP) as u32;
                    self.launch_smem(
                        stream,
                        f_up,
                        (18, mxfp8_tiles),
                        mxfp8_block,
                        mxfp8_smem,
                        &mut up_args,
                    )?;

                    // Quantize `d_b_activations` (`[b_len * TOP_K, FF]`) -> `(d_b_act_codes, d_b_act_scales)` as block-32 MXFP8
                    {
                        let mut px = self.prefill.d_b_activations.ptr;
                        let mut pq8 = self.prefill.d_b_act_codes.ptr;
                        let mut psc = self.prefill.d_b_act_scales.ptr;
                        let mut cols = FF as i32;
                        let mut q_args: [*mut c_void; 4] = [
                            (&mut px as *mut _) as *mut c_void,
                            (&mut pq8 as *mut _) as *mut c_void,
                            (&mut psc as *mut _) as *mut c_void,
                            (&mut cols as *mut _) as *mut c_void,
                        ];
                        self.launch(
                            stream,
                            f_qm,
                            ((FF / 128) as u32, (b_len * TOP_K) as u32),
                            32,
                            &mut q_args,
                        )?;
                    }
                    let mut pw2 = self.mxfp8_weight_ptr(stream, w2, FF * D, false)?;
                    let mut ps2 = s2.buffer.ptr;
                    let mut ppy = self.prefill.layer_workspace.pair_output.ptr;
                    let mut dn_args: [*mut c_void; 7] = [
                        (&mut pw2 as *mut _) as *mut c_void,
                        (&mut ps2 as *mut _) as *mut c_void,
                        (&mut pxq as *mut _) as *mut c_void,
                        (&mut pxs as *mut _) as *mut c_void,
                        (&mut pcnt as *mut _) as *mut c_void,
                        (&mut ppair as *mut _) as *mut c_void,
                        (&mut ppy as *mut _) as *mut c_void,
                    ];
                    self.launch_smem(
                        stream,
                        f_dn,
                        (18, mxfp8_tiles),
                        mxfp8_block,
                        mxfp8_smem,
                        &mut dn_args,
                    )?;
                    moe_wrote_f16 = true;
                } else if let (true, Some(f_vup), Some(f_vdn)) = (
                    active_groups_ready && b_len <= MAX_SPEC_TOKENS,
                    s13.kernels.mxfp8_verify_up_situ,
                    s2.kernels.mxfp8_verify_down,
                ) {
                    let max_active = (b_len * TOP_K).min(N_EXP) as u32;
                    let mut pw13 = self.mxfp8_weight_ptr(stream, w13, 2 * FF * D, false)?;
                    let mut ps13 = s13.buffer.ptr;
                    let mut px = self.prefill.layer_workspace.mlp_hidden;
                    let mut pact_exps = self.d_b_active_exps.ptr;
                    let mut pcnt = self.d_b_exp_counts.ptr;
                    let mut ptok = self.prefill.d_b_exp_toks.ptr;
                    let mut ppair = self.prefill.d_b_exp_pairs.ptr;
                    let mut pact = self.prefill.d_b_activations.ptr;
                    let mut up_args: [*mut c_void; 8] = [
                        (&mut pw13 as *mut _) as *mut c_void,
                        (&mut ps13 as *mut _) as *mut c_void,
                        (&mut px as *mut _) as *mut c_void,
                        (&mut pact_exps as *mut _) as *mut c_void,
                        (&mut pcnt as *mut _) as *mut c_void,
                        (&mut ptok as *mut _) as *mut c_void,
                        (&mut ppair as *mut _) as *mut c_void,
                        (&mut pact as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, f_vup, (288, max_active), 256, &mut up_args)?;

                    let mut pw2 = self.mxfp8_weight_ptr(stream, w2, FF * D, false)?;
                    let mut ps2 = s2.buffer.ptr;
                    let mut ppy = self.prefill.layer_workspace.pair_output.ptr;
                    let mut dn_args: [*mut c_void; 7] = [
                        (&mut pw2 as *mut _) as *mut c_void,
                        (&mut ps2 as *mut _) as *mut c_void,
                        (&mut pact as *mut _) as *mut c_void,
                        (&mut pact_exps as *mut _) as *mut c_void,
                        (&mut pcnt as *mut _) as *mut c_void,
                        (&mut ppair as *mut _) as *mut c_void,
                        (&mut ppy as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, f_vdn, (288, max_active), 256, &mut dn_args)?;
                } else {
                    let mut pw13 = self.mxfp8_weight_ptr(stream, w13, 2 * FF * D, false)?;
                    let mut ps13 = s13.buffer.ptr;
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
                        s13.kernels.mxfp8_grouped_up_situ,
                        (72, 128),
                        256,
                        &mut up_args,
                    )?;

                    let mut pw2 = self.mxfp8_weight_ptr(stream, w2, FF * D, false)?;
                    let mut ps2 = s2.buffer.ptr;
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
                        s2.kernels.mxfp8_grouped_down,
                        (72, 128),
                        256,
                        &mut dn_args,
                    )?;
                }
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
            if moe_wrote_f16 {
                let f_f16 = self
                    .fns
                    .moe_combine_top8_f16_b
                    .ok_or("missing ink2_moe_combine_top8_f16_b")?;
                self.launch(stream, f_f16, (1, b_u32), 288, &mut args)?;
            } else if let Some(f_vec4) = self.fns.moe_combine_top8_vec4_b {
                self.launch(stream, f_vec4, (3, b_u32), 192, &mut args)?;
            } else {
                self.launch(
                    stream,
                    self.fns.moe_combine_top8_b,
                    (9, b_u32),
                    256,
                    &mut args,
                )?;
            }
        }

        // 10. `mlp_sconv` + Residual Add into `d_b_cur`
        if self.ms_rows.get() != 0 {
            self.enqueue_short_conv4_ms(
                stream,
                self.prefill.d_b_activations.ptr,
                D as i32,
                l.mlp_conv4.ptr,
                0,
                0,
                st.hist_mlp_at(0),
                st.seq_hist_bytes,
                self.prefill.d_b_cur.ptr,
                D as i32,
                self.prefill.d_b_cur.ptr,
                D as i32,
                0,
                D as i32,
                b_len,
                0,
                1,
                slot_idx,
                4352,
            )?;
        } else if record_seq
            || !self.enqueue_short_conv4_par_b(
                stream,
                self.prefill.d_b_activations.ptr,
                D as i32,
                l.mlp_conv4.ptr,
                0,
                0,
                st.hist_mlp_at(self.cur_seq),
                self.prefill.d_b_cur.ptr,
                D as i32,
                self.prefill.d_b_cur.ptr,
                D as i32,
                0,
                D as i32,
                b_len,
                0,
                1,
            )?
        {
            let mut p_cur = self.prefill.d_b_activations.ptr;
            let mut ins = D as i32;
            let mut p_w = l.mlp_conv4.ptr;
            let mut p_dyn: CUdeviceptr = 0;
            let mut df4 = 0i32;
            let mut p_hist = st.hist_mlp_at(self.cur_seq);
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
            self.launch(stream, conv, hidden_conv_grid, 256, &mut args)?;
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

        let mut spilled_schedule = Vec::new();
        if self.tc_rows(b_len)
            && !record_seq
            && stream == self.ctx.compute_stream
            && self.tc_nvfp4
            && self.fns.quant_nvfp4_2term_b.is_some()
            && self.fns.gather_nvfp4_sorted_b.is_some()
            && self.fns.experts.nvfp4_tc_up_situ.is_some()
            && self.fns.experts.nvfp4_tc_down.is_some()
            && self.d_w13_prefetch.is_some()
        {
            for il in (0..6).chain((0..3).flat_map(|_| 6..18)).chain(18..24) {
                if matches!(self.layers[il].expert_scales, ExpertScales::Nvfp4 { .. })
                    && matches!(self.expert_weights[il][0], TierBuf::HostPinned(_))
                {
                    spilled_schedule.push(il);
                }
            }
            if let (Some(&first_il), Some(pf)) = (spilled_schedule.first(), &self.d_w13_prefetch) {
                if let TierBuf::HostPinned(hp0) = &self.expert_weights[first_il][0] {
                    const EXP_W13_BYTES: usize = 2 * FF * (D / 2);
                    let pf_n = pf.pf_exps.min(N_EXP);
                    unsafe {
                        check_cu(
                            (self.ctx.api.cuStreamWaitEvent)(
                                self.ctx.h2d_stream,
                                pf.consumed_ev,
                                0,
                            ),
                            "cuStreamWaitEvent(h2d_wait_pf_consumed_init)",
                        )?;
                        check_cu(
                            (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                                pf.buf.ptr,
                                hp0.host_ptr.cast(),
                                pf_n * EXP_W13_BYTES,
                                self.ctx.h2d_stream,
                            ),
                            "HtoD(w13_prefetch_first)",
                        )?;
                        check_cu(
                            (self.ctx.api.cuEventRecord)(pf.ready_ev, self.ctx.h2d_stream),
                            "cuEventRecord(w13_pf_ready_first)",
                        )?;
                    }
                }
            }
        }
        let mut spilled_cursor = 0usize;

        let mut n_blocks = 0usize;
        for il in 0..6 {
            self.enqueue_batch_visit(
                stream,
                il,
                il,
                b_len,
                &mut n_blocks,
                0,
                record_seq,
                &spilled_schedule,
                &mut spilled_cursor,
            )?;
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

        let precompute_all_msgs = self.small_rows(b_len);
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
                    if let Some(f_dm_m) = self.fns.depth_mix_multi_b {
                        let mut nqi = 4i32;
                        let mut m_args: [*mut c_void; 11] = [
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
                            (&mut nqi as *mut _) as *mut c_void,
                        ];
                        self.launch(stream, f_dm_m, (1, b_u32), 256, &mut m_args)?;
                    } else {
                        self.launch(stream, self.fns.depth_mix_b, (4, b_u32), 256, &mut args)?;
                    }
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
                    &spilled_schedule,
                    &mut spilled_cursor,
                )?;
            }
        }

        for il in 18..24 {
            self.enqueue_batch_visit(
                stream,
                il,
                il,
                b_len,
                &mut n_blocks,
                0,
                record_seq,
                &spilled_schedule,
                &mut spilled_cursor,
            )?;
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

        if self.ms_rows.get() != 0 {
            // Multi-sequence round: the host advances each sequence's own position.
            return Ok(());
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

    /// Every kernel may be launched with PDL (`pdl_all`), so each one must reach `pdl_wait()`
    /// before it touches memory a preceding kernel may write (#152 P0-B).
    #[test]
    fn every_kernel_waits_on_its_pdl_predecessor() {
        let src = CUDA_KERNEL_SRC;
        let mut missing = Vec::new();
        let mut seen = 0;
        let mut macro_entries = 0;
        for (at, _) in src.match_indices("extern \"C\" __global__") {
            let line_start = src[..at].rfind('\n').map_or(0, |i| i + 1);
            let prev_line_end = line_start.saturating_sub(1);
            let prev_line_start = src[..prev_line_end].rfind('\n').map_or(0, |i| i + 1);
            if src[prev_line_start..prev_line_end]
                .trim_end()
                .ends_with('\\')
            {
                let entry = src[at..].lines().next().unwrap();
                let (_, body) = entry.split_once('{').expect("kernel macro entry body");
                assert!(
                    body.trim_start().starts_with("pdl_wait();"),
                    "kernel macro reads before waiting on its predecessor: {entry}"
                );
                macro_entries += 1;
                continue;
            }
            let open = at + src[at..].find('{').expect("kernel body");
            let mut depth = 0usize;
            let mut close = open;
            for (i, c) in src[open..].char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            close = open + i;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            seen += 1;
            if !src[open..close].contains("pdl_wait()") {
                missing.push(
                    src[at..open]
                        .split('(')
                        .next()
                        .unwrap_or_default()
                        .to_string(),
                );
            }
        }
        assert!(seen > 100, "parsed only {seen} kernels");
        assert!(
            missing.is_empty(),
            "kernels without pdl_wait(): {missing:?}"
        );
        assert!(
            macro_entries >= 4,
            "parsed only {macro_entries} macro entries"
        );
        assert!(src.contains("void ink2_pdl_all_marker()"));
    }

    #[test]
    fn device_codebook_roundtrips_every_nibble_across_rans_tiles() {
        let values = [
            0, 128, 1, 129, 10, 138, 17, 145, 30, 158, 63, 191, 100, 228, 126, 254,
        ];
        let raw: Vec<u8> = (0..crate::inkling_mxfp8::TILE + 32)
            .map(|i| values[(i * 7 + i / 16) % values.len()])
            .collect();
        let encode = |raw: &[u8]| {
            let mut frame = Vec::new();
            crate::inkling_mxfp8::pack(raw, &mut frame).unwrap();
            ExpertMatrix::Packed {
                bytes: &frame,
                count: raw.len(),
                contextual: false,
            }
            .device_codebook()
        };
        let packed = encode(&raw).unwrap();
        assert_eq!(packed.len(), 16 + raw.len() / 2);
        let decoded: Vec<u8> = packed[16..]
            .iter()
            .flat_map(|&v| [packed[(v & 15) as usize], packed[(v >> 4) as usize]])
            .collect();
        assert_eq!(decoded, raw);
        let mut excessive = raw;
        excessive[0] = 2;
        assert!(encode(&excessive).is_err());
    }

    #[test]
    fn known_prompt_rounds_up_to_a_supported_workspace() {
        for (tokens, width) in [
            (0, 64),
            (1, 64),
            (64, 64),
            (65, 128),
            (255, 256),
            (257, 512),
            (513, 1024),
            (1024, 1024),
            (1025, 2048),
            (usize::MAX, 2048),
        ] {
            assert_eq!(known_prompt_limit(tokens), width);
        }
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn packed_expert_restores_bytes_to_both_memory_tiers() -> Result<(), String> {
        // Cross both the 16 MiB upload boundary and a partial final codec tile.
        let raw: Vec<u8> = (0..(16 << 20) + 37)
            .map(|i| {
                if i % 5 == 0 {
                    (i ^ (i >> 11) ^ (i >> 20)) as u8
                } else {
                    0x38
                }
            })
            .collect();
        let mut packed = Vec::new();
        crate::inkling_mxfp8::pack(&raw, &mut packed).map_err(|e| e.to_string())?;
        let matrix = ExpertMatrix::Packed {
            bytes: &packed,
            count: raw.len(),
            contextual: false,
        };
        let ctx = CudaContextHandle::init(0)?;
        for on_device in [false, true] {
            let placed = matrix.place(&ctx, on_device)?;
            assert_eq!(matches!(placed, TierBuf::Vram(_)), on_device);
            let mut actual = vec![0u8; raw.len()];
            // Both tier pointers are visible to CUDA; the allocation outlives this readback.
            unsafe {
                check_cu(
                    (ctx.api.cuMemcpyDtoH_v2)(
                        actual.as_mut_ptr().cast(),
                        placed.ptr(),
                        actual.len(),
                    ),
                    "cuMemcpyDtoH_v2",
                )?;
            }
            assert!(
                actual == raw,
                "packed expert byte mismatch on_device={on_device}"
            );
        }
        Ok(())
    }

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
        let tensors = [[ExpertMatrix::Raw(&up), ExpertMatrix::Raw(&down)]; N_LAYER];
        let placement = plan_expert_placement(&tensors, 12 * 3 + 1);
        let packed = [[
            ExpertMatrix::Packed {
                bytes: &[],
                count: up.len(),
                contextual: false,
            },
            ExpertMatrix::Packed {
                bytes: &[],
                count: down.len(),
                contextual: false,
            },
        ]; N_LAYER];
        assert_eq!(plan_expert_placement(&packed, 12 * 3 + 1), placement);
        assert!(placement[6..18].iter().all(|pair| *pair == [true, true]));
        assert_eq!(placement[NVFP4_LAYERS], [false, true]);
        assert!(placement[..6]
            .iter()
            .chain(&placement[18..NVFP4_LAYERS])
            .chain(&placement[NVFP4_LAYERS + 1..])
            .all(|pair| *pair == [false, false]));
        // Starting over with one more byte would replace this down matrix
        // with the preceding up matrix. Retain it when extending the mask.
        let extended = extend_expert_placement(&tensors, 1, placement);
        assert_eq!(extended[NVFP4_LAYERS], [false, true]);
        assert_eq!(extended[NVFP4_LAYERS + 1], [false, true]);
        assert_eq!(
            plan_expert_placement(&tensors, 12 * 3 + 2)[NVFP4_LAYERS],
            [true, false]
        );
        let after_loop_and_mxfp8 = (12 + N_LAYER - NVFP4_LAYERS) * 3;
        let placement = plan_expert_placement(&tensors, after_loop_and_mxfp8 + 8);
        assert!(placement[..6]
            .iter()
            .chain(&placement[18..NVFP4_LAYERS])
            .all(|pair| *pair == [false, true]));
        assert_eq!(plan_expert_placement(&tensors, 0), [[false; 2]; N_LAYER]);
        assert_eq!(
            plan_expert_placement(&tensors, N_LAYER * 3),
            [[true; 2]; N_LAYER]
        );
    }
}
