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
pub const MAX_BATCH_TOKENS: usize = 64;
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
    // Batched multi-token kernels (`1 <= B <= 64`)
    embed_rmsnorm_b: CUfunction,
    rmsnorm_d_b: CUfunction,
    rmsnorm_quant_q8_64_b: CUfunction,
    quant_q8_64_b: CUfunction,
    depth_mix_b: CUfunction,
    ctrl_down_b: CUfunction,
    ctrl_apply_b: CUfunction,
    q8_dp4a_gemm_b: CUfunction,
    short_conv4_b: CUfunction,
    prep_qkv_and_cache_b: CUfunction,
    gqa_rel_sink_attn_b: CUfunction,
    bf16_gemm_router_b: CUfunction,
    router_topk8_b: CUfunction,
    build_expert_groups_b: CUfunction,
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
}

unsafe impl Send for KernelFns {}
unsafe impl Sync for KernelFns {}

pub struct Inkling2GpuEngine {
    pub ctx: Arc<CudaContextHandle>,
    fns: KernelFns,
    embed: TierBuf,
    embed_norm: DeviceBuf,
    final_norm: DeviceBuf,
    unembed_q8: PlanarQ8Gpu,
    output_query: DeviceBuf,
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
    d_attn_part_ml: DeviceBuf,
    d_attn_part_o: DeviceBuf,
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
    // Multi-token batch scratch buffers (`1 <= B <= 64`) + Speculative verify buffers (`B <= 8`)
    d_b_cur: DeviceBuf,
    d_b_prelude_x: DeviceBuf,
    d_b_blocks: DeviceBuf,
    d_b_prelude_block0: DeviceBuf,
    d_b_bank: DeviceBuf,
    d_b_msgs: DeviceBuf,
    d_b_inputs: DeviceBuf,
    d_b_ctrl_g: DeviceBuf,
    d_b_qrgs: DeviceBuf,
    d_b_k_raw: DeviceBuf,
    d_b_v_raw: DeviceBuf,
    d_b_k_conv: DeviceBuf,
    d_b_v_conv: DeviceBuf,
    d_b_attn_out: DeviceBuf,
    d_b_wo_out: DeviceBuf,
    d_b_mlp_h: DeviceBuf,
    d_b_router_logits: DeviceBuf,
    d_b_sel_ids: DeviceBuf,
    d_b_sel_wts: DeviceBuf,
    d_b_exp_counts: DeviceBuf,
    d_b_exp_toks: DeviceBuf,
    d_b_exp_pairs: DeviceBuf,
    d_b_moe_act: DeviceBuf,
    d_b_pair_y: DeviceBuf,
    d_b_moe_y: DeviceBuf,
    d_b_act_q8_64: DeviceBuf,
    d_b_act_sc_64: DeviceBuf,
    d_b_act_q8_16: DeviceBuf,
    d_b_act_sc_16: DeviceBuf,
    d_spec_logits: DeviceBuf,
    d_spec_tokens_out: DeviceBuf,
    d_taps: DeviceBuf,
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
    let cols64 = cols / 64;
    let mut codes = vec![0u8; rows * cols];
    let mut scales_bytes = vec![0u8; rows * cols64 * 4];

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get().min(16))
        .unwrap_or(8);
    let rows_per_thread = rows.div_ceil(n_threads).max(1);

    std::thread::scope(|s| {
        let mut code_rest = codes.as_mut_slice();
        let mut scale_rest = scales_bytes.as_mut_slice();
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
                let (pre, src_u16, post) = unsafe { src_chunk.align_to::<u16>() };
                assert!(pre.is_empty() && post.is_empty());
                for r in 0..chunk_rows {
                    let w_row = &src_u16[r * cols..(r + 1) * cols];
                    let dst_c = &mut c_chunk[r * cols..(r + 1) * cols];
                    let dst_s = &mut s_chunk[r * cols64 * 4..(r + 1) * cols64 * 4];
                    for b in 0..cols64 {
                        let sub = &w_row[b * 64..(b + 1) * 64];
                        let mut fvals = [0.0f32; 64];
                        for (v, &u) in fvals.iter_mut().zip(sub) {
                            *v = bf16_to_f32(u);
                        }
                        for half in [1, 2, 4, 8, 16, 32] {
                            for block in fvals.chunks_exact_mut(2 * half) {
                                let (low, high) = block.split_at_mut(half);
                                for (a, b) in low.iter_mut().zip(high) {
                                    let diff = *a - *b;
                                    *a += *b;
                                    *b = diff;
                                }
                            }
                        }
                        let mut amax = 0.0f32;
                        for value in &mut fvals {
                            *value *= 0.125;
                            amax = amax.max(value.abs());
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

    Ok(PlanarQ8Gpu {
        codes: DeviceBuf::from_host_bytes(ctx, &codes)?,
        scales64: DeviceBuf::from_host_bytes(ctx, &scales_bytes)?,
        cols: cols as i32,
    })
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
        let st_path = model_dir.as_ref().join("model-00001-of-00001.safetensors");
        let st = MappedSafeTensors::open(&st_path)
            .map_err(|e| format!("open {}: {e}", st_path.display()))?;

        let embed_raw = st
            .require("model.llm.embed.weight", "BF16", &[PADDED_VOCAB, D])
            .data;
        let embed = TierBuf::HostPinned(PinnedHostBuf::from_bytes(&ctx, embed_raw)?);
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
                max_ctx.min(EXTENT + MAX_BATCH_TOKENS)
            };
            slots.push(SlotStateGpu {
                kv_cap,
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
        let d_taps = DeviceBuf::alloc_zeroed(&ctx, 4 * MAX_BATCH_TOKENS * D * 4)?;
        let d_conv_seq = if spec_k > 0 {
            DeviceBuf::alloc_zeroed(&ctx, N_SLOT * CONV_SEQ_SLOT_FLOATS * 4)?
        } else {
            DeviceBuf::alloc_zeroed(&ctx, 4)?
        };

        // Load the optional drafter before budgeting memory for tiered expert weights.
        let mtp_path = model_dir.as_ref().join("mtp.safetensors");
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
            d_msgs: DeviceBuf::alloc_zeroed(&ctx, 4 * D * 4)?,
            d_inputs: DeviceBuf::alloc_zeroed(&ctx, 3 * D * 4)?,
            d_ctrl_g: DeviceBuf::alloc_zeroed(&ctx, 3 * RANK * 4)?,
            d_qrgs: DeviceBuf::alloc_zeroed(&ctx, SWA_ALL_ROWS * 4)?,
            d_k_raw: DeviceBuf::alloc_zeroed(&ctx, 1056 * 4)?,
            d_v_raw: DeviceBuf::alloc_zeroed(&ctx, 1056 * 4)?,
            d_k_conv: DeviceBuf::alloc_zeroed(&ctx, 1024 * 4)?,
            d_v_conv: DeviceBuf::alloc_zeroed(&ctx, 1024 * 4)?,
            d_attn_out: DeviceBuf::alloc_zeroed(&ctx, 3072 * 4)?,
            attn_split_g,
            d_attn_part_ml: DeviceBuf::alloc_zeroed(
                &ctx,
                if attn_split_g > 0 {
                    MAX_BATCH_TOKENS * 24 * attn_split_g * 2 * 4
                } else {
                    4
                },
            )?,
            d_attn_part_o: DeviceBuf::alloc_zeroed(
                &ctx,
                if attn_split_g > 0 {
                    MAX_BATCH_TOKENS * 24 * attn_split_g * 128 * 4
                } else {
                    4
                },
            )?,
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
            d_b_cur: DeviceBuf::alloc_zeroed(&ctx, MAX_BATCH_TOKENS * D * 4)?,
            d_b_prelude_x: DeviceBuf::alloc_zeroed(&ctx, MAX_BATCH_TOKENS * D * 4)?,
            d_b_blocks: DeviceBuf::alloc_zeroed(&ctx, 6 * MAX_BATCH_TOKENS * D * 4)?,
            d_b_prelude_block0: DeviceBuf::alloc_zeroed(&ctx, MAX_BATCH_TOKENS * D * 4)?,
            d_b_bank: DeviceBuf::alloc_zeroed(&ctx, 2 * MAX_BATCH_TOKENS * D * 4)?,
            d_b_msgs: DeviceBuf::alloc_zeroed(&ctx, 4 * MAX_BATCH_TOKENS * D * 4)?,
            d_b_inputs: DeviceBuf::alloc_zeroed(&ctx, 3 * MAX_BATCH_TOKENS * D * 4)?,
            d_b_ctrl_g: DeviceBuf::alloc_zeroed(&ctx, 3 * MAX_BATCH_TOKENS * RANK * 4)?,
            d_b_qrgs: DeviceBuf::alloc_zeroed(&ctx, MAX_BATCH_TOKENS * SWA_ALL_ROWS * 4)?,
            d_b_k_raw: DeviceBuf::alloc_zeroed(&ctx, MAX_BATCH_TOKENS * 1056 * 4)?,
            d_b_v_raw: DeviceBuf::alloc_zeroed(&ctx, MAX_BATCH_TOKENS * 1056 * 4)?,
            d_b_k_conv: DeviceBuf::alloc_zeroed(&ctx, MAX_BATCH_TOKENS * 1024 * 4)?,
            d_b_v_conv: DeviceBuf::alloc_zeroed(&ctx, MAX_BATCH_TOKENS * 1024 * 4)?,
            d_b_attn_out: DeviceBuf::alloc_zeroed(&ctx, MAX_BATCH_TOKENS * 3072 * 4)?,
            d_b_wo_out: DeviceBuf::alloc_zeroed(&ctx, MAX_BATCH_TOKENS * D * 4)?,
            d_b_mlp_h: DeviceBuf::alloc_zeroed(&ctx, MAX_BATCH_TOKENS * D * 4)?,
            d_b_router_logits: DeviceBuf::alloc_zeroed(&ctx, MAX_BATCH_TOKENS * N_EXP * 4)?,
            d_b_sel_ids: DeviceBuf::alloc_zeroed(&ctx, MAX_BATCH_TOKENS * TOP_K * 4)?,
            d_b_sel_wts: DeviceBuf::alloc_zeroed(&ctx, MAX_BATCH_TOKENS * TOP_K * 4)?,
            d_b_exp_counts: DeviceBuf::alloc_zeroed(&ctx, N_EXP * 4)?,
            d_b_exp_toks: DeviceBuf::alloc_zeroed(&ctx, N_EXP * MAX_BATCH_TOKENS * 4)?,
            d_b_exp_pairs: DeviceBuf::alloc_zeroed(&ctx, N_EXP * MAX_BATCH_TOKENS * 4)?,
            d_b_moe_act: DeviceBuf::alloc_zeroed(&ctx, MAX_BATCH_TOKENS * TOP_K * FF * 4)?,
            d_b_pair_y: DeviceBuf::alloc_zeroed(&ctx, MAX_BATCH_TOKENS * TOP_K * D * 4)?,
            d_b_moe_y: DeviceBuf::alloc_zeroed(&ctx, MAX_BATCH_TOKENS * D * 4)?,
            d_b_act_q8_64: DeviceBuf::alloc_zeroed(&ctx, 3 * MAX_BATCH_TOKENS * 3072)?,
            d_b_act_sc_64: DeviceBuf::alloc_zeroed(&ctx, 3 * MAX_BATCH_TOKENS * 48 * 4)?,
            d_b_act_q8_16: DeviceBuf::alloc_zeroed(&ctx, MAX_BATCH_TOKENS * TOP_K * FF)?,
            d_b_act_sc_16: DeviceBuf::alloc_zeroed(&ctx, MAX_BATCH_TOKENS * TOP_K * (FF / 16) * 4)?,
            d_spec_logits: DeviceBuf::alloc_zeroed(&ctx, MAX_SPEC_TOKENS * VOCAB * 4)?,
            d_spec_tokens_out: DeviceBuf::alloc_zeroed(&ctx, MAX_SPEC_TOKENS * 4)?,
            d_taps,
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
        let resident = plan_expert_placement(&expert_views, free.saturating_sub(graph_reserve));
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
            "[LeanSonic GPU] {} | VRAM used: {:.2} GiB / {:.2} GiB | Resident expert layers: {}/24 (Spilled: {}) | Mixed VRAM/host layers: {} | Spec: {} | Loaded in {:.2}s",
            eng.ctx.device_name,
            (total_vram - free_after) as f64 / (1024.0 * 1024.0 * 1024.0),
            total_vram as f64 / (1024.0 * 1024.0 * 1024.0),
            eng.resident_expert_layers,
            eng.spilled_expert_layers,
            mixed_layers,
            drafter_tag,
            t_load.elapsed().as_secs_f64()
        );
        Ok(eng)
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
            ctrl_apply_b: get_fn("ink2_ctrl_apply_b")?,
            q8_dp4a_gemm_b: get_fn("ink2_q8_dp4a_gemm_b")?,
            short_conv4_b: get_fn("ink2_short_conv4_b")?,
            prep_qkv_and_cache_b: get_fn("ink2_prep_qkv_and_cache_b")?,
            gqa_rel_sink_attn_b: get_fn("ink2_gqa_rel_sink_attn_b")?,
            bf16_gemm_router_b: get_fn("ink2_bf16_gemm_router_b")?,
            router_topk8_b: get_fn("ink2_router_topk8_b")?,
            build_expert_groups_b: get_fn("ink2_build_expert_groups_b")?,
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
        let mut args2: [*mut c_void; 5] = [
            (&mut p_cur as *mut _) as *mut c_void,
            (&mut p_msg as *mut _) as *mut c_void,
            (&mut p_g as *mut _) as *mut c_void,
            (&mut p_gain as *mut _) as *mut c_void,
            (&mut p_forget as *mut _) as *mut c_void,
        ];
        self.launch(stream, self.fns.ctrl_apply, (9, n_ctrls), 256, &mut args2)
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
        // 16 rows per 256-thread CTA (2 rows/warp = 38 regs <= 42 SM120 full occupancy!)
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
        let mut pml = self.d_attn_part_ml.ptr;
        let mut po = self.d_attn_part_o.ptr;
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
        )
    }

    fn enqueue_visit(
        &self,
        stream: CUstream,
        il: usize,
        slot_idx: usize,
        n_blocks: &mut usize,
        has_msgs: bool,
    ) -> Result<(), String> {
        let l = &self.layers[il];
        let st = &self.slots[slot_idx];
        let d_bytes = (D * 4) as u64;

        // 1. Attention inputs `[xq, xk, xv]` (`broadcast3` eliminates extra `copy_d` launches!)
        if let Some(ref aq) = l.attn_query {
            let n_q: u32 = if l.global { 3 } else { 1 };
            let mut pb = self.d_blocks.ptr;
            let mut nb = *n_blocks as i32;
            let mut pc = self.d_cur.ptr;
            let mut ac = 1i32;
            let mut pq = aq.ptr;
            let mut pout = self.d_inputs.ptr;
            let mut b3 = i32::from(!l.global && has_msgs);
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
        } else {
            self.enqueue_copy_d(stream, self.d_cur.ptr, self.d_inputs.ptr)?;
            if l.global || has_msgs {
                self.enqueue_copy_d(stream, self.d_cur.ptr, self.d_inputs.ptr + d_bytes)?;
                self.enqueue_copy_d(stream, self.d_cur.ptr, self.d_inputs.ptr + 2 * d_bytes)?;
            }
        }

        // 2. Batched Loop Core Controller (`pass > 0`) on ALL 3 `inputs[0..3]` in ONE launch pair!
        if has_msgs {
            self.enqueue_ctrl_n(
                stream,
                self.d_inputs.ptr,
                self.d_msgs.ptr,
                l.ctrl3.as_ref().unwrap(),
                3,
            )?;
        }

        // 3. Fused Batched Input `RMSNorm + Q8_64 Quant` + Planar `Q8_64` `__dp4a` Projections!
        let distinct_k_v = l.global || has_msgs;
        let norm_count: u32 = if distinct_k_v { 3 } else { 1 };
        self.enqueue_rmsnorm_quant_q8_64(
            stream,
            self.d_inputs.ptr,
            l.norms_packed.ptr,
            norm_count,
        )?;

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

        // 5. K & V Short-Conv4
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

        // 7. GQA Relative-Bias + Sink Attention
        if self.attn_split_g == 0 {
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
                1,
            )?;
        }

        // 8. `wo` Planar `Q8_64` `__dp4a` Projection + `attn_sconv` in-place into `d_cur`!
        self.enqueue_quant_q8_64(stream, self.d_attn_out.ptr, 3072)?;
        self.enqueue_q8_gemv_slice_at(stream, &l.wo_q8, 0, D as i32, 0, self.d_wo_out.ptr, 0.0)?;

        let is_block_start = il.is_multiple_of(6);
        if is_block_start {
            let blk_dst = self.d_blocks.ptr + (*n_blocks as u64) * d_bytes;
            self.enqueue_copy_d(stream, self.d_cur.ptr, blk_dst)?;
            *n_blocks += 1;
        }
        {
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

        // 9. MLP Depth Mix + Loop Controller 3 + `mlp_norm`
        {
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
        }
        if has_msgs {
            self.enqueue_ctrl_n(
                stream,
                self.d_mlp_h.ptr,
                self.d_msgs.ptr + 3 * d_bytes,
                l.ctrl_mlp.as_ref().unwrap(),
                1,
            )?;
        }
        {
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
                    self.enqueue_quant_q8_16(stream, self.d_mlp_h.ptr, 1, D)?;
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

        if self.drafter.is_some() {
            if let Some(tap_idx) = TAP_SLOTS.iter().position(|&s| s == slot_idx) {
                let dst = self.d_taps.ptr + (tap_idx * MAX_BATCH_TOKENS * D * 4) as u64;
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
            self.enqueue_visit(stream, il, il, &mut n_blocks, false)?;
        }
        debug_assert_eq!(n_blocks, 1);
        self.enqueue_copy_d(stream, self.d_cur.ptr, self.d_prelude_x.ptr)?;
        self.enqueue_copy_d(stream, self.d_blocks.ptr, self.d_prelude_block0.ptr)?;

        for pass in 0..3 {
            if pass > 0 {
                self.enqueue_copy_d(stream, self.d_blocks.ptr + 2 * d_bytes, self.d_bank.ptr)?;
                self.enqueue_copy_d(stream, self.d_cur.ptr, self.d_bank.ptr + d_bytes)?;
                self.enqueue_copy_d(stream, self.d_prelude_x.ptr, self.d_cur.ptr)?;
                self.enqueue_copy_d(stream, self.d_prelude_block0.ptr, self.d_blocks.ptr)?;
                n_blocks = 1;
            }
            for ip in 6..18 {
                let slot = if pass == 0 {
                    ip
                } else {
                    N_LAYER + (pass - 1) * 12 + (ip - 6)
                };
                if pass > 0 {
                    let mq = self.layers[ip].msg_query.as_ref().unwrap();
                    let mut pb = self.d_bank.ptr;
                    let mut nb = 2i32;
                    let mut pc: CUdeviceptr = 0;
                    let mut ac = 0i32;
                    let mut pq = mq.ptr;
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
                    self.launch(stream, self.fns.depth_mix, (4, 1), 256, &mut args)?;
                }
                self.enqueue_visit(stream, ip, slot, &mut n_blocks, pass > 0)?;
            }
        }

        for il in 18..24 {
            self.enqueue_visit(stream, il, il, &mut n_blocks, false)?;
        }
        debug_assert_eq!(n_blocks, 4);

        if prefill_only {
            if self.drafter.is_some() {
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
                let mut p_tap3 = self.d_taps.ptr + (3 * MAX_BATCH_TOKENS * D * 4) as u64;
                let mut hg = 1i32;
                let mut rn_args: [*mut c_void; 4] = [
                    (&mut pin as *mut _) as *mut c_void,
                    (&mut pg as *mut _) as *mut c_void,
                    (&mut p_tap3 as *mut _) as *mut c_void,
                    (&mut hg as *mut _) as *mut c_void,
                ];
                self.launch(stream, self.fns.rmsnorm_d, (1, 1), 256, &mut rn_args)?;
            }
            let mut pp = self.d_pos.ptr;
            let mut args: [*mut c_void; 1] = [(&mut pp as *mut _) as *mut c_void];
            self.launch(stream, self.fns.inc_pos, (1, 1), 32, &mut args)?;
            return Ok(());
        }

        // Output Depth Mix + Fused `final_norm + Q8_64` + Planar `Q8_64` `unembed` (`softcap = 30.0`) + ArgMax
        {
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
        }
        if self.drafter.is_some() {
            let mut pin = self.d_mlp_h.ptr;
            let mut pg = self.final_norm.ptr;
            let mut p_tap3 = self.d_taps.ptr + (3 * MAX_BATCH_TOKENS * D * 4) as u64;
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
        if let Some(ref mut d) = self.drafter {
            d.enqueue_add_ctx(
                self.ctx.compute_stream,
                self.d_taps.ptr,
                MAX_BATCH_TOKENS,
                1,
            )?;
        }
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
                    if let Some(ref mut d) = self.drafter {
                        d.enqueue_add_ctx(
                            self.ctx.compute_stream,
                            self.d_taps.ptr,
                            MAX_BATCH_TOKENS,
                            1,
                        )?;
                    }
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
            let chunk_len = rem.min(MAX_BATCH_TOKENS);
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
            if let Some(ref mut d) = self.drafter {
                d.enqueue_add_ctx(
                    self.ctx.compute_stream,
                    self.d_taps.ptr,
                    MAX_BATCH_TOKENS,
                    chunk_len,
                )?;
            }
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
            if let Some(ref mut d) = self.drafter {
                d.enqueue_add_ctx(
                    self.ctx.compute_stream,
                    self.d_taps.ptr,
                    MAX_BATCH_TOKENS,
                    keep,
                )?;
            }

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
        let mut p_g = self.d_b_ctrl_g.ptr;
        let mut args1: [*mut c_void; 5] = [
            (&mut p_cur as *mut _) as *mut c_void,
            (&mut p_msg as *mut _) as *mut c_void,
            (&mut p_down as *mut _) as *mut c_void,
            (&mut p_g as *mut _) as *mut c_void,
            (&mut b_i32 as *mut _) as *mut c_void,
        ];
        self.launch(
            stream,
            self.fns.ctrl_down_b,
            (64, n_ctrls * b_u32),
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
        self.launch(
            stream,
            self.fns.ctrl_apply_b,
            (9, n_ctrls * b_u32),
            256,
            &mut args2,
        )
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
        let mut pq = self.d_b_act_q8_64.ptr;
        let mut ps = self.d_b_act_sc_64.ptr;
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
        let mut pq = self.d_b_act_q8_64.ptr;
        let mut ps = self.d_b_act_sc_64.ptr;
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
            self.d_b_act_q8_16.ptr,
            self.d_b_act_sc_16.ptr,
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
        let mut px = self.d_b_act_q8_64.ptr + (act_slot * b_len * cols) as u64;
        let mut pxs = self.d_b_act_sc_64.ptr + (act_slot * b_len * (cols / 64) * 4) as u64;
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
        has_msgs: bool,
        record_seq: bool,
    ) -> Result<(), String> {
        let l = &self.layers[il];
        let st = &self.slots[slot_idx];
        let b_u32 = b_len as u32;
        let b_i32 = b_len as i32;
        let bd_floats = b_len * D;
        let bd_bytes = (bd_floats * 4) as u64;
        let slot_seq_base = if record_seq {
            self.d_conv_seq.ptr + (slot_idx * CONV_SEQ_SLOT_FLOATS * 4) as u64
        } else {
            0
        };

        // 1. Batched Attention inputs `[xq, xk, xv]` (`[3, b_len, D]`)
        if let Some(ref aq) = l.attn_query {
            let n_q: u32 = if l.global { 3 } else { 1 };
            let mut pb = self.d_b_blocks.ptr;
            let mut nb = *n_blocks as i32;
            let mut sb = b_i32;
            let mut bo = 0i32;
            let mut bout = b_i32;
            let mut pc = self.d_b_cur.ptr;
            let mut ac = 1i32;
            let mut pq = aq.ptr;
            let mut pout = self.d_b_inputs.ptr;
            let mut b3 = i32::from(!l.global && has_msgs);
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
        } else {
            self.enqueue_copy_n(stream, self.d_b_cur.ptr, self.d_b_inputs.ptr, bd_floats)?;
            if l.global || has_msgs {
                self.enqueue_copy_n(
                    stream,
                    self.d_b_cur.ptr,
                    self.d_b_inputs.ptr + bd_bytes,
                    bd_floats,
                )?;
                self.enqueue_copy_n(
                    stream,
                    self.d_b_cur.ptr,
                    self.d_b_inputs.ptr + 2 * bd_bytes,
                    bd_floats,
                )?;
            }
        }

        // 2. Batched Loop Core Controller (`pass > 0`)
        if has_msgs {
            self.enqueue_ctrl_n_b(
                stream,
                self.d_b_inputs.ptr,
                self.d_b_msgs.ptr,
                l.ctrl3.as_ref().unwrap(),
                3,
                b_len,
            )?;
        }

        // 3. Fused Batched `RMSNorm + Q8_64` + `ink2_q8_dp4a_gemm_b`
        let distinct_k_v = l.global || has_msgs;
        let norm_count: u32 = if distinct_k_v { 3 } else { 1 };
        self.enqueue_rmsnorm_quant_q8_64_b(
            stream,
            self.d_b_inputs.ptr,
            l.norms_packed.ptr,
            norm_count,
            b_len,
        )?;

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
                self.d_b_qrgs.ptr,
                SWA_ALL_ROWS as i32,
                0.0,
            )?;
            k_raw_ptr = self.d_b_qrgs.ptr + (SWA_K_OFFSET as u64) * 4;
            k_in_stride = SWA_ALL_ROWS as i32;
            v_raw_ptr = self.d_b_qrgs.ptr + (SWA_V_OFFSET as u64) * 4;
            v_in_stride = SWA_ALL_ROWS as i32;
        } else if l.global {
            self.enqueue_q8_gemm_slice_at(
                stream,
                &l.wq_all_q8,
                0,
                QRGS_ROWS as i32,
                0,
                b_len,
                self.d_b_qrgs.ptr,
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
                self.d_b_k_raw.ptr,
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
                self.d_b_v_raw.ptr,
                1056,
                0.0,
            )?;
            k_raw_ptr = self.d_b_k_raw.ptr;
            k_in_stride = 1056;
            v_raw_ptr = self.d_b_v_raw.ptr;
            v_in_stride = 1056;
        } else {
            self.enqueue_q8_gemm_slice_at(
                stream,
                &l.wq_all_q8,
                0,
                QRGS_ROWS as i32,
                0,
                b_len,
                self.d_b_qrgs.ptr,
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
                self.d_b_k_raw.ptr,
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
                self.d_b_v_raw.ptr,
                1024,
                0.0,
            )?;
            k_raw_ptr = self.d_b_k_raw.ptr;
            k_in_stride = 1024;
            v_raw_ptr = self.d_b_v_raw.ptr;
            v_in_stride = 1024;
        }

        // 4. Register-Persistent Causal `short_conv4` for K & V
        for (raw_ptr, raw_stride, conv_w, hist_ptr, dst_ptr, seq_off_floats) in [
            (
                k_raw_ptr,
                k_in_stride,
                l.k_conv4.ptr,
                st.hist_k,
                self.d_b_k_conv.ptr,
                0usize,
            ),
            (
                v_raw_ptr,
                v_in_stride,
                l.v_conv4.ptr,
                st.hist_v,
                self.d_b_v_conv.ptr,
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
            let mut pqrgs = self.d_b_qrgs.ptr;
            let mut qstride = SWA_ALL_ROWS as i32;
            let mut pk = self.d_b_k_conv.ptr;
            let mut pv = self.d_b_v_conv.ptr;
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

        // 6. Batched GQA Relative-Bias + Sink Attention
        if self.attn_split_g == 0 {
            let mut pqrgs = self.d_b_qrgs.ptr;
            let mut qstride = SWA_ALL_ROWS as i32;
            let mut prp = l.rel_proj.ptr;
            let mut pkc = st.k_cache.ptr;
            let mut pvc = st.v_cache.ptr;
            let mut ppos = self.d_pos.ptr;
            let mut cap = st.kv_cap as i32;
            let mut ig = i32::from(l.global);
            let mut pout = self.d_b_attn_out.ptr;
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
        } else {
            self.enqueue_gqa_attn_split_impl(
                stream,
                self.d_b_qrgs.ptr,
                SWA_ALL_ROWS as i32,
                l.rel_proj.ptr,
                st.k_cache.ptr,
                st.v_cache.ptr,
                st.kv_cap as i32,
                i32::from(l.global),
                self.d_b_attn_out.ptr,
                b_len,
            )?;
        }

        // 7. `wo` Batched `Q8_64` GEMM + `attn_sconv`
        self.enqueue_quant_q8_64_b(stream, self.d_b_attn_out.ptr, 3072, b_len)?;
        self.enqueue_q8_gemm_slice_at(
            stream,
            &l.wo_q8,
            0,
            D as i32,
            0,
            b_len,
            self.d_b_wo_out.ptr,
            D as i32,
            0.0,
        )?;

        let is_block_start = il.is_multiple_of(6);
        if is_block_start {
            let blk_dst = self.d_b_blocks.ptr + (*n_blocks as u64) * bd_bytes;
            self.enqueue_copy_n(stream, self.d_b_cur.ptr, blk_dst, bd_floats)?;
            *n_blocks += 1;
        }
        {
            let mut p_cur = self.d_b_wo_out.ptr;
            let mut ins = D as i32;
            let mut p_w = l.attn_conv4.ptr;
            let mut p_dyn: CUdeviceptr = 0;
            let mut df4 = 0i32;
            let mut p_hist = st.hist_attn;
            let mut p_out = self.d_b_cur.ptr;
            let mut outs = D as i32;
            let mut p_res = if is_block_start { 0 } else { self.d_b_cur.ptr };
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

        // 8. Batched MLP Depth Mix + Loop Controller 3 + `mlp_norm`
        {
            let mut pb = self.d_b_blocks.ptr;
            let mut nb = *n_blocks as i32;
            let mut sb = b_i32;
            let mut bo = 0i32;
            let mut bout = b_i32;
            let mut pc = self.d_b_cur.ptr;
            let mut ac = 1i32;
            let mut pq = l.mlp_query.ptr;
            let mut pout = self.d_b_mlp_h.ptr;
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
        }
        if has_msgs {
            self.enqueue_ctrl_n_b(
                stream,
                self.d_b_mlp_h.ptr,
                self.d_b_msgs.ptr + 3 * bd_bytes,
                l.ctrl_mlp.as_ref().unwrap(),
                1,
                b_len,
            )?;
        }
        {
            let mut pin = self.d_b_mlp_h.ptr;
            let mut pg = l.mlp_norm.ptr;
            let mut pout = self.d_b_mlp_h.ptr;
            let mut hg = 1i32;
            let mut args: [*mut c_void; 4] = [
                (&mut pin as *mut _) as *mut c_void,
                (&mut pg as *mut _) as *mut c_void,
                (&mut pout as *mut _) as *mut c_void,
                (&mut hg as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.rmsnorm_d_b, (b_u32, 1), 256, &mut args)?;
        }

        // 9. Batched BF16 Router + TopK8 + Zero-Atomic Grouped-Expert MoE + Combine!
        {
            let mut pw = l.router_w.ptr;
            let mut px = self.d_b_mlp_h.ptr;
            let mut py = self.d_b_router_logits.ptr;
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
            let mut pl = self.d_b_router_logits.ptr;
            let mut pb = l.router_bias.ptr;
            let mut pids = self.d_b_sel_ids.ptr;
            let mut pwts = self.d_b_sel_wts.ptr;
            let mut args: [*mut c_void; 4] = [
                (&mut pl as *mut _) as *mut c_void,
                (&mut pb as *mut _) as *mut c_void,
                (&mut pids as *mut _) as *mut c_void,
                (&mut pwts as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.router_topk8_b, (b_u32, 1), 32, &mut args)?;
        }
        {
            let mut pids = self.d_b_sel_ids.ptr;
            let mut pcnt = self.d_b_exp_counts.ptr;
            let mut ptok = self.d_b_exp_toks.ptr;
            let mut ppair = self.d_b_exp_pairs.ptr;
            let mut bi = b_i32;
            let mut args: [*mut c_void; 5] = [
                (&mut pids as *mut _) as *mut c_void,
                (&mut pcnt as *mut _) as *mut c_void,
                (&mut ptok as *mut _) as *mut c_void,
                (&mut ppair as *mut _) as *mut c_void,
                (&mut bi as *mut _) as *mut c_void,
            ];
            self.launch(
                stream,
                self.fns.build_expert_groups_b,
                (1, 1),
                128,
                &mut args,
            )?;
        }

        let [w13, w2] = &self.expert_weights[il];
        match &l.expert_scales {
            ExpertScales::Nvfp4 { s13, a13, s2, a2 } => {
                self.enqueue_quant_q8_16_b(stream, self.d_b_mlp_h.ptr, b_len, D)?;
                let mut pw13 = w13.ptr();
                let mut ps13 = s13.ptr;
                let mut pa13 = a13.ptr;
                let mut pxq = self.d_b_act_q8_16.ptr;
                let mut pxs = self.d_b_act_sc_16.ptr;
                let mut pcnt = self.d_b_exp_counts.ptr;
                let mut ptok = self.d_b_exp_toks.ptr;
                let mut ppair = self.d_b_exp_pairs.ptr;
                let mut pact = self.d_b_moe_act.ptr;
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

                self.enqueue_quant_q8_16_b(stream, self.d_b_moe_act.ptr, b_len * TOP_K, FF)?;
                let mut pw2 = w2.ptr();
                let mut ps2 = s2.ptr;
                let mut pa2 = a2.ptr;
                let mut ppy = self.d_b_pair_y.ptr;
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
                let mut px = self.d_b_mlp_h.ptr;
                let mut pcnt = self.d_b_exp_counts.ptr;
                let mut ptok = self.d_b_exp_toks.ptr;
                let mut ppair = self.d_b_exp_pairs.ptr;
                let mut pact = self.d_b_moe_act.ptr;
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
                let mut ppy = self.d_b_pair_y.ptr;
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
            let mut ppy = self.d_b_pair_y.ptr;
            let mut pwts = self.d_b_sel_wts.ptr;
            let mut pout = self.d_b_moe_y.ptr;
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
            let mut p_cur = self.d_b_moe_y.ptr;
            let mut ins = D as i32;
            let mut p_w = l.mlp_conv4.ptr;
            let mut p_dyn: CUdeviceptr = 0;
            let mut df4 = 0i32;
            let mut p_hist = st.hist_mlp;
            let mut p_out = self.d_b_cur.ptr;
            let mut outs = D as i32;
            let mut p_res = self.d_b_cur.ptr;
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

        if self.drafter.is_some() {
            if let Some(tap_idx) = TAP_SLOTS.iter().position(|&s| s == slot_idx) {
                let dst = self.d_taps.ptr + (tap_idx * MAX_BATCH_TOKENS * D * 4) as u64;
                self.enqueue_copy_n(stream, self.d_b_cur.ptr, dst, bd_floats)?;
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
            let mut px = self.d_b_cur.ptr;
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
            self.enqueue_batch_visit(stream, il, il, b_len, &mut n_blocks, false, record_seq)?;
        }
        debug_assert_eq!(n_blocks, 1);
        self.enqueue_copy_n(stream, self.d_b_cur.ptr, self.d_b_prelude_x.ptr, bd_floats)?;
        self.enqueue_copy_n(
            stream,
            self.d_b_blocks.ptr,
            self.d_b_prelude_block0.ptr,
            bd_floats,
        )?;

        for pass in 0..3 {
            if pass > 0 {
                self.enqueue_copy_n(
                    stream,
                    self.d_b_blocks.ptr + 2 * bd_bytes,
                    self.d_b_bank.ptr,
                    bd_floats,
                )?;
                self.enqueue_copy_n(
                    stream,
                    self.d_b_cur.ptr,
                    self.d_b_bank.ptr + bd_bytes,
                    bd_floats,
                )?;
                self.enqueue_copy_n(stream, self.d_b_prelude_x.ptr, self.d_b_cur.ptr, bd_floats)?;
                self.enqueue_copy_n(
                    stream,
                    self.d_b_prelude_block0.ptr,
                    self.d_b_blocks.ptr,
                    bd_floats,
                )?;
                n_blocks = 1;
            }
            for ip in 6..18 {
                let slot = if pass == 0 {
                    ip
                } else {
                    N_LAYER + (pass - 1) * 12 + (ip - 6)
                };
                if pass > 0 {
                    let mq = self.layers[ip].msg_query.as_ref().unwrap();
                    let mut pb = self.d_b_bank.ptr;
                    let mut nb = 2i32;
                    let mut sb = b_i32;
                    let mut bo = 0i32;
                    let mut bout = b_i32;
                    let mut pc: CUdeviceptr = 0;
                    let mut ac = 0i32;
                    let mut pq = mq.ptr;
                    let mut pout = self.d_b_msgs.ptr;
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
                }
                self.enqueue_batch_visit(
                    stream,
                    ip,
                    slot,
                    b_len,
                    &mut n_blocks,
                    pass > 0,
                    record_seq,
                )?;
            }
        }

        for il in 18..24 {
            self.enqueue_batch_visit(stream, il, il, b_len, &mut n_blocks, false, record_seq)?;
        }
        debug_assert_eq!(n_blocks, 4);

        if self.drafter.is_some() {
            // Compute `tap 3` (`final_norm(output_query_mix)`) across all `b_len` tokens!
            let mut pb = self.d_b_blocks.ptr;
            let mut nb = n_blocks as i32;
            let mut sb = b_i32;
            let mut bo = 0i32;
            let mut bout = b_i32;
            let mut pc = self.d_b_cur.ptr;
            let mut ac = 1i32;
            let mut pq = self.output_query.ptr;
            let mut pout = self.d_b_mlp_h.ptr;
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
            let mut pin = self.d_b_mlp_h.ptr;
            let mut pg = self.final_norm.ptr;
            let mut p_tap3 = self.d_taps.ptr + (3 * MAX_BATCH_TOKENS * D * 4) as u64;
            let mut hg = 1i32;
            let mut rn_args: [*mut c_void; 4] = [
                (&mut pin as *mut _) as *mut c_void,
                (&mut pg as *mut _) as *mut c_void,
                (&mut p_tap3 as *mut _) as *mut c_void,
                (&mut hg as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.rmsnorm_d_b, (b_u32, 1), 256, &mut rn_args)?;
        }

        match head_mode {
            BatchHeadMode::None => {}
            BatchHeadMode::LastOnly => {
                let mut pb = self.d_b_blocks.ptr;
                let mut nb = n_blocks as i32;
                let mut sb = b_i32;
                let mut bo = (b_len - 1) as i32;
                let mut bout = 1i32;
                let mut pc = self.d_b_cur.ptr;
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
                self.enqueue_rmsnorm_quant_q8_64(stream, self.d_mlp_h.ptr, self.final_norm.ptr, 1)?;
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
                let mut pb = self.d_b_blocks.ptr;
                let mut nb = n_blocks as i32;
                let mut sb = b_i32;
                let mut bo = 0i32;
                let mut bout = b_i32;
                let mut pc = self.d_b_cur.ptr;
                let mut ac = 1i32;
                let mut pq = self.output_query.ptr;
                let mut pout = self.d_b_mlp_h.ptr;
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
                    self.d_b_mlp_h.ptr,
                    self.final_norm.ptr,
                    1,
                    b_len,
                )?;
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
mod placement_tests {
    use super::*;

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
