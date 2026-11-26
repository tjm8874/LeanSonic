//! Real `Inkling-2-Mini` (`19.5 GB` Official `safetensors`) Full-GPU Blackwell Engine.
//!
//! Runs ALL 24 physical layers (`48` visits: `0..19` NVFP4 MoE + `20..23` MXFP8 MoE)
//! 100% on `sm_120` Blackwell GPUs (`cuda01` RTX PRO 4500 32GB All-Resident & `cuda02`
//! RTX 5070 Ti 16GB Visit-Weighted Spilled Mode) with zero third-party Rust crates
//! and zero CPU AVX-512 dependency.

use core::ffi::c_void;
use std::ffi::CString;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::cuda_ffi::{
    check_cu, compile_cuda_to_ptx_nvrtc, CUdeviceptr, CUfunction, CUmodule, CUstream,
    CapturedGraphExec, CudaContextHandle, DeviceBuf, PinnedHostBuf,
};
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

struct CtrlGpu {
    down: DeviceBuf,
    gain: DeviceBuf,
    forget: DeviceBuf,
}

enum ExpertsGpu {
    Nvfp4 {
        w13: TierBuf,
        s13: DeviceBuf,
        a13: DeviceBuf,
        w2: TierBuf,
        s2: DeviceBuf,
        a2: DeviceBuf,
    },
    Mxfp8 {
        w13: TierBuf,
        s13: DeviceBuf,
        w2: TierBuf,
        s2: DeviceBuf,
    },
}

struct LayerGpu {
    global: bool,
    attn_query: Option<DeviceBuf>,
    mlp_query: DeviceBuf,
    norms: [DeviceBuf; 3],
    /// Fused `[wq (3072); wr (384); gate_attn (3072); sink (24)]` (`6480 x 2304` BF16)
    wq_fused: DeviceBuf,
    /// Fused `[wk (1024); k_dyn (32 if global else 0)]`
    wk_fused: DeviceBuf,
    /// Fused `[wv (1024); v_dyn (32 if global else 0)]`
    wv_fused: DeviceBuf,
    wo: DeviceBuf,
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
    ctrl: Option<[CtrlGpu; 4]>,
    msg_query: Option<DeviceBuf>,
    experts: ExpertsGpu,
}

struct SlotStateGpu {
    kv_cap: usize,
    k_cache: DeviceBuf,
    v_cache: DeviceBuf,
    hist_k: DeviceBuf,
    hist_v: DeviceBuf,
    hist_attn: DeviceBuf,
    hist_mlp: DeviceBuf,
}

struct KernelFns {
    _module: CUmodule,
    embed_rmsnorm: CUfunction,
    rmsnorm_d: CUfunction,
    depth_mix: CUfunction,
    ctrl_down: CUfunction,
    ctrl_apply: CUfunction,
    bf16_gemv: CUfunction,
    short_conv4: CUfunction,
    prep_qkv_and_cache: CUfunction,
    gqa_rel_sink_attn: CUfunction,
    router_topk8: CUfunction,
    moe_nvfp4_up_situ: CUfunction,
    moe_nvfp4_down: CUfunction,
    moe_mxfp8_up_situ: CUfunction,
    moe_mxfp8_down: CUfunction,
    copy_d: CUfunction,
    argmax_and_inc_pos: CUfunction,
}

unsafe impl Send for KernelFns {}
unsafe impl Sync for KernelFns {}

pub struct Inkling2GpuEngine {
    pub ctx: Arc<CudaContextHandle>,
    fns: KernelFns,
    embed: TierBuf,
    embed_norm: DeviceBuf,
    final_norm: DeviceBuf,
    unembed: DeviceBuf,
    output_query: DeviceBuf,
    layers: Vec<LayerGpu>,
    slots: Vec<SlotStateGpu>,
    pub resident_expert_layers: usize,
    pub spilled_expert_layers: usize,
    pub max_ctx: usize,
    // Persistent device scratch buffers (static addresses for CUDA Graph capture!)
    d_token_in: DeviceBuf,
    d_token_out: DeviceBuf,
    d_pos: DeviceBuf,
    d_cur: DeviceBuf,
    d_prelude_x: DeviceBuf,
    d_blocks: DeviceBuf,         // [6, D]
    d_prelude_block0: DeviceBuf, // [D]
    d_bank: DeviceBuf,           // [2, D]
    d_msgs: DeviceBuf,           // [4, D]
    d_inputs: DeviceBuf,         // [3, D]
    d_ctrl_g: DeviceBuf,         // [64]
    d_qrgs: DeviceBuf,           // [6480] = q(3072) + r(384) + gate(3072) + sink(24)
    d_k_raw: DeviceBuf,          // [1056] = k(1024) + k_dyn(32)
    d_v_raw: DeviceBuf,          // [1056] = v(1024) + v_dyn(32)
    d_k_conv: DeviceBuf,         // [1024]
    d_v_conv: DeviceBuf,         // [1024]
    d_attn_out: DeviceBuf,       // [3072]
    d_wo_out: DeviceBuf,         // [2304]
    d_attn_sconv: DeviceBuf,     // [2304]
    d_mlp_h: DeviceBuf,          // [2304]
    d_router_logits: DeviceBuf,  // [128]
    d_sel_ids: DeviceBuf,        // [8]
    d_sel_wts: DeviceBuf,        // [8]
    d_moe_act: DeviceBuf,        // [8 * 1152]
    d_moe_y: DeviceBuf,          // [2304]
    d_logits: DeviceBuf,         // [200058]
    host_pos: usize,
    decode_graph: Option<CapturedGraphExec>,
    pub enable_cuda_graph: bool,
}

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

impl Inkling2GpuEngine {
    pub fn load_from_dir(
        model_dir: impl AsRef<Path>,
        max_ctx: usize,
        enable_cuda_graph: bool,
    ) -> Result<Self, String> {
        let t_load = Instant::now();
        let ctx = CudaContextHandle::init(0)?;
        ctx.bind_current()?;

        // Compile `inkling2_sm120.cu` via NVRTC (`compute_120`) or load `.cubin` if set
        let module_image = if let Ok(cubin_path) = std::env::var("LEANSONIC_CUBIN_PATH") {
            std::fs::read(&cubin_path).map_err(|e| format!("read {cubin_path}: {e}"))?
        } else {
            compile_cuda_to_ptx_nvrtc(CUDA_KERNEL_SRC, "compute_120")?
        };

        let fns = Self::load_module_functions(&ctx, &module_image)?;
        let st_path = model_dir.as_ref().join("model-00001-of-00001.safetensors");
        let st = MappedSafeTensors::open(&st_path)
            .map_err(|e| format!("open {}: {e}", st_path.display()))?;

        // Embedding stays in Pinned Mapped Host RAM (`926 MB` -> saves ~1 GB VRAM while
        // single-token lookup only reads `4.6 KB` over PCIe in < 0.2 us!)
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
        // Only first `VOCAB = 200,058` rows of `unembed` are scored
        let unembed_all = st
            .require("model.llm.unembed.weight", "BF16", &[PADDED_VOCAB, D])
            .data;
        let unembed = DeviceBuf::from_host_bytes(&ctx, &unembed_all[..VOCAB * D * 2])?;
        let output_query = DeviceBuf::from_host_bytes(
            &ctx,
            st.require("model.llm.attn_res.output_query", "F32", &[D])
                .data,
        )?;

        // Allocate KV & short-conv state pools FIRST so remaining VRAM budget is exact!
        let mut slots = Vec::with_capacity(N_SLOT);
        for s_idx in 0..N_SLOT {
            let il = if s_idx < N_LAYER {
                s_idx
            } else {
                6 + ((s_idx - N_LAYER) % 12)
            };
            let is_global = GLOBAL_LAYERS.contains(&il);
            let kv_cap = if is_global { max_ctx } else { EXTENT + 64 };
            slots.push(SlotStateGpu {
                kv_cap,
                k_cache: DeviceBuf::alloc_zeroed(&ctx, kv_cap * N_KV * HD * 2)?,
                v_cache: DeviceBuf::alloc_zeroed(&ctx, kv_cap * N_KV * HD * 2)?,
                hist_k: DeviceBuf::alloc_zeroed(&ctx, N_KV * HD * 3 * 4)?,
                hist_v: DeviceBuf::alloc_zeroed(&ctx, N_KV * HD * 3 * 4)?,
                hist_attn: DeviceBuf::alloc_zeroed(&ctx, D * 3 * 4)?,
                hist_mlp: DeviceBuf::alloc_zeroed(&ctx, D * 3 * 4)?,
            });
        }

        // Determine Visit-Multiplicity-Weighted VRAM Residency Plan!
        // Priority order: Loop Core `6..18` (visited 3x/token!) -> Prelude `0..6` -> Coda `18..24`!
        let (free_vram, total_vram) = ctx.mem_info()?;
        let safety_reserve = 900 * 1024 * 1024; // 900 MiB reserve for dense + scratch + driver
        let dense_est = 1520 * 1024 * 1024; // ~1.48 GiB across 24 layers' dense weights
        let mut rem_expert_vram = free_vram.saturating_sub(safety_reserve + dense_est);

        let mut priority_order: Vec<usize> = (6..18).collect();
        priority_order.extend(0..6);
        priority_order.extend(18..24);

        let mut layer_in_vram = [false; N_LAYER];
        for &il in &priority_order {
            let exp_bytes = if il < 20 {
                N_EXP * (2 * FF * (D / 2) + D * (FF / 2))
            } else {
                N_EXP * (2 * FF * D + D * FF)
            };
            if rem_expert_vram >= exp_bytes {
                layer_in_vram[il] = true;
                rem_expert_vram -= exp_bytes;
            }
        }

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
        let mut resident_expert_layers = 0usize;
        let mut spilled_expert_layers = 0usize;

        for (il, &use_vram) in layer_in_vram.iter().enumerate() {
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

            let norms = if global {
                [
                    DeviceBuf::from_host_bytes(
                        &ctx,
                        st.require(&format!("{pfx}q_input_norm.weight"), "BF16", &[D])
                            .data,
                    )?,
                    DeviceBuf::from_host_bytes(
                        &ctx,
                        st.require(&format!("{pfx}k_input_norm.weight"), "BF16", &[D])
                            .data,
                    )?,
                    DeviceBuf::from_host_bytes(
                        &ctx,
                        st.require(&format!("{pfx}v_input_norm.weight"), "BF16", &[D])
                            .data,
                    )?,
                ]
            } else {
                let raw = st
                    .require(&format!("{pfx}attn_norm.weight"), "BF16", &[D])
                    .data;
                [
                    DeviceBuf::from_host_bytes(&ctx, raw)?,
                    DeviceBuf::from_host_bytes(&ctx, raw)?,
                    DeviceBuf::from_host_bytes(&ctx, raw)?,
                ]
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
            let wq_fused =
                DeviceBuf::from_host_bytes(&ctx, &concat_slices(&[wq, wr, gate_attn, sink]))?;

            let wk = st
                .require(&format!("{pfx}attn.wk_dv.weight"), "BF16", &[N_KV * HD, D])
                .data;
            let wv = st
                .require(&format!("{pfx}attn.wv_dv.weight"), "BF16", &[N_KV * HD, D])
                .data;
            let (wk_fused, wv_fused) = if global {
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
                    DeviceBuf::from_host_bytes(&ctx, &concat_slices(&[wk, kd]))?,
                    DeviceBuf::from_host_bytes(&ctx, &concat_slices(&[wv, vd]))?,
                )
            } else {
                (
                    DeviceBuf::from_host_bytes(&ctx, wk)?,
                    DeviceBuf::from_host_bytes(&ctx, wv)?,
                )
            };

            let wo = DeviceBuf::from_host_bytes(
                &ctx,
                st.require(
                    &format!("{pfx}attn.wo_ud.weight"),
                    "BF16",
                    &[D, N_HEAD * HD],
                )
                .data,
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

            let load_ctrl = |cname: &str| -> Result<CtrlGpu, String> {
                Ok(CtrlGpu {
                    down: DeviceBuf::from_host_bytes(
                        &ctx,
                        st.require(
                            &format!("{pfx}controllers.{cname}.down_weight"),
                            "BF16",
                            &[RANK, 2 * D],
                        )
                        .data,
                    )?,
                    gain: DeviceBuf::from_host_bytes(
                        &ctx,
                        st.require(
                            &format!("{pfx}controllers.{cname}.gain_weight"),
                            "BF16",
                            &[D, RANK],
                        )
                        .data,
                    )?,
                    forget: DeviceBuf::from_host_bytes(
                        &ctx,
                        st.require(
                            &format!("{pfx}controllers.{cname}.forget_weight"),
                            "BF16",
                            &[D, RANK],
                        )
                        .data,
                    )?,
                })
            };

            let ctrl = if in_core {
                Some([
                    load_ctrl("attn")?,
                    load_ctrl("k")?,
                    load_ctrl("v")?,
                    load_ctrl("mlp")?,
                ])
            } else {
                None
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

            if use_vram {
                resident_expert_layers += 1;
            } else {
                spilled_expert_layers += 1;
            }
            let place = |bytes: &[u8]| -> Result<TierBuf, String> {
                if use_vram {
                    Ok(TierBuf::Vram(DeviceBuf::from_host_bytes(&ctx, bytes)?))
                } else {
                    Ok(TierBuf::HostPinned(PinnedHostBuf::from_bytes(&ctx, bytes)?))
                }
            };

            let experts = if il < 20 {
                ExpertsGpu::Nvfp4 {
                    w13: place(
                        st.require(
                            &format!("{pfx}mlp.experts.w13_weight"),
                            "U8",
                            &[N_EXP, 2 * FF, D / 2],
                        )
                        .data,
                    )?,
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
                    w2: place(
                        st.require(
                            &format!("{pfx}mlp.experts.w2_weight"),
                            "U8",
                            &[N_EXP, D, FF / 2],
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
                ExpertsGpu::Mxfp8 {
                    w13: place(
                        st.require(
                            &format!("{pfx}mlp.experts.w13_weight"),
                            "F8_E4M3",
                            &[N_EXP, 2 * FF, D],
                        )
                        .data,
                    )?,
                    s13: DeviceBuf::from_host_bytes(
                        &ctx,
                        st.require(
                            &format!("{pfx}mlp.experts.w13_weight_scale"),
                            "U8",
                            &[N_EXP, 2 * FF, D / 32],
                        )
                        .data,
                    )?,
                    w2: place(
                        st.require(
                            &format!("{pfx}mlp.experts.w2_weight"),
                            "F8_E4M3",
                            &[N_EXP, D, FF],
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
                norms,
                wq_fused,
                wk_fused,
                wv_fused,
                wo,
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
                ctrl,
                msg_query,
                experts,
            });
        }

        let (free_after, _) = ctx.mem_info()?;
        eprintln!(
            "[LeanSonic GPU] {} | VRAM used: {:.2} GiB / {:.2} GiB | Resident expert layers: {}/24 (Spilled: {}) | Loaded in {:.2}s",
            ctx.device_name,
            (total_vram - free_after) as f64 / (1024.0 * 1024.0 * 1024.0),
            total_vram as f64 / (1024.0 * 1024.0 * 1024.0),
            resident_expert_layers,
            spilled_expert_layers,
            t_load.elapsed().as_secs_f64()
        );

        let mut eng = Self {
            d_token_in: DeviceBuf::alloc_zeroed(&ctx, 4)?,
            d_token_out: DeviceBuf::alloc_zeroed(&ctx, 4)?,
            d_pos: DeviceBuf::alloc_zeroed(&ctx, 4)?,
            d_cur: DeviceBuf::alloc_zeroed(&ctx, D * 4)?,
            d_prelude_x: DeviceBuf::alloc_zeroed(&ctx, D * 4)?,
            d_blocks: DeviceBuf::alloc_zeroed(&ctx, 6 * D * 4)?,
            d_prelude_block0: DeviceBuf::alloc_zeroed(&ctx, D * 4)?,
            d_bank: DeviceBuf::alloc_zeroed(&ctx, 2 * D * 4)?,
            d_msgs: DeviceBuf::alloc_zeroed(&ctx, 4 * D * 4)?,
            d_inputs: DeviceBuf::alloc_zeroed(&ctx, 3 * D * 4)?,
            d_ctrl_g: DeviceBuf::alloc_zeroed(&ctx, RANK * 4)?,
            d_qrgs: DeviceBuf::alloc_zeroed(&ctx, 6480 * 4)?,
            d_k_raw: DeviceBuf::alloc_zeroed(&ctx, 1056 * 4)?,
            d_v_raw: DeviceBuf::alloc_zeroed(&ctx, 1056 * 4)?,
            d_k_conv: DeviceBuf::alloc_zeroed(&ctx, 1024 * 4)?,
            d_v_conv: DeviceBuf::alloc_zeroed(&ctx, 1024 * 4)?,
            d_attn_out: DeviceBuf::alloc_zeroed(&ctx, 3072 * 4)?,
            d_wo_out: DeviceBuf::alloc_zeroed(&ctx, D * 4)?,
            d_attn_sconv: DeviceBuf::alloc_zeroed(&ctx, D * 4)?,
            d_mlp_h: DeviceBuf::alloc_zeroed(&ctx, D * 4)?,
            d_router_logits: DeviceBuf::alloc_zeroed(&ctx, N_EXP * 4)?,
            d_sel_ids: DeviceBuf::alloc_zeroed(&ctx, TOP_K * 4)?,
            d_sel_wts: DeviceBuf::alloc_zeroed(&ctx, TOP_K * 4)?,
            d_moe_act: DeviceBuf::alloc_zeroed(&ctx, TOP_K * FF * 4)?,
            d_moe_y: DeviceBuf::alloc_zeroed(&ctx, D * 4)?,
            d_logits: DeviceBuf::alloc_zeroed(&ctx, VOCAB * 4)?,
            ctx,
            fns,
            embed,
            embed_norm,
            final_norm,
            unembed,
            output_query,
            layers,
            slots,
            resident_expert_layers,
            spilled_expert_layers,
            max_ctx,
            host_pos: 0,
            decode_graph: None,
            enable_cuda_graph,
        };

        if enable_cuda_graph {
            let ctx_clone = Arc::clone(&eng.ctx);
            let graph = CapturedGraphExec::capture(&ctx_clone, |stream| eng.enqueue_step(stream))?;
            eng.decode_graph = Some(graph);
            // Reset any state touched during graph capture!
            eng.reset()?;
        }

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
            depth_mix: get_fn("ink2_depth_mix")?,
            ctrl_down: get_fn("ink2_ctrl_down")?,
            ctrl_apply: get_fn("ink2_ctrl_apply")?,
            bf16_gemv: get_fn("ink2_bf16_gemv")?,
            short_conv4: get_fn("ink2_short_conv4")?,
            prep_qkv_and_cache: get_fn("ink2_prep_qkv_and_cache")?,
            gqa_rel_sink_attn: get_fn("ink2_gqa_rel_sink_attn")?,
            router_topk8: get_fn("ink2_router_topk8")?,
            moe_nvfp4_up_situ: get_fn("ink2_moe_nvfp4_up_situ")?,
            moe_nvfp4_down: get_fn("ink2_moe_nvfp4_down")?,
            moe_mxfp8_up_situ: get_fn("ink2_moe_mxfp8_up_situ")?,
            moe_mxfp8_down: get_fn("ink2_moe_mxfp8_down")?,
            copy_d: get_fn("ink2_copy_d")?,
            argmax_and_inc_pos: get_fn("ink2_argmax_and_inc_pos")?,
        })
    }

    pub fn reset(&mut self) -> Result<(), String> {
        self.ctx.bind_current()?;
        self.ctx.sync_compute()?;
        self.host_pos = 0;
        unsafe {
            check_cu(
                (self.ctx.api.cuMemsetD8_v2)(self.d_pos.ptr, 0, 4),
                "zero pos",
            )?;
            for s in &self.slots {
                check_cu(
                    (self.ctx.api.cuMemsetD8_v2)(s.hist_k.ptr, 0, s.hist_k.byte_len),
                    "zero hk",
                )?;
                check_cu(
                    (self.ctx.api.cuMemsetD8_v2)(s.hist_v.ptr, 0, s.hist_v.byte_len),
                    "zero hv",
                )?;
                check_cu(
                    (self.ctx.api.cuMemsetD8_v2)(s.hist_attn.ptr, 0, s.hist_attn.byte_len),
                    "zero ha",
                )?;
                check_cu(
                    (self.ctx.api.cuMemsetD8_v2)(s.hist_mlp.ptr, 0, s.hist_mlp.byte_len),
                    "zero hm",
                )?;
            }
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

    fn enqueue_ctrl(
        &self,
        stream: CUstream,
        cur_ptr: CUdeviceptr,
        msg_ptr: CUdeviceptr,
        ctrl: &CtrlGpu,
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
        self.launch(stream, self.fns.ctrl_down, (64, 1), 256, &mut args1)?;

        let mut p_gain = ctrl.gain.ptr;
        let mut p_forget = ctrl.forget.ptr;
        let mut args2: [*mut c_void; 5] = [
            (&mut p_cur as *mut _) as *mut c_void,
            (&mut p_msg as *mut _) as *mut c_void,
            (&mut p_g as *mut _) as *mut c_void,
            (&mut p_gain as *mut _) as *mut c_void,
            (&mut p_forget as *mut _) as *mut c_void,
        ];
        self.launch(stream, self.fns.ctrl_apply, (9, 1), 256, &mut args2)
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

        // 1. Attention inputs `[xq, xk, xv]` via depthwise `attn_query` mix
        if let Some(ref aq) = l.attn_query {
            let n_q: u32 = if l.global { 3 } else { 1 };
            let mut pb = self.d_blocks.ptr;
            let mut nb = *n_blocks as i32;
            let mut pc = self.d_cur.ptr;
            let mut ac = 1i32;
            let mut pq = aq.ptr;
            let mut pout = self.d_inputs.ptr;
            let mut args: [*mut c_void; 6] = [
                (&mut pb as *mut _) as *mut c_void,
                (&mut nb as *mut _) as *mut c_void,
                (&mut pc as *mut _) as *mut c_void,
                (&mut ac as *mut _) as *mut c_void,
                (&mut pq as *mut _) as *mut c_void,
                (&mut pout as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.depth_mix, (n_q, 1), 256, &mut args)?;
            if !l.global && has_msgs {
                self.enqueue_copy_d(stream, self.d_inputs.ptr, self.d_inputs.ptr + d_bytes)?;
                self.enqueue_copy_d(stream, self.d_inputs.ptr, self.d_inputs.ptr + 2 * d_bytes)?;
            }
        } else {
            self.enqueue_copy_d(stream, self.d_cur.ptr, self.d_inputs.ptr)?;
            if l.global || has_msgs {
                self.enqueue_copy_d(stream, self.d_cur.ptr, self.d_inputs.ptr + d_bytes)?;
                self.enqueue_copy_d(stream, self.d_cur.ptr, self.d_inputs.ptr + 2 * d_bytes)?;
            }
        }

        // 2. Loop Core Controller (`pass > 0`) on `inputs[0..3]`
        if has_msgs {
            let ctrls = l.ctrl.as_ref().unwrap();
            for j in 0..3u64 {
                self.enqueue_ctrl(
                    stream,
                    self.d_inputs.ptr + j * d_bytes,
                    self.d_msgs.ptr + j * d_bytes,
                    &ctrls[j as usize],
                )?;
            }
        }

        // 3. Input RMSNorms (`norms[0..3]` if `global || has_msgs`, else just `norms[0]`)
        let distinct_k_v = l.global || has_msgs;
        let norm_count = if distinct_k_v { 3 } else { 1 };
        for j in 0..norm_count {
            let mut pin = self.d_inputs.ptr + (j as u64) * d_bytes;
            let mut pg = l.norms[j].ptr;
            let mut pout = pin;
            let mut hg = 1i32;
            let mut args: [*mut c_void; 4] = [
                (&mut pin as *mut _) as *mut c_void,
                (&mut pg as *mut _) as *mut c_void,
                (&mut pout as *mut _) as *mut c_void,
                (&mut hg as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.rmsnorm_d, (1, 1), 256, &mut args)?;
        }

        let xq_ptr = self.d_inputs.ptr;
        let xk_ptr = if distinct_k_v {
            xq_ptr + d_bytes
        } else {
            xq_ptr
        };
        let xv_ptr = if distinct_k_v {
            xq_ptr + 2 * d_bytes
        } else {
            xq_ptr
        };

        // 4. Fused `[wq; wr; gate_attn; sink]` (`6480 x 2304`) + `wk` + `wv` GEMVs
        self.enqueue_bf16_gemv(
            stream,
            l.wq_fused.ptr,
            xq_ptr,
            self.d_qrgs.ptr,
            6480,
            D as i32,
            0.0,
        )?;
        let kv_rows = if l.global { 1056 } else { 1024 };
        self.enqueue_bf16_gemv(
            stream,
            l.wk_fused.ptr,
            xk_ptr,
            self.d_k_raw.ptr,
            kv_rows,
            D as i32,
            0.0,
        )?;
        self.enqueue_bf16_gemv(
            stream,
            l.wv_fused.ptr,
            xv_ptr,
            self.d_v_raw.ptr,
            kv_rows,
            D as i32,
            0.0,
        )?;

        // 5. K & V Short-Conv4
        for (raw_ptr, conv_w, hist_ptr, dst_ptr) in [
            (
                self.d_k_raw.ptr,
                l.k_conv4.ptr,
                st.hist_k.ptr,
                self.d_k_conv.ptr,
            ),
            (
                self.d_v_raw.ptr,
                l.v_conv4.ptr,
                st.hist_v.ptr,
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

        // 7. GQA Relative-Bias + Sink Attention (`24` Q-heads)
        {
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
            self.launch(stream, self.fns.gqa_rel_sink_attn, (24, 1), 128, &mut args)?;
        }

        // 8. `wo` Projection + `attn_sconv` + Residual Block Push/Add
        self.enqueue_bf16_gemv(
            stream,
            l.wo.ptr,
            self.d_attn_out.ptr,
            self.d_wo_out.ptr,
            D as i32,
            3072,
            0.0,
        )?;
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
            let mut p_hist = st.hist_attn.ptr;
            let mut p_out = self.d_attn_sconv.ptr;
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
            self.enqueue_copy_d(stream, self.d_attn_sconv.ptr, self.d_cur.ptr)?;
        }

        // 9. MLP Depth Mix + Loop Controller 3 + `mlp_norm`
        {
            let mut pb = self.d_blocks.ptr;
            let mut nb = *n_blocks as i32;
            let mut pc = self.d_cur.ptr;
            let mut ac = 1i32;
            let mut pq = l.mlp_query.ptr;
            let mut pout = self.d_mlp_h.ptr;
            let mut args: [*mut c_void; 6] = [
                (&mut pb as *mut _) as *mut c_void,
                (&mut nb as *mut _) as *mut c_void,
                (&mut pc as *mut _) as *mut c_void,
                (&mut ac as *mut _) as *mut c_void,
                (&mut pq as *mut _) as *mut c_void,
                (&mut pout as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.depth_mix, (1, 1), 256, &mut args)?;
        }
        if has_msgs {
            let ctrls = l.ctrl.as_ref().unwrap();
            self.enqueue_ctrl(
                stream,
                self.d_mlp_h.ptr,
                self.d_msgs.ptr + 3 * d_bytes,
                &ctrls[3],
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

        // 10. Router + Full-GPU MoE (`NVFP4` for `0..19`, `MXFP8` for `20..23`)
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
            self.launch(stream, self.fns.router_topk8, (1, 1), 128, &mut args)?;
        }

        match &l.experts {
            ExpertsGpu::Nvfp4 {
                w13,
                s13,
                a13,
                w2,
                s2,
                a2,
            } => {
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
            ExpertsGpu::Mxfp8 { w13, s13, w2, s2 } => {
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

        // 11. `mlp_sconv` + Residual Add into `d_cur`
        {
            let mut p_cur = self.d_moe_y.ptr;
            let mut p_w = l.mlp_conv4.ptr;
            let mut p_dyn: CUdeviceptr = 0;
            let mut p_hist = st.hist_mlp.ptr;
            let mut p_out = self.d_attn_sconv.ptr;
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
            self.enqueue_copy_d(stream, self.d_attn_sconv.ptr, self.d_cur.ptr)?;
        }

        Ok(())
    }

    /// Enqueue the full 48-visit `Inkling-2-Mini` forward step on `stream`!
    fn enqueue_step(&self, stream: CUstream) -> Result<(), String> {
        let d_bytes = (D * 4) as u64;

        // 0. Embed + `embed_norm`
        {
            let mut pe = self.embed.ptr();
            let mut pn = self.embed_norm.ptr;
            let mut pt = self.d_token_in.ptr;
            let mut px = self.d_cur.ptr;
            let mut args: [*mut c_void; 4] = [
                (&mut pe as *mut _) as *mut c_void,
                (&mut pn as *mut _) as *mut c_void,
                (&mut pt as *mut _) as *mut c_void,
                (&mut px as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.embed_rmsnorm, (1, 1), 256, &mut args)?;
        }

        let mut n_blocks = 0usize;
        // 1. Prelude (`il = 0..6`)
        for il in 0..6 {
            self.enqueue_visit(stream, il, il, &mut n_blocks, false)?;
        }
        debug_assert_eq!(n_blocks, 1);
        self.enqueue_copy_d(stream, self.d_cur.ptr, self.d_prelude_x.ptr)?;
        self.enqueue_copy_d(stream, self.d_blocks.ptr, self.d_prelude_block0.ptr)?;

        // 2. Loop Core (`pass = 0..3`, `ip = 6..18`)
        for pass in 0..3 {
            if pass > 0 {
                // `bank = [blocks[2], cur]`; `cur = prelude_x`; `blocks = [prelude_block0]`
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
                    let mut args: [*mut c_void; 6] = [
                        (&mut pb as *mut _) as *mut c_void,
                        (&mut nb as *mut _) as *mut c_void,
                        (&mut pc as *mut _) as *mut c_void,
                        (&mut ac as *mut _) as *mut c_void,
                        (&mut pq as *mut _) as *mut c_void,
                        (&mut pout as *mut _) as *mut c_void,
                    ];
                    self.launch(stream, self.fns.depth_mix, (4, 1), 256, &mut args)?;
                }
                self.enqueue_visit(stream, ip, slot, &mut n_blocks, pass > 0)?;
            }
        }

        // 3. Coda (`il = 18..24`)
        for il in 18..24 {
            self.enqueue_visit(stream, il, il, &mut n_blocks, false)?;
        }
        debug_assert_eq!(n_blocks, 4);

        // 4. Output Depth Mix (`5` states = `blocks[0..4] + cur`) + `final_norm` + `unembed` (softcap 30) + ArgMax
        {
            let mut pb = self.d_blocks.ptr;
            let mut nb = n_blocks as i32;
            let mut pc = self.d_cur.ptr;
            let mut ac = 1i32;
            let mut pq = self.output_query.ptr;
            let mut pout = self.d_mlp_h.ptr;
            let mut args: [*mut c_void; 6] = [
                (&mut pb as *mut _) as *mut c_void,
                (&mut nb as *mut _) as *mut c_void,
                (&mut pc as *mut _) as *mut c_void,
                (&mut ac as *mut _) as *mut c_void,
                (&mut pq as *mut _) as *mut c_void,
                (&mut pout as *mut _) as *mut c_void,
            ];
            self.launch(stream, self.fns.depth_mix, (1, 1), 256, &mut args)?;
        }
        {
            let mut pin = self.d_mlp_h.ptr;
            let mut pg = self.final_norm.ptr;
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
        self.enqueue_bf16_gemv(
            stream,
            self.unembed.ptr,
            self.d_mlp_h.ptr,
            self.d_logits.ptr,
            VOCAB as i32,
            D as i32,
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

    /// Step one token and return the greedy next token ID (`0` host-side logit transfer!).
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
            self.enqueue_step(self.ctx.compute_stream)?;
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

    /// Read full `[VOCAB = 200,058]` logits of the last executed step (for diagnostic parity audit).
    pub fn read_last_logits(&self) -> Result<Vec<f32>, String> {
        self.ctx.sync_compute()?;
        let mut host = vec![0.0f32; VOCAB];
        self.d_logits.copy_to_host_f32(&mut host)?;
        Ok(host)
    }

    /// Prefill a prompt token slice and return `(greedy_first_token, prefill_ms)`.
    pub fn prefill_tokens(&mut self, tokens: &[u32]) -> Result<(u32, f64), String> {
        assert!(!tokens.is_empty());
        let t0 = Instant::now();
        let mut next_tok = 0u32;
        for &t in tokens {
            next_tok = self.forward_greedy_token(t)?;
        }
        Ok((next_tok, t0.elapsed().as_secs_f64() * 1000.0))
    }
}
