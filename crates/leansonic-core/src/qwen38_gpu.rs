//! Real `Qwen3.8-Flash-Next` / `Qwen4Exp` (`106 GB` 22-Shard `safetensors`) Full-GPU Blackwell Engine.
//!
//! Architecture (`48` layers, `D = 2560`, `hc_count = 4`, `512` experts `Top-10` + Shared Expert):
//! - Group-32 4-bit affine (`uint4` `LDG.E.128` = 32 nibbles + BF16 scale/bias) across embeddings,
//!   projections, Hyper-Connection mixers, `lm_head` (`248,320 × 2,560`), and `512`-expert MoE.
//! - Layer 1 (`ple_layer_ids = [2]`) `29.80 GiB` (`320M` rows × `160` dims) 3-Gram Host `Engram`/`PLE`
//!   table mapped zero-copy via POSIX `mmap`, dequantized (`16` heads = `10 KiB`) into a 256-slot
//!   pinned host ring, and streamed via `cuMemcpyHtoDAsync_v2`.
//! - 3-Tier `512`-Expert `switch_mlp` placement (`1,500 MiB`/layer = `70.3 GiB` total) + 48-Layer
//!   `CUDA Graph` capture (`decode_graph` + `prefill_graph`, eliminating `912` kernel launches/tok).

use core::ffi::c_void;
use std::ffi::CString;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::cuda_ffi::{
    check_cu, compile_cuda_to_ptx_nvrtc, CUdeviceptr, CUevent, CUfunction, CUmodule, CUstream,
    CapturedGraphExec, CudaContextHandle, DeviceBuf, PinnedHostBuf, CU_EVENT_DISABLE_TIMING,
};
use crate::safetensors::ShardedSafeTensors;

const CUDA_HEADER_SRC: &str = include_str!("../../../kernels/include/leansonic_verified.cuh");
const CUDA_QWEN38_SRC: &str = include_str!("../../../kernels/qwen38_sm120.cu");

pub const Q38_D: usize = 2560;
pub const Q38_HC: usize = 4;
pub const Q38_HC_D: usize = 10_240;
pub const Q38_HC_LOW: usize = 320;
pub const Q38_LAYERS: usize = 48;
pub const Q38_VOCAB: usize = 248_320;
pub const Q38_EXP: usize = 512;
pub const Q38_TOP_K: usize = 10;
pub const Q38_MOE_FF: usize = 640;
pub const Q38_PLE_SHARDS: usize = 128;
pub const Q38_PLE_SHARD_ROWS: i64 = 2_500_012;
pub const Q38_PLE_EOS_ID: i64 = 248_044;
const PLE_RING_SLOTS: usize = 256;

// Byte offsets inside a single `1,572,864,000`-byte (`1,500 MiB`) `switch_mlp` layer slab
const SW_W_BYTES: usize = 419_430_400; // 400 MiB
const SW_SB_BYTES: usize = 52_428_800; //  50 MiB
const SW_PROJ_BYTES: usize = SW_W_BYTES + 2 * SW_SB_BYTES; // 500 MiB
const SW_LAYER_BYTES: usize = 3 * SW_PROJ_BYTES; // 1,500 MiB

const OFF_GATE_W: usize = 0;
const OFF_GATE_SC: usize = SW_W_BYTES;
const OFF_GATE_BI: usize = SW_W_BYTES + SW_SB_BYTES;
const OFF_UP_W: usize = SW_PROJ_BYTES;
const OFF_UP_SC: usize = SW_PROJ_BYTES + SW_W_BYTES;
const OFF_UP_BI: usize = SW_PROJ_BYTES + SW_W_BYTES + SW_SB_BYTES;
const OFF_DOWN_W: usize = 2 * SW_PROJ_BYTES;
const OFF_DOWN_SC: usize = 2 * SW_PROJ_BYTES + SW_W_BYTES;
const OFF_DOWN_BI: usize = 2 * SW_PROJ_BYTES + SW_W_BYTES + SW_SB_BYTES;

// Byte offsets inside the `30,720,000`-byte (`29.30 MiB`) compact `Top-10` staging slab
const EX_W_BYTES: usize = SW_W_BYTES / Q38_EXP; // 819,200 B
const EX_SB_BYTES: usize = SW_SB_BYTES / Q38_EXP; // 102,400 B
const COMPACT_W_BYTES: usize = Q38_TOP_K * EX_W_BYTES;
const COMPACT_SB_BYTES: usize = Q38_TOP_K * EX_SB_BYTES;
const COMPACT_PROJ_BYTES: usize = COMPACT_W_BYTES + 2 * COMPACT_SB_BYTES;
const COMPACT_TOTAL_BYTES: usize = 3 * COMPACT_PROJ_BYTES;

const COFF_GATE_W: usize = 0;
const COFF_GATE_SC: usize = COMPACT_W_BYTES;
const COFF_GATE_BI: usize = COMPACT_W_BYTES + COMPACT_SB_BYTES;
const COFF_UP_W: usize = COMPACT_PROJ_BYTES;
const COFF_UP_SC: usize = COMPACT_PROJ_BYTES + COMPACT_W_BYTES;
const COFF_UP_BI: usize = COMPACT_PROJ_BYTES + COMPACT_W_BYTES + COMPACT_SB_BYTES;
const COFF_DOWN_W: usize = 2 * COMPACT_PROJ_BYTES;
const COFF_DOWN_SC: usize = 2 * COMPACT_PROJ_BYTES + COMPACT_W_BYTES;
const COFF_DOWN_BI: usize = 2 * COMPACT_PROJ_BYTES + COMPACT_W_BYTES + COMPACT_SB_BYTES;

macro_rules! kargs {
    ($($arg:ident),+ $(,)?) => {
        [ $(core::ptr::from_mut(&mut $arg).cast::<c_void>()),+ ]
    };
}

#[inline]
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn concat_bytes(slices: &[&[u8]]) -> Vec<u8> {
    let total: usize = slices.iter().map(|s| s.len()).sum();
    let mut out = Vec::with_capacity(total);
    for s in slices {
        out.extend_from_slice(s);
    }
    out
}

fn f32_slice_bytes(v: &[f32]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(v.as_ptr().cast::<u8>(), core::mem::size_of_val(v)) }
}

fn parallel_copy_into_pinned(dst_base: *mut u8, slices: &[(usize, &[u8])]) {
    let base_addr = dst_base as usize;
    std::thread::scope(|s| {
        for &(off, bytes) in slices {
            // Split 400 MiB `.weight` slabs across 4 parallel memcpy tasks, 50 MiB `.scales/.biases` as 1 task
            let n_chunks = if bytes.len() >= 200_000_000 { 4 } else { 1 };
            let chunk_sz = bytes.len().div_ceil(n_chunks);
            for c in 0..n_chunks {
                let start = c * chunk_sz;
                if start >= bytes.len() {
                    break;
                }
                let end = (start + chunk_sz).min(bytes.len());
                let sub = &bytes[start..end];
                let dst_addr = base_addr + off + start;
                s.spawn(move || unsafe {
                    core::ptr::copy_nonoverlapping(sub.as_ptr(), dst_addr as *mut u8, sub.len());
                });
            }
        }
    });
}

fn available_host_ram_bytes() -> usize {
    if let Ok(txt) = std::fs::read_to_string("/proc/meminfo") {
        for line in txt.lines() {
            if let Some(rest) = line.strip_prefix("MemAvailable:") {
                if let Some(kb_str) = rest.split_whitespace().next() {
                    if let Ok(kb) = kb_str.parse::<usize>() {
                        return kb.saturating_mul(1024);
                    }
                }
            }
        }
    }
    0
}

struct Q4LinearGpu {
    w: DeviceBuf,
    sc: DeviceBuf,
    bi: DeviceBuf,
    rows: i32,
    cols: i32,
}

impl Q4LinearGpu {
    fn from_sharded(
        ctx: &Arc<CudaContextHandle>,
        st: &ShardedSafeTensors,
        prefix: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self, String> {
        let w_tv = st.require(&format!("{prefix}.weight"), "U32", &[rows, cols / 8]);
        let sc_tv = st.require(&format!("{prefix}.scales"), "BF16", &[rows, cols / 32]);
        let bi_tv = st.require(&format!("{prefix}.biases"), "BF16", &[rows, cols / 32]);
        Ok(Self {
            w: DeviceBuf::from_host_bytes(ctx, w_tv.data)?,
            sc: DeviceBuf::from_host_bytes(ctx, sc_tv.data)?,
            bi: DeviceBuf::from_host_bytes(ctx, bi_tv.data)?,
            rows: rows as i32,
            cols: cols as i32,
        })
    }

    fn from_sharded_concat(
        ctx: &Arc<CudaContextHandle>,
        st: &ShardedSafeTensors,
        items: &[(&str, usize)],
        cols: usize,
    ) -> Result<Self, String> {
        let mut w_parts = Vec::with_capacity(items.len());
        let mut sc_parts = Vec::with_capacity(items.len());
        let mut bi_parts = Vec::with_capacity(items.len());
        let mut total_rows = 0usize;
        for &(prefix, r) in items {
            w_parts.push(
                st.require(&format!("{prefix}.weight"), "U32", &[r, cols / 8])
                    .data,
            );
            sc_parts.push(
                st.require(&format!("{prefix}.scales"), "BF16", &[r, cols / 32])
                    .data,
            );
            bi_parts.push(
                st.require(&format!("{prefix}.biases"), "BF16", &[r, cols / 32])
                    .data,
            );
            total_rows += r;
        }
        Ok(Self {
            w: DeviceBuf::from_host_bytes(ctx, &concat_bytes(&w_parts))?,
            sc: DeviceBuf::from_host_bytes(ctx, &concat_bytes(&sc_parts))?,
            bi: DeviceBuf::from_host_bytes(ctx, &concat_bytes(&bi_parts))?,
            rows: total_rows as i32,
            cols: cols as i32,
        })
    }
}

struct HyperConnGpu {
    hc_norm: DeviceBuf,
    down: Q4LinearGpu,
    up: Q4LinearGpu,
    inject: Option<Q4LinearGpu>,
}

enum AttentionBranchGpu {
    Gdn {
        in_all: Q4LinearGpu,
        out_proj: Q4LinearGpu,
        conv_w4: DeviceBuf,
        a_log: DeviceBuf,
        dt_bias: DeviceBuf,
        norm_w: DeviceBuf,
        conv_hist3: DeviceBuf,
        state_f32: DeviceBuf,
    },
    Qsa {
        qkv_idx: Q4LinearGpu,
        o_proj: Q4LinearGpu,
        q_norm_w: DeviceBuf,
        k_norm_w: DeviceBuf,
        idx_q_norm_w: DeviceBuf,
        idx_k_norm_w: DeviceBuf,
        k_cache_fp16: DeviceBuf,
        v_cache_fp16: DeviceBuf,
        idx_raw_k_cache: DeviceBuf,
        idx_block_k_cache: DeviceBuf,
    },
}

struct MixedExperts {
    _device: DeviceBuf,
    _host: PinnedHostBuf,
    bases: DeviceBuf,
}

impl MixedExperts {
    // One record contains the three projections, each in weights/scales/biases order.
    fn load(
        ctx: &Arc<CudaContextHandle>,
        chosen: &[usize],
        planes: &[(usize, &[u8]); 9],
    ) -> Result<Self, String> {
        let record_bytes = SW_LAYER_BYTES / Q38_EXP;
        if chosen.is_empty() || chosen.len() >= Q38_EXP {
            return Err("expected a nonempty proper subset of experts per layer".into());
        }
        let mut hot = [false; Q38_EXP];
        for &id in chosen {
            if id >= Q38_EXP || std::mem::replace(&mut hot[id], true) {
                return Err(format!("invalid or duplicate resident expert {id}"));
            }
        }
        let device = DeviceBuf::alloc_zeroed(ctx, record_bytes * chosen.len())?;
        let host = PinnedHostBuf::alloc_uninit(ctx, record_bytes * (Q38_EXP - chosen.len()))?;
        let mut packed = vec![0; device.byte_len];
        let mut bases = Vec::with_capacity(Q38_EXP * 8);
        let (mut gpu_slot, mut host_slot) = (0, 0);
        for (expert, &resident) in hot.iter().enumerate() {
            let (base, record) = if resident {
                let off = gpu_slot * record_bytes;
                gpu_slot += 1;
                (
                    device.ptr + off as u64,
                    &mut packed[off..off + record_bytes],
                )
            } else {
                let off = host_slot * record_bytes;
                host_slot += 1;
                // host_slot enumerates disjoint records inside the allocated pinned buffer.
                let slice =
                    unsafe { std::slice::from_raw_parts_mut(host.host_ptr.add(off), record_bytes) };
                (host.dev_ptr + off as u64, slice)
            };
            bases.extend_from_slice(&base.to_le_bytes());
            for &(layer_off, data) in planes {
                let width = data.len() / Q38_EXP;
                let off = layer_off / Q38_EXP;
                record[off..off + width]
                    .copy_from_slice(&data[expert * width..(expert + 1) * width]);
            }
        }
        device.copy_from_host_bytes(&packed)?;
        Ok(Self {
            _device: device,
            _host: host,
            bases: DeviceBuf::from_host_bytes(ctx, &bases)?,
        })
    }
}

enum SwitchMlpTier {
    Mixed(MixedExperts),
    Vram(DeviceBuf),
    HostPinned(PinnedHostBuf),
    MmapCompact { layer_idx: usize },
}

struct PleGpu {
    kv_proj: Q4LinearGpu,
    norm_key: DeviceBuf,
    norm_query: DeviceBuf,
    norm_conv: DeviceBuf,
    conv_w4: DeviceBuf,
    hist9: DeviceBuf,
    multipliers: [i64; 3],
    head_sizes: [i64; 16],
    head_offsets: [i64; 16],
    shard_keys: Vec<(String, String, String)>,
}

struct Qwen38LayerGpu {
    ple: Option<PleGpu>,
    attn_hc: HyperConnGpu,
    attn: AttentionBranchGpu,
    mlp_hc: HyperConnGpu,
    router_bf16: DeviceBuf,
    sh_gu_gate: Q4LinearGpu,
    sh_down: Q4LinearGpu,
    switch_mlp: SwitchMlpTier,
}

struct Qwen38KernelFns {
    _module: CUmodule,
    embed_init_hc4: CUfunction,
    q4_gemv: CUfunction,
    q4_gemv_10k: CUfunction,
    ple_inject: CUfunction,
    hc_norm4: CUfunction,
    hc_mix4: CUfunction,
    hc_inject4: CUfunction,
    gdn_conv_silu: CUfunction,
    gdn_recurrent_step: CUfunction,
    qsa_prep_and_cache: CUfunction,
    qsa_score_blocks: CUfunction,
    qsa_select_top512: CUfunction,
    qsa_attn: CUfunction,
    bf16_router_512: CUfunction,
    softmax_topk10: CUfunction,
    moe_up_swiglu11: CUfunction,
    moe_down_combine11: CUfunction,
    argmax_and_inc_pos: CUfunction,
    inc_pos: CUfunction,
}

unsafe impl Send for Qwen38KernelFns {}
unsafe impl Sync for Qwen38KernelFns {}

pub struct Qwen38GpuEngine {
    pub ctx: Arc<CudaContextHandle>,
    fns: Qwen38KernelFns,
    sharded_mmap: ShardedSafeTensors,
    embed: Q4LinearGpu,
    final_hc: HyperConnGpu,
    lm_head: Q4LinearGpu,
    inv_freq32: DeviceBuf,
    layers: Vec<Qwen38LayerGpu>,
    pub resident_expert_layers: usize,
    pub pinned_expert_layers: usize,
    pub compact_expert_layers: usize,
    pub max_ctx: usize,
    pub enable_cuda_graph: bool,
    use_gdn_sigmoid_gate: i32,
    ple_stage_ring: PinnedHostBuf, // `[PLE_RING_SLOTS, 2560]` f32 (`2.5 MiB`)
    tok_stage_ring: PinnedHostBuf, // `[PLE_RING_SLOTS]` i32
    d_ple_emb: DeviceBuf,
    d_ple_kv: DeviceBuf,
    ple_ready_event: CUevent,
    ple_prev2: i64,
    ple_prev1: i64,
    compact_host_stage: Option<PinnedHostBuf>,
    compact_dev_stage: Option<DeviceBuf>,
    d_token_in: DeviceBuf,
    d_token_out: DeviceBuf,
    d_pos: DeviceBuf,
    d_hc: DeviceBuf,
    d_hc_normed: DeviceBuf,
    d_hc_low: DeviceBuf,
    d_hc_up: DeviceBuf,
    d_hc_inj4: DeviceBuf,
    d_mixed: DeviceBuf,
    d_branch_out: DeviceBuf,
    d_gdn_in: DeviceBuf,
    d_qkv_conv: DeviceBuf,
    d_gdn_out: DeviceBuf,
    d_qsa_proj: DeviceBuf,
    d_q_rot: DeviceBuf,
    d_q_gate: DeviceBuf,
    d_idx_q: DeviceBuf,
    d_qsa_scores: DeviceBuf,
    d_qsa_mask: DeviceBuf,
    d_attn_out: DeviceBuf,
    d_router_logits: DeviceBuf,
    d_sel_ids10: DeviceBuf,
    d_sel_wts11: DeviceBuf,
    d_moe_act11: DeviceBuf,
    d_logits: DeviceBuf,
    host_pos: usize,
    decode_graph: Option<CapturedGraphExec>,
    prefill_graph: Option<CapturedGraphExec>,
}

unsafe impl Send for Qwen38GpuEngine {}

fn load_kernel_fns(ctx: &Arc<CudaContextHandle>) -> Result<Qwen38KernelFns, String> {
    let module_image: Vec<u8> = if let Ok(cubin_path) = std::env::var("LEANSONIC_QWEN38_CUBIN_PATH")
    {
        let mut raw = std::fs::read(&cubin_path).map_err(|e| format!("read {cubin_path}: {e}"))?;
        raw.push(0);
        raw
    } else {
        let inlined = CUDA_QWEN38_SRC.replace(
            "#include \"include/leansonic_verified.cuh\"",
            CUDA_HEADER_SRC,
        );
        compile_cuda_to_ptx_nvrtc(&inlined, "compute_120a")?
    };

    let mut module: CUmodule = core::ptr::null_mut();
    unsafe {
        check_cu(
            (ctx.api.cuModuleLoadData)(&mut module, module_image.as_ptr().cast()),
            "cuModuleLoadData(qwen38)",
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

    Ok(Qwen38KernelFns {
        _module: module,
        embed_init_hc4: get_fn("q38_embed_init_hc4")?,
        q4_gemv: get_fn("q38_q4_gemv")?,
        q4_gemv_10k: get_fn("q38_q4_gemv_10k")?,
        ple_inject: get_fn("q38_ple_inject")?,
        hc_norm4: get_fn("q38_hc_norm4")?,
        hc_mix4: get_fn("q38_hc_mix4")?,
        hc_inject4: get_fn("q38_hc_inject4")?,
        gdn_conv_silu: get_fn("q38_gdn_conv_silu")?,
        gdn_recurrent_step: get_fn("q38_gdn_recurrent_step")?,
        qsa_prep_and_cache: get_fn("q38_qsa_prep_and_cache")?,
        qsa_score_blocks: get_fn("q38_qsa_score_blocks")?,
        qsa_select_top512: get_fn("q38_qsa_select_top512")?,
        qsa_attn: get_fn("q38_qsa_attn")?,
        bf16_router_512: get_fn("q38_bf16_router_512")?,
        softmax_topk10: get_fn("q38_softmax_topk10")?,
        moe_up_swiglu11: get_fn(if std::env::var_os("Q38_EXPERT_PROFILE").is_some() {
            "q38_moe_up_swiglu11_mixed"
        } else {
            "q38_moe_up_swiglu11"
        })?,
        moe_down_combine11: get_fn(if std::env::var_os("Q38_EXPERT_PROFILE").is_some() {
            "q38_moe_down_combine11_mixed"
        } else {
            "q38_moe_down_combine11"
        })?,
        argmax_and_inc_pos: get_fn("q38_argmax_and_inc_pos")?,
        inc_pos: get_fn("q38_inc_pos")?,
    })
}

fn load_bf16_as_f32_buf(
    ctx: &Arc<CudaContextHandle>,
    st: &ShardedSafeTensors,
    name: &str,
    expected_shape: &[usize],
    add_one: bool,
) -> Result<DeviceBuf, String> {
    let u16s = st.require(name, "BF16", expected_shape).to_u16_vec();
    let offset = if add_one { 1.0f32 } else { 0.0f32 };
    let f32s: Vec<f32> = u16s.into_iter().map(|u| offset + bf16_to_f32(u)).collect();
    DeviceBuf::from_host_bytes(ctx, f32_slice_bytes(&f32s))
}

fn load_hyper_conn(
    ctx: &Arc<CudaContextHandle>,
    st: &ShardedSafeTensors,
    prefix: &str,
    zero_centered: bool,
    has_inject: bool,
) -> Result<HyperConnGpu, String> {
    let hc_norm = load_bf16_as_f32_buf(
        ctx,
        st,
        &format!("{prefix}.hc_norm.weight"),
        &[Q38_HC_D],
        zero_centered,
    )?;
    let down = Q4LinearGpu::from_sharded(
        ctx,
        st,
        &format!("{prefix}.input_mix_weight_down"),
        Q38_HC_LOW,
        Q38_HC_D,
    )?;
    let up = Q4LinearGpu::from_sharded(
        ctx,
        st,
        &format!("{prefix}.input_mix_weight_up"),
        Q38_HC_D,
        Q38_HC_LOW,
    )?;
    let inject = if has_inject {
        Some(Q4LinearGpu::from_sharded(
            ctx,
            st,
            &format!("{prefix}.block_inject_weight"),
            Q38_HC,
            Q38_HC_D,
        )?)
    } else {
        None
    };
    Ok(HyperConnGpu {
        hc_norm,
        down,
        up,
        inject,
    })
}

impl Qwen38GpuEngine {
    pub fn load_from_dir(dir: impl AsRef<Path>, max_ctx: usize) -> Result<Self, String> {
        Self::load_from_dir_with_opts(dir, max_ctx, true)
    }

    pub fn load_from_dir_with_opts(
        dir: impl AsRef<Path>,
        max_ctx: usize,
        enable_cuda_graph: bool,
    ) -> Result<Self, String> {
        let dir = dir.as_ref();
        let t0 = Instant::now();
        let ctx = CudaContextHandle::init(0)?;
        ctx.bind_current()?;
        let fns = load_kernel_fns(&ctx)?;
        let st = ShardedSafeTensors::open_dir(dir)
            .map_err(|e| format!("open sharded safetensors in {}: {e}", dir.display()))?;

        let cfg_text = std::fs::read_to_string(dir.join("config.json")).unwrap_or_default();
        let use_gdn_sigmoid_gate = if cfg_text.contains("\"output_gate_type\": \"silu\"") {
            0
        } else {
            1
        };

        let probe_norm = st
            .require(
                "language_model.model.layers.0.attn_hyper_connection.hc_norm.weight",
                "BF16",
                &[Q38_HC_D],
            )
            .to_u16_vec();
        let mean_probe: f32 =
            probe_norm.iter().map(|&u| bf16_to_f32(u)).sum::<f32>() / (Q38_HC_D as f32);
        let zero_centered = mean_probe < 0.5;

        let embed = Q4LinearGpu::from_sharded(
            &ctx,
            &st,
            "language_model.model.embed_tokens",
            Q38_VOCAB,
            Q38_D,
        )?;
        let final_hc = load_hyper_conn(
            &ctx,
            &st,
            "language_model.model.hyper_connection_mixer",
            zero_centered,
            false,
        )?;
        let lm_head =
            Q4LinearGpu::from_sharded(&ctx, &st, "language_model.lm_head", Q38_VOCAB, Q38_D)?;

        let inv_f: Vec<f32> = (0..32)
            .map(|i| 10_000_000.0_f64.powf(-(2.0 * (i as f64)) / 64.0) as f32)
            .collect();
        let inv_freq32 = DeviceBuf::from_host_bytes(&ctx, f32_slice_bytes(&inv_f))?;

        let ple_stage_ring = PinnedHostBuf::alloc_zeroed(&ctx, PLE_RING_SLOTS * Q38_D * 4)?;
        let tok_stage_ring = PinnedHostBuf::alloc_zeroed(&ctx, PLE_RING_SLOTS * 4)?;
        let d_ple_emb = DeviceBuf::alloc_zeroed(&ctx, Q38_D * 4)?;
        let d_ple_kv = DeviceBuf::alloc_zeroed(&ctx, (Q38_HC_D + Q38_D) * 4)?;
        let mut ple_ready_event: CUevent = core::ptr::null_mut();
        unsafe {
            check_cu(
                (ctx.api.cuEventCreate)(&mut ple_ready_event, CU_EVENT_DISABLE_TIMING),
                "cuEventCreate(ple_ready)",
            )?;
        }

        let max_blocks = (max_ctx / 4) + 16;
        let d_token_in = DeviceBuf::alloc_zeroed(&ctx, 16)?;
        let d_token_out = DeviceBuf::alloc_zeroed(&ctx, 16)?;
        let d_pos = DeviceBuf::alloc_zeroed(&ctx, 16)?;
        let d_hc = DeviceBuf::alloc_zeroed(&ctx, Q38_HC_D * 4)?;
        let d_hc_normed = DeviceBuf::alloc_zeroed(&ctx, Q38_HC_D * 4)?;
        let d_hc_low = DeviceBuf::alloc_zeroed(&ctx, Q38_HC_LOW * 4)?;
        let d_hc_up = DeviceBuf::alloc_zeroed(&ctx, Q38_HC_D * 4)?;
        let d_hc_inj4 = DeviceBuf::alloc_zeroed(&ctx, 16)?;
        let d_mixed = DeviceBuf::alloc_zeroed(&ctx, Q38_D * 4)?;
        let d_branch_out = DeviceBuf::alloc_zeroed(&ctx, Q38_D * 4)?;
        let d_gdn_in = DeviceBuf::alloc_zeroed(&ctx, 16_480 * 4)?;
        let d_qkv_conv = DeviceBuf::alloc_zeroed(&ctx, Q38_HC_D * 4)?;
        let d_gdn_out = DeviceBuf::alloc_zeroed(&ctx, 6_144 * 4)?;
        let d_qsa_proj = DeviceBuf::alloc_zeroed(&ctx, 13_952 * 4)?;
        let d_q_rot = DeviceBuf::alloc_zeroed(&ctx, 6_144 * 4)?;
        let d_q_gate = DeviceBuf::alloc_zeroed(&ctx, 6_144 * 4)?;
        let d_idx_q = DeviceBuf::alloc_zeroed(&ctx, 512 * 4)?;
        let d_qsa_scores = DeviceBuf::alloc_zeroed(&ctx, max_blocks * 4)?;
        let d_qsa_mask = DeviceBuf::alloc_zeroed(&ctx, max_blocks * 4)?;
        let d_attn_out = DeviceBuf::alloc_zeroed(&ctx, 6_144 * 4)?;
        let d_router_logits = DeviceBuf::alloc_zeroed(&ctx, Q38_EXP * 4)?;
        let d_sel_ids10 = DeviceBuf::alloc_zeroed(&ctx, Q38_TOP_K * 4)?;
        let d_sel_wts11 = DeviceBuf::alloc_zeroed(&ctx, (Q38_TOP_K + 1) * 4)?;
        let d_moe_act11 = DeviceBuf::alloc_zeroed(&ctx, (Q38_TOP_K + 1) * Q38_MOE_FF * 4)?;
        let d_logits = DeviceBuf::alloc_zeroed(&ctx, Q38_VOCAB * 4)?;

        struct PartialLayer {
            ple: Option<PleGpu>,
            attn_hc: HyperConnGpu,
            attn: AttentionBranchGpu,
            mlp_hc: HyperConnGpu,
            router_bf16: DeviceBuf,
            sh_gu_gate: Q4LinearGpu,
            sh_down: Q4LinearGpu,
        }
        let mut partials = Vec::with_capacity(Q38_LAYERS);

        for l in 0..Q38_LAYERS {
            let lp = format!("language_model.model.layers.{l}");
            let ple = if l == 1 {
                let pp = format!("{lp}.ple");
                let kv_proj = Q4LinearGpu::from_sharded_concat(
                    &ctx,
                    &st,
                    &[
                        (&format!("{pp}.key_proj"), Q38_HC_D),
                        (&format!("{pp}.value_proj"), Q38_D),
                    ],
                    Q38_D,
                )?;
                let norm_key = load_bf16_as_f32_buf(
                    &ctx,
                    &st,
                    &format!("{pp}.norm_key.weight"),
                    &[Q38_HC_D],
                    zero_centered,
                )?;
                let norm_query = load_bf16_as_f32_buf(
                    &ctx,
                    &st,
                    &format!("{pp}.norm_query.weight"),
                    &[Q38_HC_D],
                    zero_centered,
                )?;
                let norm_conv = load_bf16_as_f32_buf(
                    &ctx,
                    &st,
                    &format!("{pp}.norm_conv.weight"),
                    &[Q38_HC_D],
                    zero_centered,
                )?;
                let conv_w4 = load_bf16_as_f32_buf(
                    &ctx,
                    &st,
                    &format!("{pp}.conv1d.weight"),
                    &[Q38_HC_D, 4, 1],
                    false,
                )?;
                let hist9 = DeviceBuf::alloc_zeroed(&ctx, 9 * Q38_HC_D * 4)?;
                let mult_v = st
                    .require(
                        &format!("{pp}.ple_embedding.layer_multipliers"),
                        "I64",
                        &[3],
                    )
                    .to_i64_vec();
                let sizes_v = st
                    .require(
                        &format!("{pp}.ple_embedding.ngram_heads_vocab_sizes"),
                        "I64",
                        &[16],
                    )
                    .to_i64_vec();
                let offs_v = st
                    .require(
                        &format!("{pp}.ple_embedding.ngram_heads_offsets"),
                        "I64",
                        &[16],
                    )
                    .to_i64_vec();
                let mut multipliers = [0i64; 3];
                let mut head_sizes = [0i64; 16];
                let mut head_offsets = [0i64; 16];
                multipliers.copy_from_slice(&mult_v);
                head_sizes.copy_from_slice(&sizes_v);
                head_offsets.copy_from_slice(&offs_v);

                let mut shard_keys = Vec::with_capacity(Q38_PLE_SHARDS);
                for s in 0..Q38_PLE_SHARDS {
                    let sp = format!("{pp}.ple_embedding.ngram_embedding.shard_{s}");
                    let _ = st.require(&format!("{sp}.weight"), "U32", &[2_500_012, 20]);
                    shard_keys.push((
                        format!("{sp}.weight"),
                        format!("{sp}.scales"),
                        format!("{sp}.biases"),
                    ));
                }
                Some(PleGpu {
                    kv_proj,
                    norm_key,
                    norm_query,
                    norm_conv,
                    conv_w4,
                    hist9,
                    multipliers,
                    head_sizes,
                    head_offsets,
                    shard_keys,
                })
            } else {
                None
            };

            let attn_hc = load_hyper_conn(
                &ctx,
                &st,
                &format!("{lp}.attn_hyper_connection"),
                zero_centered,
                true,
            )?;
            let mlp_hc = load_hyper_conn(
                &ctx,
                &st,
                &format!("{lp}.mlp_hyper_connection"),
                zero_centered,
                true,
            )?;

            let attn = if (l + 1) % 4 != 0 {
                let ap = format!("{lp}.linear_attn");
                let in_all = Q4LinearGpu::from_sharded_concat(
                    &ctx,
                    &st,
                    &[
                        (&format!("{ap}.in_proj_qkv"), 10_240),
                        (&format!("{ap}.in_proj_z"), 6_144),
                        (&format!("{ap}.in_proj_b"), 48),
                        (&format!("{ap}.in_proj_a"), 48),
                    ],
                    Q38_D,
                )?;
                let out_proj =
                    Q4LinearGpu::from_sharded(&ctx, &st, &format!("{ap}.out_proj"), Q38_D, 6_144)?;
                let conv_w4 = load_bf16_as_f32_buf(
                    &ctx,
                    &st,
                    &format!("{ap}.conv1d.weight"),
                    &[Q38_HC_D, 4, 1],
                    false,
                )?;
                let a_log = load_bf16_as_f32_buf(&ctx, &st, &format!("{ap}.A_log"), &[48], false)?;
                let dt_bias =
                    load_bf16_as_f32_buf(&ctx, &st, &format!("{ap}.dt_bias"), &[48], false)?;
                let norm_w =
                    load_bf16_as_f32_buf(&ctx, &st, &format!("{ap}.norm.weight"), &[128], false)?;
                let conv_hist3 = DeviceBuf::alloc_zeroed(&ctx, Q38_HC_D * 3 * 4)?;
                let state_f32 = DeviceBuf::alloc_zeroed(&ctx, 48 * 128 * 128 * 4)?;
                AttentionBranchGpu::Gdn {
                    in_all,
                    out_proj,
                    conv_w4,
                    a_log,
                    dt_bias,
                    norm_w,
                    conv_hist3,
                    state_f32,
                }
            } else {
                let sp = format!("{lp}.self_attn");
                let qkv_idx = Q4LinearGpu::from_sharded_concat(
                    &ctx,
                    &st,
                    &[
                        (&format!("{sp}.q_proj"), 12_288),
                        (&format!("{sp}.k_proj"), 512),
                        (&format!("{sp}.v_proj"), 512),
                        (&format!("{sp}.indexer.index_qk_proj"), 640),
                    ],
                    Q38_D,
                )?;
                let o_proj =
                    Q4LinearGpu::from_sharded(&ctx, &st, &format!("{sp}.o_proj"), Q38_D, 6_144)?;
                let q_norm_w = load_bf16_as_f32_buf(
                    &ctx,
                    &st,
                    &format!("{sp}.q_norm.weight"),
                    &[256],
                    zero_centered,
                )?;
                let k_norm_w = load_bf16_as_f32_buf(
                    &ctx,
                    &st,
                    &format!("{sp}.k_norm.weight"),
                    &[256],
                    zero_centered,
                )?;
                let idx_q_norm_w = load_bf16_as_f32_buf(
                    &ctx,
                    &st,
                    &format!("{sp}.indexer.q_layernorm.weight"),
                    &[128],
                    zero_centered,
                )?;
                let idx_k_norm_w = load_bf16_as_f32_buf(
                    &ctx,
                    &st,
                    &format!("{sp}.indexer.k_layernorm.weight"),
                    &[128],
                    zero_centered,
                )?;
                let k_cache_fp16 = DeviceBuf::alloc_zeroed(&ctx, max_ctx * 512 * 2)?;
                let v_cache_fp16 = DeviceBuf::alloc_zeroed(&ctx, max_ctx * 512 * 2)?;
                let idx_raw_k_cache = DeviceBuf::alloc_zeroed(&ctx, max_ctx * 128 * 4)?;
                let idx_block_k_cache = DeviceBuf::alloc_zeroed(&ctx, max_blocks * 128 * 4)?;
                AttentionBranchGpu::Qsa {
                    qkv_idx,
                    o_proj,
                    q_norm_w,
                    k_norm_w,
                    idx_q_norm_w,
                    idx_k_norm_w,
                    k_cache_fp16,
                    v_cache_fp16,
                    idx_raw_k_cache,
                    idx_block_k_cache,
                }
            };

            let router_tv = st.require(&format!("{lp}.mlp.gate.weight"), "BF16", &[Q38_EXP, Q38_D]);
            let router_bf16 = DeviceBuf::from_host_bytes(&ctx, router_tv.data)?;
            let sh_gu_gate = Q4LinearGpu::from_sharded_concat(
                &ctx,
                &st,
                &[
                    (&format!("{lp}.mlp.shared_expert.gate_proj"), Q38_MOE_FF),
                    (&format!("{lp}.mlp.shared_expert.up_proj"), Q38_MOE_FF),
                    (&format!("{lp}.mlp.shared_expert_gate"), 1),
                ],
                Q38_D,
            )?;
            let sh_down = Q4LinearGpu::from_sharded(
                &ctx,
                &st,
                &format!("{lp}.mlp.shared_expert.down_proj"),
                Q38_D,
                Q38_MOE_FF,
            )?;

            partials.push(PartialLayer {
                ple,
                attn_hc,
                attn,
                mlp_hc,
                router_bf16,
                sh_gu_gate,
                sh_down,
            });
        }

        // Pass 2: Place `48` `switch_mlp` slabs (`1,500 MiB`/layer): VRAM -> HostPinned -> MmapCompact
        let vram_reserve_bytes: usize = 450 * 1024 * 1024; // 450 MiB headroom -> 19/48 in 32GB, 45/48 in 72GB!
        let host_ram_reserve_bytes: usize = 16 * 1024 * 1024 * 1024;
        let force_compact = std::env::var("LEANSONIC_SPILLED_MODE")
            .map(|v| v == "compact")
            .unwrap_or(false);
        let max_vram_layers: usize = std::env::var("LEANSONIC_MAX_RESIDENT_LAYERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(Q38_LAYERS);

        let profile = if let Some(path) = std::env::var_os("Q38_EXPERT_PROFILE") {
            let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
            let parsed = content
                .lines()
                .map(|line| {
                    line.split_whitespace()
                        .map(str::parse::<usize>)
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?;
            if parsed.len() != Q38_LAYERS {
                return Err("profile must have 48 lines".into());
            }
            eprintln!(
                "[mixed placement] {} device expert slots",
                parsed.iter().map(Vec::len).sum::<usize>()
            );
            Some(parsed)
        } else {
            None
        };
        let mut resident_expert_layers = 0usize;
        let mut pinned_expert_layers = 0usize;
        let mut compact_expert_layers = 0usize;
        let mut layers = Vec::with_capacity(Q38_LAYERS);

        for (l, part) in partials.into_iter().enumerate() {
            let swp = format!("language_model.model.layers.{l}.mlp.switch_mlp");
            let gw = st.require(&format!("{swp}.gate_proj.weight"), "U32", &[512, 640, 320]);
            let gsc = st.require(&format!("{swp}.gate_proj.scales"), "BF16", &[512, 640, 80]);
            let gbi = st.require(&format!("{swp}.gate_proj.biases"), "BF16", &[512, 640, 80]);
            let uw = st.require(&format!("{swp}.up_proj.weight"), "U32", &[512, 640, 320]);
            let usc = st.require(&format!("{swp}.up_proj.scales"), "BF16", &[512, 640, 80]);
            let ubi = st.require(&format!("{swp}.up_proj.biases"), "BF16", &[512, 640, 80]);
            let dw = st.require(&format!("{swp}.down_proj.weight"), "U32", &[512, 2560, 80]);
            let dsc = st.require(&format!("{swp}.down_proj.scales"), "BF16", &[512, 2560, 20]);
            let dbi = st.require(&format!("{swp}.down_proj.biases"), "BF16", &[512, 2560, 20]);

            let slices: [(usize, &[u8]); 9] = [
                (OFF_GATE_W, gw.data),
                (OFF_GATE_SC, gsc.data),
                (OFF_GATE_BI, gbi.data),
                (OFF_UP_W, uw.data),
                (OFF_UP_SC, usc.data),
                (OFF_UP_BI, ubi.data),
                (OFF_DOWN_W, dw.data),
                (OFF_DOWN_SC, dsc.data),
                (OFF_DOWN_BI, dbi.data),
            ];

            let (free_vram, _) = ctx.mem_info()?;
            let switch_mlp = if let Some(profile) = &profile {
                SwitchMlpTier::Mixed(MixedExperts::load(&ctx, &profile[l], &slices)?)
            } else if resident_expert_layers < max_vram_layers
                && free_vram > vram_reserve_bytes.saturating_add(SW_LAYER_BYTES)
            {
                let slab = DeviceBuf::alloc_zeroed(&ctx, SW_LAYER_BYTES)?;
                for &(off, bytes) in &slices {
                    slab.copy_from_host_at(off, bytes)?;
                }
                resident_expert_layers += 1;
                SwitchMlpTier::Vram(slab)
            } else if !force_compact
                && available_host_ram_bytes()
                    > host_ram_reserve_bytes.saturating_add(SW_LAYER_BYTES)
            {
                let slab = PinnedHostBuf::alloc_uninit(&ctx, SW_LAYER_BYTES)?;
                parallel_copy_into_pinned(slab.host_ptr, &slices);
                pinned_expert_layers += 1;
                SwitchMlpTier::HostPinned(slab)
            } else {
                compact_expert_layers += 1;
                SwitchMlpTier::MmapCompact { layer_idx: l }
            };

            layers.push(Qwen38LayerGpu {
                ple: part.ple,
                attn_hc: part.attn_hc,
                attn: part.attn,
                mlp_hc: part.mlp_hc,
                router_bf16: part.router_bf16,
                sh_gu_gate: part.sh_gu_gate,
                sh_down: part.sh_down,
                switch_mlp,
            });
        }

        let (compact_host_stage, compact_dev_stage) = if compact_expert_layers > 0 {
            (
                Some(PinnedHostBuf::alloc_uninit(&ctx, COMPACT_TOTAL_BYTES)?),
                Some(DeviceBuf::alloc_zeroed(&ctx, COMPACT_TOTAL_BYTES)?),
            )
        } else {
            (None, None)
        };

        let mut engine = Self {
            ctx,
            fns,
            sharded_mmap: st,
            embed,
            final_hc,
            lm_head,
            inv_freq32,
            layers,
            resident_expert_layers,
            pinned_expert_layers,
            compact_expert_layers,
            max_ctx,
            enable_cuda_graph,
            use_gdn_sigmoid_gate,
            ple_stage_ring,
            tok_stage_ring,
            d_ple_emb,
            d_ple_kv,
            ple_ready_event,
            ple_prev2: Q38_PLE_EOS_ID,
            ple_prev1: Q38_PLE_EOS_ID,
            compact_host_stage,
            compact_dev_stage,
            d_token_in,
            d_token_out,
            d_pos,
            d_hc,
            d_hc_normed,
            d_hc_low,
            d_hc_up,
            d_hc_inj4,
            d_mixed,
            d_branch_out,
            d_gdn_in,
            d_qkv_conv,
            d_gdn_out,
            d_qsa_proj,
            d_q_rot,
            d_q_gate,
            d_idx_q,
            d_qsa_scores,
            d_qsa_mask,
            d_attn_out,
            d_router_logits,
            d_sel_ids10,
            d_sel_wts11,
            d_moe_act11,
            d_logits,
            host_pos: 0,
            decode_graph: None,
            prefill_graph: None,
        };

        if enable_cuda_graph && compact_expert_layers == 0 {
            engine.decode_graph = Some(CapturedGraphExec::capture(&engine.ctx, |stream| {
                engine.enqueue_all_48_layers(stream, true, false)
            })?);
            engine.prefill_graph = Some(CapturedGraphExec::capture(&engine.ctx, |stream| {
                engine.enqueue_all_48_layers(stream, false, false)
            })?);
            engine.reset()?;
        }

        let load_s = t0.elapsed().as_secs_f64();
        eprintln!(
            "[Qwen38GpuEngine] Loaded {} ({} shards, zero_norm={}, graph={}) on {} in {:.2}s | Experts: {}/48 VRAM + {}/48 HostPinned + {}/48 MmapCompact + 29.8GiB Async Engram",
            dir.display(),
            engine.sharded_mmap.num_shards(),
            zero_centered,
            engine.decode_graph.is_some(),
            engine.ctx.device_name,
            load_s,
            resident_expert_layers,
            pinned_expert_layers,
            compact_expert_layers,
        );

        Ok(engine)
    }

    pub fn reset(&mut self) -> Result<(), String> {
        self.ctx.bind_current()?;
        self.ctx.sync_compute()?;
        self.ctx.sync_h2d()?;
        self.host_pos = 0;
        self.ple_prev2 = Q38_PLE_EOS_ID;
        self.ple_prev1 = Q38_PLE_EOS_ID;
        unsafe {
            check_cu(
                (self.ctx.api.cuMemsetD8_v2)(self.d_pos.ptr, 0, self.d_pos.byte_len),
                "zero(d_pos)",
            )?;
            for l in &self.layers {
                if let Some(ple) = &l.ple {
                    check_cu(
                        (self.ctx.api.cuMemsetD8_v2)(ple.hist9.ptr, 0, ple.hist9.byte_len),
                        "zero(ple.hist9)",
                    )?;
                }
                if let AttentionBranchGpu::Gdn {
                    conv_hist3,
                    state_f32,
                    ..
                } = &l.attn
                {
                    check_cu(
                        (self.ctx.api.cuMemsetD8_v2)(conv_hist3.ptr, 0, conv_hist3.byte_len),
                        "zero(gdn.conv_hist3)",
                    )?;
                    check_cu(
                        (self.ctx.api.cuMemsetD8_v2)(state_f32.ptr, 0, state_f32.byte_len),
                        "zero(gdn.state_f32)",
                    )?;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::manual_is_multiple_of)]
    pub fn prefill_tokens(&mut self, prompt_ids: &[u32]) -> Result<(u32, f64), String> {
        if prompt_ids.is_empty() {
            return Err("prefill_tokens requires at least 1 token".into());
        }
        if self.host_pos + prompt_ids.len() > self.max_ctx {
            return Err(format!(
                "sequence length {} exceeds max_ctx {}",
                self.host_pos + prompt_ids.len(),
                self.max_ctx
            ));
        }
        self.ctx.bind_current()?;
        let t0 = Instant::now();
        for (idx, &tok) in prompt_ids[..prompt_ids.len() - 1].iter().enumerate() {
            if idx > 0 && idx % PLE_RING_SLOTS == 0 {
                self.ctx.sync_compute()?;
            }
            self.forward_token_internal(tok, false)?;
        }
        if prompt_ids.len() > 1 {
            self.ctx.sync_compute()?;
        }
        let next = self.forward_token_internal(*prompt_ids.last().unwrap(), true)?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok((next, ms))
    }

    pub fn forward_greedy_token(&mut self, tok: u32) -> Result<u32, String> {
        if self.host_pos + 1 > self.max_ctx {
            return Err(format!("max_ctx {} reached", self.max_ctx));
        }
        self.ctx.bind_current()?;
        self.forward_token_internal(tok, true)
    }

    pub fn generate_greedy_tokens<F>(
        &mut self,
        first_tok: u32,
        max_new_tokens: usize,
        stop_ids: &[u32],
        mut on_token: F,
    ) -> Result<Vec<u32>, String>
    where
        F: FnMut(u32),
    {
        let mut out = Vec::with_capacity(max_new_tokens);
        if max_new_tokens == 0 {
            return Ok(out);
        }
        let mut cur = first_tok;
        for step in 0..max_new_tokens {
            if stop_ids.contains(&cur) {
                break;
            }
            out.push(cur);
            on_token(cur);
            if step + 1 == max_new_tokens {
                break;
            }
            cur = self.forward_greedy_token(cur)?;
        }
        Ok(out)
    }

    pub fn read_last_logits(&self) -> Result<Vec<f32>, String> {
        self.ctx.sync_compute()?;
        let mut out = vec![0.0f32; Q38_VOCAB];
        self.d_logits.copy_to_host_f32(&mut out)?;
        Ok(out)
    }

    fn decode_ple_into_ring_slot(
        &mut self,
        tok: u32,
        ring_slot: usize,
    ) -> Result<*const u8, String> {
        let s0 = tok as i64;
        let s1 = self.ple_prev1;
        let s2 = self.ple_prev2;
        if s0 == Q38_PLE_EOS_ID {
            self.ple_prev2 = Q38_PLE_EOS_ID;
            self.ple_prev1 = Q38_PLE_EOS_ID;
        } else {
            self.ple_prev2 = self.ple_prev1;
            self.ple_prev1 = s0;
        }

        let ple = self.layers[1]
            .ple
            .as_ref()
            .expect("layer 1 must carry PLE metadata");
        let m = ple.multipliers;
        let slot_ptr = unsafe { self.ple_stage_ring.host_ptr.add(ring_slot * Q38_D * 4) };
        let dst_f32 = unsafe { core::slice::from_raw_parts_mut(slot_ptr.cast::<f32>(), Q38_D) };

        for h in 0..16usize {
            let mixed = if h < 8 {
                s0.wrapping_mul(m[0]) ^ s1.wrapping_mul(m[1])
            } else {
                s0.wrapping_mul(m[0]) ^ s1.wrapping_mul(m[1]) ^ s2.wrapping_mul(m[2])
            };
            let gid = mixed.rem_euclid(ple.head_sizes[h]) + ple.head_offsets[h];
            let shard_idx = (gid / Q38_PLE_SHARD_ROWS) as usize;
            let row_idx = (gid % Q38_PLE_SHARD_ROWS) as usize;
            let (wk, sck, bik) = &ple.shard_keys[shard_idx];

            let w_tv = self
                .sharded_mmap
                .get(wk)
                .ok_or_else(|| format!("missing PLE shard {wk}"))?;
            let sc_tv = self
                .sharded_mmap
                .get(sck)
                .ok_or_else(|| format!("missing PLE shard {sck}"))?;
            let bi_tv = self
                .sharded_mmap
                .get(bik)
                .ok_or_else(|| format!("missing PLE shard {bik}"))?;

            let w_row = &w_tv.data[row_idx * 80..(row_idx + 1) * 80];
            let sc_row = &sc_tv.data[row_idx * 10..(row_idx + 1) * 10];
            let bi_row = &bi_tv.data[row_idx * 10..(row_idx + 1) * 10];
            let out_h = &mut dst_f32[h * 160..(h + 1) * 160];

            for g in 0..5usize {
                let sc = bf16_to_f32(u16::from_le_bytes([sc_row[2 * g], sc_row[2 * g + 1]]));
                let bi = bf16_to_f32(u16::from_le_bytes([bi_row[2 * g], bi_row[2 * g + 1]]));
                for wi in 0..4usize {
                    let off = (g * 4 + wi) * 4;
                    let word = u32::from_le_bytes(w_row[off..off + 4].try_into().unwrap());
                    let base = g * 32 + wi * 8;
                    for j in 0..8usize {
                        let nib = ((word >> (4 * j)) & 0xFu32) as f32;
                        out_h[base + j] = nib.mul_add(sc, bi);
                    }
                }
            }
        }
        Ok(slot_ptr.cast_const())
    }

    fn launch_q4_gemv(
        &self,
        stream: CUstream,
        lin: &Q4LinearGpu,
        x_ptr: CUdeviceptr,
        y_ptr: CUdeviceptr,
        post_mode: i32,
    ) -> Result<(), String> {
        let (mut w_p, mut sc_p, mut bi_p, mut x_p, mut y_p) =
            (lin.w.ptr, lin.sc.ptr, lin.bi.ptr, x_ptr, y_ptr);
        let (mut rows, mut cols, mut mode) = (lin.rows, lin.cols, post_mode);
        let grid_x = (rows as u32).div_ceil(8);

        if cols == (Q38_HC_D as i32) {
            let mut p = kargs![w_p, sc_p, bi_p, x_p, y_p, rows, mode];
            unsafe {
                check_cu(
                    (self.ctx.api.cuLaunchKernel)(
                        self.fns.q4_gemv_10k,
                        grid_x,
                        1,
                        1,
                        256,
                        1,
                        1,
                        0,
                        stream,
                        p.as_mut_ptr(),
                        core::ptr::null_mut(),
                    ),
                    "q38_q4_gemv_10k",
                )
            }
        } else {
            let mut p = kargs![w_p, sc_p, bi_p, x_p, y_p, rows, cols, mode];
            unsafe {
                check_cu(
                    (self.ctx.api.cuLaunchKernel)(
                        self.fns.q4_gemv,
                        grid_x,
                        1,
                        1,
                        256,
                        1,
                        1,
                        0,
                        stream,
                        p.as_mut_ptr(),
                        core::ptr::null_mut(),
                    ),
                    "q38_q4_gemv",
                )
            }
        }
    }

    fn launch_hc_pre(&self, stream: CUstream, hc: &HyperConnGpu) -> Result<(), String> {
        let (mut hc_in, mut gam, mut normed) =
            (self.d_hc.ptr, hc.hc_norm.ptr, self.d_hc_normed.ptr);
        let mut p_norm = kargs![hc_in, gam, normed];
        unsafe {
            check_cu(
                (self.ctx.api.cuLaunchKernel)(
                    self.fns.hc_norm4,
                    4,
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    stream,
                    p_norm.as_mut_ptr(),
                    core::ptr::null_mut(),
                ),
                "q38_hc_norm4",
            )?;
        }
        self.launch_q4_gemv(stream, &hc.down, self.d_hc_normed.ptr, self.d_hc_low.ptr, 1)?;
        self.launch_q4_gemv(stream, &hc.up, self.d_hc_low.ptr, self.d_hc_up.ptr, 2)?;
        let (mut up_sig, mut mixed) = (self.d_hc_up.ptr, self.d_mixed.ptr);
        let mut p_mix = kargs![normed, up_sig, mixed];
        unsafe {
            check_cu(
                (self.ctx.api.cuLaunchKernel)(
                    self.fns.hc_mix4,
                    10,
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    stream,
                    p_mix.as_mut_ptr(),
                    core::ptr::null_mut(),
                ),
                "q38_hc_mix4",
            )?;
        }
        if let Some(inj) = &hc.inject {
            self.launch_q4_gemv(stream, inj, self.d_hc_normed.ptr, self.d_hc_inj4.ptr, 0)?;
        }
        Ok(())
    }

    fn launch_hc_post(&self, stream: CUstream) -> Result<(), String> {
        let (mut hc_io, mut br, mut inj4) =
            (self.d_hc.ptr, self.d_branch_out.ptr, self.d_hc_inj4.ptr);
        let mut p = kargs![hc_io, br, inj4];
        unsafe {
            check_cu(
                (self.ctx.api.cuLaunchKernel)(
                    self.fns.hc_inject4,
                    40,
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    stream,
                    p.as_mut_ptr(),
                    core::ptr::null_mut(),
                ),
                "q38_hc_inject4",
            )
        }
    }

    fn stage_compact_top10(&self, layer_idx: usize) -> Result<CUdeviceptr, String> {
        let host_stage = self
            .compact_host_stage
            .as_ref()
            .expect("compact_host_stage allocated");
        let dev_stage = self
            .compact_dev_stage
            .as_ref()
            .expect("compact_dev_stage allocated");

        self.ctx.sync_compute()?;
        let mut sel = [0i32; Q38_TOP_K];
        unsafe {
            check_cu(
                (self.ctx.api.cuMemcpyDtoH_v2)(
                    sel.as_mut_ptr().cast(),
                    self.d_sel_ids10.ptr,
                    core::mem::size_of_val(&sel),
                ),
                "DtoH(sel_ids10)",
            )?;
        }

        let swp = format!("language_model.model.layers.{layer_idx}.mlp.switch_mlp");
        let gw = self
            .sharded_mmap
            .get(&format!("{swp}.gate_proj.weight"))
            .unwrap()
            .data;
        let gsc = self
            .sharded_mmap
            .get(&format!("{swp}.gate_proj.scales"))
            .unwrap()
            .data;
        let gbi = self
            .sharded_mmap
            .get(&format!("{swp}.gate_proj.biases"))
            .unwrap()
            .data;
        let uw = self
            .sharded_mmap
            .get(&format!("{swp}.up_proj.weight"))
            .unwrap()
            .data;
        let usc = self
            .sharded_mmap
            .get(&format!("{swp}.up_proj.scales"))
            .unwrap()
            .data;
        let ubi = self
            .sharded_mmap
            .get(&format!("{swp}.up_proj.biases"))
            .unwrap()
            .data;
        let dw = self
            .sharded_mmap
            .get(&format!("{swp}.down_proj.weight"))
            .unwrap()
            .data;
        let dsc = self
            .sharded_mmap
            .get(&format!("{swp}.down_proj.scales"))
            .unwrap()
            .data;
        let dbi = self
            .sharded_mmap
            .get(&format!("{swp}.down_proj.biases"))
            .unwrap()
            .data;

        unsafe {
            let dst = host_stage.host_ptr;
            for (k, &ex_i32) in sel.iter().enumerate() {
                let ex = ex_i32 as usize;
                core::ptr::copy_nonoverlapping(
                    gw.as_ptr().add(ex * EX_W_BYTES),
                    dst.add(COFF_GATE_W + k * EX_W_BYTES),
                    EX_W_BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    gsc.as_ptr().add(ex * EX_SB_BYTES),
                    dst.add(COFF_GATE_SC + k * EX_SB_BYTES),
                    EX_SB_BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    gbi.as_ptr().add(ex * EX_SB_BYTES),
                    dst.add(COFF_GATE_BI + k * EX_SB_BYTES),
                    EX_SB_BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    uw.as_ptr().add(ex * EX_W_BYTES),
                    dst.add(COFF_UP_W + k * EX_W_BYTES),
                    EX_W_BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    usc.as_ptr().add(ex * EX_SB_BYTES),
                    dst.add(COFF_UP_SC + k * EX_SB_BYTES),
                    EX_SB_BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    ubi.as_ptr().add(ex * EX_SB_BYTES),
                    dst.add(COFF_UP_BI + k * EX_SB_BYTES),
                    EX_SB_BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    dw.as_ptr().add(ex * EX_W_BYTES),
                    dst.add(COFF_DOWN_W + k * EX_W_BYTES),
                    EX_W_BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    dsc.as_ptr().add(ex * EX_SB_BYTES),
                    dst.add(COFF_DOWN_SC + k * EX_SB_BYTES),
                    EX_SB_BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    dbi.as_ptr().add(ex * EX_SB_BYTES),
                    dst.add(COFF_DOWN_BI + k * EX_SB_BYTES),
                    EX_SB_BYTES,
                );
            }
            check_cu(
                (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                    dev_stage.ptr,
                    dst.cast(),
                    COMPACT_TOTAL_BYTES,
                    self.ctx.compute_stream,
                ),
                "HtoD(compact_top10)",
            )?;
        }
        Ok(dev_stage.ptr)
    }

    fn enqueue_all_48_layers(
        &self,
        stream: CUstream,
        compute_logits: bool,
        wait_ple_event_at_layer1: bool,
    ) -> Result<(), String> {
        {
            let (mut ew, mut esc, mut ebi, mut tin, mut dhc) = (
                self.embed.w.ptr,
                self.embed.sc.ptr,
                self.embed.bi.ptr,
                self.d_token_in.ptr,
                self.d_hc.ptr,
            );
            let mut p = kargs![ew, esc, ebi, tin, dhc];
            unsafe {
                check_cu(
                    (self.ctx.api.cuLaunchKernel)(
                        self.fns.embed_init_hc4,
                        10,
                        1,
                        1,
                        256,
                        1,
                        1,
                        0,
                        stream,
                        p.as_mut_ptr(),
                        core::ptr::null_mut(),
                    ),
                    "q38_embed_init_hc4",
                )?;
            }
        }

        for l_idx in 0..Q38_LAYERS {
            let layer = &self.layers[l_idx];
            if let Some(ple) = &layer.ple {
                if wait_ple_event_at_layer1 {
                    unsafe {
                        check_cu(
                            (self.ctx.api.cuStreamWaitEvent)(stream, self.ple_ready_event, 0),
                            "cuStreamWaitEvent(ple_ready)",
                        )?;
                    }
                }
                self.launch_q4_gemv(
                    stream,
                    &ple.kv_proj,
                    self.d_ple_emb.ptr,
                    self.d_ple_kv.ptr,
                    0,
                )?;
                let (mut hc_io, mut k_raw, mut v_raw) = (
                    self.d_hc.ptr,
                    self.d_ple_kv.ptr,
                    self.d_ple_kv.ptr + (Q38_HC_D as u64) * 4,
                );
                let (mut nk, mut nq, mut nc, mut cw, mut h9, mut pos_p) = (
                    ple.norm_key.ptr,
                    ple.norm_query.ptr,
                    ple.norm_conv.ptr,
                    ple.conv_w4.ptr,
                    ple.hist9.ptr,
                    self.d_pos.ptr,
                );
                let mut p_ple = kargs![hc_io, k_raw, v_raw, nk, nq, nc, cw, h9, pos_p];
                unsafe {
                    check_cu(
                        (self.ctx.api.cuLaunchKernel)(
                            self.fns.ple_inject,
                            4,
                            1,
                            1,
                            256,
                            1,
                            1,
                            0,
                            stream,
                            p_ple.as_mut_ptr(),
                            core::ptr::null_mut(),
                        ),
                        "q38_ple_inject",
                    )?;
                }
            }

            self.launch_hc_pre(stream, &layer.attn_hc)?;

            match &layer.attn {
                AttentionBranchGpu::Gdn {
                    in_all,
                    out_proj,
                    conv_w4,
                    a_log,
                    dt_bias,
                    norm_w,
                    conv_hist3,
                    state_f32,
                } => {
                    self.launch_q4_gemv(stream, in_all, self.d_mixed.ptr, self.d_gdn_in.ptr, 0)?;
                    let (mut qkv_r, mut cw, mut ch3, mut qkv_c) = (
                        self.d_gdn_in.ptr,
                        conv_w4.ptr,
                        conv_hist3.ptr,
                        self.d_qkv_conv.ptr,
                    );
                    let mut p_conv = kargs![qkv_r, cw, ch3, qkv_c];
                    unsafe {
                        check_cu(
                            (self.ctx.api.cuLaunchKernel)(
                                self.fns.gdn_conv_silu,
                                40,
                                1,
                                1,
                                256,
                                1,
                                1,
                                0,
                                stream,
                                p_conv.as_mut_ptr(),
                                core::ptr::null_mut(),
                            ),
                            "q38_gdn_conv_silu",
                        )?;
                    }
                    let (mut zba, mut al, mut dtb, mut nw, mut sig_gate, mut st_p, mut gout) = (
                        self.d_gdn_in.ptr + 10_240 * 4,
                        a_log.ptr,
                        dt_bias.ptr,
                        norm_w.ptr,
                        self.use_gdn_sigmoid_gate,
                        state_f32.ptr,
                        self.d_gdn_out.ptr,
                    );
                    let mut p_rec = kargs![qkv_c, zba, al, dtb, nw, sig_gate, st_p, gout];
                    unsafe {
                        check_cu(
                            (self.ctx.api.cuLaunchKernel)(
                                self.fns.gdn_recurrent_step,
                                48,
                                1,
                                1,
                                128,
                                1,
                                1,
                                0,
                                stream,
                                p_rec.as_mut_ptr(),
                                core::ptr::null_mut(),
                            ),
                            "q38_gdn_recurrent_step",
                        )?;
                    }
                    self.launch_q4_gemv(
                        stream,
                        out_proj,
                        self.d_gdn_out.ptr,
                        self.d_branch_out.ptr,
                        0,
                    )?;
                }
                AttentionBranchGpu::Qsa {
                    qkv_idx,
                    o_proj,
                    q_norm_w,
                    k_norm_w,
                    idx_q_norm_w,
                    idx_k_norm_w,
                    k_cache_fp16,
                    v_cache_fp16,
                    idx_raw_k_cache,
                    idx_block_k_cache,
                } => {
                    self.launch_q4_gemv(stream, qkv_idx, self.d_mixed.ptr, self.d_qsa_proj.ptr, 0)?;
                    let (mut qp, mut qnw, mut knw, mut iqnw, mut iknw, mut invf, mut pos_p) = (
                        self.d_qsa_proj.ptr,
                        q_norm_w.ptr,
                        k_norm_w.ptr,
                        idx_q_norm_w.ptr,
                        idx_k_norm_w.ptr,
                        self.inv_freq32.ptr,
                        self.d_pos.ptr,
                    );
                    let (mut qrot, mut qgate, mut kc, mut vc, mut iq, mut irk, mut ibk) = (
                        self.d_q_rot.ptr,
                        self.d_q_gate.ptr,
                        k_cache_fp16.ptr,
                        v_cache_fp16.ptr,
                        self.d_idx_q.ptr,
                        idx_raw_k_cache.ptr,
                        idx_block_k_cache.ptr,
                    );
                    let mut p_prep = kargs![
                        qp, qnw, knw, iqnw, iknw, invf, pos_p, qrot, qgate, kc, vc, iq, irk, ibk
                    ];
                    unsafe {
                        check_cu(
                            (self.ctx.api.cuLaunchKernel)(
                                self.fns.qsa_prep_and_cache,
                                31,
                                1,
                                1,
                                256,
                                1,
                                1,
                                0,
                                stream,
                                p_prep.as_mut_ptr(),
                                core::ptr::null_mut(),
                            ),
                            "q38_qsa_prep_and_cache",
                        )?;
                    }

                    let seq_len = self.host_pos + 1;
                    let n_blocks = (seq_len / 4) as i32;
                    let mut use_mask: i32 = if n_blocks > 512 { 1 } else { 0 };
                    if use_mask == 1 {
                        let (mut nb, mut sc_buf, mut mask_buf) =
                            (n_blocks, self.d_qsa_scores.ptr, self.d_qsa_mask.ptr);
                        let grid_sb = (n_blocks as u32).div_ceil(8);
                        let mut p_sc = kargs![iq, ibk, nb, sc_buf];
                        let mut p_sel = kargs![sc_buf, nb, mask_buf];
                        unsafe {
                            check_cu(
                                (self.ctx.api.cuLaunchKernel)(
                                    self.fns.qsa_score_blocks,
                                    grid_sb,
                                    1,
                                    1,
                                    256,
                                    1,
                                    1,
                                    0,
                                    stream,
                                    p_sc.as_mut_ptr(),
                                    core::ptr::null_mut(),
                                ),
                                "q38_qsa_score_blocks",
                            )?;
                            check_cu(
                                (self.ctx.api.cuLaunchKernel)(
                                    self.fns.qsa_select_top512,
                                    1,
                                    1,
                                    1,
                                    256,
                                    1,
                                    1,
                                    0,
                                    stream,
                                    p_sel.as_mut_ptr(),
                                    core::ptr::null_mut(),
                                ),
                                "q38_qsa_select_top512",
                            )?;
                        }
                    }

                    let (mut mask_p, mut aout) = (self.d_qsa_mask.ptr, self.d_attn_out.ptr);
                    let mut p_attn = kargs![qrot, qgate, kc, vc, mask_p, pos_p, use_mask, aout];
                    unsafe {
                        check_cu(
                            (self.ctx.api.cuLaunchKernel)(
                                self.fns.qsa_attn,
                                24,
                                1,
                                1,
                                256,
                                1,
                                1,
                                0,
                                stream,
                                p_attn.as_mut_ptr(),
                                core::ptr::null_mut(),
                            ),
                            "q38_qsa_attn",
                        )?;
                    }
                    self.launch_q4_gemv(
                        stream,
                        o_proj,
                        self.d_attn_out.ptr,
                        self.d_branch_out.ptr,
                        0,
                    )?;
                }
            }

            self.launch_hc_post(stream)?;
            self.launch_hc_pre(stream, &layer.mlp_hc)?;

            {
                let (mut rw, mut mx, mut rlog, mut sids, mut swts) = (
                    layer.router_bf16.ptr,
                    self.d_mixed.ptr,
                    self.d_router_logits.ptr,
                    self.d_sel_ids10.ptr,
                    self.d_sel_wts11.ptr,
                );
                let mut p_rt = kargs![rw, mx, rlog];
                let mut p_tk = kargs![rlog, sids, swts];
                unsafe {
                    check_cu(
                        (self.ctx.api.cuLaunchKernel)(
                            self.fns.bf16_router_512,
                            64,
                            1,
                            1,
                            256,
                            1,
                            1,
                            0,
                            stream,
                            p_rt.as_mut_ptr(),
                            core::ptr::null_mut(),
                        ),
                        "q38_bf16_router_512",
                    )?;
                    check_cu(
                        (self.ctx.api.cuLaunchKernel)(
                            self.fns.softmax_topk10,
                            1,
                            1,
                            1,
                            256,
                            1,
                            1,
                            0,
                            stream,
                            p_tk.as_mut_ptr(),
                            core::ptr::null_mut(),
                        ),
                        "q38_softmax_topk10",
                    )?;
                }
            }

            let (
                mut gw_p,
                mut gsc_p,
                mut gbi_p,
                mut uw_p,
                mut usc_p,
                mut ubi_p,
                mut dw_p,
                mut dsc_p,
                mut dbi_p,
                mut is_compact,
            ) = match &layer.switch_mlp {
                SwitchMlpTier::Mixed(m) => (m.bases.ptr, 0, 0, 0, 0, 0, m.bases.ptr, 0, 0, 2i32),
                SwitchMlpTier::Vram(b) => (
                    b.ptr + OFF_GATE_W as u64,
                    b.ptr + OFF_GATE_SC as u64,
                    b.ptr + OFF_GATE_BI as u64,
                    b.ptr + OFF_UP_W as u64,
                    b.ptr + OFF_UP_SC as u64,
                    b.ptr + OFF_UP_BI as u64,
                    b.ptr + OFF_DOWN_W as u64,
                    b.ptr + OFF_DOWN_SC as u64,
                    b.ptr + OFF_DOWN_BI as u64,
                    0i32,
                ),
                SwitchMlpTier::HostPinned(b) => (
                    b.dev_ptr + OFF_GATE_W as u64,
                    b.dev_ptr + OFF_GATE_SC as u64,
                    b.dev_ptr + OFF_GATE_BI as u64,
                    b.dev_ptr + OFF_UP_W as u64,
                    b.dev_ptr + OFF_UP_SC as u64,
                    b.dev_ptr + OFF_UP_BI as u64,
                    b.dev_ptr + OFF_DOWN_W as u64,
                    b.dev_ptr + OFF_DOWN_SC as u64,
                    b.dev_ptr + OFF_DOWN_BI as u64,
                    0i32,
                ),
                SwitchMlpTier::MmapCompact { layer_idx } => {
                    let base = self.stage_compact_top10(*layer_idx)?;
                    (
                        base + COFF_GATE_W as u64,
                        base + COFF_GATE_SC as u64,
                        base + COFF_GATE_BI as u64,
                        base + COFF_UP_W as u64,
                        base + COFF_UP_SC as u64,
                        base + COFF_UP_BI as u64,
                        base + COFF_DOWN_W as u64,
                        base + COFF_DOWN_SC as u64,
                        base + COFF_DOWN_BI as u64,
                        1i32,
                    )
                }
            };

            let (mut sh_gu_w, mut sh_gu_sc, mut sh_gu_bi) = (
                layer.sh_gu_gate.w.ptr,
                layer.sh_gu_gate.sc.ptr,
                layer.sh_gu_gate.bi.ptr,
            );
            let (mut sh_dw, mut sh_dsc, mut sh_dbi) = (
                layer.sh_down.w.ptr,
                layer.sh_down.sc.ptr,
                layer.sh_down.bi.ptr,
            );
            let (mut mx, mut sids, mut swts, mut act11, mut mout) = (
                self.d_mixed.ptr,
                self.d_sel_ids10.ptr,
                self.d_sel_wts11.ptr,
                self.d_moe_act11.ptr,
                self.d_branch_out.ptr,
            );

            let mut p_up = kargs![
                gw_p, gsc_p, gbi_p, uw_p, usc_p, ubi_p, sh_gu_w, sh_gu_sc, sh_gu_bi, mx, sids,
                is_compact, swts, act11
            ];
            let mut p_dn = kargs![
                dw_p, dsc_p, dbi_p, sh_dw, sh_dsc, sh_dbi, act11, sids, is_compact, swts, mout
            ];
            unsafe {
                check_cu(
                    (self.ctx.api.cuLaunchKernel)(
                        self.fns.moe_up_swiglu11,
                        160,
                        11,
                        1,
                        256,
                        1,
                        1,
                        0,
                        stream,
                        p_up.as_mut_ptr(),
                        core::ptr::null_mut(),
                    ),
                    "q38_moe_up_swiglu11",
                )?;
                check_cu(
                    (self.ctx.api.cuLaunchKernel)(
                        self.fns.moe_down_combine11,
                        320,
                        1,
                        1,
                        256,
                        1,
                        1,
                        0,
                        stream,
                        p_dn.as_mut_ptr(),
                        core::ptr::null_mut(),
                    ),
                    "q38_moe_down_combine11",
                )?;
            }

            self.launch_hc_post(stream)?;
        }

        if compute_logits {
            self.launch_hc_pre(stream, &self.final_hc)?;
            self.launch_q4_gemv(
                stream,
                &self.lm_head,
                self.d_mixed.ptr,
                self.d_logits.ptr,
                0,
            )?;
            let (mut log_p, mut out_p, mut pos_p) =
                (self.d_logits.ptr, self.d_token_out.ptr, self.d_pos.ptr);
            let mut p_arg = kargs![log_p, out_p, pos_p];
            unsafe {
                check_cu(
                    (self.ctx.api.cuLaunchKernel)(
                        self.fns.argmax_and_inc_pos,
                        1,
                        1,
                        1,
                        256,
                        1,
                        1,
                        0,
                        stream,
                        p_arg.as_mut_ptr(),
                        core::ptr::null_mut(),
                    ),
                    "q38_argmax_and_inc_pos",
                )?;
            }
        } else {
            let mut pos_p = self.d_pos.ptr;
            let mut p_inc = kargs![pos_p];
            unsafe {
                check_cu(
                    (self.ctx.api.cuLaunchKernel)(
                        self.fns.inc_pos,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        stream,
                        p_inc.as_mut_ptr(),
                        core::ptr::null_mut(),
                    ),
                    "q38_inc_pos",
                )?;
            }
        }
        Ok(())
    }

    fn forward_token_internal(&mut self, tok: u32, compute_logits: bool) -> Result<u32, String> {
        let ring_slot = self.host_pos % PLE_RING_SLOTS;
        let ple_src = self.decode_ple_into_ring_slot(tok, ring_slot)?;
        let tok_slot_ptr = unsafe { self.tok_stage_ring.host_ptr.add(ring_slot * 4) };
        unsafe {
            core::ptr::copy_nonoverlapping((tok as i32).to_le_bytes().as_ptr(), tok_slot_ptr, 4);
        }

        let stream = self.ctx.compute_stream;
        let can_use_graph = self.enable_cuda_graph
            && self.compact_expert_layers == 0
            && (self.host_pos + 1) <= 2051
            && self.decode_graph.is_some();

        if can_use_graph {
            if compute_logits {
                // Single-token decode: issue `10 KiB` `Engram` HtoD on `h2d_stream` + wait on `compute_stream`
                unsafe {
                    check_cu(
                        (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                            self.d_ple_emb.ptr,
                            ple_src.cast(),
                            Q38_D * 4,
                            self.ctx.h2d_stream,
                        ),
                        "HtoD(ple_emb h2d)",
                    )?;
                    check_cu(
                        (self.ctx.api.cuEventRecord)(self.ple_ready_event, self.ctx.h2d_stream),
                        "cuEventRecord(ple_ready)",
                    )?;
                    check_cu(
                        (self.ctx.api.cuStreamWaitEvent)(stream, self.ple_ready_event, 0),
                        "cuStreamWaitEvent(ple_ready)",
                    )?;
                    check_cu(
                        (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                            self.d_token_in.ptr,
                            tok_slot_ptr.cast(),
                            4,
                            stream,
                        ),
                        "HtoD(token_in)",
                    )?;
                }
                self.decode_graph.as_ref().unwrap().replay()?;
            } else {
                // Async pipelined prefill: queue `ple_emb` + `token_in` + 48-layer `prefill_graph` in FIFO order!
                unsafe {
                    check_cu(
                        (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                            self.d_ple_emb.ptr,
                            ple_src.cast(),
                            Q38_D * 4,
                            stream,
                        ),
                        "HtoD(ple_emb prefill)",
                    )?;
                    check_cu(
                        (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                            self.d_token_in.ptr,
                            tok_slot_ptr.cast(),
                            4,
                            stream,
                        ),
                        "HtoD(token_in)",
                    )?;
                }
                self.prefill_graph.as_ref().unwrap().replay()?;
            }
        } else if compute_logits {
            // Decode / final prefill token (`compute_stream` is idle here): overlap `10 KiB` `Engram` HtoD on `h2d_stream` across Layer 0!
            unsafe {
                check_cu(
                    (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                        self.d_ple_emb.ptr,
                        ple_src.cast(),
                        Q38_D * 4,
                        self.ctx.h2d_stream,
                    ),
                    "HtoD(ple_emb)",
                )?;
                check_cu(
                    (self.ctx.api.cuEventRecord)(self.ple_ready_event, self.ctx.h2d_stream),
                    "cuEventRecord(ple_ready)",
                )?;
                check_cu(
                    (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                        self.d_token_in.ptr,
                        tok_slot_ptr.cast(),
                        4,
                        stream,
                    ),
                    "HtoD(token_in)",
                )?;
            }
            self.enqueue_all_48_layers(stream, true, true)?;
        } else {
            // Intermediate prefill token without graph: strictly FIFO on `compute_stream` (zero WAR hazard on `d_ple_emb`)
            unsafe {
                check_cu(
                    (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                        self.d_ple_emb.ptr,
                        ple_src.cast(),
                        Q38_D * 4,
                        stream,
                    ),
                    "HtoD(ple_emb fifo)",
                )?;
                check_cu(
                    (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                        self.d_token_in.ptr,
                        tok_slot_ptr.cast(),
                        4,
                        stream,
                    ),
                    "HtoD(token_in)",
                )?;
            }
            self.enqueue_all_48_layers(stream, false, false)?;
        }

        let mut next_tok = 0u32;
        if compute_logits {
            self.ctx.sync_compute()?;
            let mut out_i32 = 0i32;
            unsafe {
                check_cu(
                    (self.ctx.api.cuMemcpyDtoH_v2)(
                        (&mut out_i32 as *mut i32).cast(),
                        self.d_token_out.ptr,
                        4,
                    ),
                    "DtoH(token_out)",
                )?;
            }
            next_tok = out_i32 as u32;
        }

        self.host_pos += 1;
        Ok(next_tok)
    }
}
