//! Real `Qwen3.8-Flash-Next` / `Qwen4Exp` (`106 GB` 22-Shard `safetensors`) Full-GPU Blackwell Engine.
//!
//! Architecture (`48` layers, `D = 2560`, `hc_count = 4`, `512` experts `Top-10` + Shared Expert):
//! - Group-32 4-bit affine (`uint4` `LDG.E.128` = 32 nibbles + BF16 / lossless `24-bit` Z24 scale/bias)
//!   across projections, Hyper-Connection mixers, `lm_head` (`248,320 × 2,560`), and `512`-expert MoE.
//! - Zero-loss (`0.000000-ULP`) `24-bit` (`lo_u16 + hi_u8`) `(scale, bias)` GPU packing (`1,425 MiB`/layer
//!   vs `1,500 MiB`/layer = `-3.52 GiB` VRAM) + `PinnedHostBuf` `embed_tokens` (`-0.49 GiB` VRAM) +
//!   dynamic QSA KV cache growth (with elastic `Vram -> HostPinned` demotion at ultra-long contexts),
//!   unlocking **`48/48` full GDDR7 VRAM residency (`0` PCIe spill)** on `RTX PRO 5000 72GB`.
//! - Layer 1 (`ple_layer_ids = [2]`) `29.80 GiB` (`320M` rows × `160` dims) 3-Gram Host `Engram`/`PLE`
//!   table mapped zero-copy via POSIX `mmap` with pre-resolved shard slices (`0` HashMap lookups/step),
//!   dequantized (`16` heads = `10 KiB`) into a 256-slot pinned host ring, and streamed via `cuMemcpyHtoDAsync_v2`.
//! - Programmatic Dependent Launch (`PDL`) + 48-Layer `CUDA Graph` capture (`decode_graph` + `prefill_graph`).

use core::ffi::c_void;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::cuda_ffi::{
    check_cu, compile_cuda_to_ptx_nvrtc, CUdeviceptr, CUevent, CUfunction, CUlaunchAttribute,
    CUlaunchConfig, CUmodule, CUstream, CapturedGraphExec, CudaContextHandle, DeviceBuf,
    PinnedHostBuf, CU_EVENT_DISABLE_TIMING,
};
use crate::qwen38_engram::{make_resident, EngramBytes, EngramPolicy};
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
pub const Q38_MAX_BATCH: usize = 64;
pub const Q38_MAX_VERIFY_BATCH: usize = 8;
const PLE_RING_SLOTS: usize = 256;

// Uncompressed `1,500 MiB` `switch_mlp` layer slab fallback offsets
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

// `0-ULP` lossless `24-bit` (`Z24`: `50 MiB` `lo_u32` + `25 MiB` `hi_u32`) `1,425 MiB` `switch_mlp` slab
const SW_GROUPS_PER_PROJ: usize = 26_214_400; // `512 * 640 * 80 == 512 * 2560 * 20`
const SW_Z24_LO_BYTES: usize = SW_GROUPS_PER_PROJ * 2; // 50 MiB
const SW_Z24_HI_BYTES: usize = SW_GROUPS_PER_PROJ; // 25 MiB
const SW_Z24_PROJ_BYTES: usize = SW_W_BYTES + SW_Z24_LO_BYTES + SW_Z24_HI_BYTES; // 475 MiB
const SW_Z24_LAYER_BYTES: usize = 3 * SW_Z24_PROJ_BYTES; // 1,425 MiB

const ZOFF_GATE_W: usize = 0;
const ZOFF_GATE_LO: usize = SW_W_BYTES;
const ZOFF_GATE_HI: usize = SW_W_BYTES + SW_Z24_LO_BYTES;
const ZOFF_UP_W: usize = SW_Z24_PROJ_BYTES;
const ZOFF_UP_LO: usize = SW_Z24_PROJ_BYTES + SW_W_BYTES;
const ZOFF_UP_HI: usize = SW_Z24_PROJ_BYTES + SW_W_BYTES + SW_Z24_LO_BYTES;
const ZOFF_DOWN_W: usize = 2 * SW_Z24_PROJ_BYTES;
const ZOFF_DOWN_LO: usize = 2 * SW_Z24_PROJ_BYTES + SW_W_BYTES;
const ZOFF_DOWN_HI: usize = 2 * SW_Z24_PROJ_BYTES + SW_W_BYTES + SW_Z24_LO_BYTES;

// Byte offsets inside the `30,720,000`-byte (`29.30 MiB`) compact `Top-10` staging slab (`MmapCompact`)
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

/// Losslessly (`0-ULP`) packs `n_groups` (`n_groups % 4 == 0`) BF16 `(scale, bias)` pairs into
/// `lo_u32_out` (`n_groups / 2` `u32`s) and `hi_u32_out` (`n_groups / 4` `u32`s) whenever all
/// non-zero normal exponents fit in `<= 15` consecutive exponents (`[min_exp .. min_exp + 14]`)
/// with zero subnormals / NaN / Inf. Returns `Some((base_sc_m1, base_bi_m1))` on exact fit.
pub fn pack_z24_sb_slice(
    sc_bytes: &[u8],
    bi_bytes: &[u8],
    lo_u32_out: &mut [u32],
    hi_u32_out: &mut [u32],
) -> Option<(u8, u8)> {
    assert_eq!(sc_bytes.len(), bi_bytes.len());
    assert_eq!(sc_bytes.len() % 8, 0);
    let n_groups = sc_bytes.len() / 2;
    assert_eq!(lo_u32_out.len(), n_groups / 2);
    assert_eq!(hi_u32_out.len(), n_groups / 4);

    let scan_range = |bytes: &[u8]| -> Option<(u8, u8)> {
        let n_threads = if bytes.len() >= (1 << 20) { 8 } else { 1 };
        let chunk_len = ((bytes.len() / 2).div_ceil(n_threads) * 2).max(2);
        let parts: Vec<Option<(u8, u8)>> = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(n_threads);
            for chunk in bytes.chunks(chunk_len) {
                handles.push(s.spawn(move || {
                    let mut mn = 255u8;
                    let mut mx = 0u8;
                    for pair in chunk.as_chunks::<2>().0 {
                        let u = u16::from_le_bytes(*pair);
                        let exp = ((u >> 7) & 0xFF) as u8;
                        let mant = (u & 0x7F) as u8;
                        if exp == 255 {
                            return None;
                        }
                        if exp == 0 {
                            if mant != 0 {
                                return None;
                            }
                        } else {
                            if exp < mn {
                                mn = exp;
                            }
                            if exp > mx {
                                mx = exp;
                            }
                        }
                    }
                    Some((mn, mx))
                }));
            }
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let mut gmin = 255u8;
        let mut gmax = 0u8;
        for p in parts {
            let (mn, mx) = p?;
            if mn < gmin {
                gmin = mn;
            }
            if mx > gmax {
                gmax = mx;
            }
        }
        if gmin > gmax {
            // All elements are `±0.0`; any `base_m1` works since `code4 == 0` decodes to `exp = 0`.
            return Some((1, 1));
        }
        if gmax - gmin > 14 {
            return None;
        }
        Some((gmin, gmax))
    };

    let (min_sc, _) = scan_range(sc_bytes)?;
    let (min_bi, _) = scan_range(bi_bytes)?;
    let base_sc_m1 = min_sc - 1;
    let base_bi_m1 = min_bi - 1;

    let n_quads = n_groups / 4;
    let n_threads = if n_quads >= (1 << 18) { 8 } else { 1 };
    let quads_per_thread = n_quads.div_ceil(n_threads).max(1);

    let lo_addr = lo_u32_out.as_mut_ptr() as usize;
    let hi_addr = hi_u32_out.as_mut_ptr() as usize;

    std::thread::scope(|s| {
        for t in 0..n_threads {
            let q_start = t * quads_per_thread;
            if q_start >= n_quads {
                break;
            }
            let q_end = (q_start + quads_per_thread).min(n_quads);
            let sc_sub = &sc_bytes[q_start * 8..q_end * 8];
            let bi_sub = &bi_bytes[q_start * 8..q_end * 8];
            s.spawn(move || unsafe {
                let lo_ptr = (lo_addr as *mut u32).add(q_start * 2);
                let hi_ptr = (hi_addr as *mut u32).add(q_start);
                for q in 0..(q_end - q_start) {
                    let mut lo_vals = [0u16; 4];
                    let mut hi_vals = [0u8; 4];
                    for sub in 0..4usize {
                        let b_off = q * 8 + sub * 2;
                        let sc_u = u16::from_le_bytes([sc_sub[b_off], sc_sub[b_off + 1]]);
                        let bi_u = u16::from_le_bytes([bi_sub[b_off], bi_sub[b_off + 1]]);

                        let sc_sm = (((sc_u >> 8) & 0x80) | (sc_u & 0x7F)) as u8;
                        let sc_exp = ((sc_u >> 7) & 0xFF) as u8;
                        let sc_c = if sc_exp == 0 { 0 } else { sc_exp - base_sc_m1 };

                        let bi_sm = (((bi_u >> 8) & 0x80) | (bi_u & 0x7F)) as u8;
                        let bi_exp = ((bi_u >> 7) & 0xFF) as u8;
                        let bi_c = if bi_exp == 0 { 0 } else { bi_exp - base_bi_m1 };

                        lo_vals[sub] = ((bi_sm as u16) << 8) | (sc_sm as u16);
                        hi_vals[sub] = (bi_c << 4) | sc_c;
                    }
                    *lo_ptr.add(q * 2) = (lo_vals[0] as u32) | ((lo_vals[1] as u32) << 16);
                    *lo_ptr.add(q * 2 + 1) = (lo_vals[2] as u32) | ((lo_vals[3] as u32) << 16);
                    *hi_ptr.add(q) = u32::from_le_bytes(hi_vals);
                }
            });
        }
    });

    Some((base_sc_m1, base_bi_m1))
}

/// CPU reference mirror of CUDA `load_z24_sb_pair` (`0-ULP` bit-identity verification).
#[inline]
pub fn decode_z24_sb_pair_ref(
    lo_u32: &[u32],
    hi_u32: &[u32],
    idx: usize,
    base_sc_m1: u8,
    base_bi_m1: u8,
) -> (f32, f32) {
    let lo_pair = lo_u32[idx >> 1];
    let lo16 = (lo_pair >> ((idx & 1) * 16)) & 0xFFFF;
    let hi_quad = hi_u32[idx >> 2];
    let hi8 = (hi_quad >> ((idx & 3) * 8)) & 0xFF;

    let sc_sm = lo16 & 0xFF;
    let sc_c = hi8 & 0xF;
    let sc_exp = if sc_c == 0 {
        0
    } else {
        (base_sc_m1 as u32) + sc_c
    };
    let sc = f32::from_bits(((sc_sm & 0x80) << 24) | (sc_exp << 23) | ((sc_sm & 0x7F) << 16));

    let bi_sm = lo16 >> 8;
    let bi_c = hi8 >> 4;
    let bi_exp = if bi_c == 0 {
        0
    } else {
        (base_bi_m1 as u32) + bi_c
    };
    let bi = f32::from_bits(((bi_sm & 0x80) << 24) | (bi_exp << 23) | ((bi_sm & 0x7F) << 16));
    (sc, bi)
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

fn u32_slice_bytes(v: &[u32]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(v.as_ptr().cast::<u8>(), core::mem::size_of_val(v)) }
}

fn parallel_copy_into_pinned(dst_base: *mut u8, slices: &[(usize, &[u8])]) {
    let base_addr = dst_base as usize;
    std::thread::scope(|s| {
        for &(off, bytes) in slices {
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

#[cfg(target_os = "linux")]
fn advise_willneed_slice(bytes: &[u8]) {
    unsafe extern "C" {
        fn madvise(addr: *mut c_void, length: usize, advice: i32) -> i32;
    }
    const MADV_WILLNEED: i32 = 3;
    let page = 4096usize;
    let start = (bytes.as_ptr() as usize) & !(page - 1);
    let end = ((bytes.as_ptr() as usize) + bytes.len() + page - 1) & !(page - 1);
    if end > start {
        unsafe {
            let _ = madvise(start as *mut c_void, end - start, MADV_WILLNEED);
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn advise_willneed_slice(_bytes: &[u8]) {}

struct Q4LinearHostMapped {
    w: PinnedHostBuf,
    sc: PinnedHostBuf,
    bi: PinnedHostBuf,
}

impl Q4LinearHostMapped {
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
        let w = PinnedHostBuf::alloc_uninit(ctx, w_tv.data.len())?;
        let sc = PinnedHostBuf::alloc_uninit(ctx, sc_tv.data.len())?;
        let bi = PinnedHostBuf::alloc_uninit(ctx, bi_tv.data.len())?;
        parallel_copy_into_pinned(w.host_ptr, &[(0, w_tv.data)]);
        parallel_copy_into_pinned(sc.host_ptr, &[(0, sc_tv.data)]);
        parallel_copy_into_pinned(bi.host_ptr, &[(0, bi_tv.data)]);
        Ok(Self { w, sc, bi })
    }
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

#[derive(Clone, Copy)]
struct Z24LayerBases {
    up_packed_m1: u32,   // `[gate_sc | (gate_bi<<8) | (up_sc<<16) | (up_bi<<24)]`
    down_packed_m1: u32, // `[down_sc | (down_bi<<8)]`
}

#[path = "qwen38_mixed.rs"]
mod mixed;
use mixed::{LayerExpertPlan, MixedExperts, MixedPlacementPlan};

enum SwitchMlpTier {
    Mixed(MixedExperts),
    Vram {
        buf: DeviceBuf,
        z24: Option<Z24LayerBases>,
    },
    HostPinned {
        buf: PinnedHostBuf,
        z24: Option<Z24LayerBases>,
    },
    MmapCompact {
        layer_idx: usize,
    },
}

struct PleParameter {
    bytes: EngramBytes,
    dictionary: bool,
}

impl PleParameter {
    fn open(st: &ShardedSafeTensors, key: &str, warm: bool) -> Result<Self, String> {
        let packed = format!("{key}.q38dict1");
        let rows = Q38_PLE_SHARD_ROWS as usize;
        let (name, t, dictionary) = match (st.get(key), st.get(&packed)) {
            (Some(t), None)
                if t.dtype == "BF16" && t.shape == [rows, 5] && t.data.len() == rows * 10 =>
            {
                (key, t, false)
            }
            (None, Some(t)) if t.dtype == "U8" && t.shape == [t.data.len()] => {
                crate::qwen38_pack::Frame::open(t.data, rows * 5)
                    .map_err(|err| format!("{packed}: {err}"))?;
                (packed.as_str(), t, true)
            }
            _ => {
                return Err(format!(
                    "missing, ambiguous or malformed Engram parameter {key}"
                ))
            }
        };
        if warm {
            advise_willneed_slice(t.data);
        }
        Ok(Self {
            bytes: EngramBytes::Mapped(
                st.tensor_range(name)
                    .ok_or_else(|| format!("missing Engram parameter {name}"))?,
            ),
            dictionary,
        })
    }

    fn row(&self, st: &ShardedSafeTensors, row: usize) -> Result<[u16; 5], String> {
        let data = self.bytes.get(st);
        if self.dictionary {
            return crate::qwen38_pack::Frame::open(data, Q38_PLE_SHARD_ROWS as usize * 5)
                .and_then(|f| f.row5(row))
                .map_err(|err| format!("invalid Engram row {row}: {err}"));
        }
        let bytes = data
            .as_chunks::<10>()
            .0
            .get(row)
            .ok_or_else(|| format!("invalid Engram row {row}"))?;
        Ok(std::array::from_fn(|i| {
            u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])
        }))
    }
}

#[path = "qwen38_engram_tile.rs"]
mod engram_tile;

enum PleShard {
    Planes {
        weights: EngramBytes,
        scales: PleParameter,
        biases: PleParameter,
    },
    Tiled(engram_tile::TiledShard),
}

impl PleShard {
    fn bytes(&self, st: &ShardedSafeTensors) -> usize {
        match self {
            Self::Planes {
                weights,
                scales,
                biases,
            } => [weights, &scales.bytes, &biases.bytes]
                .iter()
                .map(|r| r.get(st).len())
                .sum(),
            Self::Tiled(shard) => shard.bytes(st).len(),
        }
    }

    fn row<'a>(
        &'a self,
        st: &'a ShardedSafeTensors,
        row: usize,
    ) -> Result<engram_tile::EngramRow<'a>, String> {
        match self {
            Self::Tiled(shard) => shard.row(st, row),
            Self::Planes {
                weights,
                scales,
                biases,
            } => {
                let w = weights
                    .get(st)
                    .as_chunks::<80>()
                    .0
                    .get(row)
                    .ok_or_else(|| format!("invalid Engram weight row {row}"))?;
                Ok((w, scales.row(st, row)?, biases.row(st, row)?))
            }
        }
    }
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
    shards: Vec<PleShard>,
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
    q4_gemv_2560: CUfunction,
    q4_gemv_10k: CUfunction,
    ple_inject: CUfunction,
    hc_norm4: CUfunction,
    hc_down_inj_10k: CUfunction,
    hc_up_mix4: CUfunction,
    hc_inject4_vec: CUfunction,
    gdn_conv_silu: CUfunction,
    gdn_recurrent_step: CUfunction,
    qsa_prep_and_cache: CUfunction,
    qsa_score_blocks: CUfunction,
    qsa_select_top512: CUfunction,
    qsa_attn: CUfunction,
    bf16_router_512: CUfunction,
    softmax_topk10: CUfunction,
    moe_up_swiglu11: CUfunction,
    moe_up_swiglu11_z24: CUfunction,
    moe_down_combine11: CUfunction,
    moe_down_combine11_z24: CUfunction,
    argmax_fast: CUfunction,
    inc_pos: CUfunction,
    embed_init_hc4_batch: CUfunction,
    q4_gemm_b8: CUfunction,
    ple_inject_seq_b: CUfunction,
    hc_norm4_batch: CUfunction,
    hc_down_inj_10k_b4: CUfunction,
    hc_up_mix4_b8: CUfunction,
    hc_inject4_vec_batch: CUfunction,
    gdn_conv_silu_seq_b: CUfunction,
    gdn_recurrent_step_seq_b: CUfunction,
    qsa_prep_batch: CUfunction,
    qsa_build_blocks_batch: CUfunction,
    qsa_score_blocks_batch: CUfunction,
    qsa_select_top512_batch: CUfunction,
    qsa_attn_batch: CUfunction,
    bf16_router_512_b4: CUfunction,
    softmax_topk10_batch: CUfunction,
    moe_build_plan_b64: CUfunction,
    moe_up_grouped_z24: CUfunction,
    moe_down_grouped_z24: CUfunction,
    moe_combine11_batch: CUfunction,
    argmax_fast_batch: CUfunction,
    add_pos: CUfunction,
    q4_gemv_2560_b2: CUfunction,
    q4_gemv_2560_b4: CUfunction,
    moe_up_swiglu11_z24_bn: CUfunction,
    moe_down_combine11_z24_bn: CUfunction,
    ple_inject_ckpt_b8: CUfunction,
    gdn_conv_silu_ckpt_b8: CUfunction,
    gdn_recurrent_step_ckpt_b8: CUfunction,
    q4_gemv_b2: CUfunction,
    hc_down_inj_10k_b2: CUfunction,
    hc_up_mix4_b2: CUfunction,
}

unsafe impl Send for Qwen38KernelFns {}
unsafe impl Sync for Qwen38KernelFns {}

struct Qwen38BatchBuffers {
    mask_stride: usize,
    d_b_tokens: DeviceBuf,
    d_b_token_out: DeviceBuf,
    d_b_ple_emb: DeviceBuf,
    d_b_ple_kv: DeviceBuf,
    d_b_hc: DeviceBuf,
    d_b_hc_normed: DeviceBuf,
    d_b_hc_low: DeviceBuf,
    d_b_hc_inj4: DeviceBuf,
    d_b_mixed: DeviceBuf,
    d_b_branch_out: DeviceBuf,
    d_b_gdn_in: DeviceBuf,
    d_b_qkv_conv: DeviceBuf,
    d_b_gdn_out: DeviceBuf,
    d_b_q_rot: DeviceBuf,
    d_b_q_gate: DeviceBuf,
    d_b_idx_q: DeviceBuf,
    d_b_qsa_scores: DeviceBuf,
    d_b_qsa_mask: DeviceBuf,
    d_b_router_logits: DeviceBuf,
    d_b_sel_ids10: DeviceBuf,
    d_b_sel_wts11: DeviceBuf,
    d_b_exp_offsets: DeviceBuf,
    d_b_exp_pairs: DeviceBuf,
    d_b_moe_act11: DeviceBuf,
    d_b_down_part: DeviceBuf,
    d_b_logits: DeviceBuf,
}

impl Qwen38BatchBuffers {
    fn alloc(ctx: &Arc<CudaContextHandle>, mask_stride: usize) -> Result<Self, String> {
        let b = Q38_MAX_BATCH;
        Ok(Self {
            mask_stride,
            d_b_tokens: DeviceBuf::alloc_zeroed(ctx, b * 4)?,
            d_b_token_out: DeviceBuf::alloc_zeroed(ctx, Q38_MAX_VERIFY_BATCH * 4)?,
            d_b_ple_emb: DeviceBuf::alloc_zeroed(ctx, b * Q38_D * 4)?,
            d_b_ple_kv: DeviceBuf::alloc_zeroed(ctx, b * (Q38_HC_D + Q38_D) * 4)?,
            d_b_hc: DeviceBuf::alloc_zeroed(ctx, b * Q38_HC_D * 4)?,
            d_b_hc_normed: DeviceBuf::alloc_zeroed(ctx, b * Q38_HC_D * 4)?,
            d_b_hc_low: DeviceBuf::alloc_zeroed(ctx, b * Q38_HC_LOW * 4)?,
            d_b_hc_inj4: DeviceBuf::alloc_zeroed(ctx, b * 4 * 4)?,
            d_b_mixed: DeviceBuf::alloc_zeroed(ctx, b * Q38_D * 4)?,
            d_b_branch_out: DeviceBuf::alloc_zeroed(ctx, b * Q38_D * 4)?,
            // `16,480 >= 13,952`, so `d_b_gdn_in` also serves as `d_b_qsa_proj`!
            d_b_gdn_in: DeviceBuf::alloc_zeroed(ctx, b * 16_480 * 4)?,
            d_b_qkv_conv: DeviceBuf::alloc_zeroed(ctx, b * Q38_HC_D * 4)?,
            // `6,144` floats/row serves both `gdn_out` and `qsa_attn_out`!
            d_b_gdn_out: DeviceBuf::alloc_zeroed(ctx, b * 6_144 * 4)?,
            d_b_q_rot: DeviceBuf::alloc_zeroed(ctx, b * 6_144 * 4)?,
            d_b_q_gate: DeviceBuf::alloc_zeroed(ctx, b * 6_144 * 4)?,
            d_b_idx_q: DeviceBuf::alloc_zeroed(ctx, b * 512 * 4)?,
            d_b_qsa_scores: DeviceBuf::alloc_zeroed(ctx, b * mask_stride * 4)?,
            d_b_qsa_mask: DeviceBuf::alloc_zeroed(ctx, b * mask_stride * 4)?,
            d_b_router_logits: DeviceBuf::alloc_zeroed(ctx, b * Q38_EXP * 4)?,
            d_b_sel_ids10: DeviceBuf::alloc_zeroed(ctx, b * Q38_TOP_K * 4)?,
            d_b_sel_wts11: DeviceBuf::alloc_zeroed(ctx, b * (Q38_TOP_K + 1) * 4)?,
            d_b_exp_offsets: DeviceBuf::alloc_zeroed(ctx, (Q38_EXP + 1) * 4)?,
            d_b_exp_pairs: DeviceBuf::alloc_zeroed(ctx, b * Q38_TOP_K * 4)?,
            d_b_moe_act11: DeviceBuf::alloc_zeroed(ctx, b * (Q38_TOP_K + 1) * Q38_MOE_FF * 4)?,
            d_b_down_part: DeviceBuf::alloc_zeroed(ctx, b * (Q38_TOP_K + 1) * Q38_D * 4)?,
            d_b_logits: DeviceBuf::alloc_zeroed(ctx, Q38_MAX_VERIFY_BATCH * Q38_VOCAB * 4)?,
        })
    }
}

/// Device-resident recurrent state snapshot (`112.57 MiB` across 36 GDN layers + Layer 1 PLE)
/// enabling `< 0.1 ms` `O(1)` speculative verification rollback!
pub struct Qwen38RecurrentSnapshot {
    ple_hist9: DeviceBuf,
    gdn_conv_hists: Vec<DeviceBuf>,
    gdn_states: Vec<DeviceBuf>,
    pub saved_pos: usize,
    pub saved_prev2: i64,
    pub saved_prev1: i64,
}

enum VerifyCkptSlab {
    Vram(DeviceBuf),
    HostMapped(PinnedHostBuf),
}

impl VerifyCkptSlab {
    fn alloc(ctx: &Arc<CudaContextHandle>, byte_len: usize) -> Result<Self, String> {
        let reserve = 96 * 1024 * 1024;
        if let Ok((free_vram, _)) = ctx.mem_info() {
            if free_vram > byte_len.saturating_add(reserve) {
                if let Ok(buf) = DeviceBuf::alloc_uninit(ctx, byte_len) {
                    return Ok(Self::Vram(buf));
                }
            }
        }
        Ok(Self::HostMapped(PinnedHostBuf::alloc_uninit(
            ctx, byte_len,
        )?))
    }

    #[inline]
    fn dev_ptr(&self) -> CUdeviceptr {
        match self {
            Self::Vram(b) => b.ptr,
            Self::HostMapped(b) => b.dev_ptr,
        }
    }

    unsafe fn copy_row_to_dev_async(
        &self,
        ctx: &CudaContextHandle,
        dst_ptr: CUdeviceptr,
        row_idx: usize,
        row_bytes: usize,
        stream: CUstream,
        op_name: &str,
    ) -> Result<(), String> {
        let off = row_idx * row_bytes;
        match self {
            Self::Vram(b) => check_cu(
                (ctx.api.cuMemcpyDtoDAsync_v2)(dst_ptr, b.ptr + (off as u64), row_bytes, stream),
                op_name,
            ),
            Self::HostMapped(b) => check_cu(
                (ctx.api.cuMemcpyHtoDAsync_v2)(
                    dst_ptr,
                    b.host_ptr.add(off).cast(),
                    row_bytes,
                    stream,
                ),
                op_name,
            ),
        }
    }
}

struct Qwen38VerifyRowCkpt {
    cap_rows: usize,
    ple_hist9_rows: VerifyCkptSlab,
    gdn_conv_rows: Vec<VerifyCkptSlab>,
    gdn_state_rows: Vec<VerifyCkptSlab>,
    last_pos0: usize,
    last_b_count: usize,
    ple_tails_after_row: [(i64, i64); Q38_MAX_VERIFY_BATCH],
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BatchLogitMode {
    NoLogits,
    LastLogitOnly,
    AllVerifyLogits,
}

pub struct Qwen38GpuEngine {
    pub ctx: Arc<CudaContextHandle>,
    fns: Qwen38KernelFns,
    pdl: bool,
    zero_centered: bool,
    all_layers_z24: bool,
    sharded_mmap: ShardedSafeTensors,
    embed: Q4LinearHostMapped,
    final_hc: HyperConnGpu,
    lm_head: Q4LinearGpu,
    inv_freq32: DeviceBuf,
    layers: Vec<Qwen38LayerGpu>,
    pub resident_expert_layers: usize,
    pub pinned_expert_layers: usize,
    pub compact_expert_layers: usize,
    pub mixed_expert_layers: usize,
    pub hot_resident_experts: usize,
    pub max_ctx: usize,
    kv_cap: usize,
    pub enable_cuda_graph: bool,
    use_gdn_sigmoid_gate: i32,
    ple_stage_ring: PinnedHostBuf,
    tok_stage_ring: PinnedHostBuf,
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
    batch_bufs: Option<Qwen38BatchBuffers>,
    verify_ckpt: Option<Qwen38VerifyRowCkpt>,
    host_pos: usize,
    decode_graph: Option<CapturedGraphExec>,
    prefill_graph: Option<CapturedGraphExec>,
}

unsafe impl Send for Qwen38GpuEngine {}

fn discover_qwen38_cubin() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("LEANSONIC_QWEN38_CUBIN_PATH") {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Some(pb);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for rel in [
                PathBuf::from("qwen38_sm120a.cubin"),
                PathBuf::from("../../build/qwen38_sm120a.cubin"),
                PathBuf::from("../build/qwen38_sm120a.cubin"),
            ] {
                let cand = dir.join(rel);
                if cand.is_file() {
                    return Some(cand);
                }
            }
        }
    }
    let cwd_cand = PathBuf::from("build/qwen38_sm120a.cubin");
    if cwd_cand.is_file() {
        return Some(cwd_cand);
    }
    None
}

fn load_kernel_fns(ctx: &Arc<CudaContextHandle>) -> Result<Qwen38KernelFns, String> {
    let module_image: Vec<u8> = if let Some(cubin_path) = discover_qwen38_cubin() {
        let mut raw = std::fs::read(&cubin_path)
            .map_err(|e| format!("read {}: {e}", cubin_path.display()))?;
        raw.push(0);
        raw
    } else {
        let header_nvrtc = CUDA_HEADER_SRC.replace("#include <stdint.h>", "");
        let inlined =
            CUDA_QWEN38_SRC.replace("#include \"include/leansonic_verified.cuh\"", &header_nvrtc);
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
        q4_gemv_2560: get_fn("q38_q4_gemv_2560")?,
        q4_gemv_10k: get_fn("q38_q4_gemv_10k")?,
        ple_inject: get_fn("q38_ple_inject")?,
        hc_norm4: get_fn("q38_hc_norm4")?,
        hc_down_inj_10k: get_fn("q38_hc_down_inj_10k")?,
        hc_up_mix4: get_fn("q38_hc_up_mix4")?,
        hc_inject4_vec: get_fn("q38_hc_inject4_vec")?,
        gdn_conv_silu: get_fn("q38_gdn_conv_silu")?,
        gdn_recurrent_step: get_fn("q38_gdn_recurrent_step")?,
        qsa_prep_and_cache: get_fn("q38_qsa_prep_and_cache")?,
        qsa_score_blocks: get_fn("q38_qsa_score_blocks")?,
        qsa_select_top512: get_fn("q38_qsa_select_top512")?,
        qsa_attn: get_fn("q38_qsa_attn")?,
        bf16_router_512: get_fn("q38_bf16_router_512")?,
        softmax_topk10: get_fn("q38_softmax_topk10")?,
        moe_up_swiglu11: get_fn("q38_moe_up_swiglu11")?,
        moe_up_swiglu11_z24: get_fn("q38_moe_up_swiglu11_z24")?,
        moe_down_combine11: get_fn("q38_moe_down_combine11")?,
        moe_down_combine11_z24: get_fn("q38_moe_down_combine11_z24")?,
        argmax_fast: get_fn("q38_argmax_fast")?,
        inc_pos: get_fn("q38_inc_pos")?,
        embed_init_hc4_batch: get_fn("q38_embed_init_hc4_batch")?,
        q4_gemm_b8: get_fn("q38_q4_gemm_b8")?,
        ple_inject_seq_b: get_fn("q38_ple_inject_seq_b")?,
        hc_norm4_batch: get_fn("q38_hc_norm4_batch")?,
        hc_down_inj_10k_b4: get_fn("q38_hc_down_inj_10k_b4")?,
        hc_up_mix4_b8: get_fn("q38_hc_up_mix4_b8")?,
        hc_inject4_vec_batch: get_fn("q38_hc_inject4_vec_batch")?,
        gdn_conv_silu_seq_b: get_fn("q38_gdn_conv_silu_seq_b")?,
        gdn_recurrent_step_seq_b: get_fn("q38_gdn_recurrent_step_seq_b")?,
        qsa_prep_batch: get_fn("q38_qsa_prep_batch")?,
        qsa_build_blocks_batch: get_fn("q38_qsa_build_blocks_batch")?,
        qsa_score_blocks_batch: get_fn("q38_qsa_score_blocks_batch")?,
        qsa_select_top512_batch: get_fn("q38_qsa_select_top512_batch")?,
        qsa_attn_batch: get_fn("q38_qsa_attn_batch")?,
        bf16_router_512_b4: get_fn("q38_bf16_router_512_b4")?,
        softmax_topk10_batch: get_fn("q38_softmax_topk10_batch")?,
        moe_build_plan_b64: get_fn("q38_moe_build_plan_b64")?,
        moe_up_grouped_z24: get_fn("q38_moe_up_grouped_z24")?,
        moe_down_grouped_z24: get_fn("q38_moe_down_grouped_z24")?,
        moe_combine11_batch: get_fn("q38_moe_combine11_batch")?,
        argmax_fast_batch: get_fn("q38_argmax_fast_batch")?,
        add_pos: get_fn("q38_add_pos")?,
        q4_gemv_2560_b2: get_fn("q38_q4_gemv_2560_b2")?,
        q4_gemv_2560_b4: get_fn("q38_q4_gemv_2560_b4")?,
        moe_up_swiglu11_z24_bn: get_fn("q38_moe_up_swiglu11_z24_bn")?,
        moe_down_combine11_z24_bn: get_fn("q38_moe_down_combine11_z24_bn")?,
        ple_inject_ckpt_b8: get_fn("q38_ple_inject_ckpt_b8")?,
        gdn_conv_silu_ckpt_b8: get_fn("q38_gdn_conv_silu_ckpt_b8")?,
        gdn_recurrent_step_ckpt_b8: get_fn("q38_gdn_recurrent_step_ckpt_b8")?,
        q4_gemv_b2: get_fn("q38_q4_gemv_b2")?,
        hc_down_inj_10k_b2: get_fn("q38_hc_down_inj_10k_b2")?,
        hc_up_mix4_b2: get_fn("q38_hc_up_mix4_b2")?,
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
        Self::load_from_dir_for_spec(dir, max_ctx, true, false)
    }

    pub fn load_from_dir_with_opts(
        dir: impl AsRef<Path>,
        max_ctx: usize,
        enable_cuda_graph: bool,
    ) -> Result<Self, String> {
        Self::load_from_dir_for_spec(dir, max_ctx, enable_cuda_graph, false)
    }

    /// Loads `Qwen3.8-Flash-Next`; when `reserve_for_mtp == true` (`--spec K`, `K >= 1`), reserves
    /// `2,048 MiB` of GDDR7 VRAM on `72 GB` GPUs so `Qwen38MtpDrafter`'s Z24 MoE slab lives 100% in VRAM!
    pub fn load_from_dir_for_spec(
        dir: impl AsRef<Path>,
        max_ctx: usize,
        enable_cuda_graph: bool,
        reserve_for_mtp: bool,
    ) -> Result<Self, String> {
        let dir = dir.as_ref();
        let t0 = Instant::now();
        let ctx = CudaContextHandle::init(0)?;
        ctx.bind_current()?;
        let fns = load_kernel_fns(&ctx)?;
        let pdl = ctx.api.cuLaunchKernelEx.is_some()
            && std::env::var("LEANSONIC_NO_PDL")
                .map(|v| v != "1")
                .unwrap_or(true);
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

        // Place `496.6 MiB` `embed_tokens` in GPU-mapped `PinnedHostBuf` (only `1,600 B` read/token!),
        // saving `0.49 GiB` of GDDR7 VRAM so all `48/48` `switch_mlp` slabs fit on `RTX PRO 5000 72GB`!
        let embed = Q4LinearHostMapped::from_sharded(
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

        // Start QSA KV cache at `max_ctx.min(4096)` (`129 MiB` total across 12 QSA layers) and grow
        // elastically via `ensure_kv_capacity` if a prompt/generation exceeds `4,096` tokens.
        let kv_cap = max_ctx.clamp(256, 4096);
        let init_blocks = (kv_cap / 4) + 16;
        let d_token_in = DeviceBuf::alloc_zeroed(&ctx, 16)?;
        let d_token_out = DeviceBuf::alloc_zeroed(&ctx, 16)?;
        let d_pos = DeviceBuf::alloc_zeroed(&ctx, 16)?;
        let d_hc = DeviceBuf::alloc_zeroed(&ctx, Q38_HC_D * 4)?;
        let d_hc_normed = DeviceBuf::alloc_zeroed(&ctx, Q38_HC_D * 4)?;
        let d_hc_low = DeviceBuf::alloc_zeroed(&ctx, Q38_HC_LOW * 4)?;
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
        let d_qsa_scores = DeviceBuf::alloc_zeroed(&ctx, init_blocks * 4)?;
        let d_qsa_mask = DeviceBuf::alloc_zeroed(&ctx, init_blocks * 4)?;
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
        let warm_engram = available_host_ram_bytes() > 40 * 1024 * 1024 * 1024;

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

                let mut shards = Vec::with_capacity(Q38_PLE_SHARDS);
                for s in 0..Q38_PLE_SHARDS {
                    let sp = format!("{pp}.ple_embedding.ngram_embedding.shard_{s}");
                    let tiled = format!("{sp}.q38tile4");
                    if st.get(&tiled).is_some() {
                        for suffix in [
                            "weight",
                            "scales",
                            "biases",
                            "scales.q38dict1",
                            "biases.q38dict1",
                        ] {
                            if st.get(&format!("{sp}.{suffix}")).is_some() {
                                return Err(format!("ambiguous Engram layout: {sp}"));
                            }
                        }
                        let shard = engram_tile::TiledShard::open(
                            &st,
                            &tiled,
                            Q38_PLE_SHARD_ROWS as usize,
                        )?;
                        if warm_engram {
                            advise_willneed_slice(shard.bytes(&st));
                        }
                        shards.push(PleShard::Tiled(shard));
                        continue;
                    }
                    let key = format!("{sp}.weight");
                    let w_tv = st.require(&key, "U32", &[2_500_012, 20]);
                    if warm_engram {
                        advise_willneed_slice(w_tv.data);
                    }
                    shards.push(PleShard::Planes {
                        weights: EngramBytes::Mapped(
                            st.tensor_range(&key)
                                .ok_or_else(|| format!("missing Engram weights {key}"))?,
                        ),
                        scales: PleParameter::open(&st, &format!("{sp}.scales"), warm_engram)?,
                        biases: PleParameter::open(&st, &format!("{sp}.biases"), warm_engram)?,
                    });
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
                    shards,
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
                let k_cache_fp16 = DeviceBuf::alloc_zeroed(&ctx, kv_cap * 512 * 2)?;
                let v_cache_fp16 = DeviceBuf::alloc_zeroed(&ctx, kv_cap * 512 * 2)?;
                let idx_raw_k_cache = DeviceBuf::alloc_zeroed(&ctx, kv_cap * 128 * 4)?;
                let idx_block_k_cache = DeviceBuf::alloc_zeroed(&ctx, init_blocks * 128 * 4)?;
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

        // Pass 2: Place `48` `switch_mlp` slabs (`1,425 MiB`/layer lossless Z24):
        //   - `72 GB` (`--spec 0`): `48/48` contiguous `Vram` (`0` PCIe spill).
        //   - `72 GB` (`--spec K`): `45/48` contiguous `Vram` + `3/48` calibrated `Mixed`, leaving `2 GiB` VRAM for `100%` VRAM MTP!
        //   - `32 GB` / `16 GB`: `48/48` calibrated `Mixed` (`~9,728` hot experts on `PRO 4500 32GB`, `3.15x` Solo Decode!).
        let vram_reserve_bytes: usize = if reserve_for_mtp {
            2_048 * 1024 * 1024
        } else {
            320 * 1024 * 1024
        };
        let host_ram_reserve_bytes: usize = 16 * 1024 * 1024 * 1024;
        let force_compact = std::env::var("LEANSONIC_SPILLED_MODE")
            .map(|v| v == "compact")
            .unwrap_or(false);
        let disable_z24 = std::env::var("LEANSONIC_NO_Z24")
            .map(|v| v == "1")
            .unwrap_or(false);
        let max_vram_layers: usize = std::env::var("LEANSONIC_MAX_RESIDENT_LAYERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(Q38_LAYERS);

        let mixed_plan = MixedPlacementPlan::plan_for_device(
            &ctx,
            vram_reserve_bytes,
            host_ram_reserve_bytes,
            max_vram_layers,
            force_compact,
            disable_z24,
        )?;

        let mut z24_lo_scratch = vec![0u32; SW_GROUPS_PER_PROJ / 2];
        let mut z24_hi_scratch = vec![0u32; SW_GROUPS_PER_PROJ / 4];

        let mut resident_expert_layers = 0usize;
        let mut pinned_expert_layers = 0usize;
        let mut compact_expert_layers = 0usize;
        let mut mixed_expert_layers = 0usize;
        let mut hot_resident_experts = 0usize;
        let mut z24_layers = 0usize;
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

            if let Some(LayerExpertPlan::Mixed(mask)) = mixed_plan.as_ref().map(|mp| &mp.layers[l])
            {
                let hot_in_layer = mask.iter().filter(|&&h| h).count();
                let mixed_tier = MixedExperts::load(
                    &ctx,
                    mask,
                    l,
                    [
                        [gw.data, gsc.data, gbi.data],
                        [uw.data, usc.data, ubi.data],
                        [dw.data, dsc.data, dbi.data],
                    ],
                    &mut z24_lo_scratch,
                    &mut z24_hi_scratch,
                )?;
                mixed_expert_layers += 1;
                hot_resident_experts += hot_in_layer;
                z24_layers += 1;
                layers.push(Qwen38LayerGpu {
                    ple: part.ple,
                    attn_hc: part.attn_hc,
                    attn: part.attn,
                    mlp_hc: part.mlp_hc,
                    router_bf16: part.router_bf16,
                    sh_gu_gate: part.sh_gu_gate,
                    sh_down: part.sh_down,
                    switch_mlp: SwitchMlpTier::Mixed(mixed_tier),
                });
                continue;
            }

            let (free_vram, _) = ctx.mem_info()?;
            let want_vram = resident_expert_layers < max_vram_layers
                && free_vram > vram_reserve_bytes.saturating_add(SW_Z24_LAYER_BYTES);
            let want_pinned = !want_vram
                && !force_compact
                && available_host_ram_bytes()
                    > host_ram_reserve_bytes.saturating_add(SW_Z24_LAYER_BYTES);

            let switch_mlp = if want_vram || want_pinned {
                // Attempt `0-ULP` lossless `24-bit` (`Z24`) packing first (`1,425 MiB` vs `1,500 MiB`).
                let z24_gate = if !disable_z24 {
                    pack_z24_sb_slice(gsc.data, gbi.data, &mut z24_lo_scratch, &mut z24_hi_scratch)
                } else {
                    None
                };

                if let Some((g_sc_m1, g_bi_m1)) = z24_gate {
                    if want_vram {
                        let slab = DeviceBuf::alloc_uninit(&ctx, SW_Z24_LAYER_BYTES)?;
                        slab.copy_from_host_at(ZOFF_GATE_W, gw.data)?;
                        slab.copy_from_host_at(ZOFF_GATE_LO, u32_slice_bytes(&z24_lo_scratch))?;
                        slab.copy_from_host_at(ZOFF_GATE_HI, u32_slice_bytes(&z24_hi_scratch))?;

                        let (u_sc_m1, u_bi_m1) = pack_z24_sb_slice(
                            usc.data,
                            ubi.data,
                            &mut z24_lo_scratch,
                            &mut z24_hi_scratch,
                        )
                        .ok_or_else(|| format!("layer {l} up_proj exceeded Z24 exponent span"))?;
                        slab.copy_from_host_at(ZOFF_UP_W, uw.data)?;
                        slab.copy_from_host_at(ZOFF_UP_LO, u32_slice_bytes(&z24_lo_scratch))?;
                        slab.copy_from_host_at(ZOFF_UP_HI, u32_slice_bytes(&z24_hi_scratch))?;

                        let (d_sc_m1, d_bi_m1) = pack_z24_sb_slice(
                            dsc.data,
                            dbi.data,
                            &mut z24_lo_scratch,
                            &mut z24_hi_scratch,
                        )
                        .ok_or_else(|| format!("layer {l} down_proj exceeded Z24 exponent span"))?;
                        slab.copy_from_host_at(ZOFF_DOWN_W, dw.data)?;
                        slab.copy_from_host_at(ZOFF_DOWN_LO, u32_slice_bytes(&z24_lo_scratch))?;
                        slab.copy_from_host_at(ZOFF_DOWN_HI, u32_slice_bytes(&z24_hi_scratch))?;

                        resident_expert_layers += 1;
                        hot_resident_experts += Q38_EXP;
                        z24_layers += 1;
                        SwitchMlpTier::Vram {
                            buf: slab,
                            z24: Some(Z24LayerBases {
                                up_packed_m1: (g_sc_m1 as u32)
                                    | ((g_bi_m1 as u32) << 8)
                                    | ((u_sc_m1 as u32) << 16)
                                    | ((u_bi_m1 as u32) << 24),
                                down_packed_m1: (d_sc_m1 as u32) | ((d_bi_m1 as u32) << 8),
                            }),
                        }
                    } else {
                        let slab = PinnedHostBuf::alloc_uninit(&ctx, SW_Z24_LAYER_BYTES)?;
                        parallel_copy_into_pinned(
                            slab.host_ptr,
                            &[
                                (ZOFF_GATE_W, gw.data),
                                (ZOFF_GATE_LO, u32_slice_bytes(&z24_lo_scratch)),
                                (ZOFF_GATE_HI, u32_slice_bytes(&z24_hi_scratch)),
                            ],
                        );
                        let (u_sc_m1, u_bi_m1) = pack_z24_sb_slice(
                            usc.data,
                            ubi.data,
                            &mut z24_lo_scratch,
                            &mut z24_hi_scratch,
                        )
                        .ok_or_else(|| format!("layer {l} up_proj exceeded Z24 exponent span"))?;
                        parallel_copy_into_pinned(
                            slab.host_ptr,
                            &[
                                (ZOFF_UP_W, uw.data),
                                (ZOFF_UP_LO, u32_slice_bytes(&z24_lo_scratch)),
                                (ZOFF_UP_HI, u32_slice_bytes(&z24_hi_scratch)),
                            ],
                        );
                        let (d_sc_m1, d_bi_m1) = pack_z24_sb_slice(
                            dsc.data,
                            dbi.data,
                            &mut z24_lo_scratch,
                            &mut z24_hi_scratch,
                        )
                        .ok_or_else(|| format!("layer {l} down_proj exceeded Z24 exponent span"))?;
                        parallel_copy_into_pinned(
                            slab.host_ptr,
                            &[
                                (ZOFF_DOWN_W, dw.data),
                                (ZOFF_DOWN_LO, u32_slice_bytes(&z24_lo_scratch)),
                                (ZOFF_DOWN_HI, u32_slice_bytes(&z24_hi_scratch)),
                            ],
                        );

                        pinned_expert_layers += 1;
                        z24_layers += 1;
                        SwitchMlpTier::HostPinned {
                            buf: slab,
                            z24: Some(Z24LayerBases {
                                up_packed_m1: (g_sc_m1 as u32)
                                    | ((g_bi_m1 as u32) << 8)
                                    | ((u_sc_m1 as u32) << 16)
                                    | ((u_bi_m1 as u32) << 24),
                                down_packed_m1: (d_sc_m1 as u32) | ((d_bi_m1 as u32) << 8),
                            }),
                        }
                    }
                } else {
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
                    if resident_expert_layers < max_vram_layers
                        && free_vram > vram_reserve_bytes.saturating_add(SW_LAYER_BYTES)
                    {
                        let slab = DeviceBuf::alloc_uninit(&ctx, SW_LAYER_BYTES)?;
                        for &(off, bytes) in &slices {
                            slab.copy_from_host_at(off, bytes)?;
                        }
                        resident_expert_layers += 1;
                        SwitchMlpTier::Vram {
                            buf: slab,
                            z24: None,
                        }
                    } else if !force_compact
                        && available_host_ram_bytes()
                            > host_ram_reserve_bytes.saturating_add(SW_LAYER_BYTES)
                    {
                        let slab = PinnedHostBuf::alloc_uninit(&ctx, SW_LAYER_BYTES)?;
                        parallel_copy_into_pinned(slab.host_ptr, &slices);
                        pinned_expert_layers += 1;
                        SwitchMlpTier::HostPinned {
                            buf: slab,
                            z24: None,
                        }
                    } else {
                        compact_expert_layers += 1;
                        SwitchMlpTier::MmapCompact { layer_idx: l }
                    }
                }
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
        drop(z24_lo_scratch);
        drop(z24_hi_scratch);

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
            pdl,
            zero_centered,
            all_layers_z24: z24_layers == Q38_LAYERS,
            sharded_mmap: st,
            embed,
            final_hc,
            lm_head,
            inv_freq32,
            layers,
            resident_expert_layers,
            pinned_expert_layers,
            compact_expert_layers,
            mixed_expert_layers,
            hot_resident_experts,
            max_ctx,
            kv_cap,
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
            batch_bufs: None,
            verify_ckpt: None,
            host_pos: 0,
            decode_graph: None,
            prefill_graph: None,
        };

        engine.rebuild_graphs_if_enabled()?;

        let engram_bytes: usize = engine
            .layers
            .iter()
            .filter_map(|layer| layer.ple.as_ref())
            .flat_map(|ple| &ple.shards)
            .map(|s| s.bytes(&engine.sharded_mmap))
            .sum();
        let engram_policy = EngramPolicy::choose(
            std::env::var("LEANSONIC_Q38_ENGRAM").ok().as_deref(),
            engram_bytes,
            available_host_ram_bytes(),
        );
        if engram_policy == EngramPolicy::Resident {
            let t_copy = Instant::now();
            let Qwen38GpuEngine {
                layers,
                sharded_mmap,
                ..
            } = &mut engine;
            let mut entries = Vec::new();
            for ple in layers.iter_mut().filter_map(|layer| layer.ple.as_mut()) {
                for shard in &mut ple.shards {
                    match shard {
                        PleShard::Tiled(tile) => entries.push(&mut tile.storage),
                        PleShard::Planes {
                            weights,
                            scales,
                            biases,
                        } => entries.extend([weights, &mut scales.bytes, &mut biases.bytes]),
                    }
                }
            }
            let threads = std::thread::available_parallelism()
                .map_or(4, usize::from)
                .min(16);
            make_resident(sharded_mmap, &mut entries, threads);
            eprintln!(
                "[Qwen38GpuEngine] Engram copied to host memory in {:.2}s (LEANSONIC_Q38_ENGRAM=mmap keeps the file mapping)",
                t_copy.elapsed().as_secs_f64()
            );
        }
        let engram_kind = if engram_policy == EngramPolicy::Resident {
            "Resident"
        } else {
            "Async"
        };
        let engram_gib = engram_bytes as f64 / (1u64 << 30) as f64;
        let load_s = t0.elapsed().as_secs_f64();
        eprintln!(
            "[Qwen38GpuEngine] Loaded {} ({} shards, zero_norm={}, pdl={}, z24={}/48, graph={}) on {} in {:.2}s | Experts: {}/48 VRAM + {}/48 CalibratedMixed ({}/24576 hot VRAM) + {}/48 HostPinned + {}/48 MmapCompact + {engram_gib:.1}GiB {engram_kind} Engram",
            dir.display(),
            engine.sharded_mmap.num_shards(),
            zero_centered,
            engine.pdl,
            z24_layers,
            engine.decode_graph.is_some(),
            engine.ctx.device_name,
            load_s,
            resident_expert_layers,
            mixed_expert_layers,
            hot_resident_experts,
            pinned_expert_layers,
            compact_expert_layers,
        );

        Ok(engine)
    }

    fn rebuild_graphs_if_enabled(&mut self) -> Result<(), String> {
        self.decode_graph = None;
        self.prefill_graph = None;
        if self.enable_cuda_graph && self.compact_expert_layers == 0 {
            self.decode_graph = Some(CapturedGraphExec::capture(&self.ctx, |stream| {
                self.enqueue_all_48_layers(stream, true, false)
            })?);
            self.prefill_graph = Some(CapturedGraphExec::capture(&self.ctx, |stream| {
                self.enqueue_all_48_layers(stream, false, false)
            })?);
            self.reset()?;
        }
        Ok(())
    }

    /// Elastically expands QSA KV caches when sequence length crosses `self.kv_cap`.
    /// If VRAM is tight at `32K–128K` context, automatically demotes trailing `Vram` expert slabs
    /// to `HostPinned` so ultra-long contexts never OOM.
    fn ensure_kv_capacity(&mut self, needed_tokens: usize) -> Result<(), String> {
        if needed_tokens <= self.kv_cap {
            return Ok(());
        }
        if needed_tokens > self.max_ctx {
            return Err(format!(
                "sequence length {needed_tokens} exceeds max_ctx {}",
                self.max_ctx
            ));
        }

        self.ctx.bind_current()?;
        self.ctx.sync_compute()?;
        self.ctx.sync_h2d()?;
        self.decode_graph = None;
        self.prefill_graph = None;

        let mut new_cap = self.kv_cap.max(4096);
        while new_cap < needed_tokens {
            new_cap = (new_cap * 2).min(self.max_ctx);
        }
        let delta_tokens = new_cap.saturating_sub(self.kv_cap);
        // 12 QSA layers × (`1024 B` K + `1024 B` V + `512 B` idx_raw_k + `128 B` idx_block_k) = `32,256 B/tok`
        let extra_vram_needed = delta_tokens * 32_256 + 192 * 1024 * 1024;
        let (mut free_vram, _) = self.ctx.mem_info()?;

        while free_vram < extra_vram_needed && self.resident_expert_layers > 0 {
            let Some(demote_idx) = self
                .layers
                .iter()
                .rposition(|ly| matches!(ly.switch_mlp, SwitchMlpTier::Vram { .. }))
            else {
                break;
            };
            let old_tier = core::mem::replace(
                &mut self.layers[demote_idx].switch_mlp,
                SwitchMlpTier::MmapCompact {
                    layer_idx: demote_idx,
                },
            );
            if let SwitchMlpTier::Vram { buf, z24 } = old_tier {
                let pinned = PinnedHostBuf::alloc_uninit(&self.ctx, buf.byte_len)?;
                unsafe {
                    check_cu(
                        (self.ctx.api.cuMemcpyDtoH_v2)(
                            pinned.host_ptr.cast(),
                            buf.ptr,
                            buf.byte_len,
                        ),
                        "DtoH(demote switch_mlp)",
                    )?;
                }
                drop(buf);
                self.layers[demote_idx].switch_mlp = SwitchMlpTier::HostPinned { buf: pinned, z24 };
                self.resident_expert_layers -= 1;
                self.pinned_expert_layers += 1;
                (free_vram, _) = self.ctx.mem_info()?;
            }
        }

        let new_blocks = (new_cap / 4) + 16;
        let valid_toks = self.host_pos.min(self.kv_cap);
        let valid_blocks = valid_toks / 4;

        for layer in &mut self.layers {
            if let AttentionBranchGpu::Qsa {
                k_cache_fp16,
                v_cache_fp16,
                idx_raw_k_cache,
                idx_block_k_cache,
                ..
            } = &mut layer.attn
            {
                let grow_buf = |old: &DeviceBuf,
                                new_bytes: usize,
                                copy_bytes: usize|
                 -> Result<DeviceBuf, String> {
                    let nb = DeviceBuf::alloc_zeroed(&self.ctx, new_bytes)?;
                    if copy_bytes > 0 {
                        unsafe {
                            check_cu(
                                (self.ctx.api.cuMemcpyDtoDAsync_v2)(
                                    nb.ptr,
                                    old.ptr,
                                    copy_bytes,
                                    self.ctx.compute_stream,
                                ),
                                "DtoD(grow QSA KV)",
                            )?;
                        }
                        self.ctx.sync_compute()?;
                    }
                    Ok(nb)
                };

                let nk = grow_buf(k_cache_fp16, new_cap * 512 * 2, valid_toks * 512 * 2)?;
                *k_cache_fp16 = nk;
                let nv = grow_buf(v_cache_fp16, new_cap * 512 * 2, valid_toks * 512 * 2)?;
                *v_cache_fp16 = nv;
                let nraw = grow_buf(idx_raw_k_cache, new_cap * 128 * 4, valid_toks * 128 * 4)?;
                *idx_raw_k_cache = nraw;
                let nblk = grow_buf(
                    idx_block_k_cache,
                    new_blocks * 128 * 4,
                    valid_blocks * 128 * 4,
                )?;
                *idx_block_k_cache = nblk;
            }
        }

        self.d_qsa_scores = DeviceBuf::alloc_zeroed(&self.ctx, new_blocks * 4)?;
        self.d_qsa_mask = DeviceBuf::alloc_zeroed(&self.ctx, new_blocks * 4)?;
        if let Some(bb) = &mut self.batch_bufs {
            if bb.mask_stride < new_blocks {
                bb.d_b_qsa_scores =
                    DeviceBuf::alloc_zeroed(&self.ctx, Q38_MAX_BATCH * new_blocks * 4)?;
                bb.d_b_qsa_mask =
                    DeviceBuf::alloc_zeroed(&self.ctx, Q38_MAX_BATCH * new_blocks * 4)?;
                bb.mask_stride = new_blocks;
            }
        }
        self.kv_cap = new_cap;

        if self.enable_cuda_graph && self.compact_expert_layers == 0 {
            // Re-capture graphs against the new KV cache pointers WITHOUT resetting live `host_pos`/states!
            let saved_pos = self.host_pos;
            let saved_p1 = self.ple_prev1;
            let saved_p2 = self.ple_prev2;
            self.decode_graph = Some(CapturedGraphExec::capture(&self.ctx, |stream| {
                self.enqueue_all_48_layers(stream, true, false)
            })?);
            self.prefill_graph = Some(CapturedGraphExec::capture(&self.ctx, |stream| {
                self.enqueue_all_48_layers(stream, false, false)
            })?);
            self.host_pos = saved_pos;
            self.ple_prev1 = saved_p1;
            self.ple_prev2 = saved_p2;
        }
        Ok(())
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

    /// Benchmark-only helper (`LEANSONIC_BENCH_DECODE_AT=<pos>`): moves decode position to `pos`
    /// after ensuring QSA KV buffers cover `pos + 256`.
    pub fn bench_jump_position(&mut self, pos: usize) -> Result<(), String> {
        let target_cap = (pos + 256).min(self.max_ctx);
        self.ensure_kv_capacity(target_cap)?;
        self.host_pos = pos;
        let pos_i32 = pos as i32;
        self.d_pos.copy_from_host_bytes(&pos_i32.to_le_bytes())
    }

    // ------------------------------------------------------------------------
    // MTP Speculative Drafter Integration Hooks (`Track 3`, `#248`)
    // ------------------------------------------------------------------------

    /// Shared CUDA context handle (`compute_stream`, `h2d_stream`, driver API).
    #[inline]
    pub fn cuda_ctx(&self) -> &Arc<CudaContextHandle> {
        &self.ctx
    }

    /// Loaded `qwen38_sm120a` CUDA module handle so `qwen38_mtp` can resolve shared kernels.
    #[inline]
    pub fn cuda_module(&self) -> CUmodule {
        self.fns._module
    }

    #[inline]
    pub fn pdl_enabled(&self) -> bool {
        self.pdl
    }

    /// Device pointer to `[4, 2560]` FP32 Hyper-Connection state right after Layer 47
    /// (before `final_hc`), ready for `language_model.mtp` fusion!
    #[inline]
    pub fn hc_state_ptr(&self) -> CUdeviceptr {
        self.d_hc.ptr
    }

    /// Device pointer to row `row_idx` (`0 <= row_idx < Q38_MAX_BATCH`) of the batched
    /// `[B, 4, 2560]` HC state produced by `forward_verify_tokens`.
    #[inline]
    pub fn hc_batch_row_ptr(&self, row_idx: usize) -> Option<CUdeviceptr> {
        self.batch_bufs
            .as_ref()
            .map(|b| b.d_b_hc.ptr + ((row_idx * Q38_HC_D * 4) as u64))
    }

    /// Copies row `row_idx` of `d_b_hc` into `self.d_hc` so subsequent `hc_state_ptr()` / decode
    /// steps observe the exact accepted speculative token's `[4, 2560]` HC hidden state.
    pub fn copy_batch_hc_row_to_main(&mut self, row_idx: usize) -> Result<(), String> {
        let Some(src) = self.hc_batch_row_ptr(row_idx) else {
            return Err("batch_bufs not initialized".into());
        };
        unsafe {
            check_cu(
                (self.ctx.api.cuMemcpyDtoDAsync_v2)(
                    self.d_hc.ptr,
                    src,
                    Q38_HC_D * 4,
                    self.ctx.compute_stream,
                ),
                "DtoD(copy_batch_hc_row_to_main)",
            )
        }
    }

    /// Shared Q4 embedding table device pointers `(w_u128, sc_u32, bi_u32)` (`[248320, 2560]`).
    #[inline]
    pub fn embed_ptrs(&self) -> (CUdeviceptr, CUdeviceptr, CUdeviceptr) {
        (
            self.embed.w.dev_ptr,
            self.embed.sc.dev_ptr,
            self.embed.bi.dev_ptr,
        )
    }

    /// Shared Q4 `lm_head` device pointers `(w_u128, sc_u32, bi_u32, rows, cols)` (`[248320, 2560]`).
    #[inline]
    pub fn lm_head_ptrs(&self) -> (CUdeviceptr, CUdeviceptr, CUdeviceptr, i32, i32) {
        (
            self.lm_head.w.ptr,
            self.lm_head.sc.ptr,
            self.lm_head.bi.ptr,
            self.lm_head.rows,
            self.lm_head.cols,
        )
    }

    #[inline]
    pub fn current_pos(&self) -> usize {
        self.host_pos
    }

    #[inline]
    pub fn ple_tail_state(&self) -> (i64, i64) {
        (self.ple_prev2, self.ple_prev1)
    }

    /// Frees ~`1,425 MiB` of GDDR7 VRAM by demoting one trailing `Vram` expert slab to `HostPinned`,
    /// making room for `language_model.mtp`'s 1-layer MoE slab on `RTX PRO 5000 72GB`.
    pub fn demote_one_expert_layer_for_drafter(&mut self) -> Result<bool, String> {
        self.ctx.bind_current()?;
        self.ctx.sync_compute()?;
        self.ctx.sync_h2d()?;
        let Some(demote_idx) = self
            .layers
            .iter()
            .rposition(|ly| matches!(ly.switch_mlp, SwitchMlpTier::Vram { .. }))
        else {
            return Ok(false);
        };
        let saved_pos = self.host_pos;
        let saved_p1 = self.ple_prev1;
        let saved_p2 = self.ple_prev2;
        self.decode_graph = None;
        self.prefill_graph = None;

        let old_tier = core::mem::replace(
            &mut self.layers[demote_idx].switch_mlp,
            SwitchMlpTier::MmapCompact {
                layer_idx: demote_idx,
            },
        );
        if let SwitchMlpTier::Vram { buf, z24 } = old_tier {
            let pinned = PinnedHostBuf::alloc_uninit(&self.ctx, buf.byte_len)?;
            unsafe {
                check_cu(
                    (self.ctx.api.cuMemcpyDtoH_v2)(pinned.host_ptr.cast(), buf.ptr, buf.byte_len),
                    "DtoH(demote layer for MTP)",
                )?;
            }
            drop(buf);
            self.layers[demote_idx].switch_mlp = SwitchMlpTier::HostPinned { buf: pinned, z24 };
            self.resident_expert_layers -= 1;
            self.pinned_expert_layers += 1;
        }
        if self.enable_cuda_graph && self.compact_expert_layers == 0 {
            self.decode_graph = Some(CapturedGraphExec::capture(&self.ctx, |stream| {
                self.enqueue_all_48_layers(stream, true, false)
            })?);
            self.prefill_graph = Some(CapturedGraphExec::capture(&self.ctx, |stream| {
                self.enqueue_all_48_layers(stream, false, false)
            })?);
            self.host_pos = saved_pos;
            self.ple_prev1 = saved_p1;
            self.ple_prev2 = saved_p2;
        }
        Ok(true)
    }

    pub fn alloc_recurrent_snapshot(&self) -> Result<Qwen38RecurrentSnapshot, String> {
        let ple_hist9 = DeviceBuf::alloc_zeroed(&self.ctx, 9 * Q38_HC_D * 4)?;
        let mut gdn_conv_hists = Vec::with_capacity(36);
        let mut gdn_states = Vec::with_capacity(36);
        for l in &self.layers {
            if let AttentionBranchGpu::Gdn {
                conv_hist3,
                state_f32,
                ..
            } = &l.attn
            {
                gdn_conv_hists.push(DeviceBuf::alloc_zeroed(&self.ctx, conv_hist3.byte_len)?);
                gdn_states.push(DeviceBuf::alloc_zeroed(&self.ctx, state_f32.byte_len)?);
            }
        }
        Ok(Qwen38RecurrentSnapshot {
            ple_hist9,
            gdn_conv_hists,
            gdn_states,
            saved_pos: 0,
            saved_prev2: Q38_PLE_EOS_ID,
            saved_prev1: Q38_PLE_EOS_ID,
        })
    }

    pub fn snapshot_recurrent_state(
        &self,
        snap: &mut Qwen38RecurrentSnapshot,
    ) -> Result<(), String> {
        snap.saved_pos = self.host_pos;
        snap.saved_prev2 = self.ple_prev2;
        snap.saved_prev1 = self.ple_prev1;
        let stream = self.ctx.compute_stream;
        let mut gdn_idx = 0usize;
        unsafe {
            for l in &self.layers {
                if let Some(ple) = &l.ple {
                    check_cu(
                        (self.ctx.api.cuMemcpyDtoDAsync_v2)(
                            snap.ple_hist9.ptr,
                            ple.hist9.ptr,
                            ple.hist9.byte_len,
                            stream,
                        ),
                        "DtoD(snapshot ple.hist9)",
                    )?;
                }
                if let AttentionBranchGpu::Gdn {
                    conv_hist3,
                    state_f32,
                    ..
                } = &l.attn
                {
                    check_cu(
                        (self.ctx.api.cuMemcpyDtoDAsync_v2)(
                            snap.gdn_conv_hists[gdn_idx].ptr,
                            conv_hist3.ptr,
                            conv_hist3.byte_len,
                            stream,
                        ),
                        "DtoD(snapshot gdn.conv_hist3)",
                    )?;
                    check_cu(
                        (self.ctx.api.cuMemcpyDtoDAsync_v2)(
                            snap.gdn_states[gdn_idx].ptr,
                            state_f32.ptr,
                            state_f32.byte_len,
                            stream,
                        ),
                        "DtoD(snapshot gdn.state_f32)",
                    )?;
                    gdn_idx += 1;
                }
            }
        }
        Ok(())
    }

    pub fn restore_recurrent_state(
        &mut self,
        snap: &Qwen38RecurrentSnapshot,
    ) -> Result<(), String> {
        self.host_pos = snap.saved_pos;
        self.ple_prev2 = snap.saved_prev2;
        self.ple_prev1 = snap.saved_prev1;
        let pos_i32 = snap.saved_pos as i32;
        let stream = self.ctx.compute_stream;
        let mut gdn_idx = 0usize;
        unsafe {
            for l in &self.layers {
                if let Some(ple) = &l.ple {
                    check_cu(
                        (self.ctx.api.cuMemcpyDtoDAsync_v2)(
                            ple.hist9.ptr,
                            snap.ple_hist9.ptr,
                            ple.hist9.byte_len,
                            stream,
                        ),
                        "DtoD(restore ple.hist9)",
                    )?;
                }
                if let AttentionBranchGpu::Gdn {
                    conv_hist3,
                    state_f32,
                    ..
                } = &l.attn
                {
                    check_cu(
                        (self.ctx.api.cuMemcpyDtoDAsync_v2)(
                            conv_hist3.ptr,
                            snap.gdn_conv_hists[gdn_idx].ptr,
                            conv_hist3.byte_len,
                            stream,
                        ),
                        "DtoD(restore gdn.conv_hist3)",
                    )?;
                    check_cu(
                        (self.ctx.api.cuMemcpyDtoDAsync_v2)(
                            state_f32.ptr,
                            snap.gdn_states[gdn_idx].ptr,
                            state_f32.byte_len,
                            stream,
                        ),
                        "DtoD(restore gdn.state_f32)",
                    )?;
                    gdn_idx += 1;
                }
            }
        }
        self.d_pos.copy_from_host_bytes(&pos_i32.to_le_bytes())
    }

    fn ensure_batch_buffers(&mut self) -> Result<(), String> {
        if self.batch_bufs.is_none() {
            let mask_stride = (self.kv_cap / 4) + 16;
            self.batch_bufs = Some(Qwen38BatchBuffers::alloc(&self.ctx, mask_stride)?);
        }
        Ok(())
    }

    fn ensure_verify_row_ckpt(&mut self, needed_rows: usize) -> Result<(), String> {
        if needed_rows == 0 {
            return Ok(());
        }
        let cur_cap = self.verify_ckpt.as_ref().map(|c| c.cap_rows).unwrap_or(0);
        if cur_cap >= needed_rows {
            return Ok(());
        }
        self.ctx.sync_compute()?;
        self.verify_ckpt = None;
        let ple_hist9_rows = VerifyCkptSlab::alloc(&self.ctx, needed_rows * 9 * Q38_HC_D * 4)?;
        let mut gdn_conv_rows = Vec::with_capacity(36);
        let mut gdn_state_rows = Vec::with_capacity(36);
        for l in &self.layers {
            if let AttentionBranchGpu::Gdn {
                conv_hist3,
                state_f32,
                ..
            } = &l.attn
            {
                gdn_conv_rows.push(VerifyCkptSlab::alloc(
                    &self.ctx,
                    needed_rows * conv_hist3.byte_len,
                )?);
                gdn_state_rows.push(VerifyCkptSlab::alloc(
                    &self.ctx,
                    needed_rows * state_f32.byte_len,
                )?);
            }
        }
        self.verify_ckpt = Some(Qwen38VerifyRowCkpt {
            cap_rows: needed_rows,
            ple_hist9_rows,
            gdn_conv_rows,
            gdn_state_rows,
            last_pos0: 0,
            last_b_count: 0,
            ple_tails_after_row: [(Q38_PLE_EOS_ID, Q38_PLE_EOS_ID); Q38_MAX_VERIFY_BATCH],
        });
        Ok(())
    }

    /// Commits or rolls back the engine after `forward_verify_tokens(&cand)` (`B = cand.len()`, `0 <= accepted_row < B`):
    /// - Sets `self.d_hc` to `d_b_hc[accepted_row]` (`[4, 2560]` HC state right after `cand[accepted_row]`).
    /// - If `accepted_row + 1 == B` (all draft tokens accepted), recurrent states & `host_pos` are already at `pos0 + B` (`< 5 µs`!).
    /// - If `accepted_row + 1 < B`, restores `ple.hist9`, `36` GDN `(conv_hist3, state_f32)`, `(ple_prev2, ple_prev1)`,
    ///   and `(host_pos, d_pos) = pos0 + accepted_row + 1` in `~0.09 ms` `DtoDAsync` — ZERO pre-snapshot & ZERO mismatch re-forward!
    pub fn commit_or_rollback_verify_row(&mut self, accepted_row: usize) -> Result<(), String> {
        self.ctx.bind_current()?;
        self.copy_batch_hc_row_to_main(accepted_row)?;
        let Some(ckpt) = self.verify_ckpt.as_ref() else {
            return Ok(());
        };
        if ckpt.last_b_count == 0 {
            return Err("commit_or_rollback_verify_row called before forward_verify_tokens".into());
        }
        if accepted_row >= ckpt.last_b_count {
            return Err(format!(
                "accepted_row {accepted_row} out of range for verify batch {}",
                ckpt.last_b_count
            ));
        }
        if accepted_row + 1 == ckpt.last_b_count {
            return Ok(());
        }

        let new_pos = ckpt.last_pos0 + accepted_row + 1;
        let (p2, p1) = ckpt.ple_tails_after_row[accepted_row];
        let stream = self.ctx.compute_stream;
        let ple_row_bytes = 9 * Q38_HC_D * 4;
        let mut gdn_idx = 0usize;
        unsafe {
            for l in &self.layers {
                if let Some(ple) = &l.ple {
                    ckpt.ple_hist9_rows.copy_row_to_dev_async(
                        &self.ctx,
                        ple.hist9.ptr,
                        accepted_row,
                        ple_row_bytes,
                        stream,
                        "rollback verify ple.hist9",
                    )?;
                }
                if let AttentionBranchGpu::Gdn {
                    conv_hist3,
                    state_f32,
                    ..
                } = &l.attn
                {
                    let c_bytes = conv_hist3.byte_len;
                    let s_bytes = state_f32.byte_len;
                    ckpt.gdn_conv_rows[gdn_idx].copy_row_to_dev_async(
                        &self.ctx,
                        conv_hist3.ptr,
                        accepted_row,
                        c_bytes,
                        stream,
                        "rollback verify gdn.conv_hist3",
                    )?;
                    ckpt.gdn_state_rows[gdn_idx].copy_row_to_dev_async(
                        &self.ctx,
                        state_f32.ptr,
                        accepted_row,
                        s_bytes,
                        stream,
                        "rollback verify gdn.state_f32",
                    )?;
                    gdn_idx += 1;
                }
            }
        }
        self.host_pos = new_pos;
        self.ple_prev2 = p2;
        self.ple_prev1 = p1;
        let pos_i32 = new_pos as i32;
        self.d_pos.copy_from_host_bytes(&pos_i32.to_le_bytes())
    }

    #[inline]
    pub fn supports_batched_verify(&self) -> bool {
        self.all_layers_z24 && self.compact_expert_layers == 0
    }

    #[inline]
    pub fn sharded_safetensors(&self) -> &ShardedSafeTensors {
        &self.sharded_mmap
    }

    #[inline]
    pub fn zero_centered_norms(&self) -> bool {
        self.zero_centered
    }

    /// Runs one `<= Q38_MAX_BATCH` (`64`) prefill chunk for MTP KV warm-up, leaving all
    /// `slice.len()` post-Layer-47 HC rows in `d_b_hc[0..slice.len()]` (`NoLogits` for earlier
    /// chunks, `LastLogitOnly` for the final chunk)!
    pub fn prefill_chunk_for_mtp(
        &mut self,
        slice: &[u32],
        is_last_chunk: bool,
    ) -> Result<Option<u32>, String> {
        if slice.is_empty() || slice.len() > Q38_MAX_BATCH {
            return Err(format!(
                "prefill_chunk_for_mtp requires 1..={Q38_MAX_BATCH} tokens, got {}",
                slice.len()
            ));
        }
        if !self.supports_batched_verify() {
            return Err("prefill_chunk_for_mtp requires Z24 batched verify engine".into());
        }
        self.ensure_kv_capacity(self.host_pos + slice.len())?;
        self.ctx.bind_current()?;
        self.ensure_batch_buffers()?;
        let mode = if is_last_chunk {
            BatchLogitMode::LastLogitOnly
        } else {
            BatchLogitMode::NoLogits
        };
        let out = self.forward_batch_chunk(slice, mode)?;
        Ok(if is_last_chunk { Some(out[0]) } else { None })
    }

    /// `0-ULP` multi-row target verification (`1 <= tokens.len() <= Q38_MAX_VERIFY_BATCH`):
    /// Evaluates all `B` tokens in ONE 48-layer weight sweep (recording per-row GDN/PLE checkpoints on the fly)
    /// and returns the `B` greedy argmax IDs! Pair with `commit_or_rollback_verify_row(accepted_row)`.
    pub fn forward_verify_tokens(&mut self, tokens: &[u32]) -> Result<Vec<u32>, String> {
        if tokens.is_empty() || tokens.len() > Q38_MAX_VERIFY_BATCH {
            return Err(format!(
                "forward_verify_tokens requires 1..={Q38_MAX_VERIFY_BATCH} tokens, got {}",
                tokens.len()
            ));
        }
        self.ensure_kv_capacity(self.host_pos + tokens.len())?;
        self.ctx.bind_current()?;
        if !self.supports_batched_verify() {
            if let Some(ckpt) = self.verify_ckpt.as_mut() {
                ckpt.last_b_count = 0;
            }
            let mut out = Vec::with_capacity(tokens.len());
            for &t in tokens {
                out.push(self.forward_token_internal(t, true)?);
            }
            return Ok(out);
        }
        self.ensure_batch_buffers()?;
        if tokens.len() >= 2 {
            self.ensure_verify_row_ckpt(tokens.len() - 1)?;
        }
        self.forward_batch_chunk(tokens, BatchLogitMode::AllVerifyLogits)
    }

    #[allow(clippy::manual_is_multiple_of)]
    pub fn prefill_tokens(&mut self, prompt_ids: &[u32]) -> Result<(u32, f64), String> {
        if prompt_ids.is_empty() {
            return Err("prefill_tokens requires at least 1 token".into());
        }
        let needed = self.host_pos + prompt_ids.len();
        self.ensure_kv_capacity(needed)?;
        self.ctx.bind_current()?;

        let force_serial = std::env::var("LEANSONIC_Q38_SERIAL_PREFILL")
            .map(|v| v == "1")
            .unwrap_or(false);

        let use_batched = !force_serial
            && prompt_ids.len() > 1
            && self.all_layers_z24
            && self.compact_expert_layers == 0;
        if use_batched {
            self.ensure_batch_buffers()?;
        }

        let t0 = Instant::now();
        if use_batched {
            let n = prompt_ids.len();
            let mut cursor = 0usize;
            while cursor < n {
                let end = (cursor + Q38_MAX_BATCH).min(n);
                let slice = &prompt_ids[cursor..end];
                if end < n {
                    let _ = self.forward_batch_chunk(slice, BatchLogitMode::NoLogits)?;
                } else {
                    let out = self.forward_batch_chunk(slice, BatchLogitMode::LastLogitOnly)?;
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    return Ok((out[0], ms));
                }
                cursor = end;
            }
            unreachable!()
        }

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
        self.ensure_kv_capacity(self.host_pos + 1)?;
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

    fn launch_kern(
        &self,
        stream: CUstream,
        func: CUfunction,
        grid: (u32, u32),
        block: u32,
        params: &mut [*mut c_void],
        op_name: &str,
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
                        op_name,
                    )
                };
            }
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
                op_name,
            )
        }
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
            let sp = &ple.shards[shard_idx];
            let (w_row, scales, biases) = sp.row(&self.sharded_mmap, row_idx)?;
            let out_h = &mut dst_f32[h * 160..(h + 1) * 160];

            for g in 0..5usize {
                let sc = bf16_to_f32(scales[g]);
                let bi = bf16_to_f32(biases[g]);
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
            self.launch_kern(
                stream,
                self.fns.q4_gemv_10k,
                (grid_x, 1),
                256,
                &mut p,
                "q38_q4_gemv_10k",
            )
        } else if cols == (Q38_D as i32) {
            let mut p = kargs![w_p, sc_p, bi_p, x_p, y_p, rows, mode];
            self.launch_kern(
                stream,
                self.fns.q4_gemv_2560,
                (grid_x, 1),
                256,
                &mut p,
                "q38_q4_gemv_2560",
            )
        } else {
            let mut p = kargs![w_p, sc_p, bi_p, x_p, y_p, rows, cols, mode];
            self.launch_kern(
                stream,
                self.fns.q4_gemv,
                (grid_x, 1),
                256,
                &mut p,
                "q38_q4_gemv",
            )
        }
    }

    fn launch_hc_pre(&self, stream: CUstream, hc: &HyperConnGpu) -> Result<(), String> {
        let (mut hc_in, mut gam, mut normed) =
            (self.d_hc.ptr, hc.hc_norm.ptr, self.d_hc_normed.ptr);
        let mut p_norm = kargs![hc_in, gam, normed];
        self.launch_kern(
            stream,
            self.fns.hc_norm4,
            (4, 1),
            256,
            &mut p_norm,
            "q38_hc_norm4",
        )?;

        let (mut dw_p, mut dsc_p, mut dbi_p) = (hc.down.w.ptr, hc.down.sc.ptr, hc.down.bi.ptr);
        let (mut iw_p, mut isc_p, mut ibi_p, mut has_inj) = if let Some(inj) = &hc.inject {
            (inj.w.ptr, inj.sc.ptr, inj.bi.ptr, 1i32)
        } else {
            (0u64, 0u64, 0u64, 0i32)
        };
        let (mut low_p, mut inj4_p) = (self.d_hc_low.ptr, self.d_hc_inj4.ptr);
        let grid_di = if has_inj == 1 { 81u32 } else { 80u32 };
        let mut p_di =
            kargs![dw_p, dsc_p, dbi_p, iw_p, isc_p, ibi_p, normed, low_p, inj4_p, has_inj];
        self.launch_kern(
            stream,
            self.fns.hc_down_inj_10k,
            (grid_di, 1),
            256,
            &mut p_di,
            "q38_hc_down_inj_10k",
        )?;

        let (mut uw_p, mut usc_p, mut ubi_p, mut mixed) =
            (hc.up.w.ptr, hc.up.sc.ptr, hc.up.bi.ptr, self.d_mixed.ptr);
        let mut p_um = kargs![uw_p, usc_p, ubi_p, low_p, normed, mixed];
        self.launch_kern(
            stream,
            self.fns.hc_up_mix4,
            (320, 1),
            256,
            &mut p_um,
            "q38_hc_up_mix4",
        )
    }

    fn launch_hc_post(&self, stream: CUstream) -> Result<(), String> {
        let (mut hc_io, mut br, mut inj4) =
            (self.d_hc.ptr, self.d_branch_out.ptr, self.d_hc_inj4.ptr);
        let mut p = kargs![hc_io, br, inj4];
        self.launch_kern(
            stream,
            self.fns.hc_inject4_vec,
            (3, 1),
            256,
            &mut p,
            "q38_hc_inject4_vec",
        )
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
                self.embed.w.dev_ptr,
                self.embed.sc.dev_ptr,
                self.embed.bi.dev_ptr,
                self.d_token_in.ptr,
                self.d_hc.ptr,
            );
            let mut p = kargs![ew, esc, ebi, tin, dhc];
            self.launch_kern(
                stream,
                self.fns.embed_init_hc4,
                (10, 1),
                256,
                &mut p,
                "q38_embed_init_hc4",
            )?;
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
                self.launch_kern(
                    stream,
                    self.fns.ple_inject,
                    (4, 1),
                    256,
                    &mut p_ple,
                    "q38_ple_inject",
                )?;
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
                    self.launch_kern(
                        stream,
                        self.fns.gdn_conv_silu,
                        (40, 1),
                        256,
                        &mut p_conv,
                        "q38_gdn_conv_silu",
                    )?;
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
                    self.launch_kern(
                        stream,
                        self.fns.gdn_recurrent_step,
                        (48, 1),
                        128,
                        &mut p_rec,
                        "q38_gdn_recurrent_step",
                    )?;
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
                    self.launch_kern(
                        stream,
                        self.fns.qsa_prep_and_cache,
                        (31, 1),
                        256,
                        &mut p_prep,
                        "q38_qsa_prep_and_cache",
                    )?;

                    let seq_len = self.host_pos + 1;
                    let n_blocks = (seq_len / 4) as i32;
                    let mut use_mask: i32 = if n_blocks > 512 { 1 } else { 0 };
                    if use_mask == 1 {
                        let (mut nb, mut sc_buf, mut mask_buf) =
                            (n_blocks, self.d_qsa_scores.ptr, self.d_qsa_mask.ptr);
                        let grid_sb = (n_blocks as u32).div_ceil(8);
                        let mut p_sc = kargs![iq, ibk, nb, sc_buf];
                        let mut p_sel = kargs![sc_buf, nb, mask_buf];
                        self.launch_kern(
                            stream,
                            self.fns.qsa_score_blocks,
                            (grid_sb, 1),
                            256,
                            &mut p_sc,
                            "q38_qsa_score_blocks",
                        )?;
                        self.launch_kern(
                            stream,
                            self.fns.qsa_select_top512,
                            (1, 1),
                            256,
                            &mut p_sel,
                            "q38_qsa_select_top512",
                        )?;
                    }

                    let (mut mask_p, mut aout) = (self.d_qsa_mask.ptr, self.d_attn_out.ptr);
                    let mut p_attn = kargs![qrot, qgate, kc, vc, mask_p, pos_p, use_mask, aout];
                    self.launch_kern(
                        stream,
                        self.fns.qsa_attn,
                        (24, 1),
                        256,
                        &mut p_attn,
                        "q38_qsa_attn",
                    )?;
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
                self.launch_kern(
                    stream,
                    self.fns.bf16_router_512,
                    (64, 1),
                    256,
                    &mut p_rt,
                    "q38_bf16_router_512",
                )?;
                self.launch_kern(
                    stream,
                    self.fns.softmax_topk10,
                    (1, 1),
                    256,
                    &mut p_tk,
                    "q38_softmax_topk10",
                )?;
            }

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

            if let Some((
                [mut gw_p, mut glo_p, mut ghi_p, mut uw_p, mut ulo_p, mut uhi_p, mut dw_p, mut dlo_p, mut dhi_p],
                z24_bases,
            )) = layer.switch_mlp.z24_parts()
            {
                let mut up_bases = z24_bases.up_packed_m1;
                let mut dn_bases = z24_bases.down_packed_m1;

                let mut p_up = kargs![
                    gw_p, glo_p, ghi_p, uw_p, ulo_p, uhi_p, up_bases, sh_gu_w, sh_gu_sc, sh_gu_bi,
                    mx, sids, swts, act11
                ];
                let mut p_dn = kargs![
                    dw_p, dlo_p, dhi_p, dn_bases, sh_dw, sh_dsc, sh_dbi, act11, sids, swts, mout
                ];
                self.launch_kern(
                    stream,
                    self.fns.moe_up_swiglu11_z24,
                    (160, 11),
                    256,
                    &mut p_up,
                    "q38_moe_up_swiglu11_z24",
                )?;
                self.launch_kern(
                    stream,
                    self.fns.moe_down_combine11_z24,
                    (80, 1),
                    256,
                    &mut p_dn,
                    "q38_moe_down_combine11_z24",
                )?;
            } else {
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
                    SwitchMlpTier::Mixed(_) => unreachable!("Mixed always provides z24_parts"),
                    SwitchMlpTier::Vram { buf: b, .. } => (
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
                    SwitchMlpTier::HostPinned { buf: b, .. } => (
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

                let mut p_up = kargs![
                    gw_p, gsc_p, gbi_p, uw_p, usc_p, ubi_p, sh_gu_w, sh_gu_sc, sh_gu_bi, mx, sids,
                    is_compact, swts, act11
                ];
                let mut p_dn = kargs![
                    dw_p, dsc_p, dbi_p, sh_dw, sh_dsc, sh_dbi, act11, sids, is_compact, swts, mout
                ];
                self.launch_kern(
                    stream,
                    self.fns.moe_up_swiglu11,
                    (160, 11),
                    256,
                    &mut p_up,
                    "q38_moe_up_swiglu11",
                )?;
                self.launch_kern(
                    stream,
                    self.fns.moe_down_combine11,
                    (320, 1),
                    256,
                    &mut p_dn,
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
            self.launch_kern(
                stream,
                self.fns.argmax_fast,
                (1, 1),
                1024,
                &mut p_arg,
                "q38_argmax_fast",
            )?;
        } else {
            let mut pos_p = self.d_pos.ptr;
            let mut p_inc = kargs![pos_p];
            self.launch_kern(
                stream,
                self.fns.inc_pos,
                (1, 1),
                1,
                &mut p_inc,
                "q38_inc_pos",
            )?;
        }
        Ok(())
    }

    fn launch_q4_gemm_b8(
        &self,
        stream: CUstream,
        lin: &Q4LinearGpu,
        x_batch_ptr: CUdeviceptr,
        y_batch_ptr: CUdeviceptr,
        b_count: usize,
        post_mode: i32,
    ) -> Result<(), String> {
        if b_count == 1 {
            return self.launch_q4_gemv(stream, lin, x_batch_ptr, y_batch_ptr, post_mode);
        }
        let (mut w_p, mut sc_p, mut bi_p, mut x_p, mut y_p) =
            (lin.w.ptr, lin.sc.ptr, lin.bi.ptr, x_batch_ptr, y_batch_ptr);
        let (mut rows, mut cols, mut bc, mut mode) =
            (lin.rows, lin.cols, b_count as i32, post_mode);
        if cols == (Q38_D as i32) && b_count == 2 {
            let grid_x = (rows as u32).div_ceil(32);
            let mut p = kargs![w_p, sc_p, bi_p, x_p, y_p, rows, bc, mode];
            return self.launch_kern(
                stream,
                self.fns.q4_gemv_2560_b2,
                (grid_x, 1),
                512,
                &mut p,
                "q38_q4_gemv_2560_b2",
            );
        }
        if cols == (Q38_D as i32) && b_count <= 4 {
            let grid_x = (rows as u32).div_ceil(32);
            let mut p = kargs![w_p, sc_p, bi_p, x_p, y_p, rows, bc, mode];
            return self.launch_kern(
                stream,
                self.fns.q4_gemv_2560_b4,
                (grid_x, 1),
                512,
                &mut p,
                "q38_q4_gemv_2560_b4",
            );
        }
        if b_count == 2 {
            let grid_x = (rows as u32).div_ceil(8);
            let mut p = kargs![w_p, sc_p, bi_p, x_p, y_p, rows, cols, mode];
            return self.launch_kern(
                stream,
                self.fns.q4_gemv_b2,
                (grid_x, 1),
                256,
                &mut p,
                "q38_q4_gemv_b2",
            );
        }
        let grid_x = (rows as u32).div_ceil(32);
        let grid_y = (b_count as u32).div_ceil(8);
        let mut p = kargs![w_p, sc_p, bi_p, x_p, y_p, rows, cols, bc, mode];
        self.launch_kern(
            stream,
            self.fns.q4_gemm_b8,
            (grid_x, grid_y),
            512,
            &mut p,
            "q38_q4_gemm_b8",
        )
    }

    fn launch_hc_pre_batch(
        &self,
        stream: CUstream,
        hc: &HyperConnGpu,
        bb: &Qwen38BatchBuffers,
        b_count: usize,
    ) -> Result<(), String> {
        let b_u32 = b_count as u32;
        let mut bc = b_count as i32;
        let (mut hc_in, mut gam, mut normed) =
            (bb.d_b_hc.ptr, hc.hc_norm.ptr, bb.d_b_hc_normed.ptr);
        let mut p_norm = kargs![hc_in, gam, normed];
        self.launch_kern(
            stream,
            self.fns.hc_norm4_batch,
            (4, b_u32),
            256,
            &mut p_norm,
            "q38_hc_norm4_batch",
        )?;

        let (mut dw_p, mut dsc_p, mut dbi_p) = (hc.down.w.ptr, hc.down.sc.ptr, hc.down.bi.ptr);
        let (mut iw_p, mut isc_p, mut ibi_p, mut has_inj) = if let Some(inj) = &hc.inject {
            (inj.w.ptr, inj.sc.ptr, inj.bi.ptr, 1i32)
        } else {
            (0u64, 0u64, 0u64, 0i32)
        };
        let (mut low_p, mut inj4_p) = (bb.d_b_hc_low.ptr, bb.d_b_hc_inj4.ptr);
        if b_count == 1 {
            let grid_di = if has_inj == 1 { 81u32 } else { 80u32 };
            let mut p_di =
                kargs![dw_p, dsc_p, dbi_p, iw_p, isc_p, ibi_p, normed, low_p, inj4_p, has_inj];
            self.launch_kern(
                stream,
                self.fns.hc_down_inj_10k,
                (grid_di, 1),
                256,
                &mut p_di,
                "q38_hc_down_inj_10k",
            )?;
            let (mut uw_p, mut usc_p, mut ubi_p, mut mixed) =
                (hc.up.w.ptr, hc.up.sc.ptr, hc.up.bi.ptr, bb.d_b_mixed.ptr);
            let mut p_um = kargs![uw_p, usc_p, ubi_p, low_p, normed, mixed];
            return self.launch_kern(
                stream,
                self.fns.hc_up_mix4,
                (320, 1),
                256,
                &mut p_um,
                "q38_hc_up_mix4",
            );
        }
        if b_count == 2 {
            let grid_di = if has_inj == 1 { 81u32 } else { 80u32 };
            let mut p_di =
                kargs![dw_p, dsc_p, dbi_p, iw_p, isc_p, ibi_p, normed, low_p, inj4_p, has_inj];
            self.launch_kern(
                stream,
                self.fns.hc_down_inj_10k_b2,
                (grid_di, 1),
                256,
                &mut p_di,
                "q38_hc_down_inj_10k_b2",
            )?;
            let (mut uw_p, mut usc_p, mut ubi_p, mut mixed) =
                (hc.up.w.ptr, hc.up.sc.ptr, hc.up.bi.ptr, bb.d_b_mixed.ptr);
            let mut p_um = kargs![uw_p, usc_p, ubi_p, low_p, normed, mixed];
            return self.launch_kern(
                stream,
                self.fns.hc_up_mix4_b2,
                (320, 1),
                256,
                &mut p_um,
                "q38_hc_up_mix4_b2",
            );
        }
        let grid_di = if has_inj == 1 { 41u32 } else { 40u32 };
        let grid_b4 = b_u32.div_ceil(4);
        let mut p_di =
            kargs![dw_p, dsc_p, dbi_p, iw_p, isc_p, ibi_p, normed, low_p, inj4_p, bc, has_inj];
        self.launch_kern(
            stream,
            self.fns.hc_down_inj_10k_b4,
            (grid_di, grid_b4),
            256,
            &mut p_di,
            "q38_hc_down_inj_10k_b4",
        )?;

        let (mut uw_p, mut usc_p, mut ubi_p, mut mixed) =
            (hc.up.w.ptr, hc.up.sc.ptr, hc.up.bi.ptr, bb.d_b_mixed.ptr);
        let grid_b8 = b_u32.div_ceil(8);
        let mut p_um = kargs![uw_p, usc_p, ubi_p, low_p, normed, mixed, bc];
        self.launch_kern(
            stream,
            self.fns.hc_up_mix4_b8,
            (320, grid_b8),
            256,
            &mut p_um,
            "q38_hc_up_mix4_b8",
        )
    }

    fn launch_hc_post_batch(
        &self,
        stream: CUstream,
        bb: &Qwen38BatchBuffers,
        b_count: usize,
    ) -> Result<(), String> {
        let (mut hc_io, mut br, mut inj4) =
            (bb.d_b_hc.ptr, bb.d_b_branch_out.ptr, bb.d_b_hc_inj4.ptr);
        let mut p = kargs![hc_io, br, inj4];
        self.launch_kern(
            stream,
            self.fns.hc_inject4_vec_batch,
            (3, b_count as u32),
            256,
            &mut p,
            "q38_hc_inject4_vec_batch",
        )
    }

    fn enqueue_batch_48_layers(
        &self,
        stream: CUstream,
        bb: &Qwen38BatchBuffers,
        ckpt: Option<&Qwen38VerifyRowCkpt>,
        b_count: usize,
    ) -> Result<(), String> {
        let b_u32 = b_count as u32;
        let mut bc = b_count as i32;
        let is_small_verify = b_count <= Q38_MAX_VERIFY_BATCH;
        {
            let (mut ew, mut esc, mut ebi, mut tin, mut dhc) = (
                self.embed.w.dev_ptr,
                self.embed.sc.dev_ptr,
                self.embed.bi.dev_ptr,
                bb.d_b_tokens.ptr,
                bb.d_b_hc.ptr,
            );
            let mut p = kargs![ew, esc, ebi, tin, dhc];
            self.launch_kern(
                stream,
                self.fns.embed_init_hc4_batch,
                (10, b_u32),
                256,
                &mut p,
                "q38_embed_init_hc4_batch",
            )?;
        }

        let mut gdn_idx = 0usize;
        for l_idx in 0..Q38_LAYERS {
            let layer = &self.layers[l_idx];
            if let Some(ple) = &layer.ple {
                self.launch_q4_gemm_b8(
                    stream,
                    &ple.kv_proj,
                    bb.d_b_ple_emb.ptr,
                    bb.d_b_ple_kv.ptr,
                    b_count,
                    0,
                )?;
                let (mut hc_io, mut kv_b) = (bb.d_b_hc.ptr, bb.d_b_ple_kv.ptr);
                let (mut nk, mut nq, mut nc, mut cw, mut h9, mut pos_p) = (
                    ple.norm_key.ptr,
                    ple.norm_query.ptr,
                    ple.norm_conv.ptr,
                    ple.conv_w4.ptr,
                    ple.hist9.ptr,
                    self.d_pos.ptr,
                );
                if is_small_verify {
                    let mut p_ckpt: CUdeviceptr = ckpt
                        .filter(|_| b_count >= 2)
                        .map(|c| c.ple_hist9_rows.dev_ptr())
                        .unwrap_or(0);
                    let mut p_ple = kargs![hc_io, kv_b, nk, nq, nc, cw, h9, p_ckpt, pos_p, bc];
                    self.launch_kern(
                        stream,
                        self.fns.ple_inject_ckpt_b8,
                        (4, 1),
                        256,
                        &mut p_ple,
                        "q38_ple_inject_ckpt_b8",
                    )?;
                } else {
                    let mut p_ple = kargs![hc_io, kv_b, nk, nq, nc, cw, h9, pos_p, bc];
                    self.launch_kern(
                        stream,
                        self.fns.ple_inject_seq_b,
                        (4, 1),
                        256,
                        &mut p_ple,
                        "q38_ple_inject_seq_b",
                    )?;
                }
            }

            self.launch_hc_pre_batch(stream, &layer.attn_hc, bb, b_count)?;

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
                    self.launch_q4_gemm_b8(
                        stream,
                        in_all,
                        bb.d_b_mixed.ptr,
                        bb.d_b_gdn_in.ptr,
                        b_count,
                        0,
                    )?;
                    let (mut gin, mut cw, mut ch3, mut qkv_c) = (
                        bb.d_b_gdn_in.ptr,
                        conv_w4.ptr,
                        conv_hist3.ptr,
                        bb.d_b_qkv_conv.ptr,
                    );
                    let (mut al, mut dtb, mut nw, mut sig_gate, mut st_p, mut gout) = (
                        a_log.ptr,
                        dt_bias.ptr,
                        norm_w.ptr,
                        self.use_gdn_sigmoid_gate,
                        state_f32.ptr,
                        bb.d_b_gdn_out.ptr,
                    );
                    if is_small_verify {
                        let (mut c_ckpt, mut s_ckpt) = ckpt
                            .filter(|_| b_count >= 2)
                            .map(|c| {
                                (
                                    c.gdn_conv_rows[gdn_idx].dev_ptr(),
                                    c.gdn_state_rows[gdn_idx].dev_ptr(),
                                )
                            })
                            .unwrap_or((0, 0));
                        let mut p_conv = kargs![gin, cw, ch3, c_ckpt, qkv_c, bc];
                        self.launch_kern(
                            stream,
                            self.fns.gdn_conv_silu_ckpt_b8,
                            (40, 1),
                            256,
                            &mut p_conv,
                            "q38_gdn_conv_silu_ckpt_b8",
                        )?;
                        let mut p_rec =
                            kargs![qkv_c, gin, al, dtb, nw, sig_gate, bc, st_p, s_ckpt, gout];
                        self.launch_kern(
                            stream,
                            self.fns.gdn_recurrent_step_ckpt_b8,
                            (48, 1),
                            128,
                            &mut p_rec,
                            "q38_gdn_recurrent_step_ckpt_b8",
                        )?;
                    } else {
                        let mut p_conv = kargs![gin, cw, ch3, qkv_c, bc];
                        self.launch_kern(
                            stream,
                            self.fns.gdn_conv_silu_seq_b,
                            (40, 1),
                            256,
                            &mut p_conv,
                            "q38_gdn_conv_silu_seq_b",
                        )?;
                        let mut p_rec = kargs![qkv_c, gin, al, dtb, nw, sig_gate, bc, st_p, gout];
                        self.launch_kern(
                            stream,
                            self.fns.gdn_recurrent_step_seq_b,
                            (48, 1),
                            128,
                            &mut p_rec,
                            "q38_gdn_recurrent_step_seq_b",
                        )?;
                    }
                    gdn_idx += 1;
                    self.launch_q4_gemm_b8(
                        stream,
                        out_proj,
                        bb.d_b_gdn_out.ptr,
                        bb.d_b_branch_out.ptr,
                        b_count,
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
                    self.launch_q4_gemm_b8(
                        stream,
                        qkv_idx,
                        bb.d_b_mixed.ptr,
                        bb.d_b_gdn_in.ptr,
                        b_count,
                        0,
                    )?;
                    let (mut qp, mut qnw, mut knw, mut iqnw, mut invf, mut pos_p) = (
                        bb.d_b_gdn_in.ptr,
                        q_norm_w.ptr,
                        k_norm_w.ptr,
                        idx_q_norm_w.ptr,
                        self.inv_freq32.ptr,
                        self.d_pos.ptr,
                    );
                    let (mut qrot, mut qgate, mut kc, mut vc, mut iq, mut irk) = (
                        bb.d_b_q_rot.ptr,
                        bb.d_b_q_gate.ptr,
                        k_cache_fp16.ptr,
                        v_cache_fp16.ptr,
                        bb.d_b_idx_q.ptr,
                        idx_raw_k_cache.ptr,
                    );
                    let mut p_prep =
                        kargs![qp, qnw, knw, iqnw, invf, pos_p, qrot, qgate, kc, vc, iq, irk];
                    self.launch_kern(
                        stream,
                        self.fns.qsa_prep_batch,
                        (31, b_u32),
                        256,
                        &mut p_prep,
                        "q38_qsa_prep_batch",
                    )?;

                    let (mut iknw, mut ibk) = (idx_k_norm_w.ptr, idx_block_k_cache.ptr);
                    let mut p_blk = kargs![irk, iknw, invf, pos_p, ibk];
                    self.launch_kern(
                        stream,
                        self.fns.qsa_build_blocks_batch,
                        (b_u32, 1),
                        128,
                        &mut p_blk,
                        "q38_qsa_build_blocks_batch",
                    )?;

                    let max_seq_in_chunk = self.host_pos + b_count;
                    let max_blocks = (max_seq_in_chunk / 4) as u32;
                    let mut mstride = bb.mask_stride as i32;
                    let (mut sc_b, mut mask_b) = (bb.d_b_qsa_scores.ptr, bb.d_b_qsa_mask.ptr);
                    if max_blocks > 512 {
                        let grid_sb = max_blocks.div_ceil(8);
                        let mut p_sc = kargs![iq, ibk, pos_p, mstride, sc_b];
                        let mut p_sel = kargs![sc_b, pos_p, mstride, mask_b];
                        self.launch_kern(
                            stream,
                            self.fns.qsa_score_blocks_batch,
                            (grid_sb, b_u32),
                            256,
                            &mut p_sc,
                            "q38_qsa_score_blocks_batch",
                        )?;
                        self.launch_kern(
                            stream,
                            self.fns.qsa_select_top512_batch,
                            (1, b_u32),
                            256,
                            &mut p_sel,
                            "q38_qsa_select_top512_batch",
                        )?;
                    }

                    let mut aout = bb.d_b_gdn_out.ptr;
                    let mut p_attn = kargs![qrot, qgate, kc, vc, mask_b, pos_p, mstride, aout];
                    self.launch_kern(
                        stream,
                        self.fns.qsa_attn_batch,
                        (24, b_u32),
                        256,
                        &mut p_attn,
                        "q38_qsa_attn_batch",
                    )?;
                    self.launch_q4_gemm_b8(
                        stream,
                        o_proj,
                        bb.d_b_gdn_out.ptr,
                        bb.d_b_branch_out.ptr,
                        b_count,
                        0,
                    )?;
                }
            }

            self.launch_hc_post_batch(stream, bb, b_count)?;
            self.launch_hc_pre_batch(stream, &layer.mlp_hc, bb, b_count)?;

            let (mut rw, mut mx, mut rlog, mut sids, mut swts) = (
                layer.router_bf16.ptr,
                bb.d_b_mixed.ptr,
                bb.d_b_router_logits.ptr,
                bb.d_b_sel_ids10.ptr,
                bb.d_b_sel_wts11.ptr,
            );
            if b_count == 1 {
                let mut p_rt = kargs![rw, mx, rlog];
                self.launch_kern(
                    stream,
                    self.fns.bf16_router_512,
                    (64, 1),
                    256,
                    &mut p_rt,
                    "q38_bf16_router_512",
                )?;
            } else {
                let mut p_rt = kargs![rw, mx, rlog, bc];
                self.launch_kern(
                    stream,
                    self.fns.bf16_router_512_b4,
                    (64, b_u32.div_ceil(4)),
                    256,
                    &mut p_rt,
                    "q38_bf16_router_512_b4",
                )?;
            }
            let mut p_tk = kargs![rlog, sids, swts];
            self.launch_kern(
                stream,
                self.fns.softmax_topk10_batch,
                (1, b_u32),
                256,
                &mut p_tk,
                "q38_softmax_topk10_batch",
            )?;

            let (
                [mut gw_p, mut glo_p, mut ghi_p, mut uw_p, mut ulo_p, mut uhi_p, mut dw_p, mut dlo_p, mut dhi_p],
                z24_bases,
            ) = layer
                .switch_mlp
                .z24_parts()
                .ok_or("batched MoE requires Z24 Vram/Mixed/HostPinned tiers")?;
            let mut up_bases = z24_bases.up_packed_m1;
            let mut dn_bases = z24_bases.down_packed_m1;
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
            let (mut act11, mut mout) = (bb.d_b_moe_act11.ptr, bb.d_b_branch_out.ptr);

            if is_small_verify {
                let mut p_up = kargs![
                    gw_p, glo_p, ghi_p, uw_p, ulo_p, uhi_p, up_bases, sh_gu_w, sh_gu_sc, sh_gu_bi,
                    mx, sids, swts, act11
                ];
                self.launch_kern(
                    stream,
                    self.fns.moe_up_swiglu11_z24_bn,
                    (160, 11 * b_u32),
                    256,
                    &mut p_up,
                    "q38_moe_up_swiglu11_z24_bn",
                )?;

                let mut p_dn = kargs![
                    dw_p, dlo_p, dhi_p, dn_bases, sh_dw, sh_dsc, sh_dbi, act11, sids, swts, mout
                ];
                self.launch_kern(
                    stream,
                    self.fns.moe_down_combine11_z24_bn,
                    (80, b_u32),
                    256,
                    &mut p_dn,
                    "q38_moe_down_combine11_z24_bn",
                )?;
            } else {
                let (mut eoff, mut epairs, mut dpart) = (
                    bb.d_b_exp_offsets.ptr,
                    bb.d_b_exp_pairs.ptr,
                    bb.d_b_down_part.ptr,
                );
                let mut p_plan = kargs![sids, bc, eoff, epairs];
                self.launch_kern(
                    stream,
                    self.fns.moe_build_plan_b64,
                    (1, 1),
                    512,
                    &mut p_plan,
                    "q38_moe_build_plan_b64",
                )?;

                let mut p_up = kargs![
                    gw_p, glo_p, ghi_p, uw_p, ulo_p, uhi_p, up_bases, sh_gu_w, sh_gu_sc, sh_gu_bi,
                    mx, eoff, epairs, bc, swts, act11
                ];
                self.launch_kern(
                    stream,
                    self.fns.moe_up_grouped_z24,
                    (80, 513),
                    512,
                    &mut p_up,
                    "q38_moe_up_grouped_z24",
                )?;

                let mut p_dn = kargs![
                    dw_p, dlo_p, dhi_p, dn_bases, sh_dw, sh_dsc, sh_dbi, act11, eoff, epairs, swts,
                    bc, dpart
                ];
                self.launch_kern(
                    stream,
                    self.fns.moe_down_grouped_z24,
                    (40, 513),
                    256,
                    &mut p_dn,
                    "q38_moe_down_grouped_z24",
                )?;

                let mut p_cmb = kargs![dpart, mout];
                self.launch_kern(
                    stream,
                    self.fns.moe_combine11_batch,
                    (10, b_u32),
                    256,
                    &mut p_cmb,
                    "q38_moe_combine11_batch",
                )?;
            }

            self.launch_hc_post_batch(stream, bb, b_count)?;
        }
        Ok(())
    }

    fn forward_batch_chunk(
        &mut self,
        tokens: &[u32],
        mode: BatchLogitMode,
    ) -> Result<Vec<u32>, String> {
        let b_count = tokens.len();
        assert!((1..=Q38_MAX_BATCH).contains(&b_count));
        let pos0 = self.host_pos;

        // Decode `b_count` PLE rows into slots `0..b_count` of `ple_stage_ring` (syncing first if previous chunk was async).
        self.ctx.sync_compute()?;
        let mut tails = [(Q38_PLE_EOS_ID, Q38_PLE_EOS_ID); Q38_MAX_VERIFY_BATCH];
        for (idx, &tok) in tokens.iter().enumerate() {
            let _ = self.decode_ple_into_ring_slot(tok, idx)?;
            if idx < Q38_MAX_VERIFY_BATCH {
                tails[idx] = (self.ple_prev2, self.ple_prev1);
            }
            let tok_ptr = unsafe { self.tok_stage_ring.host_ptr.add(idx * 4) };
            unsafe {
                core::ptr::copy_nonoverlapping((tok as i32).to_le_bytes().as_ptr(), tok_ptr, 4);
            }
        }
        if mode == BatchLogitMode::AllVerifyLogits {
            if let Some(ckpt) = self.verify_ckpt.as_mut() {
                ckpt.last_pos0 = pos0;
                ckpt.last_b_count = b_count;
                ckpt.ple_tails_after_row = tails;
            }
        }

        let stream = self.ctx.compute_stream;
        let bb = self.batch_bufs.as_ref().expect("batch_bufs initialized");
        unsafe {
            check_cu(
                (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                    bb.d_b_ple_emb.ptr,
                    self.ple_stage_ring.host_ptr.cast(),
                    b_count * Q38_D * 4,
                    stream,
                ),
                "HtoD(d_b_ple_emb)",
            )?;
            check_cu(
                (self.ctx.api.cuMemcpyHtoDAsync_v2)(
                    bb.d_b_tokens.ptr,
                    self.tok_stage_ring.host_ptr.cast(),
                    b_count * 4,
                    stream,
                ),
                "HtoD(d_b_tokens)",
            )?;
        }

        let ckpt_ref = if mode == BatchLogitMode::AllVerifyLogits {
            self.verify_ckpt.as_ref()
        } else {
            None
        };
        self.enqueue_batch_48_layers(stream, bb, ckpt_ref, b_count)?;

        let mut out_ids = Vec::new();
        match mode {
            BatchLogitMode::NoLogits => {
                let (mut pos_p, mut delta) = (self.d_pos.ptr, b_count as i32);
                let mut p_add = kargs![pos_p, delta];
                self.launch_kern(
                    stream,
                    self.fns.add_pos,
                    (1, 1),
                    1,
                    &mut p_add,
                    "q38_add_pos",
                )?;
            }
            BatchLogitMode::LastLogitOnly => {
                let last_hc_src = bb.d_b_hc.ptr + (((b_count - 1) * Q38_HC_D * 4) as u64);
                unsafe {
                    check_cu(
                        (self.ctx.api.cuMemcpyDtoDAsync_v2)(
                            self.d_hc.ptr,
                            last_hc_src,
                            Q38_HC_D * 4,
                            stream,
                        ),
                        "DtoD(last prefill d_hc)",
                    )?;
                }
                if b_count > 1 {
                    let (mut pos_p, mut delta) = (self.d_pos.ptr, (b_count - 1) as i32);
                    let mut p_add = kargs![pos_p, delta];
                    self.launch_kern(
                        stream,
                        self.fns.add_pos,
                        (1, 1),
                        1,
                        &mut p_add,
                        "q38_add_pos(last-1)",
                    )?;
                }
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
                self.launch_kern(
                    stream,
                    self.fns.argmax_fast,
                    (1, 1),
                    1024,
                    &mut p_arg,
                    "q38_argmax_fast",
                )?;
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
                out_ids.push(out_i32 as u32);
            }
            BatchLogitMode::AllVerifyLogits => {
                self.launch_hc_pre_batch(stream, &self.final_hc, bb, b_count)?;
                self.launch_q4_gemm_b8(
                    stream,
                    &self.lm_head,
                    bb.d_b_mixed.ptr,
                    bb.d_b_logits.ptr,
                    b_count,
                    0,
                )?;
                let (mut log_p, mut out_p, mut pos_p, mut bc) = (
                    bb.d_b_logits.ptr,
                    bb.d_b_token_out.ptr,
                    self.d_pos.ptr,
                    b_count as i32,
                );
                let mut p_arg = kargs![log_p, out_p, pos_p, bc];
                self.launch_kern(
                    stream,
                    self.fns.argmax_fast_batch,
                    (b_count as u32, 1),
                    1024,
                    &mut p_arg,
                    "q38_argmax_fast_batch",
                )?;
                self.ctx.sync_compute()?;
                let mut raw_ids = vec![0i32; b_count];
                unsafe {
                    check_cu(
                        (self.ctx.api.cuMemcpyDtoH_v2)(
                            raw_ids.as_mut_ptr().cast(),
                            bb.d_b_token_out.ptr,
                            b_count * 4,
                        ),
                        "DtoH(d_b_token_out)",
                    )?;
                }
                out_ids = raw_ids.into_iter().map(|v| v as u32).collect();
            }
        }

        self.host_pos += b_count;
        Ok(out_ids)
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

impl Qwen38GpuEngine {
    /// Handles the MTP drafter (`qwen38_mtp`) borrows: the CUDA context, the `[4, 2560]` HC state
    /// after the last layer (`d_hc`, valid after each forward), the embedding table (host-mapped),
    /// the LM head and the mapped shards.
    pub fn mtp_view(
        &self,
    ) -> (
        Arc<CudaContextHandle>,
        CUdeviceptr,
        crate::qwen38_mtp::Q4Ref,
        crate::qwen38_mtp::Q4Ref,
        &ShardedSafeTensors,
    ) {
        let embed = crate::qwen38_mtp::Q4Ref {
            w: self.embed.w.dev_ptr,
            sc: self.embed.sc.dev_ptr,
            bi: self.embed.bi.dev_ptr,
            rows: Q38_VOCAB as i32,
            cols: Q38_D as i32,
        };
        let lm_head = crate::qwen38_mtp::Q4Ref {
            w: self.lm_head.w.ptr,
            sc: self.lm_head.sc.ptr,
            bi: self.lm_head.bi.ptr,
            rows: self.lm_head.rows,
            cols: self.lm_head.cols,
        };
        (
            Arc::clone(&self.ctx),
            self.d_hc.ptr,
            embed,
            lm_head,
            &self.sharded_mmap,
        )
    }
}

#[cfg(test)]
mod tests {
    /// Exact-size `compute-sanitizer --tool memcheck --padding 65536` harness for every
    /// `kernels/qwen38_sm120.cu` kernel (#179, #248, #249): each launch gets buffers of exactly
    /// the documented shape, with indices at the last row / expert / position, so any over-read or
    /// over-write lands in the sanitizer's padding. `short = true` drops the last row of each
    /// case's key buffer (the planted fault); `LEANSONIC_EXACT_ONLY=<kernel>` runs one kernel.
    mod exact_size {
        use super::super::*;

        struct Lcg(u64);
        impl Lcg {
            fn next(&mut self) -> u32 {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (self.0 >> 33) as u32
            }
        }

        fn only() -> Option<String> {
            std::env::var("LEANSONIC_EXACT_ONLY").ok()
        }

        fn module(ctx: &Arc<CudaContextHandle>) -> Result<CUmodule, String> {
            let image: Vec<u8> = if let Some(cubin_path) = discover_qwen38_cubin() {
                let mut raw = std::fs::read(&cubin_path)
                    .map_err(|e| format!("read {}: {e}", cubin_path.display()))?;
                raw.push(0);
                raw
            } else {
                let header_nvrtc = CUDA_HEADER_SRC.replace("#include <stdint.h>", "");
                compile_cuda_to_ptx_nvrtc(
                    &CUDA_QWEN38_SRC
                        .replace("#include \"include/leansonic_verified.cuh\"", &header_nvrtc),
                    "compute_120a",
                )?
            };
            let mut m: CUmodule = core::ptr::null_mut();
            unsafe {
                check_cu(
                    (ctx.api.cuModuleLoadData)(&mut m, image.as_ptr().cast()),
                    "load qwen38",
                )?
            };
            Ok(m)
        }

        fn run(
            ctx: &Arc<CudaContextHandle>,
            m: CUmodule,
            name: &str,
            grid: (u32, u32),
            block: u32,
            args: &mut [*mut c_void],
        ) -> Result<(), String> {
            if only().is_some_and(|k| k != name) {
                return Ok(());
            }
            let c = CString::new(name).unwrap();
            let mut f: CUfunction = core::ptr::null_mut();
            unsafe {
                check_cu((ctx.api.cuModuleGetFunction)(&mut f, m, c.as_ptr()), name)?;
                check_cu(
                    (ctx.api.cuLaunchKernel)(
                        f,
                        grid.0,
                        grid.1,
                        1,
                        block,
                        1,
                        1,
                        0,
                        ctx.compute_stream,
                        args.as_mut_ptr(),
                        core::ptr::null_mut(),
                    ),
                    name,
                )?;
            }
            ctx.sync_compute().map_err(|e| format!("{name}: {e}"))?;
            if matches!(
                name,
                "q38_moe_up_swiglu11_z24"
                    | "q38_moe_down_combine11_z24"
                    | "q38_moe_up_grouped_z24"
                    | "q38_moe_down_grouped_z24"
                    | "q38_moe_up_swiglu11_z24_bn"
                    | "q38_moe_down_combine11_z24_bn"
            ) {
                let rec_bytes = SW_Z24_PROJ_BYTES / Q38_EXP;
                let gpu_rec = DeviceBuf::alloc_zeroed(ctx, rec_bytes)?;
                let cpu_rec = PinnedHostBuf::alloc_zeroed(ctx, rec_bytes)?;
                let mut zero_ptr: u64 = 0;
                for addr in [gpu_rec.ptr, cpu_rec.dev_ptr] {
                    let tbl = DeviceBuf::from_host_bytes(ctx, &addr.to_ne_bytes().repeat(Q38_EXP))?;
                    let mut tbl_ptr = tbl.ptr;
                    let mut mixed_args = args.to_vec();
                    mixed_args[0] = (&mut tbl_ptr as *mut u64).cast();
                    mixed_args[1] = (&mut zero_ptr as *mut u64).cast();
                    mixed_args[2] = (&mut zero_ptr as *mut u64).cast();
                    if name.starts_with("q38_moe_up_") {
                        mixed_args[3] = (&mut tbl_ptr as *mut u64).cast();
                        mixed_args[4] = (&mut zero_ptr as *mut u64).cast();
                        mixed_args[5] = (&mut zero_ptr as *mut u64).cast();
                    }
                    unsafe {
                        check_cu(
                            (ctx.api.cuLaunchKernel)(
                                f,
                                grid.0,
                                grid.1,
                                1,
                                block,
                                1,
                                1,
                                0,
                                ctx.compute_stream,
                                mixed_args.as_mut_ptr(),
                                core::ptr::null_mut(),
                            ),
                            name,
                        )?;
                    }
                    ctx.sync_compute()
                        .map_err(|e| format!("{name} (mixed): {e}"))?;
                }
            }
            Ok(())
        }

        fn noise(
            ctx: &Arc<CudaContextHandle>,
            rng: &mut Lcg,
            n: usize,
        ) -> Result<DeviceBuf, String> {
            let bytes: Vec<u8> = (0..n).map(|_| rng.next() as u8).collect();
            DeviceBuf::from_host_bytes(ctx, &bytes)
        }

        fn floats(
            ctx: &Arc<CudaContextHandle>,
            rng: &mut Lcg,
            n: usize,
        ) -> Result<DeviceBuf, String> {
            let bytes: Vec<u8> = (0..n)
                .flat_map(|_| ((rng.next() % 10_000) as f32 / 10_000.0 - 0.5).to_le_bytes())
                .collect();
            DeviceBuf::from_host_bytes(ctx, &bytes)
        }

        fn ints(ctx: &Arc<CudaContextHandle>, v: &[i32]) -> Result<DeviceBuf, String> {
            DeviceBuf::from_host_bytes(
                ctx,
                &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
            )
        }

        fn zeros(ctx: &Arc<CudaContextHandle>, n: usize) -> Result<DeviceBuf, String> {
            DeviceBuf::alloc_zeroed(ctx, n)
        }

        #[allow(clippy::too_many_arguments)]
        fn q4_case(
            ctx: &Arc<CudaContextHandle>,
            m: CUmodule,
            rng: &mut Lcg,
            rows: usize,
            cols: usize,
            mode: i32,
            short: usize,
        ) -> Result<(), String> {
            let w = noise(ctx, rng, (rows - short) * (cols / 32) * 16)?;
            let sc = noise(ctx, rng, rows * (cols / 64) * 4)?;
            let bi = noise(ctx, rng, rows * (cols / 64) * 4)?;
            let x = floats(ctx, rng, cols)?;
            let y = zeros(ctx, rows * 4)?;
            let (mut pw, mut ps, mut pb, mut px, mut py) = (w.ptr, sc.ptr, bi.ptr, x.ptr, y.ptr);
            let (mut r, mut c, mut md) = (rows as i32, cols as i32, mode);
            let grid = ((rows as u32).div_ceil(8), 1);
            if cols == Q38_HC_D {
                run(
                    ctx,
                    m,
                    "q38_q4_gemv_10k",
                    grid,
                    256,
                    &mut kargs![pw, ps, pb, px, py, r, md],
                )?;
            } else {
                run(
                    ctx,
                    m,
                    "q38_q4_gemv",
                    grid,
                    256,
                    &mut kargs![pw, ps, pb, px, py, r, c, md],
                )?;
                if cols == Q38_D {
                    run(
                        ctx,
                        m,
                        "q38_q4_gemv_2560",
                        grid,
                        256,
                        &mut kargs![pw, ps, pb, px, py, r, md],
                    )?;
                }
            }
            Ok(())
        }

        fn cases(short: bool) -> Result<(), String> {
            let ctx = CudaContextHandle::init(0)?;
            let ctx = &ctx;
            let m = module(ctx)?;
            let mut rng = Lcg(41);
            let rng = &mut rng;
            let s = usize::from(short);
            let (d, hcd) = (Q38_D, Q38_HC_D);

            // Embedding: `[vocab, 80]` u128 codes + `[vocab, 40]` scale / bias words, last token.
            {
                let w = noise(ctx, rng, (Q38_VOCAB - s) * 80 * 16)?;
                let sc = noise(ctx, rng, Q38_VOCAB * 40 * 4)?;
                let bi = noise(ctx, rng, Q38_VOCAB * 40 * 4)?;
                let tok = ints(ctx, &[Q38_VOCAB as i32 - 1])?;
                let out = zeros(ctx, hcd * 4)?;
                let (mut pw, mut ps, mut pb, mut pt, mut po) =
                    (w.ptr, sc.ptr, bi.ptr, tok.ptr, out.ptr);
                run(
                    ctx,
                    m,
                    "q38_embed_init_hc4",
                    (10, 1),
                    256,
                    &mut kargs![pw, ps, pb, pt, po],
                )?;
            }

            // Q4 GEMVs (`q38_q4_gemv`, `q38_q4_gemv_2560`, `q38_q4_gemv_10k`) across all shapes + 17-row tail.
            for (rows, cols, mode) in [
                (320, hcd, 1),
                (4, hcd, 0),
                (hcd, 320, 2),
                (16_480, d, 0),
                (d, 6144, 0),
                (13_952, d, 0),
                (5 * d, d, 0),
                (17, 6144, 1),
                (19, d, 2),
                (Q38_VOCAB, d, 0),
            ] {
                q4_case(ctx, m, rng, rows, cols, mode, s)?;
            }

            // HC pre / post (both original + fused/vectorized) and PLE inject.
            {
                let hc = floats(ctx, rng, 4 * d - s * 4)?;
                let gamma = floats(ctx, rng, 4 * d)?;
                let normed = zeros(ctx, 4 * d * 4)?;
                let (mut ph, mut pg, mut pn) = (hc.ptr, gamma.ptr, normed.ptr);
                run(ctx, m, "q38_hc_norm4", (4, 1), 256, &mut kargs![ph, pg, pn])?;

                // Fused `q38_hc_down_inj_10k` (both `has_inject = 1` and `0`)
                {
                    let dw = noise(ctx, rng, (320 - s) * 320 * 16)?;
                    let dsc = noise(ctx, rng, 320 * 160 * 4)?;
                    let dbi = noise(ctx, rng, 320 * 160 * 4)?;
                    let iw = noise(ctx, rng, (4 - s) * 320 * 16)?;
                    let isc = noise(ctx, rng, 4 * 160 * 4)?;
                    let ibi = noise(ctx, rng, 4 * 160 * 4)?;
                    let hc_norm_in = floats(ctx, rng, hcd)?;
                    let low_out = zeros(ctx, 320 * 4)?;
                    let inj4_out = zeros(ctx, 4 * 4)?;
                    let (mut pdw, mut pds, mut pdb) = (dw.ptr, dsc.ptr, dbi.ptr);
                    let (mut piw, mut pis, mut pib) = (iw.ptr, isc.ptr, ibi.ptr);
                    let (mut pni, mut plo, mut pio4) = (hc_norm_in.ptr, low_out.ptr, inj4_out.ptr);
                    let mut hi1 = 1i32;
                    run(
                        ctx,
                        m,
                        "q38_hc_down_inj_10k",
                        (81, 1),
                        256,
                        &mut kargs![pdw, pds, pdb, piw, pis, pib, pni, plo, pio4, hi1],
                    )?;
                    let mut hi0 = 0i32;
                    let mut null_p = 0u64;
                    run(
                        ctx,
                        m,
                        "q38_hc_down_inj_10k",
                        (80, 1),
                        256,
                        &mut kargs![pdw, pds, pdb, null_p, null_p, null_p, pni, plo, pio4, hi0],
                    )?;

                    // Fused `q38_hc_up_mix4`
                    let uw = noise(ctx, rng, (hcd - s) * 10 * 16)?;
                    let usc = noise(ctx, rng, hcd * 5 * 4)?;
                    let ubi = noise(ctx, rng, hcd * 5 * 4)?;
                    let mixed_f = zeros(ctx, d * 4)?;
                    let (mut puw, mut pus, mut pub_, mut pmf) =
                        (uw.ptr, usc.ptr, ubi.ptr, mixed_f.ptr);
                    run(
                        ctx,
                        m,
                        "q38_hc_up_mix4",
                        (320, 1),
                        256,
                        &mut kargs![puw, pus, pub_, plo, pni, pmf],
                    )?;
                }

                let up = floats(ctx, rng, 4 * d - s)?;
                let mixed = zeros(ctx, d * 4)?;
                let (mut pu, mut pm) = (up.ptr, mixed.ptr);
                run(ctx, m, "q38_hc_mix4", (10, 1), 256, &mut kargs![pn, pu, pm])?;
                let branch = floats(ctx, rng, d - s * 4)?;
                let inj = floats(ctx, rng, 4)?;
                let (mut pbr, mut pi) = (branch.ptr, inj.ptr);
                run(
                    ctx,
                    m,
                    "q38_hc_inject4",
                    (40, 1),
                    256,
                    &mut kargs![ph, pbr, pi],
                )?;
                run(
                    ctx,
                    m,
                    "q38_hc_inject4_vec",
                    (3, 1),
                    256,
                    &mut kargs![ph, pbr, pi],
                )?;

                let kv = floats(ctx, rng, 5 * d)?;
                let (nk, nq, nc) = (
                    floats(ctx, rng, 4 * d)?,
                    floats(ctx, rng, 4 * d)?,
                    floats(ctx, rng, 4 * d)?,
                );
                let cw = floats(ctx, rng, 4 * 4 * d)?;
                let hist = floats(ctx, rng, 9 * hcd - s)?;
                let pos = ints(ctx, &[1_000_003])?;
                let (mut pk, mut pv) = (kv.ptr, kv.ptr + (4 * d * 4) as u64);
                let (mut pnk, mut pnq, mut pnc, mut pcw, mut phi, mut pp) =
                    (nk.ptr, nq.ptr, nc.ptr, cw.ptr, hist.ptr, pos.ptr);
                run(
                    ctx,
                    m,
                    "q38_ple_inject",
                    (4, 1),
                    256,
                    &mut kargs![ph, pk, pv, pnk, pnq, pnc, pcw, phi, pp],
                )?;
            }

            // GDN: conv over `[10240]` channels, recurrent step over 48 V-heads.
            {
                let raw = floats(ctx, rng, hcd)?;
                let cw = floats(ctx, rng, 4 * hcd)?;
                let hist = floats(ctx, rng, 3 * hcd - s)?;
                let conv = zeros(ctx, hcd * 4)?;
                let (mut pr, mut pc, mut ph, mut po) = (raw.ptr, cw.ptr, hist.ptr, conv.ptr);
                run(
                    ctx,
                    m,
                    "q38_gdn_conv_silu",
                    (40, 1),
                    256,
                    &mut kargs![pr, pc, ph, po],
                )?;
                let zba = floats(ctx, rng, 6240)?;
                let (al, dt, nw) = (
                    floats(ctx, rng, 48)?,
                    floats(ctx, rng, 48)?,
                    floats(ctx, rng, 128)?,
                );
                let state = floats(ctx, rng, 48 * 128 * 128 - s)?;
                let out = zeros(ctx, 6144 * 4)?;
                let (mut pz, mut pa, mut pd, mut pn, mut ps, mut pout) =
                    (zba.ptr, al.ptr, dt.ptr, nw.ptr, state.ptr, out.ptr);
                for mut gate in [0i32, 1] {
                    run(
                        ctx,
                        m,
                        "q38_gdn_recurrent_step",
                        (48, 1),
                        128,
                        &mut kargs![po, pz, pa, pd, pn, gate, ps, pout],
                    )?;
                }
            }

            // QSA at the last cache row: `max_ctx = 4100` (1025 key blocks > 512).
            {
                let max_ctx = 4100usize;
                let n_blocks = max_ctx / 4;
                let proj = floats(ctx, rng, 13_952)?;
                let (qn, kn) = (floats(ctx, rng, 256)?, floats(ctx, rng, 256)?);
                let (iqn, ikn) = (floats(ctx, rng, 128)?, floats(ctx, rng, 128)?);
                let invf = floats(ctx, rng, 32)?;
                let pos = ints(ctx, &[max_ctx as i32 - 1])?;
                let (qrot, qgate) = (zeros(ctx, 24 * 256 * 4)?, zeros(ctx, 24 * 256 * 4)?);
                let kc = noise(ctx, rng, (max_ctx - s) * 512 * 2)?;
                let vc = noise(ctx, rng, max_ctx * 512 * 2)?;
                let iq = zeros(ctx, 4 * 128 * 4)?;
                let irk = zeros(ctx, max_ctx * 128 * 4)?;
                let ibk = floats(ctx, rng, n_blocks * 128)?;
                let (mut pp, mut pqn, mut pkn, mut piq, mut pik, mut pif, mut ppos) = (
                    proj.ptr, qn.ptr, kn.ptr, iqn.ptr, ikn.ptr, invf.ptr, pos.ptr,
                );
                let (mut pqr, mut pqg, mut pkc, mut pvc, mut pio, mut pirk, mut pibk) = (
                    qrot.ptr, qgate.ptr, kc.ptr, vc.ptr, iq.ptr, irk.ptr, ibk.ptr,
                );
                run(
                    ctx,
                    m,
                    "q38_qsa_prep_and_cache",
                    (31, 1),
                    256,
                    &mut kargs![
                        pp, pqn, pkn, piq, pik, pif, ppos, pqr, pqg, pkc, pvc, pio, pirk, pibk
                    ],
                )?;
                let scores = zeros(ctx, (n_blocks - s) * 4)?;
                let mask = zeros(ctx, n_blocks * 4)?;
                let (mut psc, mut pmask, mut nb) = (scores.ptr, mask.ptr, n_blocks as i32);
                run(
                    ctx,
                    m,
                    "q38_qsa_score_blocks",
                    ((n_blocks as u32).div_ceil(8), 1),
                    256,
                    &mut kargs![pio, pibk, nb, psc],
                )?;
                run(
                    ctx,
                    m,
                    "q38_qsa_select_top512",
                    (1, 1),
                    256,
                    &mut kargs![psc, nb, pmask],
                )?;
                let out = zeros(ctx, 24 * 256 * 4)?;
                let mut pout = out.ptr;
                for mut use_mask in [0i32, 1] {
                    run(
                        ctx,
                        m,
                        "q38_qsa_attn",
                        (24, 1),
                        256,
                        &mut kargs![pqr, pqg, pkc, pvc, pmask, ppos, use_mask, pout],
                    )?;
                }
            }

            // Router + Top-10 + MoE (both raw BF16 and Z24 24-bit lossless): expert 511 selected.
            {
                let rw = noise(ctx, rng, (Q38_EXP - s) * 320 * 16)?;
                let x = floats(ctx, rng, d)?;
                let logits = zeros(ctx, Q38_EXP * 4)?;
                let (mut prw, mut px, mut pl) = (rw.ptr, x.ptr, logits.ptr);
                run(
                    ctx,
                    m,
                    "q38_bf16_router_512",
                    (64, 1),
                    256,
                    &mut kargs![prw, px, pl],
                )?;
                let lg = floats(ctx, rng, Q38_EXP - s)?;
                let ids_out = zeros(ctx, 10 * 4)?;
                let wts = floats(ctx, rng, 11)?;
                let (mut plg, mut pid, mut pwt) = (lg.ptr, ids_out.ptr, wts.ptr);
                run(
                    ctx,
                    m,
                    "q38_softmax_topk10",
                    (1, 1),
                    256,
                    &mut kargs![plg, pid, pwt],
                )?;

                let sel = ints(ctx, &[511, 0, 7, 300, 511, 64, 2, 499, 128, 510])?;
                let sh_gu = noise(ctx, rng, 1281 * 80 * 16)?;
                let (sh_sc, sh_bi) = (
                    noise(ctx, rng, 1281 * 40 * 4)?,
                    noise(ctx, rng, 1281 * 40 * 4)?,
                );
                let sh_d = noise(ctx, rng, d * 20 * 16)?;
                let (sh_dsc, sh_dbi) = (noise(ctx, rng, d * 10 * 4)?, noise(ctx, rng, d * 10 * 4)?);
                let act = zeros(ctx, 11 * Q38_MOE_FF * 4)?;
                let out = zeros(ctx, d * 4)?;
                for (experts, compact) in [(Q38_EXP, 0i32), (Q38_TOP_K, 1)] {
                    let gw = noise(ctx, rng, (experts * 640 - s) * 80 * 16)?;
                    let uw = noise(ctx, rng, experts * 640 * 80 * 16)?;
                    let gsc = noise(ctx, rng, experts * 640 * 40 * 4)?;
                    let gbi = noise(ctx, rng, experts * 640 * 40 * 4)?;
                    let usc = noise(ctx, rng, experts * 640 * 40 * 4)?;
                    let ubi = noise(ctx, rng, experts * 640 * 40 * 4)?;
                    let (mut pgw, mut pgs, mut pgb, mut puw, mut pus, mut pub_) =
                        (gw.ptr, gsc.ptr, gbi.ptr, uw.ptr, usc.ptr, ubi.ptr);
                    let (mut psw, mut pss, mut psb, mut px2, mut psel, mut pw2, mut pa) = (
                        sh_gu.ptr, sh_sc.ptr, sh_bi.ptr, x.ptr, sel.ptr, wts.ptr, act.ptr,
                    );
                    let mut cmp = compact;
                    run(
                        ctx,
                        m,
                        "q38_moe_up_swiglu11",
                        (160, 11),
                        256,
                        &mut kargs![
                            pgw, pgs, pgb, puw, pus, pub_, psw, pss, psb, px2, psel, cmp, pw2, pa
                        ],
                    )?;
                    if compact == 0 {
                        let glo = noise(ctx, rng, experts * 640 * 40 * 4)?;
                        let ghi = noise(ctx, rng, experts * 640 * 20 * 4)?;
                        let ulo = noise(ctx, rng, experts * 640 * 40 * 4)?;
                        let uhi = noise(ctx, rng, experts * 640 * 20 * 4)?;
                        let (mut pglo, mut pghi, mut pulo, mut puhi) =
                            (glo.ptr, ghi.ptr, ulo.ptr, uhi.ptr);
                        let mut up_bases = 0x76727672u32;
                        run(
                            ctx,
                            m,
                            "q38_moe_up_swiglu11_z24",
                            (160, 11),
                            256,
                            &mut kargs![
                                pgw, pglo, pghi, puw, pulo, puhi, up_bases, psw, pss, psb, px2,
                                psel, pw2, pa
                            ],
                        )?;
                    }
                    drop((gw, uw, gsc, gbi, usc, ubi));
                    let dw = noise(ctx, rng, (experts * d - s) * 20 * 16)?;
                    let dsc = noise(ctx, rng, experts * d * 10 * 4)?;
                    let dbi = noise(ctx, rng, experts * d * 10 * 4)?;
                    let (mut pdw, mut pds, mut pdb, mut phw, mut phs, mut phb, mut po) = (
                        dw.ptr, dsc.ptr, dbi.ptr, sh_d.ptr, sh_dsc.ptr, sh_dbi.ptr, out.ptr,
                    );
                    run(
                        ctx,
                        m,
                        "q38_moe_down_combine11",
                        (320, 1),
                        256,
                        &mut kargs![pdw, pds, pdb, phw, phs, phb, pa, psel, cmp, pw2, po],
                    )?;
                    if compact == 0 {
                        let dlo = noise(ctx, rng, experts * d * 10 * 4)?;
                        let dhi = noise(ctx, rng, experts * d * 5 * 4)?;
                        let (mut pdlo, mut pdhi) = (dlo.ptr, dhi.ptr);
                        let mut dn_bases = 0x7672u32;
                        run(
                            ctx,
                            m,
                            "q38_moe_down_combine11_z24",
                            (80, 1),
                            256,
                            &mut kargs![
                                pdw, pdlo, pdhi, dn_bases, phw, phs, phb, pa, psel, pw2, po
                            ],
                        )?;
                    }
                }
            }

            // Greedy head: `[vocab]` logits (`q38_argmax_and_inc_pos` and `q38_argmax_fast`).
            {
                let logits = floats(ctx, rng, Q38_VOCAB - s * 4)?;
                let tok = zeros(ctx, 4)?;
                let pos = ints(ctx, &[7])?;
                let (mut pl, mut pt, mut pp) = (logits.ptr, tok.ptr, pos.ptr);
                run(
                    ctx,
                    m,
                    "q38_argmax_and_inc_pos",
                    (1, 1),
                    256,
                    &mut kargs![pl, pt, pp],
                )?;
                run(
                    ctx,
                    m,
                    "q38_argmax_fast",
                    (1, 1),
                    1024,
                    &mut kargs![pl, pt, pp],
                )?;
                run(ctx, m, "q38_inc_pos", (1, 1), 1, &mut kargs![pp])?;
                let mut pp_add = pos.ptr + (s * 16) as u64;
                let mut delta = 5i32;
                run(ctx, m, "q38_add_pos", (1, 1), 1, &mut kargs![pp_add, delta])?;
            }

            // Batched Prefill & Verify Kernels (`B = 5` ragged non-power-of-two batch!)
            {
                let b_cnt = 5usize;
                let mut bc = b_cnt as i32;
                let toks = ints(ctx, &[0, 7, 42, 1024, Q38_VOCAB as i32 - 1])?;
                let ew = noise(ctx, rng, (Q38_VOCAB - s) * 80 * 16)?;
                let esc = noise(ctx, rng, Q38_VOCAB * 40 * 4)?;
                let ebi = noise(ctx, rng, Q38_VOCAB * 40 * 4)?;
                let hc_b = zeros(ctx, b_cnt * hcd * 4)?;
                let (mut pew, mut pesc, mut pebi, mut ptoks, mut phcb) =
                    (ew.ptr, esc.ptr, ebi.ptr, toks.ptr, hc_b.ptr);
                run(
                    ctx,
                    m,
                    "q38_embed_init_hc4_batch",
                    (10, b_cnt as u32),
                    256,
                    &mut kargs![pew, pesc, pebi, ptoks, phcb],
                )?;
                drop((ew, esc, ebi));

                // `q38_q4_gemm_b8` (`19` rows × `2560` cols × `5` tokens)
                {
                    let rows = 19usize;
                    let w = noise(ctx, rng, (rows - s) * 80 * 16)?;
                    let sc = noise(ctx, rng, rows * 40 * 4)?;
                    let bi = noise(ctx, rng, rows * 40 * 4)?;
                    let xb = floats(ctx, rng, b_cnt * d)?;
                    let yb = zeros(ctx, b_cnt * rows * 4)?;
                    let (mut pw, mut psc, mut pbi, mut pxb, mut pyb) =
                        (w.ptr, sc.ptr, bi.ptr, xb.ptr, yb.ptr);
                    let (mut r_i, mut c_i, mut md) = (rows as i32, d as i32, 0i32);
                    run(
                        ctx,
                        m,
                        "q38_q4_gemm_b8",
                        (1, 1),
                        512,
                        &mut kargs![pw, psc, pbi, pxb, pyb, r_i, c_i, bc, md],
                    )?;
                }

                // Batched HC + PLE
                {
                    let hcb_in = floats(ctx, rng, b_cnt * hcd - s * 4)?;
                    let gam = floats(ctx, rng, hcd)?;
                    let norm_b = zeros(ctx, b_cnt * hcd * 4)?;
                    let (mut phin, mut pgam, mut pnb) = (hcb_in.ptr, gam.ptr, norm_b.ptr);
                    run(
                        ctx,
                        m,
                        "q38_hc_norm4_batch",
                        (4, b_cnt as u32),
                        256,
                        &mut kargs![phin, pgam, pnb],
                    )?;

                    let dw = noise(ctx, rng, (320 - s) * 320 * 16)?;
                    let dsc = noise(ctx, rng, 320 * 160 * 4)?;
                    let dbi = noise(ctx, rng, 320 * 160 * 4)?;
                    let iw = noise(ctx, rng, (4 - s) * 320 * 16)?;
                    let isc = noise(ctx, rng, 4 * 160 * 4)?;
                    let ibi = noise(ctx, rng, 4 * 160 * 4)?;
                    let low_b = zeros(ctx, b_cnt * 320 * 4)?;
                    let inj_b = zeros(ctx, b_cnt * 4 * 4)?;
                    let (mut pdw, mut pds, mut pdb) = (dw.ptr, dsc.ptr, dbi.ptr);
                    let (mut piw, mut pis, mut pib) = (iw.ptr, isc.ptr, ibi.ptr);
                    let (mut plb, mut pinjb) = (low_b.ptr, inj_b.ptr);
                    let mut hi1 = 1i32;
                    run(
                        ctx,
                        m,
                        "q38_hc_down_inj_10k_b4",
                        (41, 2),
                        256,
                        &mut kargs![pdw, pds, pdb, piw, pis, pib, pnb, plb, pinjb, bc, hi1],
                    )?;

                    let uw = noise(ctx, rng, (hcd - s) * 10 * 16)?;
                    let usc = noise(ctx, rng, hcd * 5 * 4)?;
                    let ubi = noise(ctx, rng, hcd * 5 * 4)?;
                    let mix_b = zeros(ctx, b_cnt * d * 4)?;
                    let (mut puw, mut pus, mut pub_, mut pmxb) =
                        (uw.ptr, usc.ptr, ubi.ptr, mix_b.ptr);
                    run(
                        ctx,
                        m,
                        "q38_hc_up_mix4_b8",
                        (320, 1),
                        256,
                        &mut kargs![puw, pus, pub_, plb, pnb, pmxb, bc],
                    )?;

                    let br_b = floats(ctx, rng, b_cnt * d - s * 4)?;
                    let mut pbrb = br_b.ptr;
                    run(
                        ctx,
                        m,
                        "q38_hc_inject4_vec_batch",
                        (3, b_cnt as u32),
                        256,
                        &mut kargs![phin, pbrb, pinjb],
                    )?;

                    let kv_b = floats(ctx, rng, b_cnt * 5 * d)?;
                    let (nk, nq, nc) = (
                        floats(ctx, rng, hcd)?,
                        floats(ctx, rng, hcd)?,
                        floats(ctx, rng, hcd)?,
                    );
                    let cw = floats(ctx, rng, 4 * hcd)?;
                    let hist = floats(ctx, rng, 9 * hcd - s)?;
                    let pos0 = ints(ctx, &[100])?;
                    let (mut pkvb, mut pnk, mut pnq, mut pnc, mut pcw, mut phist, mut ppos0) =
                        (kv_b.ptr, nk.ptr, nq.ptr, nc.ptr, cw.ptr, hist.ptr, pos0.ptr);
                    run(
                        ctx,
                        m,
                        "q38_ple_inject_seq_b",
                        (4, 1),
                        256,
                        &mut kargs![phcb, pkvb, pnk, pnq, pnc, pcw, phist, ppos0, bc],
                    )?;
                }

                // Batched GDN
                {
                    let gin_b = floats(ctx, rng, b_cnt * 16_480)?;
                    let cw = floats(ctx, rng, 4 * hcd)?;
                    let hist3 = floats(ctx, rng, 3 * hcd - s)?;
                    let qkv_c = zeros(ctx, b_cnt * hcd * 4)?;
                    let (mut pgin, mut pcw, mut ph3, mut pqkvc) =
                        (gin_b.ptr, cw.ptr, hist3.ptr, qkv_c.ptr);
                    run(
                        ctx,
                        m,
                        "q38_gdn_conv_silu_seq_b",
                        (40, 1),
                        256,
                        &mut kargs![pgin, pcw, ph3, pqkvc, bc],
                    )?;

                    let (al, dt, nw) = (
                        floats(ctx, rng, 48)?,
                        floats(ctx, rng, 48)?,
                        floats(ctx, rng, 128)?,
                    );
                    let st = floats(ctx, rng, 48 * 128 * 128 - s)?;
                    let gout = zeros(ctx, b_cnt * 6144 * 4)?;
                    let (mut pal, mut pdt, mut pnw, mut pst, mut pgout) =
                        (al.ptr, dt.ptr, nw.ptr, st.ptr, gout.ptr);
                    let mut sig = 1i32;
                    run(
                        ctx,
                        m,
                        "q38_gdn_recurrent_step_seq_b",
                        (48, 1),
                        128,
                        &mut kargs![pqkvc, pgin, pal, pdt, pnw, sig, bc, pst, pgout],
                    )?;
                }

                // Batched QSA at long context (`pos0 = 4095 .. 4099`, crossing 4-token block boundary!)
                {
                    let max_ctx = 4100usize;
                    let mstride = max_ctx / 4;
                    let qp_b = floats(ctx, rng, b_cnt * 13_952)?;
                    let (qn, kn, iqn, ikn, invf) = (
                        floats(ctx, rng, 256)?,
                        floats(ctx, rng, 256)?,
                        floats(ctx, rng, 128)?,
                        floats(ctx, rng, 128)?,
                        floats(ctx, rng, 32)?,
                    );
                    let pos0 = ints(ctx, &[(max_ctx - b_cnt) as i32])?;
                    let (qrot, qgate) =
                        (zeros(ctx, b_cnt * 6144 * 4)?, zeros(ctx, b_cnt * 6144 * 4)?);
                    let kc = noise(ctx, rng, (max_ctx - s) * 512 * 2)?;
                    let vc = noise(ctx, rng, max_ctx * 512 * 2)?;
                    let iqb = zeros(ctx, b_cnt * 512 * 4)?;
                    let irk = floats(ctx, rng, max_ctx * 128)?;
                    let ibk = floats(ctx, rng, (mstride - s) * 128)?;
                    let (mut pqp, mut pqn, mut pkn, mut piqn, mut pikn, mut pinvf, mut ppos0) = (
                        qp_b.ptr, qn.ptr, kn.ptr, iqn.ptr, ikn.ptr, invf.ptr, pos0.ptr,
                    );
                    let (mut pqr, mut pqg, mut pkc, mut pvc, mut piqb, mut pirk, mut pibk) = (
                        qrot.ptr, qgate.ptr, kc.ptr, vc.ptr, iqb.ptr, irk.ptr, ibk.ptr,
                    );
                    run(
                        ctx,
                        m,
                        "q38_qsa_prep_batch",
                        (31, b_cnt as u32),
                        256,
                        &mut kargs![
                            pqp, pqn, pkn, piqn, pinvf, ppos0, pqr, pqg, pkc, pvc, piqb, pirk
                        ],
                    )?;
                    run(
                        ctx,
                        m,
                        "q38_qsa_build_blocks_batch",
                        (b_cnt as u32, 1),
                        128,
                        &mut kargs![pirk, pikn, pinvf, ppos0, pibk],
                    )?;
                    let sc_b = zeros(ctx, (b_cnt * mstride - s) * 4)?;
                    let msk_b = ints(ctx, &vec![1i32; b_cnt * mstride])?;
                    let aout_b = zeros(ctx, b_cnt * 6144 * 4)?;
                    let (mut pscb, mut pmskb, mut paoutb, mut mst_i) =
                        (sc_b.ptr, msk_b.ptr, aout_b.ptr, mstride as i32);
                    run(
                        ctx,
                        m,
                        "q38_qsa_score_blocks_batch",
                        ((mstride as u32).div_ceil(8), b_cnt as u32),
                        256,
                        &mut kargs![piqb, pibk, ppos0, mst_i, pscb],
                    )?;
                    run(
                        ctx,
                        m,
                        "q38_qsa_select_top512_batch",
                        (1, b_cnt as u32),
                        256,
                        &mut kargs![pscb, ppos0, mst_i, pmskb],
                    )?;
                    run(
                        ctx,
                        m,
                        "q38_qsa_attn_batch",
                        (24, b_cnt as u32),
                        256,
                        &mut kargs![pqr, pqg, pkc, pvc, pmskb, ppos0, mst_i, paoutb],
                    )?;
                }

                // Batched Router + Grouped Active-Expert Z24 MoE + Batched ArgMax
                {
                    let rw = noise(ctx, rng, (Q38_EXP - s) * 320 * 16)?;
                    let xb = floats(ctx, rng, b_cnt * d)?;
                    let rlog_b = floats(ctx, rng, b_cnt * Q38_EXP - s)?;
                    let (mut prw, mut pxb, mut prlb) = (rw.ptr, xb.ptr, rlog_b.ptr);
                    run(
                        ctx,
                        m,
                        "q38_bf16_router_512_b4",
                        (64, 2),
                        256,
                        &mut kargs![prw, pxb, prlb, bc],
                    )?;

                    let mut ids_vec = Vec::with_capacity(b_cnt * 10);
                    for t in 0..b_cnt {
                        ids_vec.extend_from_slice(&[
                            511,
                            0,
                            (t * 3 + 1) as i32,
                            300,
                            511,
                            64,
                            2,
                            499,
                            128,
                            510,
                        ]);
                    }
                    let sids_b = ints(ctx, &ids_vec)?;
                    let swts_b = floats(ctx, rng, b_cnt * 11)?;
                    let (mut psids, mut pswts) = (sids_b.ptr, swts_b.ptr);
                    run(
                        ctx,
                        m,
                        "q38_softmax_topk10_batch",
                        (1, b_cnt as u32),
                        256,
                        &mut kargs![prlb, psids, pswts],
                    )?;
                    // Re-seed `sids_b` so expert 511 is guaranteed active for exact-size boundary check!
                    let sids_511 = ints(ctx, &ids_vec)?;
                    let mut ps511 = sids_511.ptr;
                    let eoff = zeros(ctx, 513 * 4)?;
                    let epairs = zeros(ctx, (b_cnt * 10 - s) * 4)?;
                    let (mut peoff, mut pepairs) = (eoff.ptr, epairs.ptr);
                    run(
                        ctx,
                        m,
                        "q38_moe_build_plan_b64",
                        (1, 1),
                        512,
                        &mut kargs![ps511, bc, peoff, pepairs],
                    )?;

                    // Deterministic host-built active-expert plan (`expert 511` active!) for grouped MoE kernels
                    let mut counts = vec![0i32; 512];
                    for &e in &ids_vec {
                        counts[e as usize] += 1;
                    }
                    let mut offsets_vec = vec![0i32; 513];
                    for e in 0..512 {
                        offsets_vec[e + 1] = offsets_vec[e] + counts[e];
                    }
                    let mut cur = offsets_vec[..512].to_vec();
                    let mut pairs_vec = vec![0i32; b_cnt * 10];
                    for (idx, &e) in ids_vec.iter().enumerate() {
                        let b_i = (idx / 10) as i32;
                        let k_i = (idx % 10) as i32;
                        let slot = cur[e as usize] as usize;
                        cur[e as usize] += 1;
                        pairs_vec[slot] = (b_i << 8) | k_i;
                    }
                    let eoff_ok = ints(ctx, &offsets_vec)?;
                    let epairs_ok = ints(ctx, &pairs_vec)?;
                    let (mut peoff_ok, mut pep_ok) = (eoff_ok.ptr, epairs_ok.ptr);

                    let sh_gu = noise(ctx, rng, 1281 * 80 * 16)?;
                    let (sh_sc, sh_bi) = (
                        noise(ctx, rng, 1281 * 40 * 4)?,
                        noise(ctx, rng, 1281 * 40 * 4)?,
                    );
                    let act_b = zeros(ctx, b_cnt * 11 * Q38_MOE_FF * 4)?;
                    {
                        let gw = noise(ctx, rng, (Q38_EXP * 640 - s) * 80 * 16)?;
                        let uw = noise(ctx, rng, Q38_EXP * 640 * 80 * 16)?;
                        let glo = noise(ctx, rng, Q38_EXP * 640 * 40 * 4)?;
                        let ghi = noise(ctx, rng, Q38_EXP * 640 * 20 * 4)?;
                        let ulo = noise(ctx, rng, Q38_EXP * 640 * 40 * 4)?;
                        let uhi = noise(ctx, rng, Q38_EXP * 640 * 20 * 4)?;
                        let (mut pgw, mut pglo, mut pghi, mut puw, mut pulo, mut puhi) =
                            (gw.ptr, glo.ptr, ghi.ptr, uw.ptr, ulo.ptr, uhi.ptr);
                        let (mut psw, mut pss, mut psb, mut pactb) =
                            (sh_gu.ptr, sh_sc.ptr, sh_bi.ptr, act_b.ptr);
                        let mut up_bases = 0x76727672u32;
                        run(
                            ctx,
                            m,
                            "q38_moe_up_grouped_z24",
                            (80, 513),
                            512,
                            &mut kargs![
                                pgw, pglo, pghi, puw, pulo, puhi, up_bases, psw, pss, psb, pxb,
                                peoff_ok, pep_ok, bc, pswts, pactb
                            ],
                        )?;
                    }
                    {
                        let dw = noise(ctx, rng, (Q38_EXP * d - s) * 20 * 16)?;
                        let dlo = noise(ctx, rng, Q38_EXP * d * 10 * 4)?;
                        let dhi = noise(ctx, rng, Q38_EXP * d * 5 * 4)?;
                        let sh_d = noise(ctx, rng, d * 20 * 16)?;
                        let (sh_dsc, sh_dbi) =
                            (noise(ctx, rng, d * 10 * 4)?, noise(ctx, rng, d * 10 * 4)?);
                        let dpart = zeros(ctx, b_cnt * 11 * d * 4)?;
                        let mout_b = zeros(ctx, (b_cnt * d - s) * 4)?;
                        let (mut pdw, mut pdlo, mut pdhi, mut phd, mut phds, mut phdb) =
                            (dw.ptr, dlo.ptr, dhi.ptr, sh_d.ptr, sh_dsc.ptr, sh_dbi.ptr);
                        let (mut pactb, mut pdpart, mut pmout) = (act_b.ptr, dpart.ptr, mout_b.ptr);
                        let mut dn_bases = 0x7672u32;
                        run(
                            ctx,
                            m,
                            "q38_moe_down_grouped_z24",
                            (40, 513),
                            256,
                            &mut kargs![
                                pdw, pdlo, pdhi, dn_bases, phd, phds, phdb, pactb, peoff_ok,
                                pep_ok, pswts, bc, pdpart
                            ],
                        )?;
                        run(
                            ctx,
                            m,
                            "q38_moe_combine11_batch",
                            (10, b_cnt as u32),
                            256,
                            &mut kargs![pdpart, pmout],
                        )?;
                    }

                    let log_b = floats(ctx, rng, b_cnt * Q38_VOCAB - s * 4)?;
                    let out_t = zeros(ctx, b_cnt * 4)?;
                    let pos_b = ints(ctx, &[12])?;
                    let (mut plogb, mut poutt, mut pposb) = (log_b.ptr, out_t.ptr, pos_b.ptr);
                    run(
                        ctx,
                        m,
                        "q38_argmax_fast_batch",
                        (b_cnt as u32, 1),
                        1024,
                        &mut kargs![plogb, poutt, pposb, bc],
                    )?;

                    // Fast Multi-Row Speculative Verify & `O(1)` Per-Row Checkpoint Kernels (`B <= 8`)
                    {
                        let rows = 19usize;
                        let w = noise(ctx, rng, (rows - s) * 80 * 16)?;
                        let sc = noise(ctx, rng, rows * 40 * 4)?;
                        let bi = noise(ctx, rng, rows * 40 * 4)?;
                        let xb2 = floats(ctx, rng, 2 * d)?;
                        let yb2 = zeros(ctx, 2 * rows * 4)?;
                        let xb4 = floats(ctx, rng, 4 * d)?;
                        let yb4 = zeros(ctx, 4 * rows * 4)?;
                        let (mut pw, mut psc, mut pbi) = (w.ptr, sc.ptr, bi.ptr);
                        let (mut px2, mut py2, mut px4, mut py4) =
                            (xb2.ptr, yb2.ptr, xb4.ptr, yb4.ptr);
                        let (mut r_i, mut b2_i, mut b4_i, mut md) = (rows as i32, 2i32, 4i32, 0i32);
                        run(
                            ctx,
                            m,
                            "q38_q4_gemv_2560_b2",
                            (1, 1),
                            512,
                            &mut kargs![pw, psc, pbi, px2, py2, r_i, b2_i, md],
                        )?;
                        run(
                            ctx,
                            m,
                            "q38_q4_gemv_2560_b4",
                            (1, 1),
                            512,
                            &mut kargs![pw, psc, pbi, px4, py4, r_i, b4_i, md],
                        )?;
                        let cols_6k = 6144usize;
                        let w6 = noise(ctx, rng, (rows - s) * 192 * 16)?;
                        let sc6 = noise(ctx, rng, rows * 96 * 4)?;
                        let bi6 = noise(ctx, rng, rows * 96 * 4)?;
                        let x6_b2 = floats(ctx, rng, 2 * cols_6k)?;
                        let y6_b2 = zeros(ctx, 2 * rows * 4)?;
                        let (mut pw6, mut psc6, mut pbi6, mut px6, mut py6) =
                            (w6.ptr, sc6.ptr, bi6.ptr, x6_b2.ptr, y6_b2.ptr);
                        let mut c6_i = cols_6k as i32;
                        run(
                            ctx,
                            m,
                            "q38_q4_gemv_b2",
                            (3, 1),
                            256,
                            &mut kargs![pw6, psc6, pbi6, px6, py6, r_i, c6_i, md],
                        )?;

                        let dw = noise(ctx, rng, 320 * 320 * 16)?;
                        let dsc = noise(ctx, rng, 320 * 160 * 4)?;
                        let dbi = noise(ctx, rng, 320 * 160 * 4)?;
                        let iw = noise(ctx, rng, (4 - s) * 320 * 16)?;
                        let isc = noise(ctx, rng, 4 * 160 * 4)?;
                        let ibi = noise(ctx, rng, 4 * 160 * 4)?;
                        let n_b2 = floats(ctx, rng, 2 * hcd)?;
                        let low_b2 = zeros(ctx, 2 * 320 * 4)?;
                        let inj_b2 = zeros(ctx, 2 * 4 * 4)?;
                        let (mut pdw, mut pds, mut pdb) = (dw.ptr, dsc.ptr, dbi.ptr);
                        let (mut piw, mut pis, mut pib) = (iw.ptr, isc.ptr, ibi.ptr);
                        let (mut pnb2, mut plb2, mut pib2) = (n_b2.ptr, low_b2.ptr, inj_b2.ptr);
                        let mut hi1 = 1i32;
                        run(
                            ctx,
                            m,
                            "q38_hc_down_inj_10k_b2",
                            (81, 1),
                            256,
                            &mut kargs![pdw, pds, pdb, piw, pis, pib, pnb2, plb2, pib2, hi1],
                        )?;

                        let uw = noise(ctx, rng, (hcd - s) * 10 * 16)?;
                        let usc = noise(ctx, rng, hcd * 5 * 4)?;
                        let ubi = noise(ctx, rng, hcd * 5 * 4)?;
                        let mix_b2 = zeros(ctx, 2 * d * 4)?;
                        let (mut puw, mut pus, mut pub_, mut pmxb2) =
                            (uw.ptr, usc.ptr, ubi.ptr, mix_b2.ptr);
                        run(
                            ctx,
                            m,
                            "q38_hc_up_mix4_b2",
                            (320, 1),
                            256,
                            &mut kargs![puw, pus, pub_, plb2, pnb2, pmxb2],
                        )?;
                    }
                    {
                        let gw = noise(ctx, rng, (Q38_EXP * 640 - s) * 80 * 16)?;
                        let uw = noise(ctx, rng, Q38_EXP * 640 * 80 * 16)?;
                        let glo = noise(ctx, rng, Q38_EXP * 640 * 40 * 4)?;
                        let ghi = noise(ctx, rng, Q38_EXP * 640 * 20 * 4)?;
                        let ulo = noise(ctx, rng, Q38_EXP * 640 * 40 * 4)?;
                        let uhi = noise(ctx, rng, Q38_EXP * 640 * 20 * 4)?;
                        let (mut pgw, mut pglo, mut pghi, mut puw, mut pulo, mut puhi) =
                            (gw.ptr, glo.ptr, ghi.ptr, uw.ptr, ulo.ptr, uhi.ptr);
                        let (mut psw, mut pss, mut psb, mut pactb) =
                            (sh_gu.ptr, sh_sc.ptr, sh_bi.ptr, act_b.ptr);
                        let mut up_bases = 0x76727672u32;
                        run(
                            ctx,
                            m,
                            "q38_moe_up_swiglu11_z24_bn",
                            (160, (11 * b_cnt) as u32),
                            256,
                            &mut kargs![
                                pgw, pglo, pghi, puw, pulo, puhi, up_bases, psw, pss, psb, pxb,
                                ps511, pswts, pactb
                            ],
                        )?;
                    }
                    {
                        let dw = noise(ctx, rng, (Q38_EXP * d - s) * 20 * 16)?;
                        let dlo = noise(ctx, rng, Q38_EXP * d * 10 * 4)?;
                        let dhi = noise(ctx, rng, Q38_EXP * d * 5 * 4)?;
                        let sh_d = noise(ctx, rng, d * 20 * 16)?;
                        let (sh_dsc, sh_dbi) =
                            (noise(ctx, rng, d * 10 * 4)?, noise(ctx, rng, d * 10 * 4)?);
                        let mout_b = zeros(ctx, (b_cnt * d - s) * 4)?;
                        let (mut pdw, mut pdlo, mut pdhi, mut phd, mut phds, mut phdb) =
                            (dw.ptr, dlo.ptr, dhi.ptr, sh_d.ptr, sh_dsc.ptr, sh_dbi.ptr);
                        let (mut pactb, mut pmout) = (act_b.ptr, mout_b.ptr);
                        let mut dn_bases = 0x7672u32;
                        run(
                            ctx,
                            m,
                            "q38_moe_down_combine11_z24_bn",
                            (80, b_cnt as u32),
                            256,
                            &mut kargs![
                                pdw, pdlo, pdhi, dn_bases, phd, phds, phdb, pactb, ps511, pswts,
                                pmout
                            ],
                        )?;
                    }
                    {
                        let hc_ple = floats(ctx, rng, 3 * hcd)?;
                        let kv_ple = floats(ctx, rng, 3 * (hcd + d))?;
                        let (nk, nq, nc) = (
                            floats(ctx, rng, hcd)?,
                            floats(ctx, rng, hcd)?,
                            floats(ctx, rng, hcd)?,
                        );
                        let cw4 = floats(ctx, rng, hcd * 4)?;
                        let h9 = floats(ctx, rng, 9 * hcd)?;
                        let ckpt_ple = zeros(ctx, (2 * 9 * hcd - s) * 4)?;
                        let pos_p = ints(ctx, &[5])?;
                        let mut b3 = 3i32;
                        let (mut phc, mut pkv, mut pnk, mut pnq, mut pnc, mut pcw, mut ph9) = (
                            hc_ple.ptr, kv_ple.ptr, nk.ptr, nq.ptr, nc.ptr, cw4.ptr, h9.ptr,
                        );
                        let (mut pck, mut ppos) = (ckpt_ple.ptr, pos_p.ptr);
                        run(
                            ctx,
                            m,
                            "q38_ple_inject_ckpt_b8",
                            (4, 1),
                            256,
                            &mut kargs![phc, pkv, pnk, pnq, pnc, pcw, ph9, pck, ppos, b3],
                        )?;
                    }
                    {
                        let gin = floats(ctx, rng, 3 * 16_480)?;
                        let cw4 = floats(ctx, rng, hcd * 4)?;
                        let ch3 = floats(ctx, rng, hcd * 3)?;
                        let c_ckpt = zeros(ctx, (2 * hcd * 3 - s) * 4)?;
                        let qkvc = zeros(ctx, 3 * hcd * 4)?;
                        let (al, dtb, nw) = (
                            floats(ctx, rng, 48)?,
                            floats(ctx, rng, 48)?,
                            floats(ctx, rng, 128)?,
                        );
                        let st = floats(ctx, rng, 48 * 128 * 128)?;
                        let s_ckpt = zeros(ctx, (2 * 48 * 128 * 128 - s) * 4)?;
                        let gout = zeros(ctx, 3 * 6144 * 4)?;
                        let (mut b3, mut sig) = (3i32, 0i32);
                        let (mut pgin, mut pcw, mut pch3, mut pcck, mut pqkvc) =
                            (gin.ptr, cw4.ptr, ch3.ptr, c_ckpt.ptr, qkvc.ptr);
                        let (mut pal, mut pdt, mut pnw, mut pst, mut psck, mut pgout) =
                            (al.ptr, dtb.ptr, nw.ptr, st.ptr, s_ckpt.ptr, gout.ptr);
                        run(
                            ctx,
                            m,
                            "q38_gdn_conv_silu_ckpt_b8",
                            (40, 1),
                            256,
                            &mut kargs![pgin, pcw, pch3, pcck, pqkvc, b3],
                        )?;
                        run(
                            ctx,
                            m,
                            "q38_gdn_recurrent_step_ckpt_b8",
                            (48, 1),
                            128,
                            &mut kargs![pqkvc, pgin, pal, pdt, pnw, sig, b3, pst, psck, pgout],
                        )?;
                    }
                }
            }
            Ok(())
        }

        #[test]
        #[ignore = "requires a CUDA device; run under compute-sanitizer --tool memcheck --padding 65536"]
        fn exact_size_qwen38_kernels() -> Result<(), String> {
            cases(false)
        }

        #[test]
        #[ignore = "planted fault: memcheck --padding 65536 must report an out-of-bounds access"]
        fn exact_size_qwen38_red_control() -> Result<(), String> {
            let _ = cases(true);
            Ok(())
        }
    }

    #[test]
    #[ignore = "requires Qwen3.8 checkpoint (`LEANSONIC_Q38_MODEL`) and CUDA device"]
    fn verify_batch_cost_and_ckpt_0ulp() -> Result<(), String> {
        use super::*;
        let dir = std::env::var("LEANSONIC_Q38_MODEL")
            .map_err(|_| "set LEANSONIC_Q38_MODEL".to_string())?;
        let prompt: Vec<u32> = (0..32).map(|i| ((i * 7 + 100) % 50_000) as u32).collect();
        let mut eng = Qwen38GpuEngine::load_from_dir_with_opts(&dir, 2048, true)?;

        // 1. Greedy ground truth (24 tokens)
        eng.reset()?;
        let (first, _) = eng.prefill_tokens(&prompt)?;
        let mut ref_ids = vec![first];
        let mut cur = first;
        for _ in 1..24 {
            cur = eng.forward_greedy_token(cur)?;
            ref_ids.push(cur);
        }

        // 2. Simulated speculative verify (`B = 4` with alternating deliberate mismatches at `acc = 0, 1, 2, 3`)
        eng.reset()?;
        let (first2, _) = eng.prefill_tokens(&prompt)?;
        assert_eq!(first2, first);
        let mut sim_ids = vec![first2];
        let mut step = 0usize;
        let mut cur2 = first2;
        while sim_ids.len() < 24 {
            let target_acc = step % 4; // 0, 1, 2, 3 accepted drafts!
            let mut cand = vec![cur2];
            for k in 0..3usize {
                let true_tok = if sim_ids.len() + k < ref_ids.len() {
                    ref_ids[sim_ids.len() + k]
                } else {
                    1
                };
                cand.push(if k < target_acc {
                    true_tok
                } else {
                    true_tok ^ 12345
                });
            }
            let v_out = eng.forward_verify_tokens(&cand)?;
            let mut acc = 0usize;
            while acc < 3 && v_out[acc] == cand[acc + 1] {
                acc += 1;
            }
            for &d in &cand[1..=acc] {
                if sim_ids.len() < 24 {
                    sim_ids.push(d);
                }
            }
            let bonus = v_out[acc];
            if sim_ids.len() < 24 {
                sim_ids.push(bonus);
            }
            eng.commit_or_rollback_verify_row(acc)?;
            cur2 = bonus;
            step += 1;
        }
        assert_eq!(
            sim_ids, ref_ids,
            "O(1) verify row checkpoint diverged from solo greedy!"
        );
        eprintln!("[PASS] 0-ULP O(1) commit_or_rollback_verify_row matched 24/24 greedy tokens across acc=0,1,2,3!");

        // 3. Measure steady-state `forward_verify_tokens` latency for `B = 1, 2, 3, 4, 8`
        for &b in &[1usize, 2, 3, 4, 8] {
            eng.reset()?;
            let _ = eng.prefill_tokens(&prompt)?;
            let dummy: Vec<u32> = (0..b).map(|i| (1000 + i * 13) as u32).collect();
            for _ in 0..3 {
                let _ = eng.forward_verify_tokens(&dummy)?;
            }
            let iters = 16usize;
            let t0 = Instant::now();
            for _ in 0..iters {
                let _ = eng.forward_verify_tokens(&dummy)?;
            }
            let avg_ms = t0.elapsed().as_secs_f64() * 1000.0 / (iters as f64);
            eprintln!("forward_verify_tokens B={b}: {avg_ms:.2} ms");
        }
        Ok(())
    }
}

#[cfg(test)]
mod direct_engram_tests {
    use super::*;
    use std::io::{Seek, SeekFrom, Write};

    #[test]
    fn mapped_raw_and_dictionary_parameters_use_the_same_row_api() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("leansonic-engram-{}-{stamp}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let rows = Q38_PLE_SHARD_ROWS as usize;
        let bytes = rows * 10;
        // A constant dictionary has no index bits. Encode the represented length in its header.
        let mut frame = crate::qwen38_pack::encode(&0x8081u16.to_le_bytes()).unwrap();
        frame[8..16].copy_from_slice(&((rows * 5) as u64).to_le_bytes());
        let n = frame.len();
        let end = bytes + n;
        let header = format!(
            r#"{{"raw":{{"dtype":"BF16","shape":[{rows},5],"data_offsets":[0,{bytes}]}},"encoded.q38dict1":{{"dtype":"U8","shape":[{n}],"data_offsets":[{bytes},{end}]}}}}"#
        );
        let mut f = std::fs::File::create(dir.join("model.safetensors")).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        f.seek(SeekFrom::Current(bytes as i64 - 10)).unwrap();
        for word in [1, 2, 3, 4, 5u16] {
            f.write_all(&word.to_le_bytes()).unwrap();
        }
        f.write_all(&frame).unwrap();
        drop(f);
        let st = ShardedSafeTensors::open_dir(&dir).unwrap();
        let raw = PleParameter::open(&st, "raw", false).unwrap();
        let packed = PleParameter::open(&st, "encoded", false).unwrap();
        assert_eq!(raw.row(&st, rows - 1).unwrap(), [1, 2, 3, 4, 5]);
        assert_eq!(packed.row(&st, rows - 1).unwrap(), [0x8081; 5]);
        assert!(raw.row(&st, rows).is_err());
        assert!(packed.row(&st, rows).is_err());
        assert!(PleParameter::open(&st, "missing", false).is_err());
        drop(st);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
