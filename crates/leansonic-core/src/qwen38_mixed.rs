//! Profile-guided per-expert Z24 placement across GDDR7 VRAM and pinned host memory (`#248`, `#262`).
use super::*;

const EMBEDDED_CALIBRATED_19L: &str = include_str!("../profiles/q38_calibrated_19layers.txt");
const EXPERT_BYTES: usize = SW_Z24_PROJ_BYTES / Q38_EXP; // 972,800 B per projection (`2.78 MiB` across 3 projs)
const TABLE_BYTES: usize = Q38_EXP * core::mem::size_of::<u64>(); // 4,096 B per projection

pub(super) enum LayerExpertPlan {
    FullVram,
    Mixed(Vec<bool>),
}

pub(super) struct MixedPlacementPlan {
    pub layers: Vec<LayerExpertPlan>,
}

fn parse_profile_lists(text: &str) -> Result<Vec<Vec<usize>>, String> {
    let mut out = Vec::with_capacity(Q38_LAYERS);
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() && out.len() == Q38_LAYERS {
            break;
        }
        let mut row = Vec::with_capacity(256);
        let mut seen = [false; Q38_EXP];
        for tok in trimmed.split_whitespace() {
            let id: usize = tok
                .parse()
                .map_err(|_| format!("invalid expert index `{tok}`"))?;
            if id >= Q38_EXP || seen[id] {
                return Err(format!(
                    "invalid or duplicate expert {id} in layer {}",
                    out.len()
                ));
            }
            seen[id] = true;
            row.push(id);
        }
        out.push(row);
    }
    if out.len() != Q38_LAYERS {
        return Err(format!(
            "expected {Q38_LAYERS} profile rows, got {}",
            out.len()
        ));
    }
    Ok(out)
}

fn build_mask_with_quota(cal_ids: &[usize], quota: usize) -> Vec<bool> {
    let mut selected = vec![false; Q38_EXP];
    let mut chosen = 0usize;
    for &id in cal_ids {
        if chosen >= quota {
            break;
        }
        if !selected[id] {
            selected[id] = true;
            chosen += 1;
        }
    }
    if chosen < quota {
        for slot in selected.iter_mut() {
            if chosen >= quota {
                break;
            }
            if !*slot {
                *slot = true;
                chosen += 1;
            }
        }
    }
    selected
}

impl MixedPlacementPlan {
    pub(super) fn plan_for_device(
        ctx: &CudaContextHandle,
        vram_reserve_bytes: usize,
        host_ram_reserve_bytes: usize,
        max_vram_layers: usize,
        force_compact: bool,
        disable_z24: bool,
    ) -> Result<Option<Self>, String> {
        if force_compact || disable_z24 {
            return Ok(None);
        }
        if std::env::var("LEANSONIC_NO_MIXED")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            return Ok(None);
        }

        if let Ok(profile_env) = std::env::var("Q38_EXPERT_PROFILE") {
            if profile_env.eq_ignore_ascii_case("none") || profile_env == "0" {
                return Ok(None);
            }
            if !profile_env.eq_ignore_ascii_case("auto") {
                let text = std::fs::read_to_string(&profile_env)
                    .map_err(|e| format!("read Q38_EXPERT_PROFILE `{profile_env}`: {e}"))?;
                let lists = parse_profile_lists(&text)?;
                let mut total_hot = 0usize;
                let mut layers = Vec::with_capacity(Q38_LAYERS);
                for row in lists {
                    total_hot += row.len();
                    let mut mask = vec![false; Q38_EXP];
                    for id in row {
                        mask[id] = true;
                    }
                    layers.push(LayerExpertPlan::Mixed(mask));
                }
                let needed_vram = 3 * (total_hot * EXPERT_BYTES + Q38_LAYERS * TABLE_BYTES);
                let needed_host = 3 * (Q38_LAYERS * Q38_EXP - total_hot) * EXPERT_BYTES;
                let (free_vram, _) = ctx.mem_info()?;
                if free_vram <= needed_vram.saturating_add(vram_reserve_bytes)
                    || available_host_ram_bytes()
                        <= needed_host.saturating_add(host_ram_reserve_bytes)
                {
                    return Err(
                        "Q38_EXPERT_PROFILE exceeds available VRAM or pinned host RAM budget"
                            .into(),
                    );
                }
                return Ok(Some(Self { layers }));
            }
        }

        let (free_vram, _) = ctx.mem_info()?;
        let table_overhead = 3 * Q38_LAYERS * TABLE_BYTES; // ~576 KiB
                                                           // Leave an extra `64 MiB` allocator alignment slack across per-layer allocations.
        let slack = 64 * 1024 * 1024;
        let usable_vram =
            free_vram.saturating_sub(vram_reserve_bytes.saturating_add(table_overhead + slack));
        let max_hot_total =
            (usable_vram / (3 * EXPERT_BYTES)).min(max_vram_layers.saturating_mul(Q38_EXP));

        // All 48 layers fit cleanly as contiguous `SwitchMlpTier::Vram`!
        if max_hot_total >= Q38_LAYERS * Q38_EXP {
            return Ok(None);
        }
        // Too little VRAM or host RAM for pinned spillover -> let engine fall back cleanly.
        if max_hot_total < Q38_LAYERS * 16 {
            return Ok(None);
        }
        let cold_total = Q38_LAYERS * Q38_EXP - max_hot_total;
        let needed_host = 3 * cold_total * EXPERT_BYTES;
        if available_host_ram_bytes() <= needed_host.saturating_add(host_ram_reserve_bytes) {
            return Ok(None);
        }

        let cal_lists = parse_profile_lists(EMBEDDED_CALIBRATED_19L)?;

        // Near-full VRAM (e.g. `RTX PRO 5000 72GB` with `--spec K` reserving VRAM for MTP):
        // Keep early layers `100%` contiguous `FullVram` and apply calibrated `Mixed` only to the trailing `3` layers!
        if max_hot_total >= 44 * Q38_EXP {
            let full_layers = (max_hot_total / Q38_EXP)
                .saturating_sub(1)
                .min(Q38_LAYERS - 1);
            let rem_layers = Q38_LAYERS - full_layers;
            let rem_hot = max_hot_total.saturating_sub(full_layers * Q38_EXP);
            let base_q = (rem_hot / rem_layers).min(Q38_EXP - 1);
            let extra_q = rem_hot % rem_layers;
            let mut layers = Vec::with_capacity(Q38_LAYERS);
            for (l, cal) in cal_lists.iter().enumerate() {
                if l < full_layers {
                    layers.push(LayerExpertPlan::FullVram);
                } else {
                    let idx = l - full_layers;
                    let q = (base_q + usize::from(idx < extra_q)).min(Q38_EXP - 1);
                    layers.push(LayerExpertPlan::Mixed(build_mask_with_quota(cal, q)));
                }
            }
            return Ok(Some(Self { layers }));
        }

        // Spilled GPUs (`32 GB` `RTX PRO 4500`, `16 GB` `RTX 5070 Ti` / `5060 Ti`):
        // Spread hot VRAM quota evenly across all `48` layers in calibrated router-frequency order!
        let base_q = max_hot_total / Q38_LAYERS;
        let extra_q = max_hot_total % Q38_LAYERS;
        let mut layers = Vec::with_capacity(Q38_LAYERS);
        for (l, cal) in cal_lists.iter().enumerate() {
            let q = (base_q + usize::from(l < extra_q)).min(Q38_EXP - 1);
            layers.push(LayerExpertPlan::Mixed(build_mask_with_quota(cal, q)));
        }
        Ok(Some(Self { layers }))
    }
}

pub(super) struct MixedExperts {
    _hot: DeviceBuf,
    _cold: PinnedHostBuf,
    pointers: DeviceBuf,
    bases: Z24LayerBases,
}

impl MixedExperts {
    pub(super) fn load(
        ctx: &Arc<CudaContextHandle>,
        selected: &[bool],
        layer: usize,
        projections: [[&[u8]; 3]; 3],
        low: &mut [u32],
        high: &mut [u32],
    ) -> Result<Self, String> {
        let hot_count = selected.iter().filter(|&&hot| hot).count();
        let cold_count = Q38_EXP - hot_count;
        let hot = DeviceBuf::alloc_uninit(ctx, (3 * hot_count * EXPERT_BYTES).max(16))?;
        let cold = PinnedHostBuf::alloc_uninit(ctx, (3 * cold_count * EXPERT_BYTES).max(16))?;
        let mut pointers = Vec::with_capacity(3 * TABLE_BYTES);
        let mut bases = [0u32; 3];
        let mut staging = vec![0u8; hot_count * EXPERT_BYTES];
        let host = unsafe {
            core::slice::from_raw_parts_mut(
                cold.host_ptr.cast::<core::mem::MaybeUninit<u8>>(),
                cold.byte_len,
            )
        };
        for (proj, [weights, scales, biases]) in projections.into_iter().enumerate() {
            let (sc, bi) = pack_z24_sb_slice(scales, biases, low, high).ok_or_else(|| {
                format!("layer {layer}, projection {proj} exceeds Z24 exponent span")
            })?;
            bases[proj] = u32::from(sc) | (u32::from(bi) << 8);
            let planes = [weights, u32_slice_bytes(low), u32_slice_bytes(high)];
            let sizes = [
                EX_W_BYTES,
                SW_Z24_LO_BYTES / Q38_EXP,
                SW_Z24_HI_BYTES / Q38_EXP,
            ];
            let mut hot_idx = 0usize;
            let mut cold_idx = 0usize;
            for (expert, &is_hot) in selected.iter().enumerate() {
                let (start, address) = if is_hot {
                    let offset = hot_idx * EXPERT_BYTES;
                    hot_idx += 1;
                    (offset, hot.ptr + (proj * staging.len() + offset) as u64)
                } else {
                    let offset = (proj * cold_count + cold_idx) * EXPERT_BYTES;
                    cold_idx += 1;
                    (offset, cold.dev_ptr + offset as u64)
                };
                pointers.extend_from_slice(&address.to_ne_bytes());
                let mut offset = start;
                for (plane, size) in planes.iter().zip(sizes) {
                    let src = &plane[expert * size..(expert + 1) * size];
                    if is_hot {
                        staging[offset..offset + size].copy_from_slice(src);
                    } else {
                        for (dst_b, &src_b) in host[offset..offset + size].iter_mut().zip(src) {
                            dst_b.write(src_b);
                        }
                    }
                    offset += size;
                }
            }
            if !staging.is_empty() {
                hot.copy_from_host_at(proj * staging.len(), &staging)?;
            }
        }
        Ok(Self {
            _hot: hot,
            _cold: cold,
            pointers: DeviceBuf::from_host_bytes(ctx, &pointers)?,
            bases: Z24LayerBases {
                up_packed_m1: bases[0] | (bases[1] << 16),
                down_packed_m1: bases[2],
            },
        })
    }
}

impl SwitchMlpTier {
    pub(super) fn z24_parts(&self) -> Option<([u64; 9], Z24LayerBases)> {
        let (ptr, bases) = match self {
            Self::Mixed(m) => {
                let base = m.pointers.ptr;
                return Some((
                    [
                        base,
                        0,
                        0,
                        base + TABLE_BYTES as u64,
                        0,
                        0,
                        base + 2 * TABLE_BYTES as u64,
                        0,
                        0,
                    ],
                    m.bases,
                ));
            }
            Self::Vram { buf, z24: Some(b) } => (buf.ptr, *b),
            Self::HostPinned { buf, z24: Some(b) } => (buf.dev_ptr, *b),
            _ => return None,
        };
        Some((
            [
                ZOFF_GATE_W,
                ZOFF_GATE_LO,
                ZOFF_GATE_HI,
                ZOFF_UP_W,
                ZOFF_UP_LO,
                ZOFF_UP_HI,
                ZOFF_DOWN_W,
                ZOFF_DOWN_LO,
                ZOFF_DOWN_HI,
            ]
            .map(|offset| ptr + offset as u64),
            bases,
        ))
    }
}
