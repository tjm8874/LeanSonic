//! Profile-guided placement of lossless Z24 expert records.
use super::*;
#[path = "qwen38_adaptive.rs"]
mod adaptive;
use adaptive::AdaptiveLayer;

const EXPERT_BYTES: usize = SW_Z24_PROJ_BYTES / Q38_EXP;
const TABLE_BYTES: usize = Q38_EXP * size_of::<u64>();

pub(super) struct MixedProfile {
    layers: Vec<Vec<bool>>,
    hot_count: usize,
    reserves: Vec<Vec<usize>>,
    pub(super) slots: usize,
}

impl MixedProfile {
    pub(super) fn from_env() -> Result<Option<Self>, String> {
        let slots = std::env::var("Q38_ADAPT_SLOTS")
            .ok()
            .map(|s| {
                s.parse::<usize>()
                    .map_err(|_| "invalid adaptive expert count")
            })
            .transpose()?
            .unwrap_or(0);
        let Some(path) = std::env::var_os("Q38_EXPERT_PROFILE") else {
            if slots > 0 {
                return Err("adaptive slots require Q38_EXPERT_PROFILE".into());
            }
            return Ok(None);
        };
        let content =
            std::fs::read_to_string(path).map_err(|e| format!("read expert profile: {e}"))?;
        let mut layers = Vec::new();
        let mut hot_count = 0;
        let mut reserves = Vec::new();
        for line in content.lines() {
            let mut priority = Vec::new();
            let mut selected = vec![false; Q38_EXP];
            for entry in line.split_whitespace() {
                let id: usize = entry.parse().map_err(|_| "invalid expert index")?;
                if id >= Q38_EXP || selected[id] {
                    return Err(format!(
                        "invalid or duplicate expert {id} in layer {}",
                        layers.len()
                    ));
                }
                priority.push(id);
                selected[id] = true;
                hot_count += 1;
            }
            if slots > priority.len() {
                return Err("adaptive capacity exceeds layer quota".into());
            }
            reserves.push(priority.into_iter().rev().take(slots).collect());
            layers.push(selected);
        }
        if layers.len() != Q38_LAYERS {
            return Err(format!(
                "expected {Q38_LAYERS} profile rows, got {}",
                layers.len()
            ));
        }
        Ok(Some(Self {
            layers,
            hot_count,
            reserves,
            slots,
        }))
    }

    pub(super) fn hot_count(&self) -> usize {
        self.hot_count
    }

    pub(super) fn check_capacity(
        &self,
        ctx: &CudaContextHandle,
        gpu_reserve: usize,
        host_reserve: usize,
    ) -> Result<(), String> {
        let resident = 3 * (self.hot_count * EXPERT_BYTES + Q38_LAYERS * TABLE_BYTES);
        let pinned = 3 * (Q38_LAYERS * (Q38_EXP + self.slots) - self.hot_count) * EXPERT_BYTES;
        if ctx.mem_info()?.0 <= resident + gpu_reserve
            || available_host_ram_bytes() <= pinned + host_reserve
        {
            return Err(
                "selected expert profile exceeds available VRAM or pinned memory budget".into(),
            );
        }
        Ok(())
    }
}

pub(super) struct MixedExperts {
    _hot: DeviceBuf,
    _cold: PinnedHostBuf,
    pointers: DeviceBuf,
    adaptive: Option<Box<AdaptiveLayer>>,
    bases: Z24LayerBases,
}

impl MixedExperts {
    pub(super) fn load(
        ctx: &Arc<CudaContextHandle>,
        profile: &MixedProfile,
        layer: usize,
        projections: [[&[u8]; 3]; 3],
        low: &mut [u32],
        high: &mut [u32],
    ) -> Result<Self, String> {
        let selected = &profile.layers[layer];
        let hot_count = selected.iter().filter(|&&hot| hot).count();
        let cold_count = Q38_EXP - hot_count;
        let hot = DeviceBuf::alloc_uninit(ctx, 3 * hot_count * EXPERT_BYTES)?;
        let cold = PinnedHostBuf::alloc_uninit(ctx, 3 * cold_count * EXPERT_BYTES)?;
        let mut pointers = Vec::with_capacity(3 * TABLE_BYTES);
        let mut bases = [0u32; 3];
        let mut staging = vec![0u8; hot_count * EXPERT_BYTES];
        // Allocation and writes are restricted to the lifetime and size of this pinned buffer.
        let host = unsafe {
            std::slice::from_raw_parts_mut(
                cold.host_ptr.cast::<std::mem::MaybeUninit<u8>>(),
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
            let mut hot_idx = 0;
            let mut cold_idx = 0;
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
                        host[offset..offset + size].write_copy_of_slice(src);
                    }
                    offset += size;
                }
            }
            if !staging.is_empty() {
                hot.copy_from_host_at(proj * staging.len(), &staging)?;
            }
        }
        let adaptive = if profile.slots > 0 {
            Some(Box::new(AdaptiveLayer::new(
                ctx,
                &profile.reserves[layer],
                selected,
                &cold,
                &pointers,
            )?))
        } else {
            None
        };
        Ok(Self {
            adaptive,
            _hot: hot,
            _cold: cold,
            pointers: DeviceBuf::from_host_bytes(ctx, &pointers)?,
            bases: Z24LayerBases {
                up_packed_m1: bases[0] | (bases[1] << 16),
                down_packed_m1: bases[2],
            },
        })
    }
    pub(super) fn adapt(&mut self, ctx: &CudaContextHandle, ids: &[i32]) -> Result<usize, String> {
        self.adaptive
            .as_mut()
            .ok_or("adaptive expert storage missing")?
            .apply(ctx, &self.pointers, ids)
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
