//! Prompt-driven replacement of a bounded set of expert records.
use super::*;

pub(super) struct AdaptiveLayer {
    _backing: PinnedHostBuf,
    original: Vec<usize>,
    occupants: Vec<Option<usize>>,
    destinations: Vec<[u64; 3]>,
    fallback: Vec<u64>,
    sources: Vec<usize>,
    table_published: bool,
}

fn table_bytes(entries: &[u64]) -> &[u8] {
    // The device ABI table contains native u64 pointers.
    unsafe { std::slice::from_raw_parts(entries.as_ptr().cast(), std::mem::size_of_val(entries)) }
}

impl AdaptiveLayer {
    pub(super) fn new(
        ctx: &Arc<CudaContextHandle>,
        evictable: &[usize],
        selected: &[bool],
        cold: &PinnedHostBuf,
        encoded_table: &[u8],
    ) -> Result<Self, String> {
        let mut fallback: Vec<u64> = encoded_table
            .chunks_exact(8)
            .map(|v| u64::from_ne_bytes(v.try_into().unwrap()))
            .collect();
        let mut sources = vec![0; fallback.len()];
        for (idx, &addr) in fallback.iter().enumerate() {
            if !selected[idx % Q38_EXP] {
                let offset = usize::try_from(addr - cold.dev_ptr)
                    .map_err(|_| "invalid cold record offset")?;
                if offset + EXPERT_BYTES > cold.byte_len {
                    return Err("cold record exceeds its backing".into());
                }
                sources[idx] = cold.host_ptr as usize + offset;
            }
        }
        let backing = PinnedHostBuf::alloc_uninit(ctx, 3 * evictable.len() * EXPERT_BYTES)?;
        let mut destinations = Vec::with_capacity(evictable.len());
        for (slot, &expert) in evictable.iter().enumerate() {
            let mut dest = [0; 3];
            for (proj, d) in dest.iter_mut().enumerate() {
                let idx = proj * Q38_EXP + expert;
                let offset = (3 * slot + proj) * EXPERT_BYTES;
                *d = fallback[idx];
                let host = unsafe { backing.host_ptr.add(offset) };
                unsafe {
                    check_cu(
                        (ctx.api.cuMemcpyDtoH_v2)(host.cast(), *d, EXPERT_BYTES),
                        "backup evictable expert",
                    )?;
                }
                sources[idx] = host as usize;
                fallback[idx] = backing.dev_ptr + offset as u64;
            }
            destinations.push(dest);
        }
        Ok(Self {
            _backing: backing,
            original: evictable.to_vec(),
            occupants: evictable.iter().copied().map(Some).collect(),
            destinations,
            fallback,
            sources,
            table_published: true,
        })
    }

    pub(super) fn apply(
        &mut self,
        ctx: &CudaContextHandle,
        device_table: &DeviceBuf,
        ids: &[i32],
    ) -> Result<usize, String> {
        let mut counts = [0u16; Q38_EXP];
        for &id in ids {
            let count = counts
                .get_mut(id as usize)
                .ok_or("invalid prefill router ID")?;
            *count += 1;
        }
        let mut candidates: Vec<_> = (0..Q38_EXP)
            .filter(|&e| counts[e] >= 2 && self.sources[e] != 0 && !self.original.contains(&e))
            .collect();
        candidates.sort_unstable_by_key(|&e| (std::cmp::Reverse(counts[e]), e));
        let mut desired = self.original.clone();
        // Backups are ordered from the lowest calibration priority to the highest.
        let victims = self
            .original
            .iter()
            .enumerate()
            .filter(|&(_, &e)| counts[e] == 0)
            .map(|(i, _)| i);
        for (slot, expert) in victims.zip(candidates) {
            desired[slot] = expert;
        }
        if self.table_published
            && self
                .occupants
                .iter()
                .copied()
                .eq(desired.iter().copied().map(Some))
        {
            return Ok(0);
        }

        // While records move, all mutable slots use immutable pinned copies. If a
        // transfer fails, subsequent launches still read the correct model weights.
        device_table.copy_from_host_bytes(table_bytes(&self.fallback))?;
        self.table_published = false;
        let mut new_table = self.fallback.clone();
        let mut moved = 0;
        for (slot, &expert) in desired.iter().enumerate() {
            let copy = self.occupants[slot] != Some(expert);
            if copy {
                self.occupants[slot] = None;
            }
            for proj in 0..3 {
                let idx = proj * Q38_EXP + expert;
                let dest = self.destinations[slot][proj];
                if copy {
                    unsafe {
                        check_cu(
                            (ctx.api.cuMemcpyHtoD_v2)(
                                dest,
                                self.sources[idx] as *const c_void,
                                EXPERT_BYTES,
                            ),
                            "promote pinned expert",
                        )?;
                    }
                }
                new_table[idx] = dest;
            }
            self.occupants[slot] = Some(expert);
            moved += usize::from(copy);
        }
        device_table.copy_from_host_bytes(table_bytes(&new_table))?;
        self.table_published = true;
        Ok(moved)
    }
}
