//! Lossless MXFP8 expert codes conditioned on their existing E8M0 block scale.
use super::*;

const BLOCK: usize = 32;
const MAX_BANKS: usize = 8;

fn bank_map(cuts: &[u8]) -> [u8; 256] {
    std::array::from_fn(|value| cuts.partition_point(|&cut| usize::from(cut) <= value) as u8)
}

fn boundaries(scales: &[u8]) -> Vec<u8> {
    let mut hist = [0usize; 256];
    for &scale in scales {
        hist[usize::from(scale)] += 1;
    }
    let used: Vec<_> = (0..=u8::MAX)
        .filter(|s| hist[usize::from(*s)] != 0)
        .collect();
    if used.len() <= MAX_BANKS {
        return used.into_iter().skip(1).collect();
    }
    let mut cuts = Vec::new();
    let mut previous = None;
    let mut seen = 0u64;
    for scale in used {
        let count = hist[usize::from(scale)] as u64;
        let bucket = (seen + count / 2) * MAX_BANKS as u64 / scales.len() as u64;
        if previous.is_some_and(|p| p != bucket) {
            cuts.push(scale);
        }
        previous = Some(bucket);
        seen += count;
    }
    cuts
}

pub fn pack_mxscale_codes(raw: &[u8], scales: &[u8], mut output: impl Write) -> io::Result<usize> {
    if raw.is_empty()
        || raw.len() > u32::MAX as usize
        || scales.len().checked_mul(BLOCK) != Some(raw.len())
    {
        return Err(io::Error::other("invalid MXFP8 code/scale dimensions"));
    }
    let cuts = boundaries(scales);
    let mut header = [0u8; MAX_BANKS];
    header[0] = cuts.len() as u8;
    header[1..1 + cuts.len()].copy_from_slice(&cuts);
    output.write_all(&header)?;
    let banks = bank_map(&cuts);
    scale_context::encode_with_context(raw, output, header.len(), cuts.len() + 1, |_, at, first| {
        usize::from(banks[usize::from(scales[(first + at) / BLOCK])])
    })
}

struct BlockScales(Vec<u8>);

impl scale_context::ScaleDecoder for BlockScales {
    fn decode(
        &self,
        lookup: &[[u32; 4096]],
        bytes: &[u8],
        mut states: [u32; 4],
        first: usize,
        dst: &mut [MaybeUninit<u8>],
    ) -> Result<(), String> {
        let mut cursor = 0;
        for (block, out) in dst.as_chunks_mut::<BLOCK>().0.iter_mut().enumerate() {
            let table = &lookup[usize::from(self.0[first / BLOCK + block])];
            for group in out.as_chunks_mut::<4>().0 {
                for (slot, x) in group.iter_mut().zip(&mut states) {
                    let entry = table[(*x & (TOTAL - 1)) as usize];
                    slot.write((entry >> 24) as u8);
                    *x = ((entry & 4095) + 1) * (*x >> PRECISION) + ((entry >> 12) & 4095);
                    while *x < LOWER {
                        let byte = *bytes.get(cursor).ok_or("short MXFP8 block-scale payload")?;
                        *x = (*x << 8) | u32::from(byte);
                        cursor += 1;
                    }
                }
            }
        }
        if cursor != bytes.len() || states != [LOWER; 4] {
            return Err("invalid MXFP8 block-scale terminal state".into());
        }
        Ok(())
    }
}

impl<'a> Matrix<'a> {
    /// Decode an independent byte-rANS or scale-conditioned MXFP8 matrix
    /// straight into the device's two-indices-per-byte representation.
    pub(crate) fn write_mxfp8_indices(
        mut self,
        index: &[u8; 256],
        packed: &mut [u8],
    ) -> Result<(), String> {
        if packed.len().checked_mul(2) != Some(self.count) {
            return Err("invalid MXFP8 device index extent".into());
        }
        // For these two formats the emitted byte is independent of the next
        // table selection. Return the device index directly from the rANS LUT.
        for table in &mut self.lookup {
            for entry in table {
                let symbol = (*entry >> 24) as usize;
                *entry = (*entry & 0x00ff_ffff) | (u32::from(index[symbol]) << 24);
            }
        }
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(24));
        let group = self.frames.len().div_ceil(threads);
        let matrix = &self;
        std::thread::scope(|scope| {
            let mut workers = Vec::new();
            for (part, (frames, output)) in self
                .frames
                .chunks(group)
                .zip(packed.chunks_mut(group * (TILE / 2)))
                .enumerate()
            {
                workers.push(scope.spawn(move || {
                    let mut scratch = vec![MaybeUninit::uninit(); TILE];
                    for (i, (frame, target)) in
                        frames.iter().zip(output.chunks_mut(TILE / 2)).enumerate()
                    {
                        let slots = &mut scratch[..target.len() * 2];
                        matrix.decode_tile((part * group + i) * TILE, frame, slots)?;
                        for (dst, pair) in target.iter_mut().zip(slots.as_chunks::<2>().0) {
                            // Successful decode_tile initialized both indices.
                            *dst = unsafe { pair[0].assume_init() | (pair[1].assume_init() << 4) };
                        }
                    }
                    Ok::<_, String>(())
                }));
            }
            for worker in workers {
                worker
                    .join()
                    .map_err(|_| "MXFP8 index decoder thread panicked")??;
            }
            Ok(())
        })
    }

    pub(crate) fn parse_mxscale(
        data: &'a [u8],
        count: usize,
        mut scales: Vec<u8>,
    ) -> Result<(Self, &'a [u8]), String> {
        if scales.len().checked_mul(BLOCK) != Some(count) {
            return Err("invalid MXFP8 code/scale dimensions".into());
        }
        let (header, frames) = data
            .split_first_chunk::<MAX_BANKS>()
            .ok_or("missing MXFP8 scale boundaries")?;
        let n = usize::from(header[0]);
        if n >= MAX_BANKS
            || header[1..=n].windows(2).any(|p| p[0] >= p[1])
            || header[n + 1..].iter().any(|&b| b != 0)
        {
            return Err("invalid MXFP8 scale boundaries".into());
        }
        let banks = bank_map(&header[1..=n]);
        for scale in &mut scales {
            *scale = banks[usize::from(*scale)];
        }
        let mut matrix = Self::parse_tables(frames, count, n + 1)?;
        matrix.scale_context = Some(Box::new(BlockScales(scales)));
        Ok((matrix, &frames[..(n + 1) * 512]))
    }
}

pub fn unpack_mxscale_codes(frame: &[u8], scales: Vec<u8>) -> Result<Vec<u8>, String> {
    let count = scales
        .len()
        .checked_mul(BLOCK)
        .ok_or("invalid scale count")?;
    unpack_codes(Matrix::parse_mxscale(frame, count, scales)?.0, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_across_tiles_and_scale_distributions() {
        let count = TILE * 2 + BLOCK;
        let codes: Vec<_> = (0..count)
            .map(|i| ((i * 17 + i / 19) % 251) as u8)
            .collect();
        for stride in [0, 1, 37] {
            let scales: Vec<_> = (0..count / BLOCK)
                .map(|i| ((i / 7) * stride) as u8)
                .collect();
            let mut frame = Vec::new();
            assert_eq!(
                pack_mxscale_codes(&codes, &scales, &mut frame).unwrap(),
                frame.len()
            );
            assert_eq!(unpack_mxscale_codes(&frame, scales.clone()).unwrap(), codes);
            assert!(unpack_mxscale_codes(&frame[..frame.len() - 8], scales.clone()).is_err());
            frame[0] = 8;
            assert!(unpack_mxscale_codes(&frame, scales).is_err());
        }
    }
}
