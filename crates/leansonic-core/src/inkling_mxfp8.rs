//! Lossless 4-lane byte rANS for on-disk MXFP8 expert weights. The 256 LE
//! u16 frequencies sum to 4096. Each independent 128 KiB of raw data has a
//! LE u32 frame size, four LE u32 states and byte-normalization payload.
//! The tensor is padded with zero bytes to an eight-byte boundary.
use std::io::{self, Write};
use std::mem::MaybeUninit;

mod mean_scales;
mod scale_split;
pub use scale_split::{pack_scale_split, unpack_scale_split};
mod scale_context;
pub use mean_scales::pack_mean_scales;
pub use scale_context::pack_scale_context;

pub const TILE: usize = 128 * 1024;
const PRECISION: u32 = 12;
const TOTAL: u32 = 1 << PRECISION;
const LOWER: u32 = 1 << 23;

fn frequencies(bytes: &[u8]) -> [u16; 256] {
    let mut histogram = [0i64; 256];
    for &b in bytes {
        histogram[b as usize] += 1;
    }
    normalize(&histogram)
}

fn normalize(histogram: &[i64; 256]) -> [u16; 256] {
    let count: i64 = histogram.iter().sum();
    if count == 0 {
        // An unreachable NVFP4 context still has a complete on-disk table.
        let mut f = [0; 256];
        f[0] = TOTAL as u16;
        return f;
    }
    let mut freq = [0u16; 256];
    let mut error = [0i64; 256];
    for (i, &n) in histogram.iter().enumerate() {
        if n != 0 {
            let ideal = n * i64::from(TOTAL);
            freq[i] = (ideal / count).max(1) as u16;
            error[i] = ideal - i64::from(freq[i]) * count;
        }
    }
    let mut sum: i32 = freq.iter().map(|&f| i32::from(f)).sum();
    while sum != TOTAL as i32 {
        if sum < TOTAL as i32 {
            let i = (0..256)
                .filter(|&i| freq[i] != 0)
                .min_by_key(|&i| -error[i])
                .unwrap();
            freq[i] += 1;
            error[i] -= count;
            sum += 1;
        } else {
            let i = (0..256)
                .filter(|&i| freq[i] > 1)
                .min_by_key(|&i| error[i])
                .unwrap();
            freq[i] -= 1;
            error[i] += count;
            sum -= 1;
        }
    }
    freq
}

pub fn pack(raw: &[u8], mut output: impl Write) -> io::Result<usize> {
    if raw.is_empty() || raw.len() > u32::MAX as usize {
        return Err(io::Error::other("invalid MXFP8 matrix size"));
    }
    let freq = frequencies(raw);
    let base = write_table(&freq, &mut output)?;
    write_frames(raw, output, 512, |tile, states, payload| {
        for (i, &value) in tile.iter().enumerate().rev() {
            let idx = value as usize;
            encode_symbol(&mut states[i % 4], freq[idx], base[idx], payload);
        }
    })
}

fn write_table(freq: &[u16; 256], output: &mut impl Write) -> io::Result<[u32; 256]> {
    let mut base = [0u32; 256];
    let mut cum = 0u32;
    for (i, &f) in freq.iter().enumerate() {
        output.write_all(&f.to_le_bytes())?;
        base[i] = cum;
        cum += u32::from(f);
    }
    Ok(base)
}

#[inline]
fn encode_symbol(x: &mut u32, frequency: u16, start: u32, payload: &mut Vec<u8>) {
    let f = u32::from(frequency);
    while *x >= ((LOWER >> PRECISION) << 8) * f {
        payload.push(*x as u8);
        *x >>= 8;
    }
    *x = ((*x / f) << PRECISION) + *x % f + start;
}

fn write_frames(
    raw: &[u8],
    mut output: impl Write,
    mut written: usize,
    mut encode: impl FnMut(&[u8], &mut [u32; 4], &mut Vec<u8>),
) -> io::Result<usize> {
    let mut payload = Vec::with_capacity(TILE);
    for tile in raw.chunks(TILE) {
        payload.clear();
        let mut states = [LOWER; 4];
        encode(tile, &mut states, &mut payload);
        let len = 16 + payload.len();
        output.write_all(&(len as u32).to_le_bytes())?;
        for x in states {
            output.write_all(&x.to_le_bytes())?;
        }
        payload.reverse();
        output.write_all(&payload)?;
        written += 4 + len;
    }
    let pad = written.next_multiple_of(8) - written;
    output.write_all(&[0; 7][..pad])?;
    Ok(written + pad)
}

const NVFP4_CONTEXTS: usize = 64;

// Bits 0..2 hold the largest preceding magnitude in this eight-byte block.
#[inline]
fn nvfp4_prefix<const SPLIT: bool>(previous: u8, value: u8) -> u8 {
    let lo = value & 7;
    let hi = (value >> 4) & 7;
    let largest = (previous & 7).max(lo).max(hi);
    if SPLIT {
        // A 0.5 or 1.5 magnitude proves the block did not use the trim mapping.
        let seen = (previous & 8 != 0) || matches!(lo, 1 | 3) || matches!(hi, 1 | 3);
        largest | (u8::from(seen) << 3)
    } else {
        largest
    }
}

// The split decoder updates four independent prefix states. Keep all transitions
// in 4 KiB so the hot loop needs only one operation per decoded symbol.
const TRIM_PREFIX: [[u8; 256]; 16] = {
    let mut table = [[0; 256]; 16];
    let mut previous = 0;
    while previous < 16 {
        let mut symbol = 0;
        while symbol < 256 {
            let lo = symbol as u8 & 7;
            let hi = (symbol as u8 >> 4) & 7;
            let preceding = previous as u8 & 7;
            let pair = if lo > hi { lo } else { hi };
            let largest = if preceding > pair { preceding } else { pair };
            let seen = previous & 8 != 0 || matches!(lo, 1 | 3) || matches!(hi, 1 | 3);
            table[previous][symbol] = largest | ((seen as u8) << 3);
            symbol += 1;
        }
        previous += 1;
    }
    table
};

/// Four independent eight-byte NVFP4 blocks are interleaved. The column and
/// previous largest E2M1 magnitude select a table; both sign bits are retained.
pub fn pack_nvfp4_context(raw: &[u8], output: impl Write) -> io::Result<usize> {
    pack_nvfp4_tables::<false>(raw, &[], NVFP4_CONTEXTS, output)
}

/// Lossless NVFP4 coding for planes containing both trimmed and untrimmed blocks.
pub fn pack_nvfp4_trim_context(raw: &[u8], output: impl Write) -> io::Result<usize> {
    pack_nvfp4_tables::<true>(raw, &[], 2 * NVFP4_CONTEXTS, output)
}

// Each byte selects one of two banks of 64 contextual tables for an eight-byte
// NVFP4 block. Transform the scale buffer in place so peak storage is bounded.
fn scale_banks(scales: &mut [u8], cuts: &[u8; 128]) -> Result<(), String> {
    if scales.is_empty() || !scales.len().is_multiple_of(128) || cuts.iter().any(|&v| v > 127) {
        return Err("invalid scale-conditioned dimensions or cutoff".into());
    }
    let per_expert = scales.len() / 128;
    for (row, &cut) in scales.chunks_exact_mut(per_expert).zip(cuts) {
        for v in row {
            *v = if *v < cut { NVFP4_CONTEXTS as u8 } else { 0 };
        }
    }
    Ok(())
}

/// Encode weights in their original block order using the shipped expert scales.
pub fn pack_scale_direct(
    raw: &[u8],
    mut scales: Vec<u8>,
    cuts: &[u8; 128],
    mut output: impl Write,
) -> io::Result<usize> {
    if scales.len().checked_mul(8) != Some(raw.len()) {
        return Err(io::Error::other(
            "mismatched expert scale and weight shapes",
        ));
    }
    scale_banks(&mut scales, cuts).map_err(io::Error::other)?;
    output.write_all(cuts)?;
    Ok(128 + pack_nvfp4_tables::<false>(raw, &scales, 2 * NVFP4_CONTEXTS, output)?)
}

pub fn unpack_scale_direct(frame: &[u8], scales: Vec<u8>) -> Result<Vec<u8>, String> {
    let count = scales
        .len()
        .checked_mul(8)
        .ok_or("weight extent overflow")?;
    unpack_codes(Matrix::parse_scale_direct(frame, count, scales)?, count)
}

// A trained cutoff separates trimmed blocks. Divide each side again at its
// expert-specific median scale and store the three boundaries in the frame.
fn four_scale_banks(scales: &mut [u8], cuts: &[u8; 384]) -> Result<(), String> {
    if scales.is_empty() || !scales.len().is_multiple_of(128) {
        return Err("invalid four-bank scale dimensions".into());
    }
    let per_expert = scales.len() / 128;
    for (expert, row) in scales.chunks_exact_mut(per_expert).enumerate() {
        let cut = cuts[expert];
        let lo = cuts[128 + expert];
        let hi = cuts[256 + expert];
        if lo > cut || hi < cut || hi > 127 {
            return Err("invalid four-bank scale boundaries".into());
        }
        for v in row {
            let bank = if *v < cut {
                u8::from(*v >= lo)
            } else {
                2 + u8::from(*v >= hi)
            };
            *v = bank * NVFP4_CONTEXTS as u8;
        }
    }
    Ok(())
}

pub fn pack_four_scale_banks(
    raw: &[u8],
    scales: Vec<u8>,
    base: &[u8; 128],
    output: impl Write,
) -> io::Result<usize> {
    pack_four_scale_quantiles(raw, scales, base, [2, 2], output)
}

/// Split each side of the original scale cutoff at a chosen quartile.
/// The frame layout and decoder are shared with the median-cut encoder.
pub fn pack_four_scale_quantiles(
    raw: &[u8],
    scales: Vec<u8>,
    base: &[u8; 128],
    quartiles: [u8; 2],
    output: impl Write,
) -> io::Result<usize> {
    if quartiles.iter().any(|q| !(1..=3).contains(q)) {
        return Err(io::Error::other("invalid scale quartile"));
    }
    if scales.is_empty()
        || !scales.len().is_multiple_of(128)
        || scales.len().checked_mul(8) != Some(raw.len())
    {
        return Err(io::Error::other("invalid four-bank weight dimensions"));
    }
    let mut cuts = [0u8; 384];
    cuts[..128].copy_from_slice(base);
    for (expert, row) in scales.chunks_exact(scales.len() / 128).enumerate() {
        let cut = usize::from(base[expert]);
        let mut counts = [0usize; 128];
        for &v in row {
            *counts
                .get_mut(usize::from(v))
                .ok_or_else(|| io::Error::other("invalid scale code"))? += 1;
        }
        if cut > 127 {
            return Err(io::Error::other("invalid scale cutoff"));
        }
        for (side, (lo, hi)) in [(0, cut), (cut, 128)].into_iter().enumerate() {
            let bins = &counts[lo..hi];
            let total = bins.iter().sum::<usize>();
            let mut prefix = 0;
            let target = usize::from(quartiles[side]) * total;
            let mut best = (target, lo);
            for (i, &n) in bins.iter().enumerate() {
                if lo + i < 127 {
                    prefix += n;
                    best = best.min(((4 * prefix).abs_diff(target), lo + i + 1));
                }
            }
            cuts[(side + 1) * 128 + expert] = best.1 as u8;
        }
    }
    pack_four_scale_boundaries(raw, scales, &cuts, output)
}

/// Encode the existing four-bank format using explicit per-expert boundaries.
pub fn pack_four_scale_boundaries(
    raw: &[u8],
    mut scales: Vec<u8>,
    cuts: &[u8; 384],
    mut output: impl Write,
) -> io::Result<usize> {
    if scales.len().checked_mul(8) != Some(raw.len()) {
        return Err(io::Error::other("invalid four-bank weight dimensions"));
    }
    four_scale_banks(&mut scales, cuts).map_err(io::Error::other)?;
    output.write_all(cuts)?;
    Ok(cuts.len() + pack_nvfp4_tables::<false>(raw, &scales, 4 * NVFP4_CONTEXTS, output)?)
}

pub fn unpack_four_scale_banks(frame: &[u8], scales: Vec<u8>) -> Result<Vec<u8>, String> {
    let count = scales
        .len()
        .checked_mul(8)
        .ok_or("weight extent overflow")?;
    unpack_codes(Matrix::parse_four_scale_banks(frame, count, scales)?, count)
}

fn pack_nvfp4_tables<const SPLIT: bool>(
    raw: &[u8],
    banks: &[u8],
    table_count: usize,
    mut output: impl Write,
) -> io::Result<usize> {
    let states_per_column = if SPLIT { 16 } else { 8 };
    if raw.is_empty() || !raw.len().is_multiple_of(32) || raw.len() > u32::MAX as usize {
        return Err(io::Error::other("invalid contextual NVFP4 matrix size"));
    }
    let mut hist = vec![[0i64; 256]; table_count];
    for (index, block) in raw.as_chunks::<8>().0.iter().enumerate() {
        let bank = banks.get(index).copied().unwrap_or(0) as usize;
        let mut prev = 0;
        for (column, &v) in block.iter().enumerate() {
            hist[bank + column * states_per_column + usize::from(prev)][usize::from(v)] += 1;
            prev = nvfp4_prefix::<SPLIT>(prev, v);
        }
    }
    let mut tables = Vec::with_capacity(table_count);
    for h in &hist {
        let freq = normalize(h);
        let base = write_table(&freq, &mut output)?;
        tables.push((freq, base));
    }
    let mut first_block = 0;
    write_frames(raw, output, 512 * table_count, |tile, states, payload| {
        for (group_id, group) in tile.as_chunks::<32>().0.iter().enumerate().rev() {
            let mut contexts = [0; 32];
            for (lane, block) in group.as_chunks::<8>().0.iter().enumerate() {
                let bank = banks
                    .get(first_block + 4 * group_id + lane)
                    .copied()
                    .unwrap_or(0) as usize;
                let mut prev = 0;
                for (j, &v) in block.iter().enumerate() {
                    contexts[8 * lane + j] = bank + states_per_column * j + usize::from(prev);
                    prev = nvfp4_prefix::<SPLIT>(prev, v);
                }
            }
            for j in (0..8).rev() {
                for lane in (0..4).rev() {
                    let i = 8 * lane + j;
                    let v = usize::from(group[i]);
                    let (freq, base) = &tables[contexts[i]];
                    encode_symbol(&mut states[lane], freq[v], base[v], payload);
                }
            }
        }
        first_block += tile.len() / 8;
    })
}

/// E2M1 codes 0 and 8 both decode to integer zero in the NVFP4 dot products.
pub fn canonicalize_nvfp4_zeros(raw: &mut [u8]) {
    for byte in raw {
        if *byte & 0x0f == 0x08 {
            *byte &= 0xf0;
        }
        if *byte & 0xf0 == 0x80 {
            *byte &= 0x0f;
        }
    }
}

/// Offline packer input from an existing rANS or block-context rANS plane.
pub fn unpack_nvfp4_codes(data: &[u8], count: usize, contextual: bool) -> Result<Vec<u8>, String> {
    let matrix = if contextual {
        Matrix::parse_nvfp4(data, count)?
    } else {
        Matrix::parse(data, count)?
    };
    unpack_codes(matrix, count)
}

/// Offline verification of the separately tagged trim-aware NVFP4 format.
pub fn unpack_nvfp4_trim_context(data: &[u8], count: usize) -> Result<Vec<u8>, String> {
    unpack_codes(Matrix::parse_nvfp4_trim(data, count)?, count)
}

/// Decode the existing eight-column mean-context scale frame for offline packing.
pub fn unpack_mean8_scale_codes(
    data: &[u8],
    count: usize,
    width: usize,
) -> Result<Vec<u8>, String> {
    unpack_codes(Matrix::parse_mean_scales(data, count, width)?, count)
}

fn unpack_codes(matrix: Matrix<'_>, count: usize) -> Result<Vec<u8>, String> {
    let mut raw = Vec::with_capacity(count);
    matrix.for_each_chunk(|_, tile| {
        raw.extend_from_slice(tile);
        Ok(())
    })?;
    Ok(raw)
}

pub(crate) struct Matrix<'a> {
    scale_banks: Option<Vec<u8>>,
    scale_context: Option<Box<dyn scale_context::ScaleDecoder>>,
    lookup: Vec<[u32; 4096]>,
    frames: Vec<&'a [u8]>,
    count: usize,
}

// A nonzero frequency makes this byte reachable; zero-frequency entries never decode.
pub(crate) fn ternary_scale_tables(frame: &[u8], four_banks: bool) -> Result<bool, String> {
    let (offset, banks) = if four_banks { (384, 4) } else { (128, 2) };
    let tables = frame
        .get(offset..offset + banks * NVFP4_CONTEXTS * 512)
        .ok_or("missing scale-conditioned NVFP4 frequency tables")?;
    Ok(tables.as_chunks::<512>().0.iter().all(|table| {
        table
            .as_chunks::<2>()
            .0
            .iter()
            .enumerate()
            .all(|(byte, freq)| {
                *freq == [0, 0]
                    || (matches!(byte & 15, 0 | 7 | 15) && matches!(byte >> 4, 0 | 7 | 15))
            })
    }))
}

impl<'a> Matrix<'a> {
    pub(crate) fn parse(data: &'a [u8], count: usize) -> Result<Self, String> {
        Self::parse_tables(data, count, 1)
    }

    pub(crate) fn parse_nvfp4(data: &'a [u8], count: usize) -> Result<Self, String> {
        Self::parse_nvfp4_tables(data, count, NVFP4_CONTEXTS)
    }

    pub(crate) fn parse_nvfp4_trim(data: &'a [u8], count: usize) -> Result<Self, String> {
        Self::parse_nvfp4_tables(data, count, 2 * NVFP4_CONTEXTS)
    }

    pub(crate) fn parse_scale_direct(
        data: &'a [u8],
        count: usize,
        mut scales: Vec<u8>,
    ) -> Result<Self, String> {
        if scales.len().checked_mul(8) != Some(count) {
            return Err("scale/weight shape mismatch".into());
        }
        let (cuts, payload) = data
            .split_first_chunk::<128>()
            .ok_or("missing NVFP4 cutoffs")?;
        scale_banks(&mut scales, cuts)?;
        let mut matrix = Self::parse_nvfp4_tables(payload, count, 2 * NVFP4_CONTEXTS)?;
        matrix.scale_banks = Some(scales);
        Ok(matrix)
    }

    pub(crate) fn parse_four_scale_banks(
        data: &'a [u8],
        count: usize,
        mut scales: Vec<u8>,
    ) -> Result<Self, String> {
        if scales.len().checked_mul(8) != Some(count) {
            return Err("scale/weight shape mismatch".into());
        }
        let (cuts, payload) = data
            .split_first_chunk::<384>()
            .ok_or("missing four-bank boundaries")?;
        four_scale_banks(&mut scales, cuts)?;
        let mut matrix = Self::parse_nvfp4_tables(payload, count, 4 * NVFP4_CONTEXTS)?;
        matrix.scale_banks = Some(scales);
        Ok(matrix)
    }

    fn parse_nvfp4_tables(data: &'a [u8], count: usize, tables: usize) -> Result<Self, String> {
        if !count.is_multiple_of(32) {
            return Err("invalid contextual NVFP4 matrix size".into());
        }
        Self::parse_tables(data, count, tables)
    }

    fn parse_tables(data: &'a [u8], count: usize, tables: usize) -> Result<Self, String> {
        if count == 0 || !data.len().is_multiple_of(8) {
            return Err("invalid rANS matrix dimensions".into());
        }
        let (header, mut rest) = data
            .split_at_checked(tables * 512)
            .ok_or("missing rANS frequency table")?;
        let mut lookup = vec![[0; 4096]; tables];
        for (table, dst) in header.as_chunks::<512>().0.iter().zip(&mut lookup) {
            let mut pos = 0;
            for (symbol, &f) in table.as_chunks::<2>().0.iter().enumerate() {
                let freq = u16::from_le_bytes(f) as usize;
                let entries = dst
                    .get_mut(pos..pos + freq)
                    .ok_or("invalid rANS frequency sum")?;
                for (slot, e) in entries.iter_mut().enumerate() {
                    *e = ((symbol as u32) << 24) | ((slot as u32) << 12) | (freq as u32 - 1);
                }
                pos += freq;
            }
            if pos != TOTAL as usize {
                return Err("incomplete rANS frequency table".into());
            }
        }
        let mut frames = Vec::with_capacity(count.div_ceil(TILE));
        for first in (0..count).step_by(TILE) {
            let (size, src) = rest
                .split_at_checked(4)
                .ok_or("missing MXFP8 frame length")?;
            let size = u32::from_le_bytes(size.try_into().unwrap()) as usize;
            if !(16..=16 + 2 * TILE.min(count - first)).contains(&size) {
                return Err("invalid MXFP8 frame size".into());
            }
            let (frame, tail) = src.split_at_checked(size).ok_or("truncated MXFP8 frame")?;
            frames.push(frame);
            rest = tail;
        }
        if rest.len() >= 8 || rest.iter().any(|&v| v != 0) {
            return Err("invalid MXFP8 tensor padding".into());
        }
        Ok(Self {
            scale_context: None,
            scale_banks: None,
            lookup,
            frames,
            count,
        })
    }

    fn decode_tile(
        &self,
        first: usize,
        frame: &[u8],
        dst: &mut [MaybeUninit<u8>],
    ) -> Result<(), String> {
        let (words, bytes) = frame.split_at(16);
        let mut states = [0u32; 4];
        for (x, &word) in states.iter_mut().zip(words.as_chunks::<4>().0) {
            *x = u32::from_le_bytes(word);
            if !(LOWER..LOWER * 256).contains(x) {
                return Err("invalid MXFP8 rANS state".into());
            }
        }
        if let Some(context) = &self.scale_context {
            return context.decode(&self.lookup, bytes, states, first, dst);
        }
        if let Some(banks) = &self.scale_banks {
            let start = first / 8;
            return self.decode_nvfp4::<false, true>(
                bytes,
                states,
                dst,
                &banks[start..start + dst.len() / 8],
            );
        }
        if self.lookup.len() == NVFP4_CONTEXTS {
            return self.decode_nvfp4::<false, false>(bytes, states, dst, &[]);
        }
        if self.lookup.len() == 2 * NVFP4_CONTEXTS {
            return self.decode_nvfp4::<true, false>(bytes, states, dst, &[]);
        }
        let lookup = &self.lookup[0];
        let mut cursor = 0;
        // Four independent states share one byte stream. Each symbol needs at
        // most two normalization bytes. With eight input bytes in hand we can
        // advance all four states before consuming their bytes in lane order.
        let mut output = dst;
        while let (Some(payload), Some(group)) = (
            bytes[cursor..].first_chunk::<8>(),
            output.first_chunk_mut::<4>(),
        ) {
            for (state, byte) in states.iter_mut().zip(group) {
                let entry = lookup[(*state & (TOTAL - 1)) as usize];
                byte.write((entry >> 24) as u8);
                *state = ((entry & 4095) + 1) * (*state >> PRECISION) + ((entry >> 12) & 4095);
            }
            let mut consumed = 0;
            for state in &mut states {
                // Below LOWER / 256 a single byte cannot normalize the state.
                let needed = usize::from(*state < LOWER) + usize::from(*state < (LOWER >> 8));
                let word = u16::from_be_bytes([payload[consumed], payload[consumed + 1]]);
                *state = (*state << (8 * needed)) | (u32::from(word) >> (16 - 8 * needed));
                consumed += needed;
            }
            cursor += consumed;
            output = &mut output[4..];
        }
        // The final bytes and any partial four-symbol group need exact bounds checks.
        for (i, out) in output.iter_mut().enumerate() {
            let x = &mut states[i % 4];
            let entry = lookup[(*x & (TOTAL - 1)) as usize];
            out.write((entry >> 24) as u8);
            *x = ((entry & 4095) + 1) * (*x >> PRECISION) + ((entry >> 12) & 4095);
            // Most symbols consume zero or one normalization byte. A zero
            // lookahead also covers the final symbols that need no more input.
            let first = usize::from(*x < LOWER);
            let b = bytes.get(cursor).copied().unwrap_or(0);
            *x = (*x << (8 * first)) | (u32::from(b) & 0u32.wrapping_sub(first as u32));
            cursor += first;
            if *x < LOWER {
                let b = *bytes
                    .get(cursor)
                    .ok_or("truncated MXFP8 normalization bytes")?;
                *x = (*x << 8) | u32::from(b);
                cursor += 1;
            }
        }
        if cursor != bytes.len() || states != [LOWER; 4] {
            return Err("invalid terminal MXFP8 state".into());
        }
        Ok(())
    }

    fn decode_nvfp4<const SPLIT: bool, const SCALE: bool>(
        &self,
        bytes: &[u8],
        mut states: [u32; 4],
        dst: &mut [MaybeUninit<u8>],
        banks: &[u8],
    ) -> Result<(), String> {
        let mut cursor = 0;
        for (index, group) in dst.as_chunks_mut::<32>().0.iter_mut().enumerate() {
            let bank = if SCALE {
                banks[4 * index..4 * index + 4]
                    .first_chunk::<4>()
                    .copied()
                    .unwrap()
            } else {
                [0; 4]
            };
            let mut prev = [0u8; 4];
            for column in 0..8 {
                for (lane, state) in states.iter_mut().enumerate() {
                    let stride = if SPLIT { 16 } else { 8 };
                    let table = &self.lookup
                        [usize::from(bank[lane]) + stride * column + usize::from(prev[lane])];
                    let entry = table[(*state & (TOTAL - 1)) as usize];
                    let v = (entry >> 24) as u8;
                    group[8 * lane + column].write(v);
                    prev[lane] = if SPLIT {
                        TRIM_PREFIX[usize::from(prev[lane])][usize::from(v)]
                    } else {
                        nvfp4_prefix::<false>(prev[lane], v)
                    };
                    *state = ((entry & 4095) + 1) * (*state >> PRECISION) + ((entry >> 12) & 4095);
                    while *state < LOWER {
                        let byte = bytes.get(cursor).ok_or("short contextual NVFP4 payload")?;
                        *state = (*state << 8) | u32::from(*byte);
                        cursor += 1;
                    }
                }
            }
        }
        if cursor != bytes.len() || states != [LOWER; 4] {
            return Err("invalid terminal contextual NVFP4 state".into());
        }
        Ok(())
    }

    /// Convert four decoded NVFP4 nibbles to the resident signed two-bit layout.
    pub(crate) fn pack_ternary_weights(&self) -> Result<Vec<u8>, String> {
        let mut packed = Vec::with_capacity(self.count / 2);
        // A selector indexes the eight literal bytes in the CUDA PRMT table.
        // For two negative weights selector 15 replicates byte 7's sign.
        let mut table = [0xff; 256];
        for (selector, byte) in [0x00, 0x07, 0x0f, 0x70, 0x77, 0x7f, 0xf0, 0xf7]
            .into_iter()
            .enumerate()
        {
            table[byte] = selector as u8;
        }
        table[0xff] = 15;
        self.for_each_chunk(|_, tile| {
            let (pairs, tail) = tile.as_chunks::<2>();
            if !tail.is_empty() {
                return Err("unaligned ternary weight tile".into());
            }
            for &[a, b] in pairs {
                let x = table[usize::from(a)];
                let y = table[usize::from(b)];
                if x == 255 || y == 255 {
                    return Err("requested ternary plane contains a non-ternary weight".into());
                }
                packed.push(x | (y << 4));
            }
            Ok(())
        })?;
        Ok(packed)
    }

    /// Deliver decoded bytes in order while bounding temporary host memory.
    pub(crate) fn for_each_chunk(
        &self,
        mut emit: impl FnMut(usize, &[u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        let chunk = self.count.min(16 << 20);
        let mut staging = vec![0u8; chunk];
        for offset in (0..self.count).step_by(chunk) {
            let len = chunk.min(self.count - offset);
            let bytes = &mut staging[..len];
            // decode_at writes initialized bytes and emit runs only after a successful decode.
            let dst = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast(), len) };
            self.decode_at(offset, dst)?;
            emit(offset, bytes)?;
        }
        Ok(())
    }

    /// The output covers complete tiles except for the last tile of the matrix.
    pub(crate) fn decode_at(
        &self,
        offset: usize,
        dst: &mut [MaybeUninit<u8>],
    ) -> Result<(), String> {
        let end = offset
            .checked_add(dst.len())
            .ok_or("invalid MXFP8 output extent")?;
        if dst.is_empty()
            || !offset.is_multiple_of(TILE)
            || end > self.count
            || (end != self.count && !end.is_multiple_of(TILE))
        {
            return Err("invalid MXFP8 output extent".into());
        }
        let frames = &self.frames[offset / TILE..end.div_ceil(TILE)];
        if let [frame] = frames {
            return self.decode_tile(offset, frame, dst);
        }
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(24));
        let group = frames.len().div_ceil(threads);
        std::thread::scope(|scope| {
            let mut workers = Vec::new();
            for (part, (s, d)) in frames
                .chunks(group)
                .zip(dst.chunks_mut(group * TILE))
                .enumerate()
            {
                workers.push(scope.spawn(move || {
                    for (i, (&f, out)) in s.iter().zip(d.chunks_mut(TILE)).enumerate() {
                        self.decode_tile(offset + (part * group + i) * TILE, f, out)?;
                    }
                    Ok::<_, String>(())
                }));
            }
            for w in workers {
                w.join().map_err(|_| "MXFP8 decoder thread panicked")??;
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scale_direct_roundtrip_crosses_tile_and_expert_boundaries() {
        let raw: Vec<_> = (0..TILE + 1024).map(|i| (i * 13 + i / 7) as u8).collect();
        let scales: Vec<_> = (0..raw.len() / 8).map(|i| (i % 127) as u8).collect();
        let cuts = std::array::from_fn(|i| (i % 127) as u8);
        let mut frame = Vec::new();
        let n = pack_scale_direct(&raw, scales.clone(), &cuts, &mut frame).unwrap();
        assert_eq!(n, frame.len());
        assert_eq!(unpack_scale_direct(&frame, scales.clone()).unwrap(), raw);
        assert!(unpack_scale_direct(&frame[..frame.len() - 8], scales.clone()).is_err());
        frame[17] = 255;
        assert!(unpack_scale_direct(&frame, scales).is_err());
    }

    #[test]
    fn ternary_placement_agrees_across_direct_and_four_bank_frames() {
        let count = TILE + 1024;
        let scales: Vec<_> = (0..count / 8).map(|i| (i % 128) as u8).collect();
        let cuts = [64; 128];
        let mut raw = Vec::with_capacity(count);
        let mut expected = Vec::with_capacity(count / 2);
        for i in 0..count / 2 {
            let mut n = i % 81;
            let mut nibbles = 0u16;
            let selector = |index| if index == 8 { 15 } else { index as u8 };
            let packed = selector(n % 9) | (selector(n / 9) << 4);
            for j in 0..4 {
                let digit = n % 3;
                n /= 3;
                nibbles |= [0u16, 7, 15][digit] << (4 * j);
            }
            raw.extend(nibbles.to_le_bytes());
            expected.push(packed);
        }
        for mixed in [false, true] {
            if mixed {
                *raw.last_mut().unwrap() = 0x24;
            }
            for four_banks in [false, true] {
                let mut frame = Vec::new();
                let matrix = if four_banks {
                    pack_four_scale_banks(&raw, scales.clone(), &cuts, &mut frame).unwrap();
                    assert_eq!(ternary_scale_tables(&frame, true).unwrap(), !mixed);
                    Matrix::parse_four_scale_banks(&frame, count, scales.clone()).unwrap()
                } else {
                    pack_scale_direct(&raw, scales.clone(), &cuts, &mut frame).unwrap();
                    assert_eq!(ternary_scale_tables(&frame, false).unwrap(), !mixed);
                    Matrix::parse_scale_direct(&frame, count, scales.clone()).unwrap()
                };
                let result = matrix.pack_ternary_weights();
                if mixed {
                    assert!(result.unwrap_err().contains("non-ternary weight"));
                } else {
                    assert_eq!(result.unwrap(), expected);
                }
            }
        }
    }

    #[test]
    fn four_scale_banks_roundtrip_and_reject_corrupt_boundaries() {
        let raw: Vec<_> = (0..TILE + 1024).map(|i| (i * 17 + i / 11) as u8).collect();
        let scales: Vec<_> = (0..raw.len() / 8).map(|i| (i % 128) as u8).collect();
        let cuts = std::array::from_fn(|e| e as u8);
        let mut frame = Vec::new();
        let len = pack_four_scale_banks(&raw, scales.clone(), &cuts, &mut frame).unwrap();
        assert_eq!(frame.len(), len);
        assert_eq!(
            unpack_four_scale_banks(&frame, scales.clone()).unwrap(),
            raw
        );
        assert!(unpack_four_scale_banks(&frame[..len - 8], scales.clone()).is_err());
        frame[128] = 1;
        assert!(unpack_four_scale_banks(&frame, scales).is_err());
    }

    #[test]
    fn nonmedian_quartiles_use_the_existing_decoder() {
        let scales: Vec<u8> = (0..128).cycle().take(128 * 128).collect();
        let raw = vec![0x65; 8 * scales.len()];
        let cuts = [64; 128];
        let mut frame = Vec::new();
        pack_four_scale_quantiles(&raw, scales.clone(), &cuts, [1, 3], &mut frame).unwrap();
        assert_eq!(&frame[128..256], &[16; 128]);
        assert_eq!(&frame[256..384], &[112; 128]);
        assert_eq!(
            unpack_four_scale_banks(&frame, scales.clone()).unwrap(),
            raw
        );
        assert!(pack_four_scale_quantiles(&raw, scales, &cuts, [0, 3], Vec::new()).is_err());
    }

    #[test]
    fn trim_lookup_matches_all_encoder_transitions() {
        for (previous, transitions) in TRIM_PREFIX.iter().enumerate() {
            for (symbol, &next) in transitions.iter().enumerate() {
                assert_eq!(next, nvfp4_prefix::<true>(previous as u8, symbol as u8));
            }
        }
    }

    #[test]
    fn nvfp4_zero_canonicalization_only_changes_negative_zero_nibbles() {
        let mut raw: Vec<u8> = (0..=255).collect();
        canonicalize_nvfp4_zeros(&mut raw);
        let magnitudes = [0, 1, 2, 3, 4, 6, 8, 12];
        let decode = |n: u8| magnitudes[usize::from(n & 7)] * if n & 8 != 0 { -1 } else { 1 };
        for (original, &value) in (0u8..=255).zip(&raw) {
            for shift in [0, 4] {
                let source = (original >> shift) & 15;
                let dest = (value >> shift) & 15;
                assert_eq!(decode(source), decode(dest));
                assert_eq!(dest, if source == 8 { 0 } else { source });
            }
        }
    }

    #[test]
    fn nvfp4_context_restores_signed_nibbles_and_rejects_truncated_frames() {
        let raw: Vec<u8> = (0..TILE + 96)
            .map(|i| match i % 32 {
                0..=7 => 0x80,
                8..=15 => 0xf7,
                _ => (i.wrapping_mul(104729) ^ (i >> 11)) as u8,
            })
            .collect();
        let mut packed = Vec::new();
        assert_eq!(pack_nvfp4_context(&raw, &mut packed).unwrap(), packed.len());
        let m = Matrix::parse_nvfp4(&packed, raw.len()).unwrap();
        let mut out = Vec::new();
        m.for_each_chunk(|_, tile| {
            out.extend_from_slice(tile);
            Ok(())
        })
        .unwrap();
        assert_eq!(out, raw);
        assert!(Matrix::parse_nvfp4(&packed, raw.len() - 1).is_err());
        assert!(Matrix::parse_nvfp4(&packed[..packed.len() - 8], raw.len()).is_err());
        packed[512 * NVFP4_CONTEXTS + 4..512 * NVFP4_CONTEXTS + 8].fill(0);
        assert!(Matrix::parse_nvfp4(&packed, raw.len())
            .unwrap()
            .for_each_chunk(|_, _| Ok(()))
            .is_err());
    }

    #[test]
    fn trim_context_restores_every_byte_and_rejects_corrupt_frames() {
        let raw: Vec<_> = (0..TILE + 64)
            .map(|i| match (i / 8) % 3 {
                0 => 0x48,
                1 => 0x31,
                _ => (i.wrapping_mul(227) ^ (i >> 5)) as u8,
            })
            .collect();
        let mut packed = Vec::new();
        assert_eq!(
            pack_nvfp4_trim_context(&raw, &mut packed).unwrap(),
            packed.len()
        );
        assert_eq!(unpack_nvfp4_trim_context(&packed, raw.len()).unwrap(), raw);
        assert!(unpack_nvfp4_trim_context(&packed, raw.len() - 1).is_err());
        assert!(unpack_nvfp4_trim_context(&packed[..packed.len() - 8], raw.len()).is_err());
        let state = 2 * NVFP4_CONTEXTS * 512 + 4;
        packed[state..state + 4].fill(0);
        assert!(unpack_nvfp4_trim_context(&packed, raw.len()).is_err());
    }

    #[test]
    fn missing_normalization_byte_is_rejected_even_with_valid_framing() {
        // The table, states, frame length and padding still parse after truncation.
        let raw: Vec<u8> = (0..=255).cycle().take(4096).collect();
        let mut packed = Vec::new();
        pack(&raw, &mut packed).unwrap();
        let size = u32::from_le_bytes(packed[512..516].try_into().unwrap());
        assert!(size > 16);
        packed[512..516].copy_from_slice(&(size - 1).to_le_bytes());
        packed.truncate(516 + size as usize - 1);
        packed.resize(packed.len().next_multiple_of(8), 0);
        let matrix = Matrix::parse(&packed, raw.len()).unwrap();
        let mut out = vec![MaybeUninit::uninit(); raw.len()];
        assert!(matrix.decode_at(0, &mut out).is_err());
    }

    #[test]
    fn mixed_rare_symbols_restore_across_four_lane_and_frame_tails() {
        // Frequent zeroes need little input, while rare symbols can require two bytes.
        let input: Vec<u8> = (0..TILE + 67)
            .map(|i| if i % 31 < 4 { (i % 251 + 1) as u8 } else { 0 })
            .collect();
        for tail in 0..4 {
            let raw = &input[..TILE + 64 + tail];
            let mut packed = Vec::new();
            pack(raw, &mut packed).unwrap();
            let matrix = Matrix::parse(&packed, raw.len()).unwrap();
            let mut out = vec![MaybeUninit::uninit(); raw.len()];
            matrix.decode_at(0, &mut out).unwrap();
            assert!(out.iter().zip(raw).all(|(decoded, expected)| {
                // A successful decode_at initializes every output byte.
                unsafe { decoded.assume_init() == *expected }
            }));
        }
    }

    #[test]
    fn frames_restore_every_code_and_reject_damage() {
        for raw in [
            vec![0x9d; TILE + 7],
            (0..TILE * 2 + 13)
                .map(|i| if i % 7 == 0 { 3 } else { (i % 256) as u8 })
                .collect(),
        ] {
            let mut packed = Vec::new();
            assert_eq!(pack(&raw, &mut packed).unwrap(), packed.len());
            let matrix = Matrix::parse(&packed, raw.len()).unwrap();
            let mut out = vec![MaybeUninit::uninit(); raw.len()];
            for (i, d) in out.chunks_mut(TILE).enumerate() {
                matrix.decode_at(i * TILE, d).unwrap();
            }
            assert!(out
                .iter()
                .zip(&raw)
                .all(|(a, b)| unsafe { a.assume_init() == *b }));
            let mut bad = packed.clone();
            bad[..512].fill(0);
            assert!(Matrix::parse(&bad, raw.len()).is_err());
            bad = packed.clone();
            bad[516..520].fill(0);
            assert!(Matrix::parse(&bad, raw.len())
                .unwrap()
                .decode_at(0, &mut out)
                .is_err());
            assert!(Matrix::parse(&packed[..packed.len() - 8], raw.len()).is_err());
        }
    }
}
