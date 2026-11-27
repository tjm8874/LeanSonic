//! Lossless 4-lane byte rANS for on-disk MXFP8 expert weights. The 256 LE
//! u16 frequencies sum to 4096. Each independent 128 KiB of raw data has a
//! LE u32 frame size, four LE u32 states and byte-normalization payload.
//! The tensor is padded with zero bytes to an eight-byte boundary.
use std::io::{self, Write};
use std::mem::MaybeUninit;

pub const TILE: usize = 128 * 1024;
const PRECISION: u32 = 12;
const TOTAL: u32 = 1 << PRECISION;
const LOWER: u32 = 1 << 23;

fn frequencies(bytes: &[u8]) -> [u16; 256] {
    let mut histogram = [0i64; 256];
    for &b in bytes {
        histogram[b as usize] += 1;
    }
    let count = bytes.len() as i64;
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
    let mut base = [0u32; 256];
    let mut cum = 0u32;
    for (i, &f) in freq.iter().enumerate() {
        output.write_all(&f.to_le_bytes())?;
        base[i] = cum;
        cum += u32::from(f);
    }
    let mut written = 512;
    let mut payload = Vec::with_capacity(TILE);
    for tile in raw.chunks(TILE) {
        payload.clear();
        let mut states = [LOWER; 4];
        for (i, &value) in tile.iter().enumerate().rev() {
            let idx = value as usize;
            let f = u32::from(freq[idx]);
            let x = &mut states[i % 4];
            while *x >= ((LOWER >> PRECISION) << 8) * f {
                payload.push(*x as u8);
                *x >>= 8;
            }
            *x = ((*x / f) << PRECISION) + *x % f + base[idx];
        }
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

pub(crate) struct Matrix<'a> {
    lookup: Box<[u32; 4096]>,
    frames: Vec<&'a [u8]>,
    count: usize,
}

impl<'a> Matrix<'a> {
    pub(crate) fn parse(data: &'a [u8], count: usize) -> Result<Self, String> {
        if count == 0 || !data.len().is_multiple_of(8) {
            return Err("invalid MXFP8 matrix dimensions".into());
        }
        let (table, mut rest) = data
            .split_at_checked(512)
            .ok_or("missing MXFP8 frequency table")?;
        let mut pos = 0;
        let mut lookup = Box::new([0; 4096]);
        for (symbol, &f) in table.as_chunks::<2>().0.iter().enumerate() {
            let freq = u16::from_le_bytes(f) as usize;
            let entries = lookup
                .get_mut(pos..pos + freq)
                .ok_or("invalid MXFP8 frequency sum")?;
            for (slot, e) in entries.iter_mut().enumerate() {
                *e = ((symbol as u32) << 24) | ((slot as u32) << 12) | (freq as u32 - 1);
            }
            pos += freq;
        }
        if pos != TOTAL as usize {
            return Err("incomplete MXFP8 frequency table".into());
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
            lookup,
            frames,
            count,
        })
    }

    fn decode_tile(&self, frame: &[u8], dst: &mut [MaybeUninit<u8>]) -> Result<(), String> {
        let (words, bytes) = frame.split_at(16);
        let mut states = [0u32; 4];
        for (x, &word) in states.iter_mut().zip(words.as_chunks::<4>().0) {
            *x = u32::from_le_bytes(word);
            if !(LOWER..LOWER * 256).contains(x) {
                return Err("invalid MXFP8 rANS state".into());
            }
        }
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
                let entry = self.lookup[(*state & (TOTAL - 1)) as usize];
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
            let entry = self.lookup[(*x & (TOTAL - 1)) as usize];
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
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(24));
        let group = frames.len().div_ceil(threads);
        std::thread::scope(|scope| {
            let mut workers = Vec::new();
            for (s, d) in frames.chunks(group).zip(dst.chunks_mut(group * TILE)) {
                workers.push(scope.spawn(move || {
                    for (&f, out) in s.iter().zip(d.chunks_mut(TILE)) {
                        self.decode_tile(f, out)?;
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
