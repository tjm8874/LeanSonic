//! Lossless BF16 input embeddings. Each 128 KiB of raw BF16 is a separate tile.
//! A little-endian u32 tile length precedes the low bytes, one packed sign bit
//! per value, and LSB-first prefix codes for the remaining seven high bits.
//! The tensor ends with zero padding to eight bytes.
mod rank;
pub use rank::unpack_rank_embedding;

use std::io::{self, Read, Write};
use std::mem::MaybeUninit;

const TILE: usize = 128 * 1024;
const COMMON: [u8; 7] = [63, 62, 61, 64, 60, 59, 58];

pub fn pack_embedding(
    mut input: impl Read,
    bytes: usize,
    mut output: impl Write,
) -> io::Result<usize> {
    if bytes == 0 || !bytes.is_multiple_of(16) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid embedding size",
        ));
    }
    let mut rank = [7u8; 128];
    for (r, &value) in COMMON.iter().enumerate() {
        rank[value as usize] = r as u8;
    }
    let mut raw = vec![0; TILE.min(bytes)];
    let mut frame = Vec::new();
    let mut written = 0;
    for first in (0..bytes).step_by(TILE) {
        let tile = &mut raw[..TILE.min(bytes - first)];
        input.read_exact(tile)?;
        let n = tile.len() / 2;
        frame.clear();
        frame.resize(n + n / 8, 0);
        let mut word = 0u32;
        let mut filled = 0;
        for (i, &[low, high]) in tile.as_chunks::<2>().0.iter().enumerate() {
            frame[i] = low;
            frame[n + i / 8] |= (high >> 7) << (i % 8);
            let magnitude = high & 127;
            let r = rank[magnitude as usize];
            // Common ranks have r one bits followed by a zero. Seven ones
            // escape to a verbatim magnitude, covering every BF16 bit pattern.
            let (code, bits) = if r == 7 {
                (127 | u32::from(magnitude) << 7, 14)
            } else {
                ((1u32 << r) - 1, r + 1)
            };
            word |= code << filled;
            filled += bits;
            while filled >= 8 {
                frame.push(word as u8);
                word >>= 8;
                filled -= 8;
            }
        }
        if filled != 0 {
            frame.push(word as u8);
        }
        output.write_all(&(frame.len() as u32).to_le_bytes())?;
        output.write_all(&frame)?;
        written += 4 + frame.len();
    }
    let padding = written.next_multiple_of(8) - written;
    output.write_all(&[0; 7][..padding])?;
    Ok(written + padding)
}

#[derive(Clone, Copy)]
struct PrefixGroup {
    magnitudes: [u8; 4],
    bits: u8,
}

// Decode four frequent magnitudes with one table lookup when all prefixes
// fit in twelve bits. An escape or a longer group uses the scalar decoder.
const LOOKUP_BITS: usize = 12;
const LOOKUP_SIZE: usize = 1 << LOOKUP_BITS;

const fn prefix_table() -> [PrefixGroup; LOOKUP_SIZE] {
    let empty = PrefixGroup {
        magnitudes: [0; 4],
        bits: 0,
    };
    let mut table = [empty; LOOKUP_SIZE];
    let mut word = 0;
    while word < LOOKUP_SIZE {
        let mut entry = empty;
        let mut lane = 0;
        while lane < 4 {
            let r = (((word as u32) >> entry.bits) & 127).trailing_ones() as usize;
            if r == 7 || entry.bits as usize + r + 1 > LOOKUP_BITS {
                break;
            }
            entry.magnitudes[lane] = COMMON[r];
            entry.bits += r as u8 + 1;
            lane += 1;
        }
        if lane == 4 {
            table[word] = entry;
        }
        word += 1;
    }
    table
}
static PREFIXES: [PrefixGroup; LOOKUP_SIZE] = prefix_table();

struct TileDecoder<'a> {
    low: &'a [u8],
    signs: &'a [u8],
    codes: &'a [u8],
    bit: usize,
}
impl<'a> TileDecoder<'a> {
    fn new(frame: &'a [u8], bytes: usize) -> Self {
        let words = bytes / 2;
        let (low, rest) = frame.split_at(words);
        let (signs, codes) = rest.split_at(words / 8);
        Self {
            low,
            signs,
            codes,
            bit: 0,
        }
    }

    #[inline(always)]
    fn group(&mut self, group: usize, out: &mut [MaybeUninit<u8>; 8]) -> Result<(), String> {
        let tail = self
            .codes
            .get(self.bit / 8..)
            .ok_or("truncated embedding code")?;
        let bytes = if let Some((first, _)) = tail.split_first_chunk::<8>() {
            *first
        } else {
            let mut tmp = [0; 8];
            tmp[..tail.len()].copy_from_slice(tail);
            tmp
        };
        let mut window = u64::from_le_bytes(bytes) >> (self.bit % 8);
        let entry = PREFIXES[window as usize & (LOOKUP_SIZE - 1)];
        let mut magnitudes = entry.magnitudes;
        if entry.bits != 0 {
            self.bit += entry.bits as usize;
        } else {
            for magnitude in &mut magnitudes {
                let rank = (window & 127).trailing_ones() as usize;
                let bits = if rank == 7 {
                    *magnitude = ((window >> 7) & 127) as u8;
                    14
                } else {
                    *magnitude = COMMON[rank];
                    rank + 1
                };
                window >>= bits;
                self.bit += bits;
            }
        }
        let sign = self.signs[group / 2] >> (4 * (group % 2));
        for i in 0..4 {
            out[2 * i].write(self.low[4 * group + i]);
            out[2 * i + 1].write(magnitudes[i] | (((sign >> i) & 1) << 7));
        }
        Ok(())
    }

    fn finish(self) -> Result<(), String> {
        let bit = self.bit;
        let codes = self.codes;
        if bit > codes.len() * 8 {
            return Err("truncated embedding code".into());
        }
        if bit.div_ceil(8) != codes.len()
            || (!bit.is_multiple_of(8) && codes[bit / 8] >> (bit % 8) != 0)
        {
            return Err("unused embedding code bits".into());
        }
        Ok(())
    }
}

// Each offset depends on the preceding prefix lookup. Alternating two
// independent tiles lets their lookup chains overlap on the same worker.
fn decode_pair(frames: &[&[u8]], dst: &mut [MaybeUninit<u8>]) -> Result<(), String> {
    let (first, second) = dst.split_at_mut(dst.len().min(TILE));
    let mut first_decoder = TileDecoder::new(frames[0], first.len());
    let (groups, _) = first.as_chunks_mut::<8>();
    let mut consumed = 0;
    if let Some(&frame) = frames.get(1) {
        let mut second_decoder = TileDecoder::new(frame, second.len());
        let (second_groups, _) = second.as_chunks_mut::<8>();
        for (index, (left, right)) in groups.iter_mut().zip(second_groups.iter_mut()).enumerate() {
            first_decoder.group(index, left)?;
            second_decoder.group(index, right)?;
        }
        consumed = second_groups.len();
        second_decoder.finish()?;
    }
    for (offset, out) in groups[consumed..].iter_mut().enumerate() {
        first_decoder.group(consumed + offset, out)?;
    }
    first_decoder.finish()
}

/// Decode directly into the final allocation. All output bytes are initialized
/// on success. The caller can discard the allocation on validation failure.
pub fn unpack_embedding(mut data: &[u8], dst: &mut [MaybeUninit<u8>]) -> Result<(), String> {
    if dst.is_empty() || !dst.len().is_multiple_of(16) || !data.len().is_multiple_of(8) {
        return Err("invalid packed embedding dimensions".into());
    }
    let mut frames = Vec::with_capacity(dst.len().div_ceil(TILE));
    for first in (0..dst.len()).step_by(TILE) {
        let (length, rest) = data
            .split_at_checked(4)
            .ok_or("missing embedding tile length")?;
        let length = u32::from_le_bytes(length.try_into().unwrap()) as usize;
        let n = (dst.len() - first).min(TILE) / 2;
        if !(n + n / 8 + n / 8..=n + n / 8 + (14 * n).div_ceil(8)).contains(&length) {
            return Err("invalid embedding tile length".into());
        }
        let (frame, rest) = rest
            .split_at_checked(length)
            .ok_or("truncated embedding tile")?;
        frames.push(frame);
        data = rest;
    }
    if data.len() >= 8 || data.iter().any(|&v| v != 0) {
        return Err("invalid embedding padding".into());
    }

    // Each tile has its own bitstream and a disjoint destination range.
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(8));
    let per_worker = frames.len().div_ceil(threads);
    std::thread::scope(|scope| {
        let mut workers = Vec::new();
        for (source, output) in frames
            .chunks(per_worker)
            .zip(dst.chunks_mut(per_worker * TILE))
        {
            workers.push(scope.spawn(move || {
                for (frames, tiles) in source.chunks(2).zip(output.chunks_mut(2 * TILE)) {
                    decode_pair(frames, tiles)?;
                }
                Ok::<_, String>(())
            }));
        }
        for worker in workers {
            worker
                .join()
                .map_err(|_| "embedding decoder thread panicked")??;
        }
        Ok(())
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn round_trip_all_bf16_patterns_and_partial_last_tile() {
        let raw: Vec<u8> = (0..65536u32 + 24)
            .flat_map(|v| (v as u16).to_le_bytes())
            .collect();
        let mut bytes = Vec::new();
        assert_eq!(
            pack_embedding(raw.as_slice(), raw.len(), &mut bytes).unwrap(),
            bytes.len()
        );
        let mut out = vec![MaybeUninit::uninit(); raw.len()];
        unpack_embedding(&bytes, &mut out).unwrap();
        // All output bytes were initialized by the successful decode.
        assert!(out
            .iter()
            .zip(&raw)
            .all(|(a, b)| unsafe { a.assume_init() == *b }));
    }
    #[test]
    fn pair_decodes_a_partial_second_tile_and_checks_its_tail() {
        let raw: Vec<u8> = (0..=u16::MAX)
            .cycle()
            .flat_map(u16::to_le_bytes)
            .take(TILE + 16)
            .collect();
        let mut packed = Vec::new();
        pack_embedding(&raw[..], raw.len(), &mut packed).unwrap();
        let first_len = u32::from_le_bytes(packed[..4].try_into().unwrap()) as usize;
        let (first, tail) = packed[4..].split_at(first_len);
        let second_len = u32::from_le_bytes(tail[..4].try_into().unwrap()) as usize;
        let second = &tail[4..4 + second_len];
        let mut out = vec![MaybeUninit::uninit(); raw.len()];
        decode_pair(&[first, second], &mut out).unwrap();
        // A successful pair decode initializes both tiles.
        assert!(out
            .iter()
            .zip(&raw)
            .all(|(a, b)| unsafe { a.assume_init() == *b }));
        assert!(decode_pair(&[first, &second[..second_len - 1]], &mut out).is_err());
    }

    #[test]
    fn rejects_damaged_frames_and_padding() {
        let mut bytes = Vec::new();
        pack_embedding([0u8; 16].as_slice(), 16, &mut bytes).unwrap();
        let bad = |frame: &[u8]| unpack_embedding(frame, &mut [MaybeUninit::uninit(); 16]).is_err();
        assert!(bad(&bytes[..8]));
        let mut changed = bytes.clone();
        changed[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(bad(&changed));
        changed = bytes.clone();
        changed.extend_from_slice(&[0; 8]);
        assert!(bad(&changed));
        let mut short_code = vec![0; 16];
        short_code[..4].copy_from_slice(&10u32.to_le_bytes());
        short_code[13] = 127; // Escape prefix with only one following bit.
        assert!(bad(&short_code));
        let mut bad_padding = Vec::new();
        pack_embedding([63u8; 16].as_slice(), 16, &mut bad_padding).unwrap();
        *bad_padding.last_mut().unwrap() = 1;
        assert!(bad(&bad_padding));
        bytes[4 + 8 + 1] = 0;
        assert!(bad(&bytes));
    }
}
