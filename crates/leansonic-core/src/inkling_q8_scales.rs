//! Lossless on-disk dictionary for the raw little-endian F32 words of one Q8 matrix.
//! Q8S1, u32 dictionary length, dictionary words, one index per scale, then
//! one literal word per index 255. Only zero padding to the next 8 bytes follows.
use std::collections::HashMap;

const ESC: u8 = 255;

pub fn pack(raw: &[u8]) -> Result<Vec<u8>, String> {
    if raw.is_empty() || !raw.len().is_multiple_of(4) {
        return Err("invalid Q8 scale plane length".into());
    }
    let words = || {
        raw.chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
    };
    let mut counts = HashMap::new();
    for word in words() {
        *counts.entry(word).or_insert(0usize) += 1;
    }
    let mut ranked: Vec<_> = counts.into_iter().collect();
    ranked.sort_unstable_by_key(|&(word, n)| (std::cmp::Reverse(n), word));
    ranked.truncate(usize::from(ESC));
    let mut out = b"Q8S1".to_vec();
    out.extend_from_slice(&(ranked.len() as u32).to_le_bytes());
    let mut indices = HashMap::new();
    for (i, (word, _)) in ranked.into_iter().enumerate() {
        indices.insert(word, i as u8);
        out.extend_from_slice(&word.to_le_bytes());
    }
    let mut literals = Vec::new();
    for word in words() {
        let index = indices.get(&word).copied().unwrap_or(ESC);
        out.push(index);
        if index == ESC {
            literals.extend_from_slice(&word.to_le_bytes());
        }
    }
    out.extend(literals);
    out.resize(out.len().next_multiple_of(8), 0);
    Ok(out)
}

pub fn unpack(packed: &[u8], count: usize) -> Result<Vec<u8>, String> {
    let bad = || "invalid Q8 scale dictionary".to_owned();
    if packed.len() < 8 || !packed.len().is_multiple_of(8) || &packed[..4] != b"Q8S1" || count == 0
    {
        return Err(bad());
    }
    let n = u32::from_le_bytes(packed[4..8].try_into().unwrap()) as usize;
    if !(1..=usize::from(ESC)).contains(&n) {
        return Err(bad());
    }
    let start = 8 + 4 * n;
    let end = start.checked_add(count).ok_or_else(bad)?;
    let indices = packed.get(start..end).ok_or_else(bad)?;
    let mut tail = &packed[end..];
    let mut raw = Vec::with_capacity(count.checked_mul(4).ok_or_else(bad)?);
    for &index in indices {
        if index == ESC {
            let (word, rest) = tail.split_at_checked(4).ok_or_else(bad)?;
            raw.extend_from_slice(word);
            tail = rest;
        } else if usize::from(index) < n {
            let off = 8 + 4 * usize::from(index);
            raw.extend_from_slice(&packed[off..off + 4]);
        } else {
            return Err(bad());
        }
    }
    if tail.len() >= 8 || tail.iter().any(|&b| b != 0) {
        return Err(bad());
    }
    Ok(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn round_trips_literals_special_bits_and_frequency_ties() {
        let raw: Vec<_> = (0..512u32)
            .chain([0x80000000, 0x7fc01234, u32::MAX, 0, 0])
            .flat_map(u32::to_le_bytes)
            .collect();
        let packed = pack(&raw).unwrap();
        assert_eq!(packed, pack(&raw).unwrap());
        assert_eq!(unpack(&packed, raw.len() / 4).unwrap(), raw);
        assert!(unpack(&packed[..packed.len() - 8], raw.len() / 4).is_err());
    }
    #[test]
    fn rejects_bad_index_trailer_and_extent() {
        let packed = pack(&[0; 16]).unwrap();
        for (index, value) in [(0, 0), (4, 0), (12, 1), (12, ESC)] {
            let mut bad = packed.clone();
            bad[index] = value;
            assert!(unpack(&bad, 4).is_err());
        }
        let mut bad = packed;
        bad.extend([0; 8]);
        assert!(unpack(&bad, 4).is_err());
        assert!(unpack(&bad, usize::MAX).is_err());
    }
}
