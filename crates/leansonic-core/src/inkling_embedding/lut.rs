use crate::inkling_mxfp8::{pack, unpack_nvfp4_codes, Matrix};
use std::mem::MaybeUninit;

const TABLE_BYTES: usize = 512;

fn finite(word: &[u8; 2]) -> bool {
    u16::from_le_bytes(*word) & 0x7f80 != 0x7f80
}

fn read_table(frame: &[u8]) -> Result<(&[[u8; 2]], &[u8]), String> {
    let (table, body) = frame
        .split_first_chunk::<TABLE_BYTES>()
        .ok_or("truncated embedding lookup table")?;
    let values = table.as_chunks::<2>().0;
    if !values.iter().all(finite) {
        return Err("invalid BF16 lookup table".into());
    }
    Ok((values, body))
}

fn expand(
    table: &[[u8; 2]],
    indices: &[u8],
    literals: &mut impl Iterator<Item = [u8; 2]>,
    dst: &mut [MaybeUninit<u8>],
) -> Result<(), String> {
    for (&index, out) in indices.iter().zip(dst.as_chunks_mut::<2>().0) {
        let [lo, hi] = if index == 255 {
            let literal = literals.next().ok_or("missing embedding literal")?;
            if !finite(&literal) {
                return Err("non-finite embedding literal".into());
            }
            literal
        } else {
            table[usize::from(index)]
        };
        out[0].write(lo);
        out[1].write(hi);
    }
    Ok(())
}

/// Decode a BF16 lookup table, one index per weight, then exact BF16 escapes.
pub fn unpack_lut8_embedding(frame: &[u8], dst: &mut [MaybeUninit<u8>]) -> Result<(), String> {
    let (table, body) = read_table(frame)?;
    if dst.is_empty() || !dst.len().is_multiple_of(2) {
        return Err("invalid lookup embedding dimensions".into());
    }
    let (indices, literal_data) = body
        .split_at_checked(dst.len() / 2)
        .ok_or("truncated embedding indices")?;
    let (words, trailing) = literal_data.as_chunks::<2>();
    if !trailing.is_empty() {
        return Err("truncated embedding literal".into());
    }
    let mut literals = words.iter().copied();
    expand(table, indices, &mut literals, dst)?;
    if literals.next().is_some() {
        return Err("unused embedding literals".into());
    }
    Ok(())
}

/// Losslessly transcode an existing LUT8 frame. Counts and stream lengths are LE u32.
/// Separate rANS streams store indices and the low and high literal bytes.
pub fn pack_lut8_rans(frame: &[u8], count: usize) -> Result<Vec<u8>, String> {
    let (_, body) = read_table(frame)?;
    if count == 0 || count > u32::MAX as usize {
        return Err("invalid lookup embedding dimensions".into());
    }
    let (indices, literal_data) = body
        .split_at_checked(count)
        .ok_or("truncated embedding indices")?;
    let (words, trailing) = literal_data.as_chunks::<2>();
    if !trailing.is_empty()
        || indices.iter().filter(|&&v| v == 255).count() != words.len()
        || !words.iter().all(finite)
    {
        return Err("invalid embedding literal stream".into());
    }
    let (low, high): (Vec<u8>, Vec<u8>) = words.iter().map(|&[a, b]| (a, b)).unzip();
    let mut out = frame[..TABLE_BYTES].to_vec();
    out.extend_from_slice(&(words.len() as u32).to_le_bytes());
    out.extend_from_slice(&[0; 12]);
    for (header, data) in [indices, &low, &high].into_iter().enumerate() {
        if !data.is_empty() {
            let n = pack(data, &mut out).map_err(|e| e.to_string())?;
            let length = u32::try_from(n).map_err(|_| "oversized embedding stream")?;
            let p = TABLE_BYTES + 4 + 4 * header;
            out[p..p + 4].copy_from_slice(&length.to_le_bytes());
        }
    }
    Ok(out)
}

pub fn unpack_lut8_rans(frame: &[u8], dst: &mut [MaybeUninit<u8>]) -> Result<(), String> {
    let (table, body) = read_table(frame)?;
    let (header, streams) = body
        .split_first_chunk::<16>()
        .ok_or("missing embedding stream lengths")?;
    let n = |i: usize| u32::from_le_bytes(header[4 * i..4 * i + 4].try_into().unwrap()) as usize;
    let count = dst.len() / 2;
    let escapes = n(0);
    if count == 0 || !dst.len().is_multiple_of(2) || escapes > count {
        return Err("invalid lookup embedding dimensions".into());
    }
    let (indices, rest) = streams
        .split_at_checked(n(1))
        .ok_or("truncated index stream")?;
    let (low, high) = rest
        .split_at_checked(n(2))
        .ok_or("truncated literal stream")?;
    if high.len() != n(3) {
        return Err("incorrect final literal stream length".into());
    }
    let decode_literals = |data: &[u8]| {
        if escapes == 0 {
            if !data.is_empty() {
                return Err("unexpected literal stream".into());
            }
            Ok(Vec::new())
        } else {
            unpack_nvfp4_codes(data, escapes, false)
        }
    };
    let low = decode_literals(low)?;
    let high = decode_literals(high)?;
    let mut literals = low.into_iter().zip(high).map(|(lo, hi)| [lo, hi]);
    Matrix::parse(indices, count)?.for_each_chunk(|offset, codes| {
        expand(
            table,
            codes,
            &mut literals,
            &mut dst[2 * offset..2 * (offset + codes.len())],
        )
    })?;
    if literals.next().is_some() {
        return Err("unused embedding literals".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_lookup_matches_raw_across_tiles_with_and_without_literals() {
        for escape in [false, true] {
            let n = crate::inkling_mxfp8::TILE + 19;
            let mut frame: Vec<u8> = (0..256u16).flat_map(u16::to_le_bytes).collect();
            let codes: Vec<u8> = (0..n)
                .map(|i| {
                    if escape && i % 17 == 0 {
                        255
                    } else {
                        (i % 254) as u8
                    }
                })
                .collect();
            frame.extend_from_slice(&codes);
            for (i, _) in codes.iter().enumerate().filter(|(_, c)| **c == 255) {
                frame.extend_from_slice(&(0x8000 | (i % 0x7000) as u16).to_le_bytes());
            }
            let packed = pack_lut8_rans(&frame, n).unwrap();
            let mut a = vec![MaybeUninit::uninit(); 2 * n];
            let mut b = a.clone();
            unpack_lut8_embedding(&frame, &mut a).unwrap();
            unpack_lut8_rans(&packed, &mut b).unwrap();
            // Both full destinations were initialized by the successful decodes.
            assert!(a
                .iter()
                .zip(&b)
                .all(|(a, b)| unsafe { a.assume_init() == b.assume_init() }));
            assert!(unpack_lut8_rans(&packed[..packed.len() - 1], &mut b).is_err());
            let mut bad = packed.clone();
            bad[TABLE_BYTES..TABLE_BYTES + 4].copy_from_slice(&u32::MAX.to_le_bytes());
            assert!(unpack_lut8_rans(&bad, &mut b).is_err());
        }
    }
}
