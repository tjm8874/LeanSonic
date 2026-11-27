//! Positive BF16 magnitudes and a separate sign bit for each weight.
//! Four little-endian u32s give the table count, escape count, primary frame
//! length and secondary frame length. A primary index of 255 consumes one
//! secondary index. Both streams use the existing four-lane byte rANS format.
use crate::inkling_mxfp8::{unpack_nvfp4_codes, Matrix};
use std::mem::MaybeUninit;

pub fn unpack_magnitude_embedding(frame: &[u8], dst: &mut [MaybeUninit<u8>]) -> Result<(), String> {
    let count = dst.len() / 2;
    if count == 0 || !dst.len().is_multiple_of(2) {
        return Err("invalid magnitude embedding dimensions".into());
    }
    let (header, body) = frame
        .split_first_chunk::<16>()
        .ok_or("missing magnitude header")?;
    let [entries, escapes, first_bytes, extra_bytes] = std::array::from_fn(|i| {
        u32::from_le_bytes(header[4 * i..4 * i + 4].try_into().unwrap()) as usize
    });
    if !(1..=511).contains(&entries) || escapes > count || (escapes > 0 && entries <= 255) {
        return Err("invalid magnitude table or escape count".into());
    }
    let (table, body) = body
        .split_at_checked(2 * entries)
        .ok_or("truncated magnitude table")?;
    let table = table.as_chunks::<2>().0;
    if table.iter().any(|word| u16::from_le_bytes(*word) >= 0x7f80) {
        return Err("invalid BF16 magnitude".into());
    }
    let (signs, body) = body
        .split_at_checked(count.div_ceil(8))
        .ok_or("truncated sign bits")?;
    if !count.is_multiple_of(8) && signs[signs.len() - 1] >> (count % 8) != 0 {
        return Err("unused magnitude sign bits".into());
    }
    let (first, extra) = body
        .split_at_checked(first_bytes)
        .ok_or("truncated magnitude indices")?;
    if extra.len() != extra_bytes || (escapes == 0 && !extra.is_empty()) {
        return Err("invalid secondary magnitude frame".into());
    }
    let secondary = if escapes == 0 {
        Vec::new()
    } else {
        unpack_nvfp4_codes(extra, escapes, false)?
    };
    let mut secondary = secondary.into_iter();
    Matrix::parse(first, count)?.for_each_chunk(|offset, indices| {
        for (j, (&index, out)) in indices
            .iter()
            .zip(dst[2 * offset..].chunks_exact_mut(2))
            .enumerate()
        {
            let rank = if index == 255 {
                255 + usize::from(
                    secondary
                        .next()
                        .ok_or("missing secondary magnitude index")?,
                )
            } else {
                usize::from(index)
            };
            let &[lo, hi] = table.get(rank).ok_or("magnitude index exceeds table")?;
            let i = offset + j;
            let sign = (signs[i / 8] >> (i % 8)) & 1;
            out[0].write(lo);
            out[1].write(hi | (sign << 7));
        }
        Ok(())
    })?;
    if secondary.next().is_some() {
        return Err("unused secondary magnitude indices".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inkling_mxfp8::{pack, TILE};

    fn frame(entries: u32, codes: &[u8], extra: &[u8], signs: &[u8]) -> Vec<u8> {
        let mut primary = Vec::new();
        pack(codes, &mut primary).unwrap();
        let mut secondary = Vec::new();
        if !extra.is_empty() {
            pack(extra, &mut secondary).unwrap();
        }
        let mut bytes = Vec::new();
        for n in [
            entries,
            extra.len() as u32,
            primary.len() as u32,
            secondary.len() as u32,
        ] {
            bytes.extend_from_slice(&n.to_le_bytes());
        }
        for i in 0..entries {
            bytes.extend_from_slice(&(i as u16).to_le_bytes());
        }
        bytes.extend_from_slice(signs);
        bytes.extend_from_slice(&primary);
        bytes.extend_from_slice(&secondary);
        bytes
    }

    #[test]
    fn decodes_signed_zero_and_secondary_magnitudes_across_tiles() {
        let n = TILE + 9;
        let codes: Vec<_> = (0..n)
            .map(|i| if i % 5 == 0 { 255 } else { (i % 255) as u8 })
            .collect();
        let extra = vec![255; n.div_ceil(5)];
        let mut signs = vec![0xaa; n.div_ceil(8)];
        *signs.last_mut().unwrap() = 0;
        let mut out = vec![MaybeUninit::uninit(); 2 * n];
        unpack_magnitude_embedding(&frame(511, &codes, &extra, &signs), &mut out).unwrap();
        for (i, b) in out.chunks_exact(2).enumerate() {
            // A successful decode initializes both bytes at every position.
            let word = u16::from_le_bytes(unsafe { [b[0].assume_init(), b[1].assume_init()] });
            let value = if codes[i] == 255 {
                510
            } else {
                u16::from(codes[i])
            };
            assert_eq!(
                word,
                value | (u16::from((signs[i / 8] >> (i % 8)) & 1) << 15)
            );
        }
        let mut zero = [MaybeUninit::uninit(); 2];
        unpack_magnitude_embedding(&frame(1, &[0], &[], &[1]), &mut zero).unwrap();
        assert_eq!(unsafe { zero.map(|x| x.assume_init()) }, [0, 128]);
    }

    #[test]
    fn rejects_invalid_table_signs_indices_and_lengths() {
        let valid = frame(256, &[255], &[0], &[0]);
        let mut invalid = vec![
            frame(1, &[1], &[], &[0]),
            frame(256, &[255], &[], &[0]),
            frame(256, &[0], &[0], &[0]),
            frame(256, &[255], &[1], &[0]),
            frame(1, &[0], &[], &[2]),
            valid[..valid.len() - 1].to_vec(),
        ];
        let mut nonfinite = valid.clone();
        nonfinite[16..18].copy_from_slice(&0x7f80u16.to_le_bytes());
        invalid.push(nonfinite);
        let mut trailing = valid;
        trailing.push(0);
        invalid.push(trailing);
        for bytes in invalid {
            assert!(unpack_magnitude_embedding(&bytes, &mut [MaybeUninit::uninit(); 2]).is_err());
        }
    }
}
