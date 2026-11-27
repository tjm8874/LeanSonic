//! Store the low and high BF16 bytes separately with the existing lossless byte codec.
use crate::inkling_mxfp8::{self, Matrix};
use std::io;

pub mod row_exponents;

pub fn pack(raw: &[u8]) -> io::Result<Vec<u8>> {
    if raw.is_empty() || !raw.len().is_multiple_of(2) {
        return Err(io::Error::other("invalid BF16 plane size"));
    }
    let mut packed = vec![0; 8];
    let mut plane = vec![0; raw.len() / 2];
    for byte in 0..2 {
        for (dst, word) in plane.iter_mut().zip(raw.as_chunks::<2>().0) {
            *dst = word[byte];
        }
        let length = inkling_mxfp8::pack(&plane, &mut packed)?;
        if byte == 0 {
            packed[..8].copy_from_slice(&(length as u64).to_le_bytes());
        }
    }
    Ok(packed)
}

pub(crate) fn unpack(data: &[u8], count: usize) -> Result<Vec<u8>, String> {
    let (len, frames) = data
        .split_first_chunk::<8>()
        .ok_or("missing BF16 plane header")?;
    let low_len =
        usize::try_from(u64::from_le_bytes(*len)).map_err(|_| "invalid BF16 plane offset")?;
    let (lo, hi) = frames
        .split_at_checked(low_len)
        .ok_or("invalid BF16 plane offset")?;
    let mut raw = vec![0; count.checked_mul(2).ok_or("invalid BF16 plane size")?];
    for (byte, frame) in [lo, hi].into_iter().enumerate() {
        Matrix::parse(frame, count)?.for_each_chunk(|offset, tile| {
            for (word, &value) in raw[2 * offset..]
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(tile)
            {
                word[byte] = value;
            }
            Ok(())
        })?;
    }
    Ok(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn restores_all_bf16_patterns_and_checks_both_frames() {
        let raw: Vec<u8> = (0..=u16::MAX)
            .cycle()
            .take(2 * inkling_mxfp8::TILE + 3)
            .flat_map(u16::to_le_bytes)
            .collect();
        let mut frame = pack(&raw).unwrap();
        assert_eq!(unpack(&frame, raw.len() / 2).unwrap(), raw);
        assert!(unpack(&frame[..frame.len() - 8], raw.len() / 2).is_err());
        frame[..8].fill(255);
        assert!(unpack(&frame, raw.len() / 2).is_err());
    }
}
