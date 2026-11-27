//! Lossless encoding of the 256-wide DSpark BF16 Markov head.
//! Two little-endian u64 frame lengths precede rANS streams of row means,
//! sign plus fraction, and wrapping exponent differences from the row means.
use crate::inkling_mxfp8::{self, Matrix};
use std::io;

const WIDTH: usize = 256;

pub fn pack(raw: &[u8]) -> io::Result<Vec<u8>> {
    let (rows, tail) = raw.as_chunks::<{ 2 * WIDTH }>();
    if rows.is_empty() || !tail.is_empty() {
        return Err(io::Error::other("invalid Markov row dimensions"));
    }
    let mut means = Vec::with_capacity(rows.len());
    let mut fractions = Vec::with_capacity(raw.len() / 2);
    let mut residuals = Vec::with_capacity(raw.len() / 2);
    for row in rows {
        let words = row.as_chunks::<2>().0;
        let exponents = words
            .iter()
            .map(|&word| (u16::from_le_bytes(word) >> 7) as u8);
        let sum: u32 = exponents.clone().map(u32::from).sum();
        let mean = ((sum + WIDTH as u32 / 2) / WIDTH as u32) as u8;
        means.push(mean);
        for (&[lo, hi], exponent) in words.iter().zip(exponents) {
            fractions.push((lo & 127) | (hi & 128));
            residuals.push(exponent.wrapping_sub(mean));
        }
    }
    let mut dst = vec![0; 16];
    let a = inkling_mxfp8::pack(&means, &mut dst)?;
    let b = inkling_mxfp8::pack(&fractions, &mut dst)?;
    inkling_mxfp8::pack(&residuals, &mut dst)?;
    dst[..8].copy_from_slice(&(a as u64).to_le_bytes());
    dst[8..16].copy_from_slice(&(b as u64).to_le_bytes());
    Ok(dst)
}

pub fn unpack(data: &[u8], rows: usize) -> Result<Vec<u8>, String> {
    let n = rows.checked_mul(WIDTH).ok_or("invalid Markov dimensions")?;
    let bytes = n.checked_mul(2).ok_or("invalid Markov dimensions")?;
    let (header, body) = data
        .split_first_chunk::<16>()
        .ok_or("missing Markov row header")?;
    let lengths = header.as_chunks::<8>().0;
    let cut = |b| usize::try_from(u64::from_le_bytes(b)).map_err(|_| "invalid Markov row offset");
    let (mean_frame, rest) = body
        .split_at_checked(cut(lengths[0])?)
        .ok_or("invalid mean frame")?;
    let (frac_frame, exp_frame) = rest
        .split_at_checked(cut(lengths[1])?)
        .ok_or("invalid fraction frame")?;
    let mean_matrix = Matrix::parse(mean_frame, rows)?;
    let fractions = Matrix::parse(frac_frame, n)?;
    let exponents = Matrix::parse(exp_frame, n)?;
    let mut means = Vec::with_capacity(rows);
    mean_matrix.for_each_chunk(|_, chunk| {
        means.extend_from_slice(chunk);
        Ok(())
    })?;
    let mut dst = vec![0; bytes];
    fractions.for_each_chunk(|off, chunk| {
        for (word, &f) in dst[2 * off..].as_chunks_mut::<2>().0.iter_mut().zip(chunk) {
            *word = [f & 127, f & 128];
        }
        Ok(())
    })?;
    exponents.for_each_chunk(|off, chunk| {
        for (i, (word, &residual)) in dst[2 * off..]
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(chunk)
            .enumerate()
        {
            let exp = means[(off + i) / WIDTH].wrapping_add(residual);
            word[0] |= exp << 7;
            word[1] |= exp >> 1;
        }
        Ok(())
    })?;
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn restores_all_bit_patterns_across_frame_boundaries() {
        let src: Vec<u8> = (0..=u16::MAX)
            .cycle()
            .take(513 * WIDTH)
            .flat_map(u16::to_le_bytes)
            .collect();
        let mut frame = pack(&src).unwrap();
        assert_eq!(unpack(&frame, 513).unwrap(), src);
        assert!(unpack(&frame[..frame.len() - 8], 513).is_err());
        assert!(unpack(&frame, 512).is_err());
        frame[8..16].fill(255);
        assert!(unpack(&frame, 513).is_err());
        assert!(pack(&src[2..]).is_err());
    }
}
