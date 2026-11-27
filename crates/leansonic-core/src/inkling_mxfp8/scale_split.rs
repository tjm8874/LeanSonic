//! Two lossless NVFP4 streams selected by the accompanying scale plane.
use super::{pack_nvfp4_context, unpack_nvfp4_codes};
use std::error::Error;

fn arm(scale: u8, cutoff_plus_one: u8) -> usize {
    usize::from(scale < cutoff_plus_one)
}

pub fn pack_scale_split(
    raw: &[u8],
    scales: &[u8],
    cuts: &[u8; 128],
) -> Result<Vec<u8>, Box<dyn Error>> {
    if scales.is_empty()
        || !scales.len().is_multiple_of(128)
        || scales.len().checked_mul(8) != Some(raw.len())
    {
        return Err("invalid scale-split input dimensions".into());
    }
    let mut parts = [Vec::new(), Vec::new()];
    let groups_per_expert = scales.len() / cuts.len();
    for (i, bytes) in raw.chunks_exact(8).enumerate() {
        parts[arm(scales[i], cuts[i / groups_per_expert])].extend_from_slice(bytes);
    }
    let mut frame = cuts.to_vec();
    frame.resize(144, 0);
    for (j, part) in parts.iter_mut().enumerate() {
        if part.is_empty() {
            continue;
        }
        part.resize(part.len().next_multiple_of(32), 0);
        let bytes = pack_nvfp4_context(part, &mut frame)?;
        frame[128 + 8 * j..136 + 8 * j].copy_from_slice(&(bytes as u64).to_le_bytes());
    }
    Ok(frame)
}

pub fn unpack_scale_split(frame: &[u8], scales: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    if scales.is_empty()
        || !scales.len().is_multiple_of(128)
        || scales.len().checked_mul(8).is_none()
    {
        return Err("invalid scale-split output dimensions".into());
    }
    let cuts = frame.get(..128).ok_or("missing cutoffs")?;
    if cuts.iter().any(|&v| v > 127) {
        return Err("invalid E4M3 cutoff".into());
    }
    let header = frame.get(128..144).ok_or("missing sizes")?;
    let mut sizes = [0usize; 2];
    let groups_per_expert = scales.len() / 128;
    for (i, &scale) in scales.iter().enumerate() {
        sizes[arm(scale, cuts[i / groups_per_expert])] += 8;
    }
    let mut data = &frame[144..];
    let mut parts = [Vec::new(), Vec::new()];
    for (j, field) in header.chunks_exact(8).enumerate() {
        let bytes = usize::try_from(u64::from_le_bytes(field.try_into()?))?;
        let (encoded, rest) = data.split_at_checked(bytes).ok_or("truncated stream")?;
        if sizes[j] > 0 {
            parts[j] = unpack_nvfp4_codes(encoded, sizes[j].next_multiple_of(32), true)?;
            if parts[j][sizes[j]..].iter().any(|&b| b != 0) {
                return Err("nonzero decoded padding".into());
            }
        } else if bytes != 0 {
            return Err("nonempty stream for empty group".into());
        }
        data = rest;
    }
    if !data.is_empty() {
        return Err("trailing bytes".into());
    }
    let mut pos = [0usize; 2];
    let mut out = Vec::with_capacity(scales.len() * 8);
    for (i, &scale) in scales.iter().enumerate() {
        let j = arm(scale, cuts[i / groups_per_expert]);
        out.extend_from_slice(&parts[j][pos[j]..pos[j] + 8]);
        pos[j] += 8;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn split_handles_empty_arms_ragged_counts_and_corrupt_frames() {
        let scales: Vec<_> = (0..384).map(|i| (i % 127) as u8).collect();
        let raw: Vec<_> = (0..scales.len() * 8).map(|i| (i * 17) as u8).collect();
        for cutoff in [0, 1, 37, 127] {
            let frame = pack_scale_split(&raw, &scales, &[cutoff; 128]).unwrap();
            assert_eq!(unpack_scale_split(&frame, &scales).unwrap(), raw);
            assert!(unpack_scale_split(&frame[..frame.len() - 1], &scales).is_err());
            let mut damaged = frame;
            damaged[128..136].fill(255);
            assert!(unpack_scale_split(&damaged, &scales).is_err());
        }
    }
}
