//! Optional, lossless storage for Inkling expert scale bytes.
//! The U8 plane named `<original tensor>.range_v1` starts with a 16-byte
//! header: bit width, base, two zero bytes, element count and escape count
//! (little-endian u32), then four zero bytes. Each eight inputs form a
//! little-endian group of `bits` bytes. The largest code escapes to the
//! next verbatim byte after the groups. Zero padding aligns the end to 8.
use std::io::{self, Write};

pub struct ScalePacking {
    base: u8,
    bits: u8,
    count: usize,
    escapes: usize,
}

impl ScalePacking {
    pub fn choose(values: &[u8]) -> Result<Self, String> {
        let count = values.len();
        if count == 0 || !count.is_multiple_of(8) || count > u32::MAX as usize {
            return Err("invalid expert scale count".into());
        }
        let mut cumulative = [0usize; 257];
        for &v in values {
            cumulative[v as usize + 1] += 1;
        }
        for i in 1..cumulative.len() {
            cumulative[i] += cumulative[i - 1];
        }
        let mut best = (usize::MAX, 0, 0, 0);
        for bits in 1..=7u8 {
            let span = (1usize << bits) - 1;
            for base in 0..=256 - span {
                let escapes = count - (cumulative[base + span] - cumulative[base]);
                best = best.min((
                    count / 8 * bits as usize + escapes,
                    bits,
                    base as u8,
                    escapes,
                ));
            }
        }
        Ok(Self {
            base: best.2,
            bits: best.1,
            count,
            escapes: best.3,
        })
    }

    pub fn encoded_len(&self) -> usize {
        (16 + self.count / 8 * self.bits as usize + self.escapes).next_multiple_of(8)
    }

    pub fn write(&self, values: &[u8], mut writer: impl Write) -> io::Result<()> {
        if values.len() != self.count {
            return Err(io::Error::other("scale count changed"));
        }
        let mut header = [0u8; 16];
        header[0] = self.bits;
        header[1] = self.base;
        header[4..8].copy_from_slice(&(self.count as u32).to_le_bytes());
        header[8..12].copy_from_slice(&(self.escapes as u32).to_le_bytes());
        writer.write_all(&header)?;
        let escape = (1u8 << self.bits) - 1;
        let mut tail = Vec::with_capacity(self.escapes);
        for group in values.as_chunks::<8>().0 {
            let mut word = 0u64;
            for (lane, &v) in group.iter().enumerate() {
                let delta = v.wrapping_sub(self.base);
                let code = if delta < escape {
                    delta
                } else {
                    tail.push(v);
                    escape
                };
                word |= (code as u64) << (lane * self.bits as usize);
            }
            writer.write_all(&word.to_le_bytes()[..self.bits as usize])?;
        }
        if tail.len() != self.escapes {
            return Err(io::Error::other("scale histogram changed"));
        }
        writer.write_all(&tail)?;
        let used = 16 + self.count / 8 * self.bits as usize + tail.len();
        writer.write_all(&[0; 7][..self.encoded_len() - used])
    }
}

fn expand<const B: usize>(
    groups: &[u8],
    base: u8,
    tail: &mut &[u8],
    out: &mut [u8],
) -> Result<(), String> {
    const BYTE_ONES: u64 = 0x0101_0101_0101_0101;
    let code_mask = (1u64 << B) - 1;
    let four_codes = (1u64 << (4 * B)) - 1;
    let two_codes = ((1u64 << (2 * B)) - 1) * 0x0000_0001_0000_0001;
    let one_code = code_mask * 0x0001_0001_0001_0001;
    let escapes = code_mask * BYTE_ONES;
    let base_low = u64::from(base & 127) * BYTE_ONES;
    let base_high = u64::from(base & 128) * BYTE_ONES;
    for (group, row) in groups
        .as_chunks::<B>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<8>().0)
    {
        let mut bytes = [0; 8];
        bytes[..B].copy_from_slice(group);
        let word = u64::from_le_bytes(bytes);
        // Separate the fields four at a time, then two, then one per byte.
        let halves = (word & four_codes) | ((word >> (4 * B)) << 32);
        let pairs = (halves & two_codes) | (((halves >> (2 * B)) & two_codes) << 16);
        let lanes = (pairs & one_code) | (((pairs >> B) & one_code) << 8);
        // Codes and base_low are each at most 127 per byte, so the add
        // cannot carry between lanes. Restore bit 7 after the addition.
        *row = ((lanes + base_low) ^ base_high).to_le_bytes();
        // XOR maps each escape to zero. Adding 127 sets bit 7 in every
        // nonzero lane. Visit only the exceptions, in original byte order.
        let mut hits = !((lanes ^ escapes) + 127 * BYTE_ONES) & (128 * BYTE_ONES);
        while hits != 0 {
            let (first, rest) = tail.split_first().ok_or("missing scale exception")?;
            row[hits.trailing_zeros() as usize / 8] = *first;
            *tail = rest;
            hits &= hits - 1;
        }
    }
    Ok(())
}

pub fn unpack_scales(
    data: &[u8],
    count: usize,
    mut emit: impl FnMut(usize, &[u8]) -> Result<(), String>,
) -> Result<(), String> {
    let header = data.get(..16).ok_or("short packed scale header")?;
    let bits = header[0];
    let base = header[1];
    let stored = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
    let escapes = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
    if !(1..=7).contains(&bits)
        || stored != count
        || count == 0
        || !count.is_multiple_of(8)
        || escapes > count
        || (base as u16 + (1u16 << bits) - 2) > 255
        || header[2..4] != [0; 2]
        || header[12..16] != [0; 4]
    {
        return Err("invalid packed scale header".into());
    }
    let groups_end = 16 + count / 8 * bits as usize;
    let end = groups_end + escapes;
    if data.len() != end.next_multiple_of(8) || data[end..].iter().any(|&v| v != 0) {
        return Err("invalid packed scale length or padding".into());
    }
    let mut tail = &data[groups_end..end];
    const TILE: usize = 1 << 20;
    let mut scratch = vec![0; count.min(TILE)];
    for first in (0..count).step_by(TILE) {
        let tile = &mut scratch[..TILE.min(count - first)];
        let start = 16 + first / 8 * bits as usize;
        let groups = &data[start..start + tile.len() / 8 * bits as usize];
        match bits {
            1 => expand::<1>(groups, base, &mut tail, tile)?,
            2 => expand::<2>(groups, base, &mut tail, tile)?,
            3 => expand::<3>(groups, base, &mut tail, tile)?,
            4 => expand::<4>(groups, base, &mut tail, tile)?,
            5 => expand::<5>(groups, base, &mut tail, tile)?,
            6 => expand::<6>(groups, base, &mut tail, tile)?,
            7 => expand::<7>(groups, base, &mut tail, tile)?,
            _ => unreachable!(),
        }
        emit(first, tile)?;
    }
    if !tail.is_empty() {
        return Err("unused scale exceptions".into());
    }
    Ok(())
}

// Four consecutive original bytes form one logical scale word in the CUDA
// kernels. Store eight low-bit fields together with no padding between groups.
pub(crate) fn pack_gpu_tile<const BITS: usize, const BASE: u8>(
    values: &[u8],
    out: &mut Vec<u8>,
) -> bool {
    const BYTE_ONES: u64 = 0x0101_0101_0101_0101;
    let low_mask = (1u64 << BITS) - 1;
    let fixed_bits = BYTE_ONES * (255 - low_mask);
    let (groups, rem) = values.as_chunks::<8>();
    if !rem.is_empty() {
        return false;
    }
    out.clear();
    out.reserve(groups.len() * BITS);
    for &group in groups {
        let word = u64::from_le_bytes(group);
        if word & fixed_bits != u64::from(BASE) * BYTE_ONES {
            return false;
        }
        let mut fields = 0;
        for lane in 0..8 {
            fields |= ((word >> (8 * lane)) & low_mask) << (BITS * lane);
        }
        out.extend_from_slice(&fields.to_le_bytes()[..BITS]);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_scale_words_cross_aligned_loads_and_reject_outliers() {
        fn check<const BITS: usize, const BASE: u8>() {
            let mask = (1u8 << BITS) - 1;
            let mut values: Vec<u8> = (0..256).map(|i| BASE + (i as u8 & mask)).collect();
            let mut packed = Vec::new();
            assert!(pack_gpu_tile::<BITS, BASE>(&values, &mut packed));
            assert_eq!(packed.len(), values.len() * BITS / 8);
            packed.extend_from_slice(&[0; 4]);
            for (index, original) in values.chunks_exact(4).enumerate() {
                let bit = index * 4 * BITS;
                let byte = (bit / 32) * 4;
                let lo = u32::from_le_bytes(packed[byte..byte + 4].try_into().unwrap());
                let hi = u32::from_le_bytes(packed[byte + 4..byte + 8].try_into().unwrap());
                let fields = (u64::from(lo) | u64::from(hi) << 32) >> (bit % 32);
                for (lane, &value) in original.iter().enumerate() {
                    assert_eq!(BASE + ((fields >> (lane * BITS)) as u8 & mask), value);
                }
            }
            assert!(!pack_gpu_tile::<BITS, BASE>(&values[1..], &mut packed));
            for outlier in [BASE - 1, BASE + mask + 1] {
                values[255] = outlier;
                assert!(!pack_gpu_tile::<BITS, BASE>(&values, &mut packed));
            }
        }
        check::<6, 64>();
        check::<3, 112>();
    }

    #[test]
    fn round_trip_crosses_upload_tile_and_byte_boundaries() {
        for width in 1..=7 {
            let base = 256 - ((1u16 << width) - 1);
            let count = (1 << 20) + 24;
            let mut escapes = 0;
            let values: Vec<_> = (0..count)
                .map(|i| {
                    if i % 17 == 0 {
                        escapes += 1;
                        0
                    } else {
                        (base + (i % (256 - base) as usize) as u16) as u8
                    }
                })
                .collect();
            let plan = ScalePacking {
                base: base as u8,
                bits: width,
                count,
                escapes,
            };
            let mut encoded = Vec::new();
            plan.write(&values, &mut encoded).unwrap();
            assert_eq!(encoded.len(), plan.encoded_len());
            let mut copied = 0;
            unpack_scales(&encoded, count, |first, tile| {
                assert_eq!(first, copied);
                assert_eq!(tile, &values[first..first + tile.len()]);
                copied += tile.len();
                Ok(())
            })
            .unwrap();
            assert_eq!(copied, count);
        }
    }

    #[test]
    fn all_exception_patterns_preserve_the_order_and_values() {
        for bits in 1..=7 {
            let span = (1u8 << bits) - 1;
            for base in [0, 127, 128, 255 - span + 1] {
                let mut values = Vec::new();
                let mut escapes = 0;
                for mask in 0..=255u8 {
                    for lane in 0..8u8 {
                        if mask & (1 << lane) != 0 {
                            values.push(if base > 0 { lane % base } else { 255 - lane });
                            escapes += 1;
                        } else {
                            values.push(base + mask.wrapping_add(lane) % span);
                        }
                    }
                }
                let plan = ScalePacking {
                    base,
                    bits,
                    count: values.len(),
                    escapes,
                };
                let mut frame = Vec::new();
                plan.write(&values, &mut frame).unwrap();
                unpack_scales(&frame, values.len(), |offset, tile| {
                    assert_eq!(tile, &values[offset..offset + tile.len()]);
                    Ok(())
                })
                .unwrap();
            }
        }
    }

    #[test]
    fn rejects_bad_header_codes_and_unused_exception_bytes() {
        let values = [90, 90, 90, 90, 90, 90, 90, 255];
        let plan = ScalePacking::choose(&values).unwrap();
        let mut packed = Vec::new();
        plan.write(&values, &mut packed).unwrap();
        assert_eq!((plan.bits, plan.base, plan.escapes), (1, 90, 1));
        let bad = |p: &[u8]| unpack_scales(p, 8, |_, _| Ok(())).is_err();
        for len in 0..packed.len() {
            assert!(bad(&packed[..len]));
        }
        for (index, value) in [
            (0, 0),
            (0, 8),
            (2, 1),
            (4, 16),
            (8, 2),
            (12, 1),
            (16, 0),
            (16, 255),
            (23, 1),
        ] {
            let mut copy = packed.clone();
            copy[index] = value;
            assert!(bad(&copy));
        }
        let mut invalid = packed;
        invalid[0] = 4;
        invalid[1] = 255;
        assert!(bad(&invalid));
    }
}
