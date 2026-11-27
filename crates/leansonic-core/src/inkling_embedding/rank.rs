//! Lossless BF16 frequency ranks. Eight adjacent low ranks occupy `bits` bytes.
//! The high ranks use the existing byte rANS matrix and decode into bounded scratch.
use crate::inkling_mxfp8::Matrix;
use std::mem::MaybeUninit;

pub fn unpack_rank_embedding(frame: &[u8], dst: &mut [MaybeUninit<u8>]) -> Result<(), String> {
    let (header, body) = frame
        .split_first_chunk::<16>()
        .ok_or("missing rank header")?;
    let field = |i| u32::from_le_bytes(header[i..i + 4].try_into().unwrap()) as usize;
    let (entries, bits, upper_bytes) = (field(4), field(8), field(12));
    if &header[..4] != b"BRK1"
        || !(1..=8).contains(&bits)
        || entries == 0
        || entries > (256 << bits)
        || dst.is_empty()
        || !dst.len().is_multiple_of(16)
    {
        return Err("invalid BF16 rank dimensions".into());
    }
    let count = dst.len() / 2;
    let (words, body) = body
        .split_at_checked(entries * 2)
        .ok_or("truncated BF16 rank table")?;
    let words = words.as_chunks::<2>().0;
    let (lower, upper) = body
        .split_at_checked(count / 8 * bits)
        .ok_or("truncated low ranks")?;
    if upper.len() != upper_bytes {
        return Err("invalid high rank stream length".into());
    }
    let mask = (1u64 << bits) - 1;
    Matrix::parse(upper, count)?.for_each_chunk(|offset, high| {
        let (groups, tail) = high.as_chunks::<8>();
        if !tail.is_empty() || !offset.is_multiple_of(8) {
            return Err("unaligned rank matrix chunk".into());
        }
        let low_groups = lower[offset / 8 * bits..].chunks_exact(bits);
        let outputs = dst[2 * offset..].chunks_exact_mut(16);
        for ((hi, low), output) in groups.iter().zip(low_groups).zip(outputs) {
            let mut packed = [0u8; 8];
            packed[..bits].copy_from_slice(low);
            let mut lanes = u64::from_le_bytes(packed);
            for (&h, out) in hi.iter().zip(output.chunks_exact_mut(2)) {
                let rank = ((h as usize) << bits) | (lanes & mask) as usize;
                let word = words.get(rank).ok_or("BF16 rank outside table")?;
                out[0].write(word[0]);
                out[1].write(word[1]);
                lanes >>= bits;
            }
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inkling_mxfp8::pack;

    #[test]
    fn splits_ranks_and_preserves_bf16_bits() {
        let table = [0x8000u16, 0x0000, 0x7fc1, 0xc080, 0x4100];
        let ranks = [0u8, 1, 2, 3, 4, 3, 1, 0];
        for bits in 1..=8u32 {
            let mut upper = Vec::new();
            pack(&ranks.map(|r| (u32::from(r) >> bits) as u8), &mut upper).unwrap();
            let mut frame = b"BRK1".to_vec();
            for field in [table.len() as u32, bits, upper.len() as u32] {
                frame.extend(field.to_le_bytes());
            }
            for word in table {
                frame.extend(word.to_le_bytes());
            }
            let mask = (1u64 << bits) - 1;
            let mut packed = 0u64;
            for (lane, &r) in ranks.iter().enumerate() {
                packed |= (u64::from(r) & mask) << (lane * bits as usize);
            }
            frame.extend(&packed.to_le_bytes()[..bits as usize]);
            frame.extend(upper);
            let mut dst = [MaybeUninit::uninit(); 16];
            unpack_rank_embedding(&frame, &mut dst).unwrap();
            // A successful decode initialized every output lane.
            assert_eq!(
                dst.map(|v| unsafe { v.assume_init() }).as_slice(),
                ranks
                    .map(|i| table[usize::from(i)].to_le_bytes())
                    .as_flattened()
            );
            assert!(unpack_rank_embedding(&frame[..frame.len() - 1], &mut dst).is_err());
            if bits == 8 {
                frame[16 + table.len() * 2] = 250;
                assert!(unpack_rank_embedding(&frame, &mut dst).is_err());
            }
        }
    }
}
