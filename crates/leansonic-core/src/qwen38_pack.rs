//! Lossless dictionary frames for Qwen3.8 BF16 scale and bias planes.
//! The fixed-width indices allow random access without inflating preceding rows.

const MAGIC: &[u8; 8] = b"Q38DICT1";
const HEADER: usize = 20;
const PAD: usize = 2;

fn width(entries: usize) -> usize {
    (usize::BITS - (entries - 1).leading_zeros()) as usize
}

fn payload_size(words: usize, bits: usize) -> Result<usize, &'static str> {
    words
        .checked_mul(bits)
        .map(|n| n.div_ceil(8))
        .ok_or("oversized frame")
}

pub fn encode(raw: &[u8]) -> Result<Vec<u8>, &'static str> {
    if raw.is_empty() || !raw.len().is_multiple_of(2) {
        return Err("expected nonempty BF16 words");
    }
    let mut lookup = vec![u32::MAX; 65536];
    let mut dictionary = Vec::new();
    for pair in raw.as_chunks::<2>().0 {
        let word = u16::from_le_bytes([pair[0], pair[1]]);
        if lookup[word as usize] == u32::MAX {
            lookup[word as usize] = dictionary.len() as u32;
            dictionary.push(word);
        }
    }
    let bits = width(dictionary.len());
    let mut frame = Vec::with_capacity(
        HEADER + dictionary.len() * 2 + payload_size(raw.len() / 2, bits)? + PAD,
    );
    frame.extend_from_slice(MAGIC);
    frame.extend_from_slice(&((raw.len() / 2) as u64).to_le_bytes());
    frame.extend_from_slice(&(dictionary.len() as u32).to_le_bytes());
    for word in dictionary {
        frame.extend_from_slice(&word.to_le_bytes());
    }
    let mut accumulator = 0u32;
    let mut used = 0;
    for pair in raw.as_chunks::<2>().0 {
        let word = u16::from_le_bytes([pair[0], pair[1]]);
        accumulator |= lookup[word as usize] << used;
        used += bits;
        while used >= 8 {
            frame.push(accumulator as u8);
            accumulator >>= 8;
            used -= 8;
        }
    }
    if used != 0 {
        frame.push(accumulator as u8);
    }
    frame.extend_from_slice(&[0; PAD]);
    Ok(frame)
}

pub struct Frame<'a> {
    dictionary: &'a [u8],
    payload: &'a [u8],
    words: usize,
    bits: usize,
}
impl<'a> Frame<'a> {
    pub fn open(data: &'a [u8], expected_words: usize) -> Result<Self, &'static str> {
        if data.len() < HEADER || &data[..8] != MAGIC {
            return Err("invalid frame header");
        }
        let count = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let entries = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
        if count != expected_words as u64
            || count == 0
            || !(1..=65536).contains(&entries)
            || entries > expected_words
        {
            return Err("invalid frame dimensions");
        }
        let end = HEADER + 2 * entries;
        let bits = width(entries);
        let size = payload_size(expected_words, bits)?;
        if data.len().checked_sub(end + PAD) != Some(size) || !data.ends_with(&[0; PAD]) {
            return Err("invalid payload length or padding");
        }
        Ok(Self {
            dictionary: &data[HEADER..end],
            payload: &data[end..],
            words: expected_words,
            bits,
        })
    }
    pub fn word(&self, i: usize) -> Result<u16, &'static str> {
        if i >= self.words {
            return Err("word out of bounds");
        }
        let bit = i * self.bits;
        let p = bit / 8;
        if self.bits == 0 {
            return Ok(u16::from_le_bytes([self.dictionary[0], self.dictionary[1]]));
        }
        let chunk =
            u32::from_le_bytes([self.payload[p], self.payload[p + 1], self.payload[p + 2], 0]);
        let code = (chunk >> (bit % 8)) & ((1 << self.bits) - 1);
        self.entry(code as usize)
    }

    /// Decode a contiguous BF16 word range, retaining its original wire byte order.
    pub fn decode_range(&self, start: usize, words: usize) -> Result<Vec<u8>, &'static str> {
        if start > self.words || words > self.words - start {
            return Err("word range out of bounds");
        }
        let bytes = words.checked_mul(2).ok_or("oversized word range")?;
        let mut result = vec![0; bytes];
        for (i, pair) in result.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            *pair = self.word(start + i)?.to_le_bytes();
        }
        Ok(result)
    }

    /// Decode the five group parameters of one Engram embedding row.
    pub fn row5(&self, row: usize) -> Result<[u16; 5], &'static str> {
        let i = row.checked_mul(5).ok_or("row out of bounds")?;
        if self.words.saturating_sub(i) < 5 {
            return Err("row out of bounds");
        }
        let bit = i * self.bits;
        let offset = bit / 8;
        let mut out = [0; 5];
        if self.bits <= 11 {
            if let Some(window) = self.payload.get(offset..offset + 8) {
                let mut codes = u64::from_le_bytes(window.try_into().unwrap()) >> (bit % 8);
                let mask = (1 << self.bits) - 1;
                for word in &mut out {
                    *word = self.entry((codes & mask) as usize)?;
                    codes >>= self.bits;
                }
                return Ok(out);
            }
        }
        for (j, word) in out.iter_mut().enumerate() {
            *word = self.word(i + j)?;
        }
        Ok(out)
    }

    fn entry(&self, code: usize) -> Result<u16, &'static str> {
        let j = 2 * code;
        let pair = self
            .dictionary
            .get(j..j + 2)
            .ok_or("invalid dictionary index")?;
        Ok(u16::from_le_bytes([pair[0], pair[1]]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn boundaries_and_corruption() {
        for n in [1, 3, 255, 256, 257, 8193, 65536] {
            let raw: Vec<_> = (0..n as u32)
                .chain([0, n as u32 - 1])
                .flat_map(|v| (v as u16).to_le_bytes())
                .collect();
            let frame = encode(&raw).unwrap();
            let view = Frame::open(&frame, raw.len() / 2).unwrap();
            for (i, pair) in raw.as_chunks::<2>().0.iter().enumerate() {
                assert_eq!(view.word(i).unwrap().to_le_bytes(), *pair);
            }
            for (row, group) in raw.as_chunks::<10>().0.iter().enumerate() {
                assert_eq!(
                    view.row5(row).unwrap().map(u16::to_le_bytes).concat(),
                    *group
                );
            }
            assert!(view.row5(raw.len() / 10).is_err());
            assert!(view.row5(usize::MAX).is_err());
            assert!(view.word(raw.len() / 2).is_err());
            assert_eq!(view.decode_range(1, raw.len() / 2 - 1).unwrap(), raw[2..]);
            assert_eq!(view.decode_range(raw.len() / 2, 0).unwrap(), []);
            assert!(view.decode_range(raw.len() / 2 + 1, 0).is_err());
            assert!(view.decode_range(1, usize::MAX).is_err());
            assert!(Frame::open(&frame[..frame.len() - 1], raw.len() / 2).is_err());
        }
        let mut f = encode(&[1, 0, 2, 0, 3, 0]).unwrap();
        f[HEADER + 6] |= 3;
        assert!(Frame::open(&f, 3).unwrap().word(0).is_err());
        assert!(encode(&[0]).is_err());
    }
}
