//! Four Engram rows share 320 Q4 weight bytes and two 25-byte planes of ten-bit indices.
use super::{EngramBytes, ShardedSafeTensors};

pub(super) type EngramRow<'a> = (&'a [u8; 80], [u16; 5], [u16; 5]);

pub(super) struct TiledShard {
    pub(super) storage: EngramBytes,
    layout: Layout,
}

struct Layout {
    scale_end: usize,
    bias_end: usize,
}

impl Layout {
    fn parse(data: &[u8], expected: usize) -> Result<Self, String> {
        if data.len() < 16 || &data[..8] != b"Q38TILE4" {
            return Err("invalid Engram tile header".into());
        }
        let rows = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let scales = u16::from_le_bytes(data[12..14].try_into().unwrap()) as usize;
        let biases = u16::from_le_bytes(data[14..16].try_into().unwrap()) as usize;
        let scale_end = 16 + 2 * scales;
        let bias_end = scale_end + 2 * biases;
        if rows != expected
            || rows == 0
            || !rows.is_multiple_of(4)
            || !(1..=1024).contains(&scales)
            || !(1..=1024).contains(&biases)
            || (rows / 4)
                .checked_mul(370)
                .and_then(|n| n.checked_add(bias_end))
                != Some(data.len())
        {
            return Err("invalid Engram tile dimensions".into());
        }
        Ok(Self {
            scale_end,
            bias_end,
        })
    }

    fn row<'a>(&self, data: &'a [u8], row: usize) -> Result<EngramRow<'a>, String> {
        let tiles = data[self.bias_end..].as_chunks::<370>().0;
        let tile = tiles.get(row / 4).ok_or("Engram tile row out of range")?;
        let slot = row % 4;
        let weights = tile[..320].as_chunks::<80>().0.get(slot).unwrap();
        let scales = parameters(&tile[320..345], slot, &data[16..self.scale_end])?;
        let biases = parameters(&tile[345..], slot, &data[self.scale_end..self.bias_end])?;
        Ok((weights, scales, biases))
    }
}

fn parameters(indices: &[u8], slot: usize, dictionary: &[u8]) -> Result<[u16; 5], String> {
    let mut words = [0; 5];
    for (i, word) in words.iter_mut().enumerate() {
        let bit = (slot * 5 + i) * 10;
        let off = bit / 8;
        let pair = u16::from_le_bytes([indices[off], indices[off + 1]]);
        let index = ((pair >> (bit % 8)) & 1023) as usize;
        let value = dictionary
            .as_chunks::<2>()
            .0
            .get(index)
            .ok_or("invalid Engram tile dictionary index")?;
        *word = u16::from_le_bytes(*value);
    }
    Ok(words)
}

impl TiledShard {
    pub(super) fn open(
        st: &ShardedSafeTensors,
        name: &str,
        expected: usize,
    ) -> Result<Self, String> {
        let value = st.get(name).ok_or_else(|| format!("missing {name}"))?;
        if value.dtype != "U8" || value.shape != [value.data.len()] {
            return Err(format!("invalid Engram tile type or shape: {name}"));
        }
        let layout = Layout::parse(value.data, expected)?;
        let range = st
            .tensor_range(name)
            .ok_or_else(|| format!("missing range for {name}"))?;
        Ok(Self {
            storage: EngramBytes::Mapped(range),
            layout,
        })
    }
    pub(super) fn bytes<'a>(&'a self, st: &'a ShardedSafeTensors) -> &'a [u8] {
        self.storage.get(st)
    }
    pub(super) fn row<'a>(
        &'a self,
        st: &'a ShardedSafeTensors,
        row: usize,
    ) -> Result<EngramRow<'a>, String> {
        self.layout.row(self.bytes(st), row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checks_tile_bounds_and_dictionary_codes() {
        let mut data = Vec::from(*b"Q38TILE4");
        data.extend(4u32.to_le_bytes());
        data.extend([1, 0, 1, 0, 23, 0, 45, 0]);
        data.resize(20 + 370, 0);
        data[20 + 3 * 80] = 123;
        let layout = Layout::parse(&data, 4).unwrap();
        let (w, s, b) = layout.row(&data, 3).unwrap();
        assert_eq!(w[0], 123);
        assert_eq!(s, [23; 5]);
        assert_eq!(b, [45; 5]);
        assert!(layout.row(&data, 4).is_err());
        data[20 + 369] = 255;
        assert!(layout.row(&data, 3).is_err());
        assert!(Layout::parse(&data[..data.len() - 1], 4).is_err());
        assert!(Layout::parse(&data, 8).is_err());
    }
}
