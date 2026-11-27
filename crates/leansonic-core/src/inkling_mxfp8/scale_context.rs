//! Lossless expert scales conditioned on the preceding scale in the row and
//! the scale directly above. Each tile starts with empty history.
use super::*;

pub(super) struct ScaleContext {
    bins: [u8; 256],
    categories: usize,
    width: usize,
}

impl ScaleContext {
    fn new(edges: &[u8], width: usize) -> Self {
        Self {
            bins: std::array::from_fn(|i| edges.partition_point(|&e| usize::from(e) <= i) as u8),
            categories: edges.len() + 1,
            width,
        }
    }

    fn context(&self, tile: &[u8], pos: usize, first: usize) -> usize {
        let left = if pos != 0 && !(first + pos).is_multiple_of(self.width) {
            self.bins[tile[pos - 1] as usize]
        } else {
            0
        };
        let above = if pos >= self.width {
            self.bins[tile[pos - self.width] as usize]
        } else {
            0
        };
        usize::from(left) * self.categories + usize::from(above)
    }
}

pub(super) trait ScaleDecoder: Send + Sync {
    fn decode(
        &self,
        lookup: &[[u32; 4096]],
        bytes: &[u8],
        states: [u32; 4],
        first: usize,
        dst: &mut [MaybeUninit<u8>],
    ) -> Result<(), String>;
}

impl ScaleDecoder for ScaleContext {
    fn decode(
        &self,
        lookup: &[[u32; 4096]],
        bytes: &[u8],
        mut states: [u32; 4],
        first: usize,
        dst: &mut [MaybeUninit<u8>],
    ) -> Result<(), String> {
        // Each slot holds the most recent decoded bin in its column. Clearing
        // the ring at each tile boundary makes independent parallel decoding possible.
        let mut previous_row = vec![0u8; self.width];
        let mut column = first % self.width;
        let mut left = 0;
        let mut cursor = 0;
        for (i, out) in dst.iter_mut().enumerate() {
            let context = usize::from(left) * self.categories + usize::from(previous_row[column]);
            let x = &mut states[i % 4];
            let entry = lookup[context][(*x & (TOTAL - 1)) as usize];
            let value = (entry >> 24) as u8;
            out.write(value);
            left = self.bins[value as usize];
            previous_row[column] = left;
            column += 1;
            if column == self.width {
                column = 0;
                left = 0;
            }
            *x = ((entry & 4095) + 1) * (*x >> PRECISION) + ((entry >> 12) & 4095);
            while *x < LOWER {
                let byte = bytes
                    .get(cursor)
                    .ok_or("short scale context normalization payload")?;
                *x = (*x << 8) | u32::from(*byte);
                cursor += 1;
            }
        }
        if cursor != bytes.len() || states != [LOWER; 4] {
            return Err("invalid scale context terminal state".into());
        }
        Ok(())
    }
}

pub(super) fn valid_shape(count: usize, width: usize) -> bool {
    matches!(width, 36 | 72 | 144)
        && count != 0
        && count <= u32::MAX as usize
        && count.is_multiple_of(width)
}

/// The first 16 bytes contain the boundary count, sorted quantiles and zero padding.
pub fn pack_scale_context(raw: &[u8], width: usize, mut output: impl Write) -> io::Result<usize> {
    if !valid_shape(raw.len(), width) {
        return Err(io::Error::other("invalid scale context dimensions"));
    }
    let mut hist = [0u64; 256];
    for &v in raw {
        hist[usize::from(v)] += 1;
    }
    let mut cumulative = 0;
    let mut next = 1;
    let mut edges = Vec::new();
    for (v, count) in hist.into_iter().enumerate() {
        cumulative += count;
        while next < 12 && cumulative * 12 >= next * (raw.len() as u64) {
            if edges.last() != Some(&(v as u8)) {
                edges.push(v as u8);
            }
            next += 1;
        }
    }
    let mut header = [0; 16];
    header[0] = edges.len() as u8;
    header[1..1 + edges.len()].copy_from_slice(&edges);
    output.write_all(&header)?;
    let ctx = ScaleContext::new(&edges, width);
    encode_with_context(
        raw,
        output,
        16,
        ctx.categories * ctx.categories,
        |tile, p, first| ctx.context(tile, p, first),
    )
}

pub(super) fn encode_with_context(
    raw: &[u8],
    mut output: impl Write,
    header_bytes: usize,
    n_tables: usize,
    index: impl Fn(&[u8], usize, usize) -> usize,
) -> io::Result<usize> {
    let mut counts = vec![[0; 256]; n_tables];
    for (tile_index, tile) in raw.chunks(TILE).enumerate() {
        for (p, &value) in tile.iter().enumerate() {
            counts[index(tile, p, tile_index * TILE)][usize::from(value)] += 1;
        }
    }
    let mut tables = Vec::with_capacity(counts.len());
    for h in &counts {
        let freq = normalize(h);
        let base = write_table(&freq, &mut output)?;
        tables.push((freq, base));
    }
    let mut first = 0;
    write_frames(
        raw,
        output,
        header_bytes + tables.len() * 512,
        |tile, states, payload| {
            for (p, &v) in tile.iter().enumerate().rev() {
                let (freq, base) = &tables[index(tile, p, first)];
                encode_symbol(
                    &mut states[p % 4],
                    freq[v as usize],
                    base[v as usize],
                    payload,
                );
            }
            first += tile.len();
        },
    )
}

impl<'a> Matrix<'a> {
    pub(crate) fn parse_scales(data: &'a [u8], count: usize, width: usize) -> Result<Self, String> {
        if !valid_shape(count, width) {
            return Err("invalid scale context dimensions".into());
        }
        let (header, body) = data
            .split_first_chunk::<16>()
            .ok_or("missing scale context header")?;
        let n = usize::from(header[0]);
        if !(1..=11).contains(&n) {
            return Err("invalid scale context bin count".into());
        }
        let edges = &header[1..1 + n];
        if edges.windows(2).any(|w| w[0] >= w[1]) || header[n + 1..].iter().any(|&b| b != 0) {
            return Err("invalid scale context boundaries".into());
        }
        let ctx = ScaleContext::new(edges, width);
        let mut matrix = Self::parse_tables(body, count, ctx.categories * ctx.categories)?;
        matrix.scale_context = Some(Box::new(ctx));
        Ok(matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_rows_and_tile_boundaries_restore_exact_scales() {
        for width in [36, 72, 144] {
            let n = (2 * TILE + 17).next_multiple_of(width);
            let raw: Vec<_> = (0..n)
                .map(|i| ((i % width) * 41 + i / width / 7) as u8)
                .collect();
            let mut frame = Vec::new();
            assert_eq!(
                pack_scale_context(&raw, width, &mut frame).unwrap(),
                frame.len()
            );
            let matrix = Matrix::parse_scales(&frame, n, width).unwrap();
            matrix
                .for_each_chunk(|offset, tile| {
                    assert_eq!(tile, &raw[offset..offset + tile.len()]);
                    Ok(())
                })
                .unwrap();
            let mut tail = vec![MaybeUninit::uninit(); n - TILE];
            matrix.decode_at(TILE, &mut tail).unwrap();
            assert!(tail
                .iter()
                .zip(&raw[TILE..])
                .all(|(v, b)| unsafe { v.assume_init() == *b }));
            assert!(Matrix::parse_scales(&frame[..frame.len() - 8], n, width).is_err());
            frame[15] = 1;
            assert!(Matrix::parse_scales(&frame, n, width).is_err());
        }
    }
}
