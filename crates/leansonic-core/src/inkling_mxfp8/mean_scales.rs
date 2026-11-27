//! rANS tables selected from the last eight scale bytes to the left and above.
//! Rows and tiles reset the horizontal history. Tiles also reset vertical history.
use super::scale_context::{encode_with_context, valid_shape, ScaleDecoder};
use super::*;

const MAX: usize = 8 * 255;
struct MeanContext {
    left: [u8; MAX + 1],
    above: [u8; MAX + 1],
    columns: usize,
    width: usize,
}
fn scaled_mean(sum: usize, n: usize) -> usize {
    match n {
        0 => 0,
        8 => sum,
        _ => (8 * sum + n / 2) / n,
    }
}
fn features(tile: &[u8], p: usize, first: usize, width: usize) -> [usize; 2] {
    let left = 8.min(p).min((first + p) % width);
    let above = 8.min(p / width);
    let sx = tile[p - left..p].iter().map(|&x| usize::from(x)).sum();
    let sy = (1..=above).map(|i| usize::from(tile[p - i * width])).sum();
    [scaled_mean(sx, left), scaled_mean(sy, above)]
}
impl MeanContext {
    fn new(left: &[u16], above: &[u16], width: usize) -> Self {
        let bins = |edges: &[u16]| {
            std::array::from_fn(|v| edges.partition_point(|&e| usize::from(e) <= v) as u8)
        };
        Self {
            left: bins(left),
            above: bins(above),
            columns: above.len() + 1,
            width,
        }
    }
    fn index(&self, [x, y]: [usize; 2]) -> usize {
        usize::from(self.left[x]) * self.columns + usize::from(self.above[y])
    }
}
impl ScaleDecoder for MeanContext {
    fn decode(
        &self,
        lookup: &[[u32; 4096]],
        bytes: &[u8],
        mut states: [u32; 4],
        first: usize,
        dst: &mut [MaybeUninit<u8>],
    ) -> Result<(), String> {
        let mut row_sum = 0usize;
        let mut row_count = 0usize;
        let mut left = [0u8; 8];
        let mut vertical = vec![0usize; self.width];
        let mut previous = vec![0u8; 8 * self.width];
        let mut ring = 0;
        let mut col = first % self.width;
        let mut cursor = 0;
        for (i, out) in dst.iter_mut().enumerate() {
            let n = (i / self.width).min(8);
            let context = self.index([
                scaled_mean(row_sum, row_count),
                scaled_mean(vertical[col], n),
            ]);
            let state = &mut states[i % 4];
            let entry = lookup[context][(*state & (TOTAL - 1)) as usize];
            let value = (entry >> 24) as u8;
            out.write(value);
            let slot = i % 8;
            if row_count == 8 {
                row_sum -= usize::from(left[slot]);
            } else {
                row_count += 1;
            }
            row_sum += usize::from(value);
            left[slot] = value;
            vertical[col] += usize::from(value);
            vertical[col] -= usize::from(previous[ring]);
            previous[ring] = value;
            ring += 1;
            if ring == previous.len() {
                ring = 0;
            }
            col += 1;
            if col == self.width {
                col = 0;
                row_sum = 0;
                row_count = 0;
            }
            *state = ((entry & 4095) + 1) * (*state >> PRECISION) + ((entry >> 12) & 4095);
            while *state < LOWER {
                *state = (*state << 8)
                    | u32::from(*bytes.get(cursor).ok_or("short mean-scale payload")?);
                cursor += 1;
            }
        }
        if cursor != bytes.len() || states != [LOWER; 4] {
            return Err("invalid mean-scale terminal state".into());
        }
        Ok(())
    }
}
fn boundaries(hist: &[u64; MAX + 1], count: usize) -> Vec<u16> {
    let mut result = Vec::new();
    let mut cumulative = 0;
    let mut next = 1;
    for (i, &n) in hist.iter().enumerate() {
        cumulative += n;
        while next < 12 && cumulative * 12 >= next * count as u64 {
            if result.last() != Some(&(i as u16)) {
                result.push(i as u16);
            }
            next += 1;
        }
    }
    result
}
/// Two 24-byte blocks hold the sorted LE u16 quantile boundaries and zero padding.
pub fn pack_mean_scales(raw: &[u8], width: usize, mut output: impl Write) -> io::Result<usize> {
    if !valid_shape(raw.len(), width) {
        return Err(io::Error::other("invalid mean-scale dimensions"));
    }
    let mut hist = [[0u64; MAX + 1]; 2];
    for (i, tile) in raw.chunks(TILE).enumerate() {
        for p in 0..tile.len() {
            let [x, y] = features(tile, p, i * TILE, width);
            hist[0][x] += 1;
            hist[1][y] += 1;
        }
    }
    let left = boundaries(&hist[0], raw.len());
    let above = boundaries(&hist[1], raw.len());
    for edges in [&left, &above] {
        let mut bytes = [0u8; 24];
        bytes[0] = edges.len() as u8;
        for (dst, e) in bytes[1..].as_chunks_mut::<2>().0.iter_mut().zip(edges) {
            *dst = e.to_le_bytes();
        }
        output.write_all(&bytes)?;
    }
    let ctx = MeanContext::new(&left, &above, width);
    encode_with_context(
        raw,
        output,
        48,
        (left.len() + 1) * (above.len() + 1),
        |tile, p, first| ctx.index(features(tile, p, first, width)),
    )
}
fn read_edges(h: &[u8; 24]) -> Result<Vec<u16>, String> {
    let n = usize::from(h[0]);
    if !(1..=11).contains(&n) || h[1 + 2 * n..].iter().any(|&x| x != 0) {
        return Err("invalid mean-scale header".into());
    }
    let edges: Vec<_> = h[1..1 + 2 * n]
        .as_chunks::<2>()
        .0
        .iter()
        .copied()
        .map(u16::from_le_bytes)
        .collect();
    if edges.windows(2).any(|w| w[0] >= w[1]) || usize::from(edges[n - 1]) > MAX {
        return Err("invalid mean-scale boundaries".into());
    }
    Ok(edges)
}
impl<'a> Matrix<'a> {
    pub(crate) fn parse_mean_scales(
        data: &'a [u8],
        count: usize,
        width: usize,
    ) -> Result<Self, String> {
        if !valid_shape(count, width) {
            return Err("invalid mean-scale dimensions".into());
        }
        let (x, rest) = data
            .split_first_chunk::<24>()
            .ok_or("missing mean-scale header")?;
        let (y, body) = rest
            .split_first_chunk::<24>()
            .ok_or("missing mean-scale header")?;
        let left = read_edges(x)?;
        let above = read_edges(y)?;
        let mut matrix = Self::parse_tables(body, count, (left.len() + 1) * (above.len() + 1))?;
        matrix.scale_context = Some(Box::new(MeanContext::new(&left, &above, width)));
        Ok(matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn restores_split_rows_and_extreme_scales_and_rejects_invalid_headers() {
        for w in [36, 72, 144] {
            let n = (TILE * 2 + 1).next_multiple_of(w);
            let raw: Vec<_> = (0..n)
                .map(|i| match (i / w) % 5 {
                    0 => 0,
                    1 => 255,
                    _ => ((i % w) * 31 + i / w) as u8,
                })
                .collect();
            let mut frame = Vec::new();
            assert_eq!(pack_mean_scales(&raw, w, &mut frame).unwrap(), frame.len());
            let matrix = Matrix::parse_mean_scales(&frame, n, w).unwrap();
            matrix
                .for_each_chunk(|off, b| {
                    assert_eq!(b, &raw[off..off + b.len()]);
                    Ok(())
                })
                .unwrap();
            assert!(Matrix::parse_mean_scales(&frame[..frame.len() - 8], n, w).is_err());
            frame[2] = 255;
            assert!(Matrix::parse_mean_scales(&frame, n, w).is_err());
        }
    }
}
