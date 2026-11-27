// Fixed canonical codes, ordered by length. Rare BF16 exponents carry a
// literal after the escape. The low stream stores sign and fraction.
use std::{mem::MaybeUninit, sync::LazyLock};
const EXP: [u8; 12] = [126, 127, 125, 124, 123, 128, 122, 121, 120, 119, 118, 117];
const WIDTH: [u8; 13] = [2, 2, 2, 3, 5, 5, 5, 6, 7, 8, 9, 10, 10];
const TILE: usize = 128 * 1024;

fn codes() -> [(u32, u8); 13] {
    let mut out = [(0, 0); 13];
    let mut next = 0u32;
    let mut prev = 0;
    for (i, &n) in WIDTH.iter().enumerate() {
        next <<= n - prev;
        out[i] = (next.reverse_bits() >> (32 - n), n);
        next += 1;
        prev = n;
    }
    assert_eq!(next, 1 << prev);
    out
}
static ENCODE: LazyLock<[(u32, u8); 256]> = LazyLock::new(|| {
    let c = codes();
    std::array::from_fn(|i| {
        EXP.iter()
            .position(|&x| x as usize == i)
            .map_or((c[12].0 | ((i as u32) << c[12].1), c[12].1 + 8), |j| c[j])
    })
});
static SINGLE: LazyLock<[(u8, u8); 1024]> = LazyLock::new(|| {
    let c = codes();
    std::array::from_fn(|v| {
        let j = c
            .iter()
            .position(|&(bits, n)| v as u32 & ((1 << n) - 1) == bits)
            .unwrap();
        (j as u8, WIDTH[j])
    })
});
#[derive(Clone, Copy)]
struct Group {
    parts: [u8; 8],
    bits: u8,
}
static GROUP: LazyLock<[Group; 4096]> = LazyLock::new(|| {
    std::array::from_fn(|val| {
        let mut bits = 0;
        let mut parts = [0; 8];
        for d in parts.chunks_exact_mut(2) {
            let (j, n) = SINGLE[(val >> bits) & 1023];
            if j == 12 || bits + n as usize > 12 {
                return Group {
                    parts: [0; 8],
                    bits: 0,
                };
            }
            let exp = EXP[j as usize];
            d[0] = exp << 7;
            d[1] = exp >> 1;
            bits += n as usize;
        }
        Group {
            parts,
            bits: bits as u8,
        }
    })
});
pub fn encode(src: &[u8]) -> Vec<u8> {
    assert!(!src.is_empty() && src.len().is_multiple_of(8));
    let mut out = Vec::new();
    for tile in src.chunks(TILE) {
        let start = out.len();
        out.extend_from_slice(&[0; 4]);
        let n = tile.len() / 2;
        out.resize(start + 4 + n, 0);
        let mut bits = 0u32;
        let mut count = 0;
        for (i, &[lo, hi]) in tile.as_chunks::<2>().0.iter().enumerate() {
            out[start + 4 + i] = (lo & 127) | (hi & 128);
            let (v, len) = ENCODE[((hi << 1) | (lo >> 7)) as usize];
            bits |= v << count;
            count += len;
            while count >= 8 {
                out.push(bits as u8);
                bits >>= 8;
                count -= 8;
            }
        }
        if count > 0 {
            out.push(bits as u8);
        }
        let bytes = (out.len() - start - 4) as u32;
        out[start..start + 4].copy_from_slice(&bytes.to_le_bytes());
    }
    out.resize(out.len().next_multiple_of(8), 0);
    out
}
fn tile(frame: &[u8], dst: &mut [MaybeUninit<u8>]) -> Result<(), String> {
    let (frac, codes) = frame
        .split_at_checked(dst.len() / 2)
        .ok_or("short fractions")?;
    let mut bit = 0;
    for (ff, out) in frac
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<8>().0)
    {
        let tail = codes.get(bit / 8..).ok_or("short exponent codes")?;
        // Four escaped values can consume 72 bits. A group that hits the
        // fallback reads a fresh window per symbol.
        let window = |tail: &[u8], bit: usize| {
            let bytes = if let Some((a, _)) = tail.split_first_chunk::<4>() {
                *a
            } else {
                let mut a = [0; 4];
                a[..tail.len()].copy_from_slice(tail);
                a
            };
            u32::from_le_bytes(bytes) >> (bit % 8)
        };
        let e = GROUP[(window(tail, bit) & 4095) as usize];
        let mut parts = e.parts;
        if e.bits > 0 {
            bit += e.bits as usize;
        } else {
            for ex in parts.chunks_exact_mut(2) {
                let w = window(codes.get(bit / 8..).ok_or("short exponent code")?, bit);
                let (j, n) = SINGLE[(w & 1023) as usize];
                bit += n as usize;
                let v = if j == 12 {
                    bit += 8;
                    (w >> n) as u8
                } else {
                    EXP[j as usize]
                };
                ex[0] = v << 7;
                ex[1] = v >> 1;
            }
        }
        for i in 0..4 {
            out[2 * i].write(parts[2 * i] | (ff[i] & 127));
            out[2 * i + 1].write(parts[2 * i + 1] | (ff[i] & 128));
        }
    }
    if bit.div_ceil(8) != codes.len()
        || (!bit.is_multiple_of(8) && codes[bit / 8] >> (bit % 8) != 0)
    {
        return Err("invalid exponent padding or extent".into());
    }
    Ok(())
}
pub fn decode(mut src: &[u8], dst: &mut [u8]) -> Result<(), String> {
    if dst.is_empty() || !dst.len().is_multiple_of(8) || !src.len().is_multiple_of(8) {
        return Err("bad dimensions".into());
    }
    let mut tiles = Vec::new();
    for n in (0..dst.len()).step_by(TILE) {
        let (length, rest) = src.split_first_chunk::<4>().ok_or("missing tile")?;
        let count = (dst.len() - n).min(TILE) / 2;
        let length = u32::from_le_bytes(*length) as usize;
        if !(count + (2 * count).div_ceil(8)..=count + (18 * count).div_ceil(8)).contains(&length) {
            return Err("bad tile size".into());
        }
        let (f, rest) = rest.split_at_checked(length).ok_or("short tile")?;
        tiles.push(f);
        src = rest;
    }
    if src.len() >= 8 || src.iter().any(|b| *b != 0) {
        return Err("bad frame padding".into());
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(8));
    let width = tiles.len().div_ceil(threads);
    let uninit = unsafe {
        std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<MaybeUninit<u8>>(), dst.len())
    };
    std::thread::scope(|s| {
        let mut tasks = Vec::new();
        for (frames, chunk) in tiles.chunks(width).zip(uninit.chunks_mut(width * TILE)) {
            tasks.push(s.spawn(move || -> Result<(), String> {
                for (&f, out) in frames.iter().zip(chunk.chunks_mut(TILE)) {
                    tile(f, out)?;
                }
                Ok(())
            }));
        }
        for t in tasks {
            t.join().map_err(|_| "worker panicked")??;
        }
        Ok(())
    })
}
