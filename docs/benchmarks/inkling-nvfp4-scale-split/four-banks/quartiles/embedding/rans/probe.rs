use leansonic_core::{
    inkling_embedding::pack_embedding,
    inkling_mxfp8::{pack, unpack_nvfp4_codes},
};
use std::{
    env,
    error::Error,
    fs::File,
    io::{Read, Seek, SeekFrom},
    time::Instant,
};

const WIDTH: usize = 2304;
fn encode(x: &[u8]) -> Result<(usize, Vec<u8>), Box<dyn Error>> {
    let mut bytes = Vec::new();
    pack(x, &mut bytes)?;
    let decoded = unpack_nvfp4_codes(&bytes, x.len(), false)?;
    assert_eq!(x, decoded);
    Ok((bytes.len(), decoded))
}
fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let mut f = File::open(args.next().unwrap())?;
    let start: u64 = args.next().unwrap().parse()?;
    for first in [0, 32768, 65536, 98304, 131072, 163840, 199680] {
        f.seek(SeekFrom::Start(start + first * WIDTH as u64 * 2))?;
        let mut raw = vec![0u8; 1024 * WIDTH * 2];
        f.read_exact(&mut raw)?;
        let n = raw.len() / 2;
        let prefix = pack_embedding(&raw[..], raw.len(), std::io::sink())?;
        for mode in ["bytes", "fields", "row", "col", "row-col"] {
            let timer = Instant::now();
            let center_rows = mode.starts_with("row");
            let center_cols = mode.ends_with("col");
            let mut col = vec![0i16; WIDTH];
            if center_cols {
                let mut sums = vec![0i64; WIDTH];
                for (i, b) in raw.as_chunks::<2>().0.iter().enumerate() {
                    sums[i % WIDTH] += i64::from((u16::from_le_bytes(*b) >> 7) as u8);
                }
                for (out, s) in col.iter_mut().zip(sums) {
                    *out = ((s + 512) / 1024) as i16;
                }
            }
            let mut predictors = Vec::new();
            let mut fraction = Vec::with_capacity(n);
            let mut residual = Vec::with_capacity(n);
            for row in raw.chunks_exact(2 * WIDTH) {
                let words = row.as_chunks::<2>().0;
                let mean = if center_rows {
                    let sum: i64 = words
                        .iter()
                        .zip(&col)
                        .map(|(b, c)| {
                            i64::from((u16::from_le_bytes(*b) >> 7) as u8) - i64::from(*c)
                        })
                        .sum();
                    sum.div_euclid(WIDTH as i64) as u8
                } else {
                    0
                };
                if center_rows {
                    predictors.push(mean);
                }
                for (&b, &c) in words.iter().zip(&col) {
                    if mode == "bytes" {
                        fraction.push(b[0]);
                        residual.push(b[1]);
                    } else {
                        fraction.push((b[0] & 127) | (b[1] & 128));
                        let e = (u16::from_le_bytes(b) >> 7) as u8;
                        residual.push(e.wrapping_sub(mean).wrapping_sub(c as u8));
                    }
                }
            }
            let (a, fr) = encode(&fraction)?;
            let (b, exp) = encode(&residual)?;
            let (c, rowmeans) = if center_rows {
                encode(&predictors)?
            } else {
                (0, vec![])
            };
            let (d, colmeans) = if center_cols {
                encode(&col.iter().map(|&v| v as u8).collect::<Vec<_>>())?
            } else {
                (0, vec![])
            };
            for (i, original) in raw.as_chunks::<2>().0.iter().enumerate() {
                let recovered = if mode == "bytes" {
                    [fr[i], exp[i]]
                } else {
                    let e = exp[i]
                        .wrapping_add(if center_rows { rowmeans[i / WIDTH] } else { 0 })
                        .wrapping_add(if center_cols { colmeans[i % WIDTH] } else { 0 });
                    [(fr[i] & 127) | (e << 7), (fr[i] & 128) | (e >> 1)]
                };
                assert_eq!(&recovered, original);
            }
            let frame =
                8 * (2 + usize::from(center_rows) + usize::from(center_cols)) + a + b + c + d;
            println!(
                "{first},{mode},{prefix},{frame},{a},{b},{c},{d},{:.4}",
                timer.elapsed().as_secs_f64()
            );
        }
    }
    Ok(())
}
