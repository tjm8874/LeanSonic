use leansonic_core::inkling_mxfp8::{unpack_mean8_scale_codes, unpack_scale_direct};
use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
};
fn read(f: &str, start: &str, len: usize) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut f = File::open(f)?;
    f.seek(SeekFrom::Start(start.parse()?))?;
    let mut data = vec![0; len];
    f.read_exact(&mut data)?;
    Ok(data)
}
fn scale(s: u8) -> f64 {
    (1.0 + f64::from(s % 8) / 8.) * 2f64.powi(i32::from(s / 8) - 7)
}
fn magnitude(n: u8) -> f64 {
    let m = [0., 0.5, 1., 1.5, 2., 3., 4., 6.][usize::from(n & 7)];
    if n & 8 == 0 {
        m
    } else {
        -m
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [off, wo, so, base, bo, bl, new, no, nl, po, pl, width, cuts] = args.as_slice() else {
        return Err("bad input".into());
    };
    let width: usize = width.parse()?;
    let n = 128 * 2304 * width;
    let raw = read(off, wo, 8 * n)?;
    let original = read(off, so, n)?;
    let prior = unpack_mean8_scale_codes(&read(base, bo, bl.parse()?)?, n, width)?;
    let codes = unpack_mean8_scale_codes(&read(new, no, nl.parse()?)?, n, width)?;
    let weights = unpack_scale_direct(&read(new, po, pl.parse()?)?, codes.clone())?;
    let cuts: [u8; 128] = fs::read(cuts)?
        .try_into()
        .map_err(|_| "expected 128 cuts")?;
    let mut changed = 0;
    let mut zeros = 0;
    for (expert, cut) in cuts.into_iter().enumerate() {
        for sample in 0..32 {
            let at = expert * (n / 128) + (sample * 104729 + expert * 3571 + 1217) % (n / 128);
            let old = prior[at];
            let new = codes[at];
            let mut best = (f64::INFINITY, u8::MAX, u8::MAX);
            let orig = &raw[8 * at..8 * at + 8];
            let quant = &weights[8 * at..8 * at + 8];
            for trial in 64u8..=126 {
                if cut != 255 && (old <= cut) != (trial <= cut) {
                    continue;
                }
                let mut loss = 0.;
                for (&a, &q) in orig.iter().zip(quant) {
                    for shift in [0, 4] {
                        let v = magnitude((a >> shift) & 15) * scale(original[at]);
                        let y = magnitude((q >> shift) & 15) * scale(trial);
                        loss += (v - y) * (v - y);
                    }
                }
                let score = (loss, old.abs_diff(trial), trial);
                if score < best {
                    best = score;
                }
            }
            assert_eq!(new, best.2, "expert={expert} block={at}");
            changed += usize::from(new != old);
            zeros += usize::from(quant.iter().all(|&v| v == 0));
        }
    }
    println!("4096 {changed} {zeros}");
    Ok(())
}
