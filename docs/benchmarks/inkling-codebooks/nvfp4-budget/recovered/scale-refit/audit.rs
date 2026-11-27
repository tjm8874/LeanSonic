use leansonic_core::inkling_mxfp8::unpack_mean8_scale_codes;
use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
};

fn read(path: &str, offset: &str, count: usize) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(offset.parse()?))?;
    let mut data = vec![0; count];
    f.read_exact(&mut data)?;
    Ok(data)
}
// Evaluate actual dequantized values independently, including both signs.
fn scale(b: u8) -> f64 {
    (1.0 + f64::from(b % 8) / 8.0) * 2f64.powi(i32::from(b / 8) - 7)
}
fn score(block: &[u8], source: u8, trial: u8) -> f64 {
    let mag = [0., 0.5, 1., 1.5, 2., 3., 4., 6.];
    let rounded = [0., 0., 1., 2., 2., 3., 4., 6.];
    let mut loss = 0.;
    for &byte in block {
        for nibble in [byte & 15, byte >> 4] {
            let sign = if nibble >= 8 { -1. } else { 1. };
            let i = usize::from(nibble & 7);
            let delta = sign * mag[i] * scale(source) - sign * rounded[i] * scale(trial);
            loss += delta * delta;
        }
    }
    loss
}
fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<_> = env::args().skip(1).collect();
    let [official, wo, wl, so, new, co, cl, width, cuts] = a.as_slice() else {
        return Err("invalid input".into());
    };
    let weights = read(official, wo, wl.parse()?)?;
    let n = weights.len() / 8;
    let old = read(official, so, n)?;
    let codes = unpack_mean8_scale_codes(&read(new, co, cl.parse()?)?, n, width.parse()?)?;
    let limits = fs::read(cuts)?;
    let per = n / 128;
    let (mut same, mut optimized, mut eligible) = (0, 0, 0);
    for (e, &cut) in limits.iter().enumerate() {
        for j in 0..32 {
            let pos = e * per + ((j * 104729 + e * 3571) % per);
            let s = old[pos];
            let v = codes[pos];
            let block = &weights[pos * 8..pos * 8 + 8];
            if s > cut || cut == 255 {
                assert_eq!(v, s);
                same += 1;
                continue;
            }
            eligible += 1;
            let mut best = s;
            let mut loss = score(block, s, s);
            for trial in 64..=cut {
                let candidate = score(block, s, trial);
                if candidate < loss || (candidate == loss && s.abs_diff(trial) < s.abs_diff(best)) {
                    best = trial;
                    loss = candidate;
                }
            }
            assert_eq!(v, best, "expert={e} position={pos} original={s}");
            if v != s {
                optimized += 1;
            }
        }
    }
    println!("{same} {eligible} {optimized}");
    Ok(())
}
