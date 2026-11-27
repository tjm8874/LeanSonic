use leansonic_core::inkling_mxfp8::{unpack_mean8_scale_codes, unpack_scale_direct};
use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
};

fn read(path: &str, offset: &str, size: usize) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(offset.parse()?))?;
    let mut bytes = vec![0; size];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}
fn scale(code: u8) -> f64 {
    (1. + f64::from(code % 8) / 8.) * 2f64.powi(i32::from(code / 8) - 7)
}
const MAG: [f64; 8] = [0., 0.5, 1., 1.5, 2., 3., 4., 6.];
fn reduced(original: &[u8; 16], source: u8, allowed: &[u8], penalty: i32, cut: u8) -> (u8, [u8; 16]) {
    let mut best = (f64::INFINITY, u8::MAX, u8::MAX);
    let mut chosen = [0; 16];
    let range = if penalty < 0 { source.saturating_sub(2).max(64)..=source.saturating_add(6).min(126) } else {64..=126};
    for trial in range {
        let mut loss = 0.;
        let mut codes = [0; 16];
        for (slot, &n) in codes.iter_mut().zip(original) {
            let original_value = MAG[usize::from(n & 7)] * scale(source);
            let mut best_error = f64::INFINITY;
            for &code in allowed {
                let value = MAG[usize::from(code)];
                let error = (value * scale(trial) - original_value).powi(2) + if code == 0 {0.} else { f64::from(penalty.max(0)) * (scale(source) * 0.5).powi(2) };
                if error < best_error {
                    best_error = error;
                    *slot = if code == 0 { 0 } else { (n & 8) | code };
                }
            }
            loss += best_error;
        }
        let key = (loss, source.abs_diff(trial), trial);
        if key < best {
            best = key;
            chosen = codes;
        }
    }
    if penalty < 0 {
        let old = best.2;
        let mut refit = (f64::INFINITY, 0, old);
        for c in 64u8..=126 {
            if cut != 255 && (c <= cut) != (old <= cut) {continue}
            let err = original.iter().zip(chosen).map(|(x,y)| (MAG[usize::from(x&7)]*scale(source)-MAG[usize::from(y&7)]*scale(c)).powi(2)).sum();
            let key = (err,old.abs_diff(c),c);
            if key < refit {refit=key;}
        }
        return (refit.2,chosen);
    }
    (best.2, chosen)
}
fn nibbles(bytes: &[u8]) -> [u8; 16] {
    std::array::from_fn(|i| (bytes[i / 2] >> (4 * (i % 2))) & 15)
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [src, wo, so, out, cwo, cwl, cso, csl, width, cutfile, mode, palette, penalty] = args.as_slice()
    else {
        return Err(
            "expected official, offsets, final file, two frame offsets/lengths, width, cuts, mode"
                .into(),
        );
    };
    let mut allowed = Vec::new();
    for digit in palette.chars() {
        let value = match digit {
            'h' => 0.5,
            '0' => 0.,
            '1' => 1.,
            '2' => 2.,
            '3' => 3.,
            '4' => 4.,
            '6' => 6.,
            _ => return Err("invalid alphabet".into()),
        };
        allowed.push(MAG.iter().position(|&x| x == value).unwrap() as u8);
    }
    assert!(allowed.starts_with(&[0]) && allowed.windows(2).all(|w| w[0] < w[1]));
    let width: usize = width.parse()?;
    if !matches!(width, 72 | 144) || !matches!(mode.as_str(), "control" | "low" | "all") {
        return Err("bad width or mode".into());
    }
    let n = 128 * 2304 * width;
    let orig = read(src, wo, n * 8)?;
    let original_scales = read(src, so, n)?;
    let scales = unpack_mean8_scale_codes(&read(out, cso, csl.parse()?)?, n, width)?;
    let final_weights = unpack_scale_direct(&read(out, cwo, cwl.parse()?)?, scales.clone())?;
    let cuts: [u8; 128] = fs::read(cutfile)?
        .try_into()
        .map_err(|_| "expected 128 cutoffs")?;
    let mut counts = [0usize; 3];
    for (expert, cut) in cuts.into_iter().enumerate() {
        for sample in 0..32 {
            let pos = expert * (n / 128) + (sample * 104729 + expert * 3571 + 911) % (n / 128);
            let original = nibbles(&orig[8 * pos..]);
            let source = original_scales[pos];
            let selected = cut != 255 && source <= cut;
            let expected = if mode == "all" || (mode == "low" && selected) {
                counts[2] += 1;
                reduced(&original, source, &allowed, penalty.parse()?, cut)
            } else {
                counts[usize::from(selected)] += 1;
                (
                    source,
                    original.map(|n| {
                        let m = n & 7;
                        let mag = if selected {
                            [0, 0, 2, 4, 4, 5, 6, 7][usize::from(m)]
                        } else {
                            m
                        };
                        if mag == 0 {
                            0
                        } else {
                            mag | (n & 8)
                        }
                    }),
                )
            };
            assert_eq!(
                expected,
                (scales[pos], nibbles(&final_weights[8 * pos..])),
                "expert={expert}, block={pos}"
            );
        }
    }
    println!("{} {} {}", counts[0], counts[1], counts[2]);
    Ok(())
}
