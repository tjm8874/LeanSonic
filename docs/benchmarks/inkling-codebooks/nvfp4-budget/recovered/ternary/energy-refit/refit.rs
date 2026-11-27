use leansonic_core::inkling_mxfp8::{
    pack_mean_scales, unpack_mean8_scale_codes, unpack_scale_direct,
};
use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{Read, Seek, SeekFrom, Write},
};

fn read(file: &mut File, at: &str, count: usize) -> Result<Vec<u8>, Box<dyn Error>> {
    file.seek(SeekFrom::Start(at.parse()?))?;
    let mut bytes = vec![0; count];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}
const MAG: [i64; 8] = [0, 1, 2, 3, 4, 6, 8, 12];
fn scale(s: u8) -> i64 {
    (8 + i64::from(s & 7)) << ((s >> 3) - 1)
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [official, wo, wl, so, parent, co, cl, po, pl, width, cuts, out] = args.as_slice() else {
        return Err("bad argument count".into());
    };
    let mut src = File::open(official)?;
    let mut stored = File::open(parent)?;
    let raw = read(&mut src, wo, wl.parse()?)?;
    let n = raw.len() / 8;
    let width: usize = width.parse()?;
    assert!(matches!(width, 72 | 144));
    assert_eq!(n, 128 * 2304 * width);
    let original = read(&mut src, so, n)?;
    let mut scales = unpack_mean8_scale_codes(&read(&mut stored, co, cl.parse()?)?, n, width)?;
    let frame = read(&mut stored, po, pl.parse()?)?;
    let weights = unpack_scale_direct(&frame, scales.clone())?;
    let cuts: [u8; 128] = fs::read(cuts)?
        .try_into()
        .map_err(|_| "expected 128 cuts")?;
    let mut eligible = 0u64;
    let mut changed = 0u64;
    let (mut before, mut after) = (0u128, 0u128);
    let mut energy = [0u128; 3];
    for (idx, (&source, code)) in original.iter().zip(scales.iter_mut()).enumerate() {
        let cut = cuts[idx / (2304 * width)];
        let old = *code;
        assert!((64..=126).contains(&source) && (64..=126).contains(&old));
        let (lo, hi) = if cut == 255 {
            (64, 126)
        } else if old <= cut {
            (64, cut)
        } else {
            (64.max(cut + 1), 126)
        };
        let (mut aa, mut bb) = (0i64, 0i64);
        for (&original, &ternary) in raw[idx * 8..idx * 8 + 8]
            .iter()
            .zip(&weights[idx * 8..idx * 8 + 8])
        {
            for shift in [0, 4] {
                let a = (original >> shift) & 15;
                let b = (ternary >> shift) & 15;
                assert!(b == 0 || (b & 7 == 7 && (a & 8) == (b & 8)));
                let x = MAG[usize::from(a & 7)];
                let y = MAG[usize::from(b & 7)];
                aa += x * x;
                bb += y * y;
            }
        }
        let os = scale(source);
        let error = |c: u8| {
            let s = scale(c);
            (aa * os * os - bb * s * s).abs()
        };
        let start = error(old);
        let mut best = (start, 0u8, old);
        if bb > 0 {
            eligible += 1;
            // Binary search for the two feasible codes bracketing the exact real optimum.
            let mut left = lo;
            let mut right = hi;
            while left < right {
                let mid = left + (right - left) / 2;
                if scale(mid).pow(2) * bb < aa * os.pow(2) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            for c in [left, left.saturating_sub(1).max(lo)] {
                best = best.min((error(c), old.abs_diff(c), c));
            }
        }
        assert!(best.0 >= 0 && best.0 <= start);
        *code = best.2;
        changed += u64::from(*code != old);
        energy[0] += (aa * os.pow(2)) as u128;
        energy[1] += (bb * scale(old).pow(2)) as u128;
        energy[2] += (bb * scale(*code).pow(2)) as u128;
        before += start as u128;
        after += best.0 as u128;
    }
    let mut encoded = vec![];
    pack_mean_scales(&scales, width, &mut encoded)?;
    let decoded = unpack_mean8_scale_codes(&encoded, n, width)?;
    assert_eq!(decoded, scales);
    if unpack_scale_direct(&frame, decoded)? != weights {
        return Err("reused frame decoded different weights".into());
    }
    File::create_new(out)?.write_all(&encoded)?;
    println!(
        "{} {eligible} {changed} {before} {after} {} {} {}",
        encoded.len(),
        energy[0],
        energy[1],
        energy[2]
    );
    Ok(())
}
