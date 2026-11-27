use leansonic_core::inkling_mxfp8::{
    pack_mean_scales, pack_scale_direct, unpack_mean8_scale_codes, unpack_scale_direct,
};
use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};
include!("palette.rs");
fn slice(f: &mut File, start: &str, bytes: usize) -> Result<Vec<u8>, Box<dyn Error>> {
    f.seek(SeekFrom::Start(start.parse()?))?;
    let mut v = vec![0; bytes];
    f.read_exact(&mut v)?;
    Ok(v)
}
fn store(path: &str, raw: &[u8]) -> Result<(), Box<dyn Error>> {
    File::create_new(Path::new(path))?.write_all(raw)?;
    Ok(())
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [file, wo, so, width, cuts, mode, weight_out, scale_out] = args.as_slice() else {
        return Err(
            "expected official shard, offsets, row width, cutoffs, mode and two outputs".into(),
        );
    };
    if !matches!(mode.as_str(), "control" | "low" | "all") {
        return Err("invalid mode".into());
    }
    let width: usize = width.parse()?;
    if !matches!(width, 72 | 144) {
        return Err("invalid row width".into());
    }
    let cuts: [u8; 128] = fs::read(cuts)?
        .try_into()
        .map_err(|_| "128 cutoffs required")?;
    let thresholds = cuts.map(|x| if x == 255 { 0 } else { x + 1 });
    let n = 128 * 2304 * width;
    let mut f = File::open(file)?;
    let mut weights = slice(&mut f, wo, n * 8)?;
    let mut scales = slice(&mut f, so, n)?;
    let parent = env::var("TERNARY_NONZERO_COST").as_deref() == Ok("-1");
    let options: Vec<_> = (64..=126).map(trials).collect();
    let mut count = 0u64;
    for (expert, group) in weights.chunks_exact_mut(2304 * width * 8).enumerate() {
        let offset = expert * 2304 * width;
        for (j, block) in group.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let s = &mut scales[offset + j];
            if !(64..=126).contains(s) {
                return Err("out-of-range scale".into());
            }
            let selected = cuts[expert] != 255 && *s <= cuts[expert];
            let map = if mode == "all" || (mode == "low" && selected) {
                let mut h = [0i64; 8];
                for &v in block.iter() {
                    h[usize::from(v & 7)] += 1;
                    h[usize::from((v >> 4) & 7)] += 1;
                }
                let best = options[usize::from(*s - 64)]
                    .iter()
                    .min_by_key(|t| (0..8).map(|i| h[i] * t.err[i]).sum::<i64>())
                    .unwrap();
                count += 1;
                *s = if parent { refit(&h, *s, &best.mag, best.code, cuts[expert]) } else { best.code };
                best.mag
            } else if selected {
                TRIM
            } else {
                [0, 1, 2, 3, 4, 5, 6, 7]
            };
            for b in block {
                *b = sign(*b, map[usize::from(*b & 7)])
                    | (sign(*b >> 4, map[usize::from((*b >> 4) & 7)]) << 4);
            }
        }
    }
    let (mut w, mut sc) = (vec![], vec![]);
    pack_scale_direct(&weights, scales.clone(), &thresholds, &mut w)?;
    pack_mean_scales(&scales, width, &mut sc)?;
    let decoded = unpack_mean8_scale_codes(&sc, n, width)?;
    if decoded != scales || unpack_scale_direct(&w, decoded)? != weights {
        return Err("full roundtrip mismatch".into());
    }
    store(weight_out, &w)?;
    store(scale_out, &sc)?;
    println!("{} {} {count}", w.len(), sc.len());
    Ok(())
}
