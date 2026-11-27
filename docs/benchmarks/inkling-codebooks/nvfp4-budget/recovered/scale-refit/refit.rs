use leansonic_core::inkling_mxfp8::{pack_mean_scales, unpack_mean8_scale_codes};
use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
};

// UE4M3 as an integer multiple of 1/512. The sign bit and NaN are invalid here.
fn scale_units(code: u8) -> i64 {
    let exponent = code >> 3;
    let mantissa = i64::from(code & 7);
    if exponent == 0 {
        mantissa
    } else {
        (8 + mantissa) << (exponent - 1)
    }
}

#[derive(Clone, Copy, Default)]
struct Terms {
    residual2: i64,
    residual_times_quant: i64,
    quant2: i64,
}

fn byte_terms(code: u8) -> Terms {
    // Magnitudes measured in half-units; the shared sign cancels in all products.
    let levels = [0, 1, 2, 3, 4, 6, 8, 12];
    let mut terms = Terms::default();
    for mag in [code & 7, (code >> 4) & 7] {
        let original = levels[usize::from(mag)];
        let quant = match mag {
            1 => 0,
            3 => 4,
            _ => original,
        };
        let residual = original - quant;
        terms.residual2 += residual * residual;
        terms.residual_times_quant += residual * quant;
        terms.quant2 += quant * quant;
    }
    terms
}

fn section(file: &mut File, offset: &str, len: usize) -> Result<Vec<u8>, Box<dyn Error>> {
    file.seek(SeekFrom::Start(offset.parse()?))?;
    let mut out = vec![0; len];
    file.read_exact(&mut out)?;
    Ok(out)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [official, wo, wl, so, compact, co, cl, width, cutoff_file, output] = args.as_slice()
    else {
        return Err("expected official, raw weight offset/bytes, raw scale offset, compact, stored scale offset/bytes, row width, cutoffs and output".into());
    };
    let mut src = File::open(official)?;
    let weights = section(&mut src, wo, wl.parse()?)?;
    let count = weights.len() / 8;
    let width: usize = width.parse()?;
    if !matches!(width, 72 | 144)
        || count == 0
        || !count.is_multiple_of(128 * width)
        || weights.len() != 8 * count
    {
        return Err("invalid expert dimensions".into());
    }
    let original = section(&mut src, so, count)?;
    let encoded = section(&mut File::open(compact)?, co, cl.parse()?)?;
    let mut scales = unpack_mean8_scale_codes(&encoded, count, width)?;
    if scales != original || scales.iter().any(|v| !(64..=126).contains(v)) {
        return Err("input scale mismatch or out-of-range UE4M3".into());
    }
    let cuts: [u8; 128] = fs::read(cutoff_file)?
        .try_into()
        .map_err(|_| "expected 128 cutoffs")?;
    let lut: [Terms; 256] = std::array::from_fn(|i| byte_terms(i as u8));
    let unit: [i64; 127] = std::array::from_fn(|i| scale_units(i as u8));
    let mut eligible = 0u64;
    let mut changed = 0u64;
    let (mut before, mut after) = (0u128, 0u128);
    for (expert, bank) in scales.chunks_exact_mut(count / 128).enumerate() {
        let begin = expert * bank.len() * 8;
        for (code, block) in bank.iter_mut().zip(weights[begin..].as_chunks::<8>().0) {
            let cut = cuts[expert];
            if cut == 255 || *code > cut {
                continue;
            }
            eligible += 1;
            let mut t = Terms::default();
            for &b in block {
                let e = lut[usize::from(b)];
                t.residual2 += e.residual2;
                t.residual_times_quant += e.residual_times_quant;
                t.quant2 += e.quant2;
            }
            let scale = unit[usize::from(*code)];
            let start = scale * scale * t.residual2;
            let mut best = start;
            let mut chosen = *code;
            // Rounding 0.5 to 0 and 1.5 to 2 gives a nonpositive cross term,
            // so the convex least-squares optimum is at or below the original scale.
            for trial in (64..*code).rev() {
                let dx = scale - unit[usize::from(trial)];
                let err = start + dx * (2 * scale * t.residual_times_quant + dx * t.quant2);
                if err >= best {
                    break;
                }
                best = err;
                chosen = trial;
            }
            if best < 0 || chosen > cut {
                return Err("invalid scale refit".into());
            }
            changed += u64::from(chosen != *code);
            *code = chosen;
            before += start as u128;
            after += best as u128;
        }
    }
    let mut packed = Vec::new();
    pack_mean_scales(&scales, width, &mut packed)?;
    if unpack_mean8_scale_codes(&packed, count, width)? != scales {
        return Err("scale round-trip failed".into());
    }
    use std::io::Write;
    File::create_new(output)?.write_all(&packed)?;
    println!("{} {eligible} {changed} {before} {after}", packed.len());
    Ok(())
}
