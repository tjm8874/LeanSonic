//! Test whether the existing lossless scales can select smaller NVFP4 streams.
use leansonic_core::inkling_mxfp8::{
    pack_scale_split, unpack_mean8_scale_codes, unpack_nvfp4_trim_context, unpack_scale_split,
};
use std::error::Error;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

fn section(file: &mut File, offset: &str, length: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    file.seek(SeekFrom::Start(offset.parse()?))?;
    let mut bytes = vec![0; length.parse()?];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

// Select one complete 16-weight block in each 64-block stratum. Expert
// boundaries are multiples of 64 blocks for both supported matrix shapes.
fn sample(scales: &[u8], raw: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let mut selected = Vec::with_capacity(scales.len() / 64);
    let mut weights = Vec::with_capacity(raw.len() / 64);
    let mut rng = 0x54abff05d3a21f99_u64;
    for group in (0..scales.len()).step_by(64) {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let block = group + (rng >> 58) as usize;
        selected.push(scales[block]);
        weights.extend_from_slice(&raw[8 * block..8 * block + 8]);
    }
    (selected, weights)
}

fn select_quartiles(
    raw: &[u8],
    scales: &[u8],
    cuts: &[u8; 128],
) -> Result<[u8; 2], Box<dyn Error>> {
    let (small_scales, small_raw) = sample(scales, raw);
    let mut best = (usize::MAX, true, [2, 2]);
    for low in 1..=3 {
        for high in 1..=3 {
            let pair = [low, high];
            let size = leansonic_core::inkling_mxfp8::pack_four_scale_quantiles(
                &small_raw,
                small_scales.clone(),
                cuts,
                pair,
                std::io::sink(),
            )?;
            best = best.min((size, pair != [2, 2], pair));
        }
    }
    Ok(best.2)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    let direct = args.last().is_some_and(|s| s == "--direct");
    let four_banks = args.last().is_some_and(|s| s == "--four-banks");
    let retune = args.last().is_some_and(|s| s == "--retune-four-banks");
    if direct || four_banks || retune {
        args.pop();
    }
    let destination = if args.len() == 9 { args.pop() } else { None };
    let [path, weight_off, weight_len, scale_off, scale_len, row_width, raw_len, cutoff_file] =
        args.as_slice()
    else {
        return Err("expected shard, weight and scale slices, raw length and cutoff file".into());
    };
    let count: usize = raw_len.parse()?;
    if count == 0 || !count.is_multiple_of(8 * 128) {
        return Err("invalid original NVFP4 plane length".into());
    }
    let mut file = File::open(path)?;
    let scale = unpack_mean8_scale_codes(
        &section(&mut file, scale_off, scale_len)?,
        count / 8,
        row_width.parse()?,
    )?;
    let source = section(&mut file, weight_off, weight_len)?;
    let cuts: [u8; 128] = fs::read(cutoff_file)?
        .try_into()
        .map_err(|_| "expected 128 cutoff bytes")?;
    let raw = if four_banks || retune {
        if !source.starts_with(&cuts) {
            return Err("cutoffs differ from source frame".into());
        }
        if retune {
            if !matches!(row_width.parse::<usize>()?, 72 | 144)
                || scale.len() != 128 * 2304 * row_width.parse::<usize>()?
            {
                return Err("expected full Inkling expert matrix for quartile sampling".into());
            }
            leansonic_core::inkling_mxfp8::unpack_four_scale_banks(&source, scale.clone())?
        } else {
            leansonic_core::inkling_mxfp8::unpack_scale_direct(&source, scale.clone())?
        }
    } else {
        unpack_nvfp4_trim_context(&source, count)?
    };
    let quartiles = if retune {
        select_quartiles(&raw, &scale, &cuts)?
    } else {
        [2, 2]
    };
    let frame = if four_banks || retune {
        let mut bytes = Vec::new();
        leansonic_core::inkling_mxfp8::pack_four_scale_quantiles(
            &raw,
            scale.clone(),
            &cuts,
            quartiles,
            &mut bytes,
        )?;
        bytes
    } else if direct {
        let mut bytes = Vec::new();
        leansonic_core::inkling_mxfp8::pack_scale_direct(&raw, scale.clone(), &cuts, &mut bytes)?;
        bytes
    } else {
        pack_scale_split(&raw, &scale, &cuts)?
    };
    let start = Instant::now();
    let roundtrip = if four_banks || retune {
        leansonic_core::inkling_mxfp8::unpack_four_scale_banks(&frame, scale)?
    } else if direct {
        leansonic_core::inkling_mxfp8::unpack_scale_direct(&frame, scale)?
    } else {
        unpack_scale_split(&frame, &scale)?
    };
    let elapsed = start.elapsed().as_secs_f64();
    if roundtrip != raw {
        return Err("changed a reconstructed byte".into());
    }
    if let Some(path) = destination {
        use std::io::Write;
        File::create_new(path)?.write_all(&frame)?;
    }
    if retune {
        println!(
            "{} {} {:.6} {} {}",
            frame.len(),
            roundtrip.len(),
            elapsed,
            quartiles[0],
            quartiles[1]
        );
    } else {
        println!("{} {} {:.6}", frame.len(), roundtrip.len(), elapsed);
    }
    Ok(())
}
