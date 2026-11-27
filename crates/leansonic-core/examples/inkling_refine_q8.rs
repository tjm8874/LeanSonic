//! Evaluate wider Q8 block scales on the original attention weights. Keep the native decoder.
use leansonic_core::{inkling2_gpu::pack_q8_rows, inkling_mxfp8, inkling_q8_scales};
use std::{
    env,
    error::Error,
    fs,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

fn write_new(path: &str, body: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    fs::File::create_new(path)?.write_all(body)
}

fn error(weights: &[f32], codes: &[u8], scale: f32) -> f64 {
    weights
        .iter()
        .zip(codes)
        .map(|(&w, &q)| {
            let delta = f64::from(w) - f64::from(q as i8) * f64::from(scale);
            delta * delta
        })
        .sum()
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [mode, shard, offset, size, code_out, scale_out] = args.as_slice() else {
        return Err(
            "expected control|refine|rate SHARD BYTE_OFFSET ELEMENTS CODES_OUT SCALES_OUT".into(),
        );
    };
    if !matches!(mode.as_str(), "control" | "refine" | "rate")
        || Path::new(code_out).exists()
        || Path::new(scale_out).exists()
    {
        return Err("unknown mode or output path already exists".into());
    }
    let size: usize = size.parse()?;
    if size == 0 || size > 3072 * 3072 || !size.is_multiple_of(64) {
        return Err("invalid attention matrix size".into());
    }
    let mut file = fs::File::open(shard)?;
    file.seek(SeekFrom::Start(offset.parse()?))?;
    let mut source = vec![0; size * 2];
    file.read_exact(&mut source)?;
    let mut codes = vec![0; size];
    let mut scales = vec![0; size / 64 * 4];
    pack_q8_rows(&source, 64, &mut codes, &mut scales);
    // Cross entropy under the parent tensor's code histogram. Pseudocounts keep
    // previously unseen bytes finite. Each block has the same 64-symbol length.
    let mut histogram = [1u64; 256];
    if mode == "rate" {
        for &value in &codes {
            histogram[value as usize] += 1;
        }
    }
    let symbol_cost = histogram.map(|count| -(count as f64).log2());
    let cost = |bytes: &[u8]| bytes.iter().map(|&q| symbol_cost[q as usize]).sum::<f64>();
    let mut old_error = 0.;
    let mut new_error = 0.;
    let mut counts = [0u64; 5];
    for ((raw, codes), word) in source
        .chunks_exact(128)
        .zip(codes.chunks_exact_mut(64))
        .zip(scales.chunks_exact_mut(4))
    {
        let mut w = [0.; 64];
        let mut peak = 0u16;
        for (i, b) in raw.chunks_exact(2).enumerate() {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            peak = peak.max(bits & 0x7fff);
            w[i] = f32::from_bits(u32::from(bits) << 16);
        }
        if peak >= 0x7f80 {
            return Err("nonfinite BF16 weight".into());
        }
        let mut best_scale = f32::from_le_bytes(word.try_into()?);
        let mut best = error(&w, codes, best_scale);
        old_error += best;
        let original_error = best;
        let mut best_cost = if mode == "rate" { cost(codes) } else { 0. };
        let mut choice = 0;
        if mode != "control" && peak != 0 {
            for (i, offset) in [1, 2, 4, 8].iter().enumerate() {
                let cap_bits = peak + offset;
                if cap_bits >= 0x7f80 {
                    continue;
                }
                let cap = f32::from_bits(u32::from(cap_bits) << 16);
                let scale = cap * (1. / 127.);
                let inv = 127. / cap;
                let trial = w.map(|v| (v * inv).round().clamp(-127., 127.) as i8 as u8);
                let err = error(&w, &trial, scale);
                let trial_cost = if mode == "rate" { cost(&trial) } else { 0. };
                let improves = if mode == "rate" {
                    err <= original_error
                        && (trial_cost < best_cost || (trial_cost == best_cost && err < best))
                } else {
                    err < best
                };
                if improves {
                    best_cost = trial_cost;
                    best = err;
                    best_scale = scale;
                    codes.copy_from_slice(&trial);
                    choice = i + 1;
                }
            }
        }
        counts[choice] += 1;
        new_error += best;
        word.copy_from_slice(&best_scale.to_le_bytes());
    }
    let mut frame = Vec::new();
    inkling_mxfp8::pack(&codes, &mut frame)?;
    if inkling_mxfp8::unpack_nvfp4_codes(&frame, size, false)? != codes {
        return Err("Q8 codes failed round trip".into());
    }
    let dictionary = inkling_q8_scales::pack(&scales)?;
    if inkling_q8_scales::unpack(&dictionary, size / 64)? != scales {
        return Err("scale dictionary failed round trip".into());
    }
    write_new(code_out, &frame)?;
    write_new(scale_out, &dictionary)?;
    println!("{{\"old_sse\":{old_error},\"new_sse\":{new_error},\"cap_histogram\":{counts:?}}}");
    Ok(())
}
