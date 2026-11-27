use leansonic_core::inkling_mxfp8::{
    pack_four_scale_quantiles, unpack_four_scale_banks, unpack_mean8_scale_codes,
};
use std::{
    env,
    error::Error,
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
};

fn read(file: &mut File, at: &str, size: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    file.seek(SeekFrom::Start(at.parse()?))?;
    let mut data = vec![0; size.parse()?];
    file.read_exact(&mut data)?;
    Ok(data)
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

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [src, wo, wl, so, sl, width, out] = args.as_slice() else {
        return Err("expected shard, offsets, lengths, row width, output".into());
    };
    let width: usize = width.parse()?;
    assert!(matches!(width, 72 | 144));
    let mut file = File::open(src)?;
    let scales = unpack_mean8_scale_codes(&read(&mut file, so, sl)?, 128 * 2304 * width, width)?;
    let old = read(&mut file, wo, wl)?;
    let base: &[u8; 128] = old[..128].try_into()?;
    let raw = unpack_four_scale_banks(&old, scales.clone())?;
    let (small_scales, small_raw) = sample(&scales, &raw);
    let mut trials = Vec::new();
    for low in 1..=3 {
        for high in 1..=3 {
            let pair = [low, high];
            let size = pack_four_scale_quantiles(
                &small_raw,
                small_scales.clone(),
                base,
                pair,
                std::io::sink(),
            )?;
            trials.push((size, pair != [2, 2], pair));
        }
    }
    let (_, _, pair) = *trials.iter().min().unwrap();
    let mut frame = Vec::new();
    let size = pack_four_scale_quantiles(&raw, scales.clone(), base, pair, &mut frame)?;
    assert_eq!(size, frame.len());
    assert_eq!(raw, unpack_four_scale_banks(&frame, scales)?);
    if pair == [2, 2] {
        assert_eq!(old, frame);
    }
    if size < old.len() {
        File::create_new(out)?.write_all(&frame)?;
    }
    println!(
        "{} {} {} {} {} {}",
        old.len(),
        size,
        raw.len(),
        pair[0],
        pair[1],
        trials
            .iter()
            .map(|x| x.0.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    Ok(())
}
