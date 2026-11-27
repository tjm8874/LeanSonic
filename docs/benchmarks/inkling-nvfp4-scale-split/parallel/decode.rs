use leansonic_core::inkling_mxfp8::{unpack_mean8_scale_codes, unpack_scale_split};
use std::{env, error::Error, fs::File, io::{Read, Seek, SeekFrom}, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [path, offset, size, decoded, scale_off, scale_len, width] = args.as_slice() else { return Err("expected input, offset, length and raw length".into()); };
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(offset.parse()?))?;
    let mut frame = vec![0; size.parse()?];
    file.read_exact(&mut frame)?;
    let length: usize = decoded.parse()?;
    file.seek(SeekFrom::Start(scale_off.parse()?))?;
    let mut packed_scales = vec![0; scale_len.parse()?];
    file.read_exact(&mut packed_scales)?;
    let scales = unpack_mean8_scale_codes(&packed_scales, length / 8, width.parse()?)?;
    let expected = unpack_scale_split(&frame, &scales)?;
    let fingerprint = expected.iter().fold(14695981039346656037u64, |acc, &x| {
        (acc ^ u64::from(x)).wrapping_mul(1099511628211)
    });
    for run in 0..3 {
        let start = Instant::now();
        let raw = unpack_scale_split(&frame, &scales)?;
        let elapsed = start.elapsed().as_secs_f64();
        assert_eq!(raw, expected);
        println!("{run},{length},{elapsed:.9},{fingerprint:016x}");
    }
    Ok(())
}
