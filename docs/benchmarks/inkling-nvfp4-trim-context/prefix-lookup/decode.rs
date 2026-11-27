use leansonic_core::inkling_mxfp8::unpack_nvfp4_trim_context;
use std::{env, error::Error, fs::File, io::{Read, Seek, SeekFrom}, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [path, offset, size, decoded] = args.as_slice() else { return Err("expected input, offset, length and raw length".into()); };
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(offset.parse()?))?;
    let mut frame = vec![0; size.parse()?];
    file.read_exact(&mut frame)?;
    let length = decoded.parse()?;
    let expected = unpack_nvfp4_trim_context(&frame, length)?;
    let fingerprint = expected.iter().fold(14695981039346656037u64, |acc, &x| {
        (acc ^ u64::from(x)).wrapping_mul(1099511628211)
    });
    for run in 0..3 {
        let start = Instant::now();
        let raw = unpack_nvfp4_trim_context(&frame, length)?;
        let elapsed = start.elapsed().as_secs_f64();
        assert_eq!(raw, expected);
        println!("{run},{length},{elapsed:.9},{fingerprint:016x}");
    }
    Ok(())
}
