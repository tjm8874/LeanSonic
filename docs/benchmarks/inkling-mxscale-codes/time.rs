use leansonic_core::inkling_mxfp8::{unpack_mxscale_codes, unpack_nvfp4_codes};
use std::{env, fs, hint::black_box, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [mode, path, scale, count] = args.as_slice() else {
        return Err("MODE FRAME SCALE COUNT".into());
    };
    let count: usize = count.parse()?;
    let frame = fs::read(path)?;
    let scales = if mode == "mx" {
        fs::read(scale)?
    } else {
        Vec::new()
    };
    let start = Instant::now();
    let value = match mode.as_str() {
        "mx" => unpack_mxscale_codes(&frame, unpack_nvfp4_codes(&scales, count / 32, false)?)?,
        "plain" => unpack_nvfp4_codes(&frame, count, false)?,
        _ => return Err("bad mode".into()),
    };
    let elapsed = start.elapsed().as_secs_f64();
    assert_eq!(black_box(&value).len(), count);
    println!("{elapsed:.9}");
    Ok(())
}
