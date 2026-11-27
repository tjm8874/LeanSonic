use leansonic_core::inkling_mxfp8::{
    benchmark_prior_class_frame, unpack_class_scale_banks, unpack_four_scale_banks,
    unpack_mean8_scale_codes,
};
use std::{env, fs, io::{self, Write}, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    let width: usize = args[3].parse()?;
    let frame = fs::read(&args[2])?;
    let scale_frame = fs::read(&args[4])?;
    let scales = unpack_mean8_scale_codes(&scale_frame, 128*2304*width, width)?;
    drop(scale_frame);
    let now = Instant::now();
    let weights = match args[1].as_str() {
        "control" => unpack_four_scale_banks(&frame, scales)?,
        "old" => benchmark_prior_class_frame(&frame, &scales)?,
        "new" => unpack_class_scale_banks(&frame, &scales)?,
        _ => return Err("invalid mode".into()),
    };
    eprintln!("{}", now.elapsed().as_micros());
    assert_eq!(weights.len(), 128*2304*width*8);
    io::stdout().lock().write_all(&weights)?;
    Ok(())
}
