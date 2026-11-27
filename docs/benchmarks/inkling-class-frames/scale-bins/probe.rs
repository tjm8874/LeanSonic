use leansonic_core::inkling_mxfp8::{pack_custom, sample_mean_features, unpack_mean8_scale_codes};
use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{BufWriter, Read, Seek, SeekFrom},
};
fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<_> = env::args().collect();
    let mut file = File::open(&a[1])?;
    file.seek(SeekFrom::Start(a[2].parse()?))?;
    let mut old = vec![0u8; a[3].parse()?];
    file.read_exact(&mut old)?;
    let width: usize = a[4].parse()?;
    let n = 128 * 2304 * width;
    let codes = unpack_mean8_scale_codes(&old, n, width)?;
    if a[5] == "sample" {
        sample_mean_features(&codes, width, BufWriter::new(File::create("features")?))?;
        fs::write("header", &old[..48])?;
        return Ok(());
    }
    let edges = |s: &str| {
        s.split(',')
            .map(str::parse)
            .collect::<Result<Vec<u16>, _>>()
    };
    let mut frame = vec![];
    pack_custom(&codes, width, &edges(&a[6])?, &edges(&a[7])?, &mut frame)?;
    if unpack_mean8_scale_codes(&frame, n, width)? != codes {
        return Err("scale readback mismatch".into());
    }
    println!("{} {} {}", old.len(), frame.len(), n);
    if frame.len() < old.len() {
        fs::write("chosen.frame", frame)?;
    }
    Ok(())
}
