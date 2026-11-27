use leansonic_core::inkling_mxfp8::{unpack_four_scale_banks, unpack_mean8_scale_codes};
use std::{
    env,
    error::Error,
    fs::File,
    io::{Read, Seek, SeekFrom},
};

fn slice(file: &mut File, at: &str, size: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    file.seek(SeekFrom::Start(at.parse()?))?;
    let mut output = vec![0; size.parse()?];
    file.read_exact(&mut output)?;
    Ok(output)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [parent, candidate, ao, al, bo, bl, so, sl, width] = args.as_slice() else {
        return Err(
            "expected two model paths, frame offsets/sizes, scale slice and row width".into(),
        );
    };
    let mut old = File::open(parent)?;
    let mut new = File::open(candidate)?;
    let width: usize = width.parse()?;
    assert!(matches!(width, 72 | 144));
    let scales = unpack_mean8_scale_codes(&slice(&mut old, so, sl)?, 128 * 2304 * width, width)?;
    let a = slice(&mut old, ao, al)?;
    let b = slice(&mut new, bo, bl)?;
    assert_eq!(a[..128], b[..128]);
    let moved = a[128..384]
        .iter()
        .zip(&b[128..384])
        .filter(|(x, y)| x != y)
        .count();
    let x = unpack_four_scale_banks(&a, scales.clone())?;
    let y = unpack_four_scale_banks(&b, scales)?;
    assert_eq!(x.len(), 128 * 2304 * width * 8);
    assert_eq!(x, y);
    println!("{} {moved}", x.len());
    Ok(())
}
