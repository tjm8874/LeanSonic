use leansonic_core::inkling_mxfp8::{
    unpack_four_scale_banks, unpack_mean8_scale_codes, unpack_scale_direct,
};
use std::{
    env,
    error::Error,
    fs::File,
    io::{Read, Seek, SeekFrom},
};

fn section(file: &mut File, offset: &str, len: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    file.seek(SeekFrom::Start(offset.parse()?))?;
    let mut result = vec![0; len.parse()?];
    file.read_exact(&mut result)?;
    Ok(result)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [old, new, wo, wl, no, nl, so, sl, width, codec] = args.as_slice() else {
        return Err("expected models, frame locations, scale location, width and codec".into());
    };
    let width: usize = width.parse()?;
    assert!(matches!(width, 72 | 144));
    let mut parent = File::open(old)?;
    let scales =
        unpack_mean8_scale_codes(&section(&mut parent, so, sl)?, 128 * 2304 * width, width)?;
    let old_frame = section(&mut parent, wo, wl)?;
    let new_frame = section(&mut File::open(new)?, no, nl)?;
    let reference = unpack_scale_direct(&old_frame, scales.clone())?;
    let decoded = match &codec[..] {
        "direct" => unpack_scale_direct(&new_frame, scales)?,
        "four" => unpack_four_scale_banks(&new_frame, scales)?,
        _ => return Err("unrecognized frame format".into()),
    };
    assert_eq!(reference.len(), 128 * 2304 * width * 8);
    if reference != decoded {
        return Err("decoded weight mismatch".into());
    }
    println!("{}", reference.len());
    Ok(())
}
