//! Lossy research packer for one raw finite E4M3 expert plane.
use leansonic_core::inkling_mxfp8::{pack, unpack_nvfp4_codes};
use std::{
    env,
    error::Error,
    fs,
    io::{Read, Seek, SeekFrom},
};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [source, offset, len, dest] = args.as_slice() else {
        return Err("expected SHARD OFFSET LENGTH OUTPUT".into());
    };
    let n: usize = len.parse()?;
    if n == 0 || n > 128 * 2304 * 2304 {
        return Err("invalid length".into());
    }
    let mut bytes = vec![0; n];
    let mut input = fs::File::open(source)?;
    input.seek(SeekFrom::Start(offset.parse()?))?;
    input.read_exact(&mut bytes)?;
    let mut hist = [0u64; 256];
    for value in &mut bytes {
        hist[*value as usize] += 1;
        let mag = *value & 127;
        if mag == 127 {
            return Err("nonfinite E4M3 code".into());
        }
        if mag % 2 == 1 {
            let even = if mag & 2 == 0 { mag - 1 } else { mag + 1 };
            *value = (*value & 128) | even;
        }
    }
    let output = fs::File::create_new(dest)?;
    let size = pack(&bytes, std::io::BufWriter::new(output))?;
    let frame = fs::read(dest)?;
    if frame.len() != size || unpack_nvfp4_codes(&frame, n, false)? != bytes {
        return Err("frame roundtrip mismatch".into());
    }
    println!("{size}");
    for (i, v) in hist.iter().enumerate() {
        println!("{i} {v}");
    }
    Ok(())
}
