//! Re-encode a single NVFP4 plane and compare every decoded byte before saving it.
use leansonic_core::inkling_mxfp8::{
    pack_nvfp4_trim_context, unpack_nvfp4_codes, unpack_nvfp4_trim_context,
};
use std::{
    env,
    error::Error,
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [source, offset, length, count, output] = args.as_slice() else {
        return Err("expected SOURCE OFFSET ENCODED_BYTES DECODED_BYTES OUTPUT".into());
    };
    let mut src = File::open(source)?;
    src.seek(SeekFrom::Start(offset.parse()?))?;
    let mut old = vec![0; length.parse()?];
    src.read_exact(&mut old)?;
    let count = count.parse()?;
    let raw = unpack_nvfp4_codes(&old, count, true)?;
    let mut packed = Vec::new();
    let size = pack_nvfp4_trim_context(&raw, &mut packed)?;
    if size != packed.len() || unpack_nvfp4_trim_context(&packed, count)? != raw {
        return Err("NVFP4 frame did not reconstruct every original byte".into());
    }
    if size < old.len() {
        File::create_new(output)?.write_all(&packed)?;
    }
    println!("{size}");
    Ok(())
}
