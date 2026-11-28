use leansonic_core::inkling2_gpu::pack_q8_rows;
use std::{
    env,
    error::Error,
    fs,
    io::{Read, Seek, SeekFrom, Write},
};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [filename, offset, codes, scales] = args.as_slice() else {
        return Err("expected SHARD OFFSET CODES SCALES".into());
    };
    let size = 2304 * 3072;
    let mut raw = vec![0; size * 2];
    let mut source = fs::File::open(filename)?;
    source.seek(SeekFrom::Start(offset.parse()?))?;
    source.read_exact(&mut raw)?;
    let mut q = vec![0; size];
    let mut s = vec![0; size / 64 * 4];
    pack_q8_rows(&raw, 3072, &mut q, &mut s);
    fs::File::create_new(codes)?.write_all(&q)?;
    fs::File::create_new(scales)?.write_all(&s)?;
    Ok(())
}
