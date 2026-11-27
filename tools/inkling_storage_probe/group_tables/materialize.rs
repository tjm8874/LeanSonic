//! Materialize the fitted bytes from the distributable compact frame and side table.
use leansonic_core::inkling_mxfp8::{pack, unpack_nvfp4_codes};
use std::{
    env,
    error::Error,
    fs,
    io::{BufWriter, Read, Seek, SeekFrom},
};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [source, offset, size, columns, parity, table, dest] = args.as_slice() else {
        return Err("expected shard offset size columns parity table dest".into());
    };
    let cols: usize = columns.parse()?;
    let parity: usize = parity.parse()?;
    if !matches!((cols, parity), (2304, 2) | (1152, 1)) {
        return Err("unexpected expert plane shape".into());
    }
    let mut file = fs::File::open(source)?;
    file.seek(SeekFrom::Start(offset.parse()?))?;
    let mut frame = vec![0u8; size.parse()?];
    file.read_exact(&mut frame)?;
    let mut lookup = [u8::MAX; 256];
    let mut symbols = 0;
    for (code, f) in frame
        .get(..512)
        .ok_or("short frame")?
        .chunks_exact(2)
        .enumerate()
    {
        if f != [0, 0] {
            lookup[code] = symbols;
            symbols += 1;
            if symbols > 16 {
                return Err("too many symbols".into());
            }
        }
    }
    let mut bytes = unpack_nvfp4_codes(&frame, 128 * 2304 * cols, false)?;
    let tables = fs::read(table)?;
    if tables.len() != 128 * 2304 / 16 * parity * 16 || tables.iter().any(|x| x & 127 == 127) {
        return Err("invalid table".into());
    }
    for (row, values) in bytes.chunks_exact_mut(cols).enumerate() {
        let group = (row / 16) * parity + row % parity;
        let lut = &tables[group * 16..group * 16 + 16];
        for x in values {
            *x = lut[lookup[*x as usize] as usize];
        }
    }
    let size = pack(&bytes, BufWriter::new(fs::File::create_new(dest)?))?;
    let packed = fs::read(dest)?;
    if packed.len() != size || unpack_nvfp4_codes(&packed, bytes.len(), false)? != bytes {
        return Err("roundtrip failed".into());
    }
    println!("{size}");
    Ok(())
}
