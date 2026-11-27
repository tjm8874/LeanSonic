use leansonic_core::inkling_mxfp8::{unpack_mean8_scale_codes, unpack_scale_direct};
use std::{
    env,
    error::Error,
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
};
fn section(f: &mut File, at: &str, size: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    f.seek(SeekFrom::Start(at.parse()?))?;
    let mut data = vec![0; size.parse()?];
    f.read_exact(&mut data)?;
    Ok(data)
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [file, wo, wl, so, sl, width] = args.as_slice() else {
        return Err("expected final shard, frame offsets/lengths, width".into());
    };
    let width: usize = width.parse()?;
    let n = 128 * 2304 * width;
    let mut f = File::open(file)?;
    let sf = section(&mut f, so, sl)?;
    let wf = section(&mut f, wo, wl)?;
    let scales = unpack_mean8_scale_codes(&sf, n, width)?;
    let raw = unpack_scale_direct(&wf, scales.clone())?;
    let mut packed = Vec::with_capacity(n * 4);
    let mut populations = [0u64; 3];
    for bytes in raw.as_chunks::<8>().0 {
        let mut word = 0u32;
        for i in 0..16 {
            let nib = (bytes[i / 2] >> (4 * (i % 2))) & 15;
            let code = match nib {
                0 => 0,
                7 => 1,
                15 => 3,
                _ => return Err("non-ternary nibble".into()),
            };
            populations[match code {
                0 => 0,
                1 => 1,
                _ => 2,
            }] += 1;
            word |= code << (2 * i);
        }
        packed.extend_from_slice(&word.to_le_bytes());
    }
    for (&x, &y) in raw.iter().zip(
        packed
            .iter()
            .flat_map(|&p| [p & 15, p >> 4])
            .collect::<Vec<_>>()
            .iter(),
    ) {
        let values = [0, 7, 0xff, 15];
        assert_eq!(
            x,
            values[usize::from(y & 3)] | (values[usize::from(y >> 2)] << 4)
        );
    }
    assert_eq!(raw.len(), 2 * packed.len());
    for (path, data) in [
        ("nibbles.bin", raw),
        ("ternary.bin", packed),
        ("scales.bin", scales),
    ] {
        File::create_new(path)?.write_all(&data)?;
    }
    println!("{} {} {}", populations[0], populations[1], populations[2]);
    Ok(())
}
