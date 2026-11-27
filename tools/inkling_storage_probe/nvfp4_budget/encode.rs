use leansonic_core::inkling_mxfp8::{
    canonicalize_nvfp4_zeros, pack_nvfp4_context, unpack_nvfp4_codes,
};
use std::{
    env,
    error::Error,
    fs,
    io::{BufWriter, Read, Seek, SeekFrom},
};
fn read_range(f: &mut fs::File, offset: u64, n: usize) -> std::io::Result<Vec<u8>> {
    let mut v = vec![0; n];
    f.seek(SeekFrom::Start(offset))?;
    f.read_exact(&mut v)?;
    Ok(v)
}
fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<_> = env::args().skip(1).collect();
    let [path, wo, so, n, cut, out] = a.as_slice() else {
        return Err("usage: official weight_offset scale_offset byte_count cutoffs output".into());
    };
    let mut f = fs::File::open(path)?;
    let n: usize = n.parse()?;
    if ![128 * 2304 * 1152, 128 * 2304 * 576].contains(&n) {
        return Err("invalid plane".into());
    }
    let mut v = read_range(&mut f, wo.parse()?, n)?;
    let scales = read_range(&mut f, so.parse()?, n / 8)?;
    let limits = fs::read(cut)?;
    if limits.len() != 128 {
        return Err("128 cutoffs required".into());
    }
    canonicalize_nvfp4_zeros(&mut v);
    let trim_nibble = |x: u8| match x & 7 {
        0 | 1 => 0,
        3 => (x & 8) | 4,
        _ => x,
    };
    let trim: [u8; 256] = std::array::from_fn(|i| {
        let x = i as u8;
        trim_nibble(x & 15) | (trim_nibble(x >> 4) << 4)
    });
    let mut changed = 0u64;
    for (expert, groups) in v.chunks_exact_mut(n / 128).enumerate() {
        let base = expert * (n / 128 / 8);
        let limit = limits[expert];
        for (i, w) in groups.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            if limit != 255 && scales[base + i] <= limit {
                for x in w {
                    let y = trim[*x as usize];
                    changed += u64::from((y & 15) != (*x & 15)) + u64::from((y >> 4) != (*x >> 4));
                    *x = y;
                }
            }
        }
    }
    let len = pack_nvfp4_context(&v, BufWriter::new(fs::File::create_new(out)?))?;
    let data = fs::read(out)?;
    if data.len() != len || unpack_nvfp4_codes(&data, n, true)? != v {
        return Err("frame verification failed".into());
    }
    println!("{len} {changed}");
    Ok(())
}
