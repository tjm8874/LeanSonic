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
    let [source, offset, len, dest, options @ ..] = args.as_slice() else {
        return Err("expected SHARD OFFSET LENGTH OUTPUT [CUTOFF]".into());
    };
    let lut: [u8; 256] = match options {
        [flag, file] if flag == "--lut" => {
            let table: [u8; 256] = fs::read(file)?
                .try_into()
                .map_err(|_| "expected 256 lookup entries")?;
            if (0..256).any(|i| i & 127 != 127 && table[i] & 127 == 127) {
                return Err("lookup table produces nonfinite E4M3".into());
            }
            table
        }
        [] | [_] => {
            let cutoff: u8 = if let Some(s) = options.first() {
                s.parse()?
            } else {
                125
            };
            if cutoff > 125 {
                return Err("cutoff exceeds the largest odd finite magnitude".into());
            }
            std::array::from_fn(|i| {
                let value = i as u8;
                let mag = value & 127;
                if mag % 2 == 1 && mag <= cutoff {
                    (value & 128) | if mag & 2 == 0 { mag - 1 } else { mag + 1 }
                } else {
                    value
                }
            })
        }
        _ => return Err("expected CUTOFF or --lut FILE".into()),
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
        *value = lut[*value as usize];
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
