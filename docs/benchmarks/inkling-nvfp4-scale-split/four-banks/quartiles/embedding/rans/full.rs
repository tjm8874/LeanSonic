use leansonic_core::{
    inkling_embedding::{pack_embedding, unpack_embedding},
    inkling_mxfp8::{pack, unpack_nvfp4_codes},
};
use std::{env, error::Error, fs, io::Write, mem::MaybeUninit, time::Instant};
const TILE: usize = 1024 * 2304 * 2;

fn pack_fields(raw: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut dst = Vec::new();
    for tile in raw.chunks(TILE) {
        let (mut frac, mut exp) = (
            Vec::with_capacity(tile.len() / 2),
            Vec::with_capacity(tile.len() / 2),
        );
        for &[lo, hi] in tile.as_chunks::<2>().0 {
            frac.push((lo & 127) | (hi & 128));
            exp.push((hi << 1) | (lo >> 7));
        }
        let header = dst.len();
        dst.extend_from_slice(&[0; 16]);
        let n = pack(&frac, &mut dst)?;
        let m = pack(&exp, &mut dst)?;
        dst[header..header + 8].copy_from_slice(&(n as u64).to_le_bytes());
        dst[header + 8..header + 16].copy_from_slice(&(m as u64).to_le_bytes());
    }
    Ok(dst)
}
fn unpack_fields(mut src: &[u8], dst: &mut [u8]) -> Result<(), String> {
    let mut frames = Vec::new();
    for _ in 0..dst.len().div_ceil(TILE) {
        let (header, rest) = src
            .split_first_chunk::<16>()
            .ok_or("missing field lengths")?;
        let lengths = header.as_chunks::<8>().0;
        let frame_size = |v| usize::try_from(u64::from_le_bytes(v)).map_err(|_| "invalid length");
        let (f, rest) = rest
            .split_at_checked(frame_size(lengths[0])?)
            .ok_or("truncated fractions")?;
        let (e, rest) = rest
            .split_at_checked(frame_size(lengths[1])?)
            .ok_or("truncated exponents")?;
        frames.push((f, e));
        src = rest;
    }
    if !src.is_empty() || frames.is_empty() {
        return Err("invalid field package".into());
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(8));
    let group = frames.len().div_ceil(threads);
    std::thread::scope(|s| {
        let mut tasks = Vec::new();
        for (frames, out) in frames.chunks(group).zip(dst.chunks_mut(group * TILE)) {
            tasks.push(s.spawn(move || -> Result<(), String> {
                for (&(f, e), tile) in frames.iter().zip(out.chunks_mut(TILE)) {
                    let frac = unpack_nvfp4_codes(f, tile.len() / 2, false)?;
                    let exp = unpack_nvfp4_codes(e, tile.len() / 2, false)?;
                    for ((word, f), e) in tile.as_chunks_mut::<2>().0.iter_mut().zip(frac).zip(exp)
                    {
                        *word = [(f & 127) | (e << 7), (f & 128) | (e >> 1)];
                    }
                }
                Ok(())
            }));
        }
        for task in tasks {
            task.join().map_err(|_| "field worker panicked")??;
        }
        Ok(())
    })
}
fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<String> = env::args().collect();
    let raw = fs::read(&a[1])?;
    assert_eq!(raw.len(), 201024 * 2304 * 2);
    let prefix = fs::read(&a[2])?;
    let mut encoded = Vec::new();
    pack_embedding(&raw[..], raw.len(), &mut encoded)?;
    assert_eq!(encoded, prefix);
    drop(encoded);
    let fields = pack_fields(&raw)?;
    fs::File::create_new("fields.bin")?.write_all(&fields)?;
    let mut dst = vec![0; raw.len()];
    for cycle in 0..6 {
        let order = if cycle % 2 == 0 {
            ["prefix", "fields", "fields", "prefix"]
        } else {
            ["fields", "prefix", "prefix", "fields"]
        };
        for (leg, kind) in order.iter().enumerate() {
            dst.fill(if leg % 2 == 0 { 0x55 } else { 0xaa });
            let start = Instant::now();
            if *kind == "fields" {
                unpack_fields(&fields, &mut dst)?;
            } else {
                // dst is already fully initialized and the decoder only writes bytes.
                let out = unsafe {
                    std::slice::from_raw_parts_mut(
                        dst.as_mut_ptr().cast::<MaybeUninit<u8>>(),
                        dst.len(),
                    )
                };
                unpack_embedding(&prefix, out)?;
            }
            let elapsed = start.elapsed().as_secs_f64();
            assert_eq!(
                dst.as_slice(),
                raw.as_slice(),
                "embedding differs in {kind}"
            );
            println!(
                "{cycle},{leg},{kind},{},{elapsed:.9}",
                if *kind == "fields" {
                    fields.len()
                } else {
                    prefix.len()
                }
            );
        }
    }
    Ok(())
}
