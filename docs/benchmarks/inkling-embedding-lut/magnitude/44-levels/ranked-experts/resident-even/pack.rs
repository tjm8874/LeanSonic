use leansonic_core::inkling_mxfp8::{unpack_four_scale_banks, unpack_mean8_scale_codes};
use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
};
fn read(f: &mut File, offset: &str, length: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    f.seek(SeekFrom::Start(offset.parse()?))?;
    let mut b = vec![0; length.parse()?];
    f.read_exact(&mut b)?;
    Ok(b)
}
fn word(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().unwrap())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [path, w, n, s, m, width, output] = args.as_slice() else {
        return Err("expected shard, weights/scales ranges, width and dump flag".into());
    };
    let width: usize = width.parse()?;
    let mut f = File::open(path)?;
    let w = read(&mut f, w, n)?;
    let s = read(&mut f, s, m)?;
    let scales = unpack_mean8_scale_codes(&s, 128 * 2304 * width, width)?;
    let raw = unpack_four_scale_banks(&w, scales.clone())?;
    let size = 2304 * width * 8;
    assert_eq!(raw.len(), 128 * size);
    let mut frame = vec![0u8; 512];
    let mut eligible = 0;
    for (e, expert) in raw.chunks_exact(size).enumerate() {
        let even = expert.iter().all(|&v| v & 0x11 == 0);
        let offset = u32::try_from(frame.len() / 4)?;
        assert!(offset < 1 << 31);
        frame[e * 4..e * 4 + 4].copy_from_slice(&(offset | ((even as u32) << 31)).to_le_bytes());
        if !even {
            frame.extend_from_slice(expert);
            continue;
        }
        eligible += 1;
        for tile in expert.chunks_exact(512) {
            let mut compact = [0u8; 384];
            for (i, g) in tile.chunks_exact(16).enumerate() {
                let d = word(&g[12..16]);
                for j in 0..3 {
                    let a = word(&g[4 * j..4 * j + 4]) | ((d >> (j + 1)) & 0x11111111);
                    compact[128 * j + 4 * i..128 * j + 4 * i + 4].copy_from_slice(&a.to_le_bytes());
                }
            }
            frame.extend_from_slice(&compact);
        }
    }
    // Validate from the serialized descriptor, never from the encoder's flag or offset.
    for (e, expert) in raw.chunks_exact(size).enumerate() {
        let hdr = word(&frame[e * 4..e * 4 + 4]);
        let base = (hdr & 0x7fffffff) as usize * 4;
        if hdr >> 31 == 0 {
            assert_eq!(&frame[base..base + size], expert);
            continue;
        }
        for (i, g) in expert.chunks_exact(16).enumerate() {
            let p = base + (i / 32) * 384 + i % 32 * 4;
            let a = word(&frame[p..p + 4]);
            let b = word(&frame[p + 128..p + 132]);
            let c = word(&frame[p + 256..p + 260]);
            let k = 0x11111111;
            let got = [
                a & !k,
                b & !k,
                c & !k,
                ((a & k) << 1) | ((b & k) << 2) | ((c & k) << 3),
            ];
            for (v, bytes) in got.into_iter().zip(g.chunks_exact(4)) {
                assert_eq!(v, word(bytes))
            }
        }
    }
    println!("{eligible} {} {}", raw.len(), frame.len());
    if output == "save" {
        fs::write("nibbles.bin", raw)?;
        fs::write("mixed.bin", frame)?;
        fs::write("scales.bin", scales)?
    }
    Ok(())
}
