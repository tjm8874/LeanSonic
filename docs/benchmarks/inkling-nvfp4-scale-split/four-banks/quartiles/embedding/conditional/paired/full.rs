mod conditional;
mod pair;
use leansonic_core::inkling_embedding::unpack_embedding;
use std::{env, error::Error, fs, mem::MaybeUninit, time::Instant};
fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<_> = env::args().collect();
    let raw = fs::read(&a[1])?;
    let prefix = fs::read(&a[2])?;
    let cond = fs::read(&a[3])?;
    assert_eq!(raw.len(), 201024 * 2304 * 2);
    // Exercise a lone partial frame and a partial second frame with every BF16 pattern.
    for n in [32 + 128 * 1024, 2 * 128 * 1024 + 16] {
        let sample: Vec<_> = (0..=65535u16)
            .cycle()
            .flat_map(u16::to_le_bytes)
            .take(n)
            .collect();
        for mode in 0..2 {
            let mut encoded = Vec::new();
            if mode == 0 {
                pair::pack_embedding(&sample[..], n, &mut encoded)?;
            } else {
                conditional::pack_embedding(&sample[..], n, &mut encoded)?;
            }
            let mut decoded = vec![0; n];
            let storage = unsafe {
                std::slice::from_raw_parts_mut(decoded.as_mut_ptr() as *mut MaybeUninit<u8>, n)
            };
            if mode == 0 {
                pair::unpack_embedding(&encoded, storage)?;
            } else {
                conditional::unpack_embedding(&encoded, storage)?;
            }
            assert_eq!(decoded, sample);
        }
    }
    let mut dst = vec![0; raw.len()];
    let orders = [
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [1, 0, 2],
        [0, 2, 1],
    ];
    for (c, order) in orders.iter().cycle().take(12).enumerate() {
        for (l, &arm) in order.iter().enumerate() {
            dst.fill(if l % 2 == 0 { 0x55 } else { 0xaa });
            let storage = unsafe {
                std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut MaybeUninit<u8>, dst.len())
            };
            let now = Instant::now();
            match arm {
                0 => unpack_embedding(&prefix, storage)?,
                1 => pair::unpack_embedding(&prefix, storage)?,
                _ => conditional::unpack_embedding(&cond, storage)?,
            }
            let secs = now.elapsed().as_secs_f64();
            assert_eq!(raw.as_slice(), dst.as_slice());
            println!("{c},{l},{arm},{secs:.9}");
        }
    }
    Ok(())
}
