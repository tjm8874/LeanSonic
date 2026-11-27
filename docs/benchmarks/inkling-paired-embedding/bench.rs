mod baseline;
use leansonic_core::inkling_embedding::unpack_embedding;
use std::{env, error::Error, fs, mem::MaybeUninit, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [raw_path, packed_path] = args.as_slice() else {
        return Err("expected raw and prefix files".into());
    };
    let raw = fs::read(raw_path)?;
    let packed = fs::read(packed_path)?;
    assert_eq!((raw.len(), packed.len()), (926_318_592, 618_590_352));
    let mut dst = vec![0; raw.len()];
    println!("cycle,leg,arm,ns");
    for cycle in 0..=10 {
        let order = if cycle % 2 == 1 {
            [0, 1, 1, 0]
        } else {
            [1, 0, 0, 1]
        };
        for (leg, arm) in order.into_iter().enumerate() {
            dst.fill(if leg % 2 == 0 { 0xaa } else { 0x55 });
            // MaybeUninit<u8> has the layout of u8; both decoders only write bytes.
            let storage = unsafe {
                std::slice::from_raw_parts_mut(
                    dst.as_mut_ptr().cast::<MaybeUninit<u8>>(),
                    dst.len(),
                )
            };
            let start = Instant::now();
            if arm == 0 {
                baseline::unpack_embedding(&packed, storage)?;
            } else {
                unpack_embedding(&packed, storage)?;
            }
            let elapsed = start.elapsed().as_nanos();
            assert_eq!(raw.as_slice(), dst.as_slice());
            println!("{cycle},{leg},{arm},{elapsed}");
        }
    }
    Ok(())
}
