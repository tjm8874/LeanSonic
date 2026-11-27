use leansonic_core::inkling_embedding::{pack_embedding, unpack_embedding};
use std::{env, error::Error, fs, io::Write, mem::MaybeUninit, time::Instant};
#[path = "embed_prefix_trial.rs"]
mod trial;
fn decode(data: &[u8], dst: &mut [u8]) -> Result<(), String> {
    // Initialized storage is valid as a mutable MaybeUninit byte slice.
    let u = unsafe {
        std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<MaybeUninit<u8>>(), dst.len())
    };
    trial::unpack_embedding(data, u)
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
    let sanity: Vec<u8> = (0u16..=65535).flat_map(u16::to_le_bytes).collect();
    let mut check = Vec::new();
    trial::pack_embedding(&sanity[..], sanity.len(), &mut check)?;
    let mut recovered = vec![0; sanity.len()];
    decode(&check, &mut recovered)?;
    assert_eq!(sanity, recovered);
    assert!(decode(&check[..check.len() - 8], &mut recovered).is_err());
    let mut fields = Vec::new();
    trial::pack_embedding(&raw[..], raw.len(), &mut fields)?;
    fs::File::create_new("conditional-prefix.bin")?.write_all(&fields)?;
    let mut dst = vec![0; raw.len()];
    for cycle in 0..6 {
        let order = if cycle % 2 == 0 {
            [
                "prefix",
                "conditional-prefix",
                "conditional-prefix",
                "prefix",
            ]
        } else {
            [
                "conditional-prefix",
                "prefix",
                "prefix",
                "conditional-prefix",
            ]
        };
        for (leg, kind) in order.iter().enumerate() {
            dst.fill(if leg % 2 == 0 { 0x55 } else { 0xaa });
            let start = Instant::now();
            if *kind == "conditional-prefix" {
                decode(&fields, &mut dst)?;
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
                if *kind == "conditional-prefix" {
                    fields.len()
                } else {
                    prefix.len()
                }
            );
        }
    }
    Ok(())
}
