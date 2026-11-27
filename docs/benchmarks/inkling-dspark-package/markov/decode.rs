use leansonic_core::{inkling_mxfp8::unpack_nvfp4_codes, inkling_q8_scales};
use std::{
    env,
    error::Error,
    io::{Read, Write},
};
fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<_> = env::args().collect();
    let mut b = vec![];
    std::io::stdin().read_to_end(&mut b)?;
    let n: usize = a[2].parse()?;
    let raw = match a[1].as_str() {
        "bf16planes" => {
            let lo = usize::try_from(u64::from_le_bytes(b[..8].try_into()?))?;
            let l = unpack_nvfp4_codes(&b[8..8 + lo], n / 2, false)?;
            let h = unpack_nvfp4_codes(&b[8 + lo..], n / 2, false)?;
            l.iter().zip(h).flat_map(|(&x, y)| [x, y]).collect()
        }
        "rans" => unpack_nvfp4_codes(&b, n, false)?,
        "dict" => inkling_q8_scales::unpack(&b, n / 4)?,
        _ => return Err("unknown frame".into()),
    };
    assert_eq!(raw.len(), n);
    std::io::stdout().write_all(&raw)?;
    Ok(())
}
