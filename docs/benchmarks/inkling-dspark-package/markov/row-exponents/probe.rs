use leansonic_core::inkling_mxfp8::{pack, unpack_nvfp4_codes};
use std::{env, error::Error, fs, time::Instant};

fn frame(input: &[u8]) -> Result<(Vec<u8>, Vec<u8>), Box<dyn Error>> {
    let mut encoded = Vec::new();
    pack(input, &mut encoded)?;
    let decoded = unpack_nvfp4_codes(&encoded, input.len(), false)?;
    assert_eq!(input, decoded);
    Ok((encoded, decoded))
}

fn main() -> Result<(), Box<dyn Error>> {
    let raw = fs::read(env::args().nth(1).ok_or("missing BF16 matrix")?)?;
    assert_eq!(raw.len(), 201024 * 256 * 2);
    for mode in ["bytes", "fields", "column", "column-fields", "row-fields"] {
        let start = Instant::now();
        let mut low = Vec::with_capacity(raw.len()/2);
        let mut high = Vec::with_capacity(raw.len()/2);
        let fields = mode.contains("fields");
        let column = mode.starts_with("column");
        let centered = mode == "row-fields";
        let mut biases = Vec::new();
        for row in raw.chunks_exact(512) {
            let mut words=Vec::with_capacity(256);
            for b in row.chunks_exact(2) {
                let mut w = u16::from_le_bytes([b[0],b[1]]);
                if fields {
                    // Preserve sign+fraction in low and full exponent in high byte.
                    w = (w & 127) | ((w >> 8) & 128) | ((w << 1) & 0xff00);
                }
                words.push(w);
            }
            let bias = if centered {
                let sum: u32 = words.iter().map(|x| u32::from(x>>8)).sum();
                let val = ((sum+128)/256) as u8;
                biases.push(val);
                val
            } else { 0 };
            for w in words {
                low.push(w as u8);
                high.push(((w>>8) as u8).wrapping_sub(bias));
            }
        }
        let (lo_bytes, lo) = frame(&low)?;
        let (bias_bytes, bias) = if centered { frame(&biases)? } else { (vec![],vec![]) };
        let mut hi = vec![0;high.len()];
        let mut hi_bytes = 0;
        let lanes = if column {256} else {1};
        for i in 0..lanes {
            let seq: Vec<u8> = high[i..].iter().step_by(lanes).copied().collect();
            let (f,v) = frame(&seq)?;
            hi_bytes += f.len();
            for (dst,x) in hi[i..].iter_mut().step_by(lanes).zip(v) { *dst=x; }
        }
        for (i,b) in raw.chunks_exact(2).enumerate() {
            let e = hi[i].wrapping_add(if centered { bias[i/256] } else { 0 });
            let w = if fields { u16::from(lo[i]&127) | (u16::from(lo[i]&128)<<8) | (u16::from(e)<<7) }
                else { u16::from_le_bytes([lo[i],e]) };
            assert_eq!(w.to_le_bytes(), b);
        }
        // One offset for every stream except the last, as in the two-plane control.
        let headers = 8*(lanes+usize::from(centered));
        println!("{mode},{},{},{},{headers},{:.3}", lo_bytes.len(),hi_bytes,bias_bytes.len(),start.elapsed().as_secs_f64());
    }
    Ok(())
}
