// Offline WikiText train observation only, on the normalized gate/up input before quantization.
use std::{
    fs::File,
    io::{BufWriter, Write},
    sync::{Mutex, OnceLock},
};

static ACC: OnceLock<Mutex<Accumulator>> = OnceLock::new();
const L: usize = 20;
const E: usize = 128;
const D: usize = 2304;
const K: usize = 8;
struct Accumulator {
    calls: usize,
    by_layer: Vec<usize>,
    moments: Vec<f64>,
}

pub(super) fn observe(il: usize, ids: &[i32], weights: &[f32], x: &[f32]) -> Result<(), String> {
    let mut acc = ACC
        .get_or_init(|| {
            Mutex::new(Accumulator {
                calls: 0,
                by_layer: vec![0; L],
                moments: vec![0.0; L * E * D],
            })
        })
        .lock()
        .map_err(|e| e.to_string())?;
    if ids.len() != 512 * K
        || weights.len() != ids.len()
        || x.len() != 512 * D
        || !x.iter().all(|v| v.is_finite())
        || il >= L
    {
        return Err("invalid train activation".into());
    }
    let mut square = [0.0; D];
    for ((values, experts), ws) in x
        .chunks_exact(D)
        .zip(ids.chunks_exact(K))
        .zip(weights.chunks_exact(K))
    {
        for (out, &value) in square.iter_mut().zip(values) {
            *out = f64::from(value).powi(2);
        }
        for (&expert, &w) in experts.iter().zip(ws) {
            if !(0..E as i32).contains(&expert) || !w.is_finite() || w < 0.0 {
                return Err("invalid selected expert".into());
            }
            let offset = (il * E + expert as usize) * D;
            let weight2 = f64::from(w).powi(2);
            for (out, &v) in acc.moments[offset..offset + D].iter_mut().zip(&square) {
                *out += weight2 * v;
            }
        }
    }
    acc.calls += 1;
    acc.by_layer[il] += 1;
    if acc.calls == 32 * 44 {
        for (layer, &visits) in acc.by_layer.iter().enumerate() {
            if visits != if (6..18).contains(&layer) { 96 } else { 32 } {
                return Err("invalid train visit count".into());
            }
        }
        let mut out = BufWriter::new(File::create_new("moments.tmp").map_err(|e| e.to_string())?);
        for v in &acc.moments {
            out.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        out.flush().map_err(|e| e.to_string())?;
        std::fs::rename("moments.tmp", "moments.f64").map_err(|e| e.to_string())?;
    }
    Ok(())
}
