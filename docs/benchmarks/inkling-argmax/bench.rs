use leansonic_core::inkling2_gpu::Inkling2GpuEngine;
use std::{env, error::Error, fs, time::Instant};
fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<_> = env::args().skip(1).collect();
    let [model, path] = a.as_slice() else {
        return Err("expected model and ids".into());
    };
    let ids = fs::read_to_string(path)?
        .trim()
        .split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(ids.len(), 1024);
    let mut engine = Inkling2GpuEngine::load_from_dir_with_spec(model, 2048, true, 0)?;
    for pass in 0..4 {
        for n in [32, 1024] {
            engine.reset()?;
            let start = Instant::now();
            let (next, _) = engine.prefill_tokens(&ids[..n])?;
            let pre = start.elapsed();
            let start = Instant::now();
            let (preds, tail) = engine.generate_greedy_tokens(next, 64, None, &ids[..n], |_| {})?;
            let dec = start.elapsed();
            assert_eq!(preds.len(), 64);
            let row = engine.read_last_logits()?;
            assert!(
                row.len() == 200058
                    && row.iter().all(|v| v.is_finite())
                    && row[tail as usize] == row.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            );
            let hash = row.iter().fold(14695981039346656037u64, |acc, x| {
                (acc ^ u64::from(x.to_bits())).wrapping_mul(1099511628211)
            });
            println!(
                "{pass},{n},{:.6},{:.6},{tail},{hash:016x},{}",
                pre.as_secs_f64() * 1000.,
                dec.as_secs_f64() * 1000.,
                preds
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    }
    Ok(())
}
