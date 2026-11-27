use leansonic_core::inkling2_gpu::Inkling2GpuEngine;
use std::{env, error::Error, fs, time::Instant};
fn ids(path: &str) -> Result<Vec<u32>, Box<dyn Error>> {
    Ok(fs::read_to_string(path)?
        .trim()
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?)
}
fn main() -> Result<(), Box<dyn Error>> {
    let model = env::args()
        .nth(1)
        .ok_or("expected Inkling model directory")?;
    let prompt = ids("prompt.ids")?;
    let forced = ids("anchor.ids")?;
    assert_eq!((prompt.len(), forced.len()), (1024, 32));
    let mut engine = Inkling2GpuEngine::load_from_dir_with_spec(model, 2048, true, 0)?;
    for rep in 0..8 {
        let order = if rep % 2 == 0 {
            ["forced", "greedy"]
        } else {
            ["greedy", "forced"]
        };
        for mode in order {
            engine.reset()?;
            let start = Instant::now();
            let (mut next, _) = engine.prefill_tokens(&prompt)?;
            let pref = start.elapsed();
            let start = Instant::now();
            let mut tokens = Vec::with_capacity(33);
            for &anchor in &forced {
                tokens.push(next);
                next = engine.forward_greedy_token(if mode == "forced" { anchor } else { next })?;
            }
            let dec = start.elapsed();
            tokens.push(next);
            println!(
                "{rep},{mode},{:.7},{:.7},{}",
                pref.as_secs_f64() * 1000.,
                dec.as_secs_f64() * 1000.,
                tokens
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    }
    Ok(())
}
