//! Measure prefill at workspace and Tensor Core dispatch boundaries.
use leansonic_core::inkling2_gpu::Inkling2GpuEngine;
use std::{env, error::Error, fs, path::Path, time::Instant};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [model, input] = args.as_slice() else {
        return Err("expected model and input".into());
    };
    let ids = fs::read_to_string(input)?
        .trim()
        .split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()?;
    if ids.len() != 1024 {
        return Err("invalid input".into());
    }
    let mut engine = Inkling2GpuEngine::load_from_dir_with_spec(Path::new(model), 2048, true, 0)?;
    for run in 0..4 {
        for len in [64, 65, 66, 80, 95, 96, 128, 129, 159, 160, 255, 256, 1024] {
            engine.reset()?;
            let start = Instant::now();
            let (id, _) = engine.prefill_tokens(&ids[..len])?;
            let prefill = start.elapsed();
            let start = Instant::now();
            for &token in &ids[400..408] {
                engine.forward_greedy_token(token)?;
            }
            println!(
                "{run},{len},{},{},{id}",
                prefill.as_secs_f64() * 1000.,
                start.elapsed().as_secs_f64() * 1000.
            );
        }
    }
    Ok(())
}
