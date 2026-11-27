//! Replay a previously recorded greedy continuation, even after the first divergence.
use leansonic_core::inkling2_gpu::Inkling2GpuEngine;
use std::{env, error::Error, fs, time::Instant};

fn ids(path: &str, count: usize) -> Result<Vec<u32>, Box<dyn Error>> {
    let v = fs::read_to_string(path)?
        .trim()
        .split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()?;
    if v.len() != count || v.iter().any(|&id| id >= 200058) {
        return Err("invalid fixture".into());
    }
    Ok(v)
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [model, prefix, continuation] = args.as_slice() else {
        return Err("expected MODEL PREFIX.ids CONTINUATION.ids".into());
    };
    let prompt = ids(prefix, 1024)?;
    let tokens = ids(continuation, 32)?;
    let mut eng = Inkling2GpuEngine::load_from_dir_with_spec(model, 8192, true, 0)?;
    eng.reset()?;
    let start = Instant::now();
    let (mut next, _) = eng.prefill_tokens(&prompt)?;
    let after_prefill = Instant::now();
    let mut predictions = Vec::with_capacity(33);
    for token in tokens {
        predictions.push(next);
        next = eng.forward_greedy_token(token)?;
    }
    let elapsed = after_prefill.elapsed();
    predictions.push(next);
    println!(
        "phases {:.5} {:.5}",
        (after_prefill - start).as_secs_f64() * 1000.,
        elapsed.as_secs_f64() * 1000.
    );
    for id in predictions {
        println!("T {id}");
    }
    Ok(())
}
