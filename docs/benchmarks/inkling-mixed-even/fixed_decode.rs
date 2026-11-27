// Run the fixed-token latency comparison described in docs/inkling-reference.md.
use leansonic_core::inkling2_gpu::Inkling2GpuEngine;
use std::{env, error::Error, fs, time::Instant};

fn read_ids(path: &str, count: usize) -> Result<Vec<u32>, Box<dyn Error>> {
    let ids = fs::read_to_string(path)?
        .trim()
        .split(',')
        .map(str::parse)
        .collect::<Result<Vec<u32>, _>>()?;
    if ids.len() != count || ids.iter().any(|&id| id >= 200_058) {
        return Err(format!("expected {count} Inkling token IDs in {path}").into());
    }
    Ok(ids)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    let [_, weights, prompt, continuation] = args.as_slice() else {
        return Err("usage: inkling_fixed_decode MODEL_DIR PROMPT.ids CONTINUATION.ids".into());
    };
    let prompt = read_ids(prompt, 1024)?;
    let continuation = read_ids(continuation, 32)?;
    let mut engine = Inkling2GpuEngine::load_from_dir_with_spec(weights, 8192, true, 0)?;
    for run in 0..5 {
        engine.reset()?;
        let start = Instant::now();
        let (first, _) = engine.prefill_tokens(&prompt)?;
        let prefill_end = Instant::now();
        let mut next = first;
        let mut matches = 0;
        let mut trace = [0_u32; 32];
        for (slot, &token) in trace.iter_mut().zip(&continuation) {
            *slot = next;
            matches += usize::from(next == token);
            next = engine.forward_greedy_token(token)?;
        }
        let decode_ms = prefill_end.elapsed().as_secs_f64() * 1000.0;
        let prefill_ms = (prefill_end - start).as_secs_f64() * 1000.0;
        println!("{run} {prefill_ms:.4} {decode_ms:.4} {matches} {first}");
        println!(
            "P {run} {}",
            trace
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    Ok(())
}
