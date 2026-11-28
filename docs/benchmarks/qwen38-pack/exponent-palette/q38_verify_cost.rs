use leansonic_core::qwen38_gpu::Qwen38GpuEngine;
use std::{env, error::Error, fs, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [model, fixtures] = args.as_slice() else {
        return Err("expected MODEL FIXTURES".into());
    };
    let prompts = fs::read_to_string(fixtures)?;
    let input: Vec<u32> = prompts
        .lines()
        .nth(1)
        .ok_or("missing Japanese input")?
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let mut engine = Qwen38GpuEngine::load_from_dir(model, 256)?;
    let (mut next, _) = engine.prefill_tokens(&input)?;
    let mut continuation = vec![next];
    for _ in 0..32 {
        next = engine.forward_greedy_token(next)?;
        continuation.push(next);
    }
    println!("ids {continuation:?}");
    for cycle in 0..4 {
        for width in [0usize, 1, 2, 4, 8] {
            engine.reset()?;
            let (start, _) = engine.prefill_tokens(&input)?;
            if start != continuation[0] {
                return Err("prefill changed".into());
            }
            let mut offset = 0;
            for chunk in continuation[..32].chunks(width.max(1)) {
                let before = Instant::now();
                let ids = if width == 0 {
                    vec![engine.forward_greedy_token(chunk[0])?]
                } else {
                    engine.forward_verify_tokens(chunk)?
                };
                let ns = before.elapsed().as_nanos();
                if ids != continuation[offset + 1..offset + chunk.len() + 1] {
                    return Err(format!(
                        "verify differed at cycle {cycle}, width {width}, token {offset}"
                    )
                    .into());
                }
                println!("{cycle} {width} {offset} {ns}");
                offset += chunk.len();
            }
        }
    }
    Ok(())
}
