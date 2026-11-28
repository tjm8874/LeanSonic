use leansonic_core::qwen38_gpu::Qwen38GpuEngine;
use std::{env, error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [model, fixtures] = args.as_slice() else {
        return Err("expected MODEL FIXTURE_IDS".into());
    };
    let mut engine = Qwen38GpuEngine::load_from_dir(model, 256)?;
    for (case, line) in fs::read_to_string(fixtures)?.lines().enumerate() {
        let tokens = line
            .split_whitespace()
            .map(str::parse::<u32>)
            .collect::<Result<Vec<_>, _>>()?;
        engine.reset()?;
        let serial = tokens
            .iter()
            .map(|&id| engine.forward_greedy_token(id))
            .collect::<Result<Vec<_>, _>>()?;
        println!("{case} serial {serial:?}");
        for width in 1..=8 {
            engine.reset()?;
            let mut actual = Vec::new();
            for chunk in tokens.chunks(width) {
                actual.extend(engine.forward_verify_tokens(chunk)?);
            }
            if actual.len() != tokens.len() {
                return Err("missing verify predictions".into());
            }
            println!("{case} {width} {actual:?}");
        }
    }
    Ok(())
}
