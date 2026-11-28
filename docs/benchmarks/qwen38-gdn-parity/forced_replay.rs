//! Isolate verification width and rollback on a fixed, previously recorded greedy history.
use leansonic_core::qwen38_gpu::Qwen38GpuEngine;
use std::{env, error::Error, fs};

fn ids(path: &str) -> Result<Vec<u32>, Box<dyn Error>> {
    Ok(fs::read_to_string(path)?
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [model, prompt, greedy] = args.as_slice() else {
        return Err("expected MODEL PROMPT GREEDY".into());
    };
    let prompt = ids(prompt)?;
    let history = ids(greedy)?;
    let mut engine = Qwen38GpuEngine::load_from_dir_for_spec(model, 256, true, true)?;
    for (width, commit) in [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (2, 1),
        (3, 1),
        (3, 2),
        (4, 1),
        (4, 2),
        (4, 3),
    ] {
        engine.reset()?;
        let (first, _) = engine.prefill_tokens(&prompt)?;
        let mut next = vec![first];
        let mut offset = 0;
        while offset + 1 < history.len() {
            let stop = (offset + width).min(history.len() - 1);
            let verified = engine.forward_verify_tokens(&history[offset..stop])?;
            let take = commit.min(verified.len());
            next.extend_from_slice(&verified[..take]);
            engine.commit_or_rollback_verify_row(take - 1)?;
            offset += take;
        }
        let diffs: Vec<_> = next
            .iter()
            .zip(&history)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, (a, b))| (i, *a, *b))
            .collect();
        println!("{width}\t{commit}\t{diffs:?}");
    }
    Ok(())
}
