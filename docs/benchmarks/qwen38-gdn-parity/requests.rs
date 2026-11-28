//! Replay full greedy and speculative requests with the same target placement and prompt history.
use leansonic_core::qwen38_gpu::Qwen38GpuEngine;
use leansonic_core::qwen38_mtp::{spec_generate, Qwen38MtpDrafter, SpecStats};
use std::{env, error::Error, fs, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [model, prompts, cycles] = args.as_slice() else {
        return Err("expected MODEL PROMPT_IDS CYCLES".into());
    };
    let prompt_text = fs::read_to_string(prompts)?;
    let prompts: Vec<Vec<u32>> = prompt_text
        .lines()
        .map(|line| line.split_whitespace().map(str::parse).collect())
        .collect::<Result<_, _>>()?;
    let mut target = Qwen38GpuEngine::load_from_dir_for_spec(model, 256, true, true)?;
    let refs = Qwen38MtpDrafter::target_q4_refs(&target);
    let mut drafter = Qwen38MtpDrafter::load_for_engine(&target)?;
    for cycle in 0..cycles.parse::<usize>()? {
        for (case, prompt) in prompts.iter().enumerate() {
            for k in [0, 1, 2] {
                target.reset()?;
                drafter.reset();
                let start = Instant::now();
                let mut stats = SpecStats::default();
                let ids = if k == 0 {
                    let (first, ms) = target.prefill_tokens(prompt)?;
                    stats.prompt_s = ms / 1000.;
                    target.generate_greedy_tokens(first, 128, &[], |_| {})?
                } else {
                    spec_generate(&mut target, &mut drafter, refs, prompt, 128, k, &mut stats)?
                };
                if ids.len() != 128 {
                    return Err("incomplete fixed request".into());
                }
                println!(
                    "{cycle}\t{case}\t{k}\t{:.6}\t{:.6}\t{}\t{}\t{ids:?}",
                    stats.prompt_s,
                    start.elapsed().as_secs_f64(),
                    stats.accepted,
                    stats.drafted
                );
            }
        }
    }
    Ok(())
}
