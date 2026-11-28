//! Compare complete greedy and speculative resident requests on held-out tokens.
use leansonic_core::qwen38_gpu::{Qwen38GpuEngine, Q38_HC_D};
use leansonic_core::qwen38_mtp::{spec_generate, Qwen38MtpDrafter, SpecStats};
use std::{env, error::Error, fs, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [model, fixtures, spec] = args.as_slice() else {
        return Err("expected MODEL HELDOUT_IDS K".into());
    };
    let k: usize = spec.parse()?;
    let cases = fs::read_to_string(fixtures)?
        .lines()
        .map(|line| {
            line.split_whitespace()
                .map(str::parse)
                .collect::<Result<Vec<u32>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut eng = Qwen38GpuEngine::load_from_dir_with_opts(model, 256, true)?;
    // Allocate normal target prefill buffers before placing the drafter, matching the native test.
    eng.prefill_tokens(&cases[0])?;
    eng.reset()?;
    let (ctx, _, embed, head, shards) = eng.mtp_view();
    let mut mtp = if k > 0 {
        let v = shards
            .require(
                "language_model.model.layers.0.attn_hyper_connection.hc_norm.weight",
                "BF16",
                &[Q38_HC_D],
            )
            .to_u16_vec();
        let mean = v
            .iter()
            .map(|&u| f32::from_bits(u32::from(u) << 16))
            .sum::<f32>()
            / Q38_HC_D as f32;
        let m = Qwen38MtpDrafter::load(&ctx, shards, 256, mean < 0.5, 512 << 20)?;
        eprintln!("MTP MoE in VRAM: {}", m.moe_in_vram);
        Some(m)
    } else {
        None
    };
    for cycle in 0..4 {
        for (case, prompt) in cases.iter().enumerate() {
            eng.reset()?;
            let mut stats = SpecStats::default();
            if let Some(m) = mtp.as_mut() {
                m.reset();
            }
            let start = Instant::now();
            let generated = if let Some(m) = mtp.as_mut() {
                spec_generate(&mut eng, m, (embed, head), prompt, 128, k, &mut stats)?
            } else {
                let (next, _) = eng.prefill_tokens(prompt)?;
                stats.prompt_s = start.elapsed().as_secs_f64();
                eng.generate_greedy_tokens(next, 128, &[], |_| {})?
            };
            let total = start.elapsed().as_secs_f64();
            if generated.len() != 128 {
                return Err("short native generation".into());
            }
            println!(
                "{cycle}\t{case}\t{:.6}\t{:.6}\t{}\t{}\t{}\t{}\t{generated:?}",
                stats.prompt_s, total, stats.drafted, stats.accepted, stats.cycles, stats.rollbacks
            );
        }
    }
    Ok(())
}
