use leansonic_core::inkling2_gpu::Inkling2GpuEngine;
use std::{env, error::Error, fs, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let [model, spec, fixture] = args.as_slice() else {
        return Err("expected MODEL SPEC FIXTURES".into());
    };
    let mut engine = Inkling2GpuEngine::load_for_prompt(model, 2048, true, spec.parse()?, 1024)?;
    let text = fs::read_to_string(fixture)?;
    for (index, line) in text.lines().enumerate() {
        let ids = line.split(',').map(str::parse::<u32>).collect::<Result<Vec<_>, _>>()?;
        assert_eq!(ids.len(), 1056);
        let prompt = &ids[..1024];
        for rep in 0..4 {
            engine.reset()?;
            let start = Instant::now();
            let (first, _) = engine.prefill_tokens(prompt)?;
            let prefill_ms = start.elapsed().as_secs_f64() * 1000.0;
            let before = (engine.spec_verify_steps, engine.spec_drafted_total, engine.spec_accepted_total);
            let start = Instant::now();
            let (output, next) = engine.generate_greedy_tokens(first, 64, None, prompt, |_| {})?;
            let decode_ms = start.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(output.len(), 64);
            let ids = output.iter().map(u32::to_string).collect::<Vec<_>>().join(" ");
            println!("{index},{rep},{prefill_ms:.7},{decode_ms:.7},{},{},{},{next},{ids}",
                engine.spec_verify_steps-before.0, engine.spec_drafted_total-before.1,
                engine.spec_accepted_total-before.2);
        }
    }
    Ok(())
}
