use leansonic_core::inkling2_gpu::Inkling2GpuEngine;
use std::{env, error::Error, fs, io::Write, time::Instant};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [model, input, output] = args.as_slice() else {
        return Err("expected model directory, token IDs and output prefix".into());
    };
    let ids = fs::read_to_string(input)?
        .trim()
        .split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(ids.len(), 1024);
    let start = Instant::now();
    let mut engine = Inkling2GpuEngine::load_from_dir_with_spec(model, 2048, true, 0)?;
    println!("load_ms={:.3}", start.elapsed().as_secs_f64() * 1000.);
    let mut rows = fs::File::create(format!("{output}.f32"))?;
    let mut generated = fs::File::create(format!("{output}.ids"))?;
    for length in [32, 1024] {
        engine.reset()?;
        let prompt = &ids[..length];
        let (first, _) = engine.prefill_tokens(prompt)?;
        for step in 0..2 {
            let expected = if step == 0 {
                first
            } else {
                let (tokens, tail) =
                    engine.generate_greedy_tokens(first, 64, None, prompt, |_| {})?;
                assert_eq!(tokens.len(), 64);
                writeln!(generated, "{length},{tail},{tokens:?}")?;
                tail
            };
            let row = engine.read_last_logits()?;
            assert_eq!(row.len(), 200058);
            assert!(row.iter().all(|x| x.is_finite()));
            assert_eq!(
                row[expected as usize],
                row.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            );
            let bytes: Vec<_> = row.iter().flat_map(|x| x.to_le_bytes()).collect();
            rows.write_all(&bytes)?;
        }
    }
    Ok(())
}
