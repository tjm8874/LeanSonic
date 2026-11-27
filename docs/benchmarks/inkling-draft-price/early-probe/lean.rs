use leansonic_core::inkling2_gpu::Inkling2GpuEngine;
use std::{env, error::Error, fs, time::Instant};
fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<_> = env::args().collect();
    use std::io::Write;
    let mut probes = fs::File::create_new(env::var("PROBE_COUNTER_CSV")?)?;
    let model = &a[1];
    let spec = a[2].parse()?;
    let mut e = Inkling2GpuEngine::load_from_dir_with_spec(model, 2048, true, spec)?;
    for pass in 0..2 {
        for fixture in [
            "q00_code_merge",
            "q11_json_schema",
            "ja_reasoning",
            "readme1024",
            "en_boolean",
            "en_sort",
            "ja_stock",
            "ja_name",
            "en_count",
            "en_extract",
            "ja_json",
            "ja_filter",
        ] {
            let ids: Vec<u32> = fs::read_to_string(format!("{fixture}.ids"))?
                .split_whitespace()
                .map(str::parse)
                .collect::<Result<_, _>>()?;
            e.reset()?;
            let started = Instant::now();
            let (next, _) = e.prefill_tokens(&ids)?;
            let pre = started.elapsed();
            let started = Instant::now();
            let (tokens, tail) = e.generate_greedy_tokens(next, 128, None, &ids, |_| {})?;
            let dec = started.elapsed();
            assert_eq!(tokens.len(), 128);
            writeln!(probes, "{pass},{fixture},{},{}", e.spec_probe_rounds, e.spec_probe_resumes)?;
            let row = e.read_last_logits()?;
            assert_eq!(row.len(), 200058);
            assert!(row.iter().all(|x| x.is_finite()));
            let hash = row.iter().fold(14695981039346656037u64, |h, x| {
                (h ^ u64::from(x.to_bits())).wrapping_mul(1099511628211)
            });
            let best = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            assert_eq!(best as u32, tail);
            println!(
                "{pass},{fixture},{},{:.6},{:.6},{tail},{hash:016x},{},{},{},{},{}",
                ids.len(),
                pre.as_secs_f64() * 1000.,
                dec.as_secs_f64() * 1000.,
                e.spec_drafted_total,
                e.spec_accepted_total,
                e.spec_verify_steps,
                e.spec_price_skips,
                tokens
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    }
    Ok(())
}
