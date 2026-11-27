use leansonic_core::inkling2_gpu::Inkling2GpuEngine;
use std::{env, error::Error, fs, time::Instant};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [model, file, sz, k, label] = args.as_slice() else {
        return Err("expected model, ids, width, spec and label".into());
    };
    let input: Vec<u32> = fs::read_to_string(file)?
        .trim()
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let n: usize = sz.parse()?;
    let spec: usize = k.parse()?;
    let prompt = &input[..n];
    let start = Instant::now();
    let mut e = Inkling2GpuEngine::load_from_dir_with_spec(model, 8192, true, spec)?;
    println!(
        "load_ms={:.3} has_df2={}",
        start.elapsed().as_secs_f64() * 1e3,
        e.dflash2.is_some()
    );
    assert_eq!(e.dflash2.is_some(), spec > 0);
    let mut all = Vec::new();
    for rep in 0..2 {
        e.reset()?;
        let t = Instant::now();
        let (first, _) = e.prefill_tokens(prompt)?;
        let pre = t.elapsed();
        let t = Instant::now();
        let (ids, tail) = e.generate_greedy_tokens(first, 64, None, prompt, |_| {})?;
        assert_eq!(ids.len(), 64);
        println!(
            "rep={rep} prefill_ms={:.3} decode_ms={:.3} draft={} accept={} steps={} copied={}",
            pre.as_secs_f64() * 1e3,
            t.elapsed().as_secs_f64() * 1e3,
            e.spec_drafted_total,
            e.spec_accepted_total,
            e.spec_verify_steps,
            e.spec_copy_rounds
        );
        let result = (ids, tail);
        if rep == 1 {
            assert_eq!(&result, &all[0]);
        }
        all.push(result);
    }
    fs::write(format!("{label}.ids"), format!("{:?}\n", all[0]))?;
    if spec > 0 {
        e.reset()?;
        let (first, _) = e.prefill_tokens(prompt)?;
        e.forward_greedy_token(first)?;
        fs::write(format!("{label}.taps"), e.read_scalar_drafter_taps()?)?;
    }
    Ok(())
}
