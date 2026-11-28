//! Compare serial and verified scoring after ordinary prefill of the same true history.
use leansonic_core::qwen38_gpu::Qwen38GpuEngine;
use std::{env, error::Error, fs};
fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<_> = env::args().skip(1).collect();
    let [model, prompt_file, generated_file] = a.as_slice() else {
        return Err("expected model prompt ids".into());
    };
    let ids = |p: &str| -> Result<Vec<u32>, Box<dyn Error>> {
        Ok(fs::read_to_string(p)?
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<Vec<_>, _>>()?)
    };
    let prompt = ids(prompt_file)?;
    let reference = ids(generated_file)?;
    let mut eng = Qwen38GpuEngine::load_from_dir(model, 256)?;
    let mut schedule = vec![
        ("serial", 0, 0),
        ("verify-all", 0, 127),
        ("last-alone", 79, 80),
        ("early-only", 0, 79),
        ("stock-again", 0, 0),
    ];
    for x in [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 76, 78] {
        schedule.push(("prefix", 0, x));
    }
    for (arm, start, end) in schedule {
        eng.reset()?;
        eprintln!("arm {arm} {start}..{end}");
        let mut observed = vec![eng.prefill_tokens(&prompt)?.0];
        for (i, &id) in reference[..80].iter().enumerate() {
            observed.push(if (start..end).contains(&i) {
                eng.forward_verify_tokens(&[id])?[0]
            } else {
                eng.forward_greedy_token(id)?
            });
        }
        println!("{arm}\t{start}\t{end}\t{:?}", observed);
    }
    Ok(())
}
