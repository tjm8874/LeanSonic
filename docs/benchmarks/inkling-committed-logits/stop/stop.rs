use leansonic_core::inkling2_gpu::Inkling2GpuEngine;
use leansonic_core::tiktoken::TOK_END_SAMPLING;
use std::{env, error::Error, fs, io::Write};

fn snapshot(e: &Inkling2GpuEngine, path: &str) -> Result<u32, Box<dyn Error>> {
    let v = e.read_last_logits()?;
    assert_eq!(v.len(), 200058);
    assert!(v.iter().all(|x| x.is_finite()));
    let top = v
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32;
    let mut out = std::io::BufWriter::new(fs::File::create_new(path)?);
    for x in v {
        out.write_all(&x.to_le_bytes())?;
    }
    out.flush()?;
    Ok(top)
}

fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<_> = env::args().collect();
    let mut e = Inkling2GpuEngine::load_from_dir_with_spec(&a[1], 2048, true, a[2].parse()?)?;
    fs::create_dir(&a[3])?;
    for (label, input, limit, stop) in [
        ("immediate", "q11_json_schema", 128, Some(90)),
        ("first-draft", "q11_json_schema", 128, Some(198)),
        ("mid-draft", "q11_json_schema", 128, Some(220)),
        ("eos", "ja_reasoning", 128, Some(TOK_END_SAMPLING)),
        ("near-eos", "ja_end", 16, Some(TOK_END_SAMPLING)),
        ("cap1", "q11_json_schema", 1, Some(TOK_END_SAMPLING)),
        ("cap2", "q11_json_schema", 2, Some(TOK_END_SAMPLING)),
        ("cap3", "q11_json_schema", 3, Some(TOK_END_SAMPLING)),
        ("cap7", "q11_json_schema", 7, Some(TOK_END_SAMPLING)),
        ("fixed", "q11_json_schema", 7, None),
    ] {
        let prompt: Vec<u32> = fs::read_to_string(format!("{input}.ids"))?
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        e.reset()?;
        let (first, _) = e.prefill_tokens(&prompt)?;
        let mut callback = vec![];
        let (ids, terminal) =
            e.generate_greedy_tokens(first, limit, stop, &prompt, |t| callback.push(t))?;
        assert_eq!(ids, callback);
        let top = snapshot(&e, &format!("{}/{label}.terminal", a[3]))?;
        let rounds = e.spec_verify_steps;
        let accepted = e.spec_accepted_total;
        let mut continuation = vec![];
        let mut next = terminal;
        for _ in 0..3 {
            next = e.forward_greedy_token(next)?;
            continuation.push(next);
        }
        snapshot(&e, &format!("{}/{label}.resume", a[3]))?;
        let seq = |v: &[u32]| v.iter().map(u32::to_string).collect::<Vec<_>>().join(" ");
        println!(
            "{label},{terminal},{top},{rounds},{accepted},{},{}",
            seq(&ids),
            seq(&continuation)
        );
    }
    Ok(())
}
