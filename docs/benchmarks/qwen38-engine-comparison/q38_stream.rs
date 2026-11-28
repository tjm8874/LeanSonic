//! Resident greedy runner for the matched stdin/stdout timing harness.
use leansonic_core::qwen38_gpu::{Qwen38GpuEngine, Q38_VOCAB};
use std::{
    env,
    error::Error,
    io::{self, BufRead, Write},
};
fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let model = args.next().ok_or("missing model")?;
    let context: usize = args.next().ok_or("missing context")?.parse()?;
    let mut engine = Qwen38GpuEngine::load_from_dir(model, context)?;
    let mut output = io::BufWriter::new(io::stdout().lock());
    writeln!(output, "READY {context}")?;
    output.flush()?;
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line == "QUIT" {
            break;
        }
        let mut parts = line.split_ascii_whitespace();
        if parts.next() != Some("GEN") {
            return Err("expected GEN".into());
        }
        let n: usize = parts.next().ok_or("missing output count")?.parse()?;
        let seq = parts
            .next()
            .ok_or("missing ids")?
            .split(',')
            .map(str::parse)
            .collect::<Result<Vec<u32>, _>>()?;
        if parts.next().is_some()
            || n == 0
            || seq.is_empty()
            || seq.len().checked_add(n).is_none_or(|l| l > context)
            || seq.iter().any(|&t| t as usize >= Q38_VOCAB)
        {
            return Err("invalid request".into());
        }
        engine.reset()?;
        let (mut next, _) = engine.prefill_tokens(&seq)?;
        for i in 0..n {
            if i > 0 {
                next = engine.forward_greedy_token(next)?;
            }
            writeln!(output, "T {next}")?;
            output.flush()?;
        }
        writeln!(output, "DONE {n}")?;
        output.flush()?;
    }
    Ok(())
}
