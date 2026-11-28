//! Cargo example for a scratch checkout of the pinned engine.
use leansonic_core::qwen38_gpu::{Qwen38GpuEngine, Q38_VOCAB};
use std::{
    env,
    error::Error,
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [model, fixtures, dest] = args.as_slice() else {
        return Err("expected MODEL FIXTURE_IDS NEW_OUTPUT_DIRECTORY".into());
    };
    let inputs = fs::read_to_string(fixtures)?;
    let sequences: Vec<Vec<u32>> = inputs
        .lines()
        .map(|line| line.split_whitespace().map(str::parse).collect())
        .collect::<Result<_, _>>()?;
    if sequences.is_empty()
        || sequences
            .iter()
            .any(|s| s.len() < 2 || s.iter().any(|&id| id as usize >= Q38_VOCAB))
    {
        return Err("invalid token fixture".into());
    }
    let max_ctx = sequences.iter().map(Vec::len).max().unwrap() + 8;
    let mut engine = Qwen38GpuEngine::load_from_dir(model, max_ctx)?;
    let dir = Path::new(dest);
    fs::create_dir(dir)?;
    let mut logits = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dir.join("logits.f32"))?,
    );
    let mut rows = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dir.join("rows.tsv"))?,
    );
    let mut canaries = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dir.join("canaries.txt"))?,
    );
    let adapt_before_scoring = std::env::var_os("Q38_TEACHER_ADAPT_PREFILL").is_some();
    for (case, seq) in sequences.iter().enumerate() {
        engine.reset()?;
        if adapt_before_scoring {
            engine.prefill_tokens(seq)?;
            engine.reset()?;
        }
        for (pos, pair) in seq.windows(2).enumerate() {
            let top = engine.forward_greedy_token(pair[0])?;
            let values = engine.read_last_logits()?;
            if values.len() != Q38_VOCAB || values.iter().any(|v| !v.is_finite()) {
                return Err("invalid model logits".into());
            }
            for value in values {
                logits.write_all(&value.to_le_bytes())?;
            }
            writeln!(rows, "{case}\t{pos}\t{}\t{}\t{top}", pair[0], pair[1])?;
        }
        engine.reset()?;
        let (mut next, _) = engine.prefill_tokens(seq)?;
        for _ in 0..8 {
            write!(canaries, "{next} ")?;
            next = engine.forward_greedy_token(next)?;
        }
        writeln!(canaries)?;
    }
    logits.flush()?;
    rows.flush()?;
    canaries.flush()?;
    Ok(())
}
