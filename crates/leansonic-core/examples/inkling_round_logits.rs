//! Compare each FP32 vocabulary entry from a single-sequence engine to batched rounds.
use leansonic_core::inkling2_gpu::Inkling2GpuEngine;
use std::{env, error::Error, fs, path::Path};
const VOCAB: usize = 200_058;
fn top(row: &[f32]) -> Result<u32, &'static str> {
    if row.len() != VOCAB || row.iter().any(|x| !x.is_finite()) {
        return Err("invalid logit row");
    }
    let mut best = 0;
    for i in 1..row.len() {
        if row[i] > row[best] {
            best = i;
        }
    }
    Ok(best as u32)
}
fn ordered(x: f32) -> u32 {
    let b = x.to_bits();
    if b >> 31 != 0 {
        !b
    } else {
        b ^ (1 << 31)
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [model, prompt, n] = args.as_slice() else {
        return Err("expected MODEL INPUT.ids SEQUENCES".into());
    };
    let n: usize = n.parse()?;
    if !(2..=64).contains(&n) {
        return Err("expected 2..64 sequences".into());
    }
    let ids = fs::read_to_string(prompt)?
        .trim()
        .split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()?;
    if ids.len() != 1024 || ids.iter().any(|&x| x >= VOCAB as u32) {
        return Err("expected the 1024-token fixture".into());
    }
    let prompts: Vec<_> = (0..n).map(|s| &ids[3 * s..3 * s + 32 + s]).collect();
    let forced = |slot: usize, step: usize| ids[512 + slot * 2 + step];
    env::set_var("LEANSONIC_SEQS", "1");
    let mut reference = Inkling2GpuEngine::load_from_dir_with_spec(Path::new(model), 256, true, 0)?;
    let mut golden = Vec::new();
    let mut first = Vec::new();
    for (s, p) in prompts.iter().enumerate() {
        reference.reset()?;
        first.push(reference.prefill_tokens(p)?.0);
        let mut steps = Vec::new();
        for t in 0..2 {
            let id = reference.forward_greedy_token(forced(s, t))?;
            let row = reference.read_last_logits()?;
            if top(&row)? != id {
                return Err("reference argmax mismatch".into());
            }
            steps.push((id, row));
        }
        golden.push(steps);
    }
    drop(reference);
    env::set_var("LEANSONIC_SEQS", n.to_string());
    let mut batch = Inkling2GpuEngine::load_from_dir_with_spec(Path::new(model), 256, true, 0)?;
    for (s, p) in prompts.iter().enumerate() {
        batch.select_seq(s)?;
        batch.reset()?;
        if batch.prefill_tokens(p)?.0 != first[s] {
            return Err("prefill first token changed".into());
        }
    }
    let mut total_diff = 0u64;
    let mut top_diff = 0;
    for t in 0..2 {
        let mut slots: Vec<_> = (0..n).collect();
        if t == 1 {
            slots.reverse();
        }
        let inputs: Vec<_> = slots.iter().map(|&s| forced(s, t)).collect();
        let greedy = batch.decode_round(&slots, &inputs)?;
        let output = batch.read_round_logits(n)?;
        for (i, &s) in slots.iter().enumerate() {
            let row = &output[i * VOCAB..(i + 1) * VOCAB];
            if top(row)? != greedy[i] {
                return Err("batch argmax mismatch".into());
            }
            let (id, r) = &golden[s][t];
            let mut diff = 0;
            let mut max_abs = 0f32;
            let mut max_ulp = 0;
            for (&a, &b) in r.iter().zip(row) {
                diff += u64::from(a.to_bits() != b.to_bits());
                max_abs = max_abs.max((a - b).abs());
                max_ulp = max_ulp.max(ordered(a).abs_diff(ordered(b)));
            }
            top_diff += usize::from(*id != greedy[i]);
            total_diff += diff;
            println!(
                "row,{n},{t},{s},{id},{},{diff},{max_abs},{max_ulp}",
                greedy[i]
            );
        }
    }
    println!("total,{n},{},{total_diff},{top_diff}", 2 * n * VOCAB);
    Ok(())
}
