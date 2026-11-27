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
    let (core, first_len) = match args.as_slice() {
        [a, b, c, d] => ([a, b, c, d], 32usize),
        [a, b, c, d, first] => ([a, b, c, d], first.parse()?),
        _ => {
            return Err("expected MODEL INPUT.ids SEQUENCES <64|auto|split> [FIRST_LENGTH]".into())
        }
    };
    let [model, prompt, n, prefill] = core;
    let n: usize = n.parse()?;
    if !(2..=64).contains(&n) {
        return Err("expected 2..64 sequences".into());
    }
    if !(1..=255 - n).contains(&first_len) {
        return Err("prompt and two decode steps must fit context 256".into());
    }
    match prefill.as_str() {
        "64" => env::set_var("LEANSONIC_PREFILL_CHUNK", "64"),
        "split" => env::set_var("LEANSONIC_PREFILL_CHUNK", "256"),
        "auto" => env::remove_var("LEANSONIC_PREFILL_CHUNK"),
        _ => return Err("expected prefill width 64, auto or split".into()),
    }
    let ids = fs::read_to_string(prompt)?
        .trim()
        .split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()?;
    if ids.len() != 1024 || ids.iter().any(|&x| x >= VOCAB as u32) {
        return Err("expected the 1024-token fixture".into());
    }
    let prompts: Vec<_> = (0..n).map(|s| &ids[3 * s..3 * s + first_len + s]).collect();
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
    if prefill == "split" {
        env::set_var("LEANSONIC_PREFILL_CHUNK", "64");
    }
    env::set_var("LEANSONIC_SEQS", n.to_string());
    let mut batch = Inkling2GpuEngine::load_from_dir_with_spec(Path::new(model), 256, true, 0)?;
    let mut prefill_diff = 0;
    for (s, p) in prompts.iter().enumerate() {
        batch.select_seq(s)?;
        batch.reset()?;
        let next = batch.prefill_tokens(p)?.0;
        if next != first[s] {
            prefill_diff += 1;
            println!("prefill,{s},{},{next}", first[s]);
        }
    }
    let mut total_diff = 0u64;
    let mut top_diff = 0;
    let forward: Vec<_> = (0..n).collect();
    let reverse: Vec<_> = (0..n).rev().collect();
    for (t, slots) in [forward, reverse].iter().enumerate() {
        let inputs: Vec<_> = slots.iter().map(|&s| forced(s, t)).collect();
        let (greedy, output) = batch.decode_round_with_logits(slots, &inputs)?;
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
    if total_diff != 0 || prefill_diff != 0 {
        return Err("independent sequence and batched logits differ; see CSV rows".into());
    }
    Ok(())
}
