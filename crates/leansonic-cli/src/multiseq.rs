//! `leansonic multiseq`: multi-sequence state checks and throughput (#12 / #107).
//!
//! `--mode check` runs every sequence alone (fresh state), then all of them interleaved one
//! token at a time, and requires identical greedy tokens per sequence.

use std::path::Path;
use std::time::Instant;

use leansonic_core::inkling2_gpu::Inkling2GpuEngine;
use leansonic_core::inkling2_mtp::SuffixDrafter;

pub fn run(args: &[String]) -> Result<(), String> {
    let mut dir = None;
    let mut tokens = None;
    let mut seqs = 4usize;
    let mut gen = 32usize;
    let mut round_rows = 8usize;
    let mut check = true;
    let mut spec = 0usize;
    let mut dspark = false;
    let mut i = 0;
    while i < args.len() {
        let v = args.get(i + 1).cloned();
        match args[i].as_str() {
            "--weights" => dir = v,
            "--tokens" => tokens = v,
            "--seqs" => seqs = v.and_then(|s| s.parse().ok()).ok_or("bad --seqs")?,
            "--gen" => gen = v.and_then(|s| s.parse().ok()).ok_or("bad --gen")?,
            "--round" => round_rows = v.and_then(|s| s.parse().ok()).ok_or("bad --round")?,
            "--spec" => spec = v.and_then(|s| s.parse().ok()).ok_or("bad --spec")?,
            "--dspark" => {
                dspark = true;
                i += 1;
                continue;
            }
            "--no-check" => {
                check = false;
                i += 1;
                continue;
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
        i += 2;
    }
    let dir = dir.ok_or("missing --weights <dir>")?;
    let base = crate::parse_bench_tokens(&tokens.ok_or("missing --tokens")?)?;
    if base.len() < 3 * seqs + 8 {
        return Err("prompt too short for the requested number of sequences".into());
    }
    // Sequence `s` drops its last `3 * s` prompt tokens so the sequences differ.
    let prompts: Vec<Vec<u32>> = (0..seqs)
        .map(|s| base[..base.len() - 3 * s].to_vec())
        .collect();
    std::env::set_var("LEANSONIC_SEQS", seqs.to_string());
    let max_ctx = std::env::var("LEANSONIC_CTX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8192)
        .max(base.len() + gen + 64);
    // `--dspark` loads the MTP drafter (per-sequence contexts) for `--spec K` rounds.
    let spec_k = if dspark { spec.max(1) } else { 0 };
    let mut eng =
        Inkling2GpuEngine::load_from_dir_with_spec(Path::new(&dir), max_ctx, true, spec_k)?;
    if eng.n_seqs != seqs {
        return Err(format!("engine has {} sequences", eng.n_seqs));
    }

    // 1) Each sequence alone.
    let mut alone = Vec::new();
    for (s, p) in prompts.iter().enumerate().filter(|_| check) {
        eng.select_seq(s)?;
        eng.reset()?;
        let (mut tok, _) = eng.prefill_tokens(p)?;
        let mut out = vec![tok];
        for _ in 1..gen {
            tok = eng.forward_greedy_token(tok)?;
            out.push(tok);
        }
        alone.push(out);
    }
    // 2) All sequences interleaved token by token.
    let mut cur: Vec<u32> = Vec::new();
    let mut inter: Vec<Vec<u32>> = Vec::new();
    for (s, p) in prompts.iter().enumerate().filter(|_| check) {
        eng.select_seq(s)?;
        eng.reset()?;
        let (tok, _) = eng.prefill_tokens(p)?;
        cur.push(tok);
        inter.push(vec![tok]);
    }
    let t0 = Instant::now();
    for _ in (1..gen).filter(|_| check) {
        for s in 0..seqs {
            eng.select_seq(s)?;
            let t = eng.forward_greedy_token(cur[s])?;
            cur[s] = t;
            inter[s].push(t);
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    // 3) All sequences in batched rounds (`decode_round`, up to 8 rows per pass).
    let mut cur3: Vec<u32> = Vec::new();
    let mut round: Vec<Vec<u32>> = Vec::new();
    for (s, p) in prompts.iter().enumerate() {
        eng.select_seq(s)?;
        eng.reset()?;
        let (tok, _) = eng.prefill_tokens(p)?;
        cur3.push(tok);
        round.push(vec![tok]);
    }
    let t1 = Instant::now();
    let (mut drafted, mut accepted, mut verify_rounds) = (0usize, 0usize, 0usize);
    let mut draft_ms = 0f64;
    if spec == 0 {
        for _ in 1..gen {
            for chunk in (0..seqs).collect::<Vec<_>>().chunks(round_rows) {
                let toks: Vec<u32> = chunk.iter().map(|&s| cur3[s]).collect();
                let next = eng.decode_round(chunk, &toks)?;
                for (k, &s) in chunk.iter().enumerate() {
                    cur3[s] = next[k];
                    round[s].push(next[k]);
                }
            }
        }
    } else {
        // `--spec K`: suffix (n-gram) drafts per sequence, verified for all sequences in
        // shared `verify_round`s of at most `--round` rows (#107 stage 3).
        loop {
            let mut runs: Vec<(usize, Vec<u32>)> = Vec::new();
            let mut rows = 0usize;
            for s in 0..seqs {
                let rem = gen.saturating_sub(round[s].len());
                if rem == 0 {
                    continue;
                }
                let mut run = vec![cur3[s]];
                if dspark {
                    let td = Instant::now();
                    run.extend(eng.draft_for_seq(s, cur3[s], spec.min(rem - 1))?);
                    draft_ms += td.elapsed().as_secs_f64() * 1e3;
                } else {
                    let mut hist = prompts[s].clone();
                    hist.extend_from_slice(&round[s][..round[s].len() - 1]);
                    run.extend(SuffixDrafter::draft_from_history(
                        &hist,
                        cur3[s],
                        spec.min(rem - 1),
                    ));
                }
                if rows + run.len() > round_rows {
                    if rows + 1 > round_rows {
                        break;
                    }
                    run.truncate(round_rows - rows);
                }
                rows += run.len();
                runs.push((s, run));
            }
            if runs.is_empty() {
                break;
            }
            let refs: Vec<(usize, &[u32])> = runs.iter().map(|(s, r)| (*s, r.as_slice())).collect();
            let preds = eng.verify_round(&refs)?;
            verify_rounds += 1;
            let mut keep = Vec::with_capacity(runs.len());
            for ((s, run), p) in runs.iter().zip(&preds) {
                let mut a = 0;
                while a + 1 < run.len() && p[a] == run[a + 1] {
                    a += 1;
                }
                drafted += run.len() - 1;
                accepted += a;
                round[*s].extend_from_slice(&p[..=a]);
                cur3[*s] = p[a];
                keep.push(a + 1);
            }
            eng.commit_verify_round(&keep)?;
        }
        for r in &mut round {
            r.truncate(gen);
        }
        println!("spec: {verify_rounds} verify rounds, accepted {accepted}/{drafted} drafts, drafting {draft_ms:.1} ms");
    }
    let ms3 = t1.elapsed().as_secs_f64() * 1e3;
    println!(
        "batched rounds: {ms3:.1} ms for {} tokens ({:.1} tok/s total)",
        seqs * (gen - 1),
        (seqs * (gen - 1)) as f64 / (ms3 / 1e3)
    );
    let mut ok = true;
    for s in (0..seqs).filter(|_| check) {
        let first = alone[s].iter().zip(&round[s]).position(|(a, b)| a != b);
        println!("seq {s}: alone vs batched rounds first diff {first:?}");
        ok &= first.is_none();
    }
    for s in (0..seqs).filter(|_| check) {
        let first = alone[s].iter().zip(&inter[s]).position(|(a, b)| a != b);
        let h = alone[s].iter().fold(0xcbf29ce484222325u64, |h, &t| {
            (h ^ t as u64).wrapping_mul(0x100000001b3)
        });
        println!(
            "seq {s}: prompt {} tokens, alone vs interleaved first diff {:?}, hash {h:016x}",
            prompts[s].len(),
            first
        );
        ok &= first.is_none();
    }
    println!(
        "interleaved decode: {ms:.1} ms for {} tokens",
        seqs * (gen - 1)
    );
    if ok {
        println!("MULTISEQ CHECK PASS");
        Ok(())
    } else {
        Err("MULTISEQ CHECK FAIL".into())
    }
}
