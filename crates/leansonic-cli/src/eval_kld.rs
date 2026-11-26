//! Score teacher-forced Inkling logits against a llama.cpp _logits_ reference.
use leansonic_core::inkling2_gpu::{Inkling2GpuEngine, VOCAB};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

struct Reference {
    tokens: BufReader<File>,
    rows: BufReader<File>,
    ctx: usize,
    chunks: usize,
}

impl Reference {
    fn open(path: &Path) -> Result<Self> {
        let mut tokens = BufReader::new(File::open(path)?);
        let mut header = [0u8; 20];
        tokens.read_exact(&mut header)?;
        if &header[..8] != b"_logits_" {
            return Err("expected a llama.cpp _logits_ reference".into());
        }
        let u32_at = |i| u32::from_le_bytes(header[i..i + 4].try_into().unwrap());
        let ctx = u32_at(8) as usize;
        let vocab = u32_at(12) as usize;
        let chunks = u32_at(16) as usize;
        if vocab != VOCAB || ctx < 3 || chunks == 0 {
            return Err(format!(
                "incompatible reference: ctx={ctx}, vocab={vocab}, chunks={chunks}"
            )
            .into());
        }
        let first_row = 20 + (chunks as u128) * (ctx as u128) * 4;
        let row_count = (chunks as u128) * ((ctx - 1 - ctx / 2) as u128);
        let expected = first_row + row_count * row_bytes(vocab) as u128;
        if u128::from(tokens.get_ref().metadata()?.len()) != expected {
            return Err("file length differs from its declared reference dimensions".into());
        }
        let mut rows = BufReader::new(File::open(path)?);
        rows.seek(SeekFrom::Start(first_row as u64))?;
        Ok(Self {
            tokens,
            rows,
            ctx,
            chunks,
        })
    }

    fn seek_chunk(&mut self, chunk: usize) -> Result<()> {
        if chunk >= self.chunks {
            return Err(format!("start chunk {chunk} exceeds reference bounds").into());
        }
        // open() validated the entire extent against the actual u64 file length.
        let token_bytes = self.ctx as u64 * 4;
        let rows_per_chunk = (self.ctx - 1 - self.ctx / 2) as u64;
        let row_start = 20 + self.chunks as u64 * token_bytes;
        self.tokens
            .seek(SeekFrom::Start(20 + chunk as u64 * token_bytes))?;
        self.rows.seek(SeekFrom::Start(
            row_start + chunk as u64 * rows_per_chunk * row_bytes(VOCAB) as u64,
        ))?;
        Ok(())
    }

    fn next_tokens(&mut self) -> Result<Vec<u32>> {
        let mut buf = vec![0; self.ctx * 4];
        self.tokens.read_exact(&mut buf)?;
        let tokens: Vec<_> = buf
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&bytes| u32::from_le_bytes(bytes))
            .collect();
        if tokens.iter().any(|&id| id as usize >= VOCAB) {
            return Err("reference token outside the model vocabulary".into());
        }
        Ok(tokens)
    }
}

fn row_bytes(vocab: usize) -> usize {
    8 + 2 * (vocab + vocab % 2)
}

struct Score {
    nll: f64,
    nll_ref: f64,
    kld: f64,
    same_top: bool,
    target_at_ref_floor: bool,
}

fn score_row(logits: &[f32], row: &[u8], target: usize) -> Result<Score> {
    let scale = f32::from_le_bytes(row[..4].try_into()?);
    let min_lp = f32::from_le_bytes(row[4..8].try_into()?);
    if !scale.is_finite() || scale < 0.0 || !min_lp.is_finite() {
        return Err("invalid reference scale or offset".into());
    }
    let mut top = 0;
    for (i, &value) in logits.iter().enumerate() {
        if !value.is_finite() {
            return Err("non-finite model logit".into());
        }
        if value > logits[top] {
            top = i;
        }
    }
    let max = logits[top];
    let sum: f64 = logits.iter().map(|&v| f64::from((v - max).exp())).sum();
    let lse = max + sum.ln() as f32;
    let code = |i: usize| {
        let start = 8 + 2 * i;
        u16::from_le_bytes([row[start], row[start + 1]])
    };
    let lp = |i| scale * f32::from(code(i)) + min_lp;
    let mut ref_top = 0;
    let mut ref_max = f32::NEG_INFINITY;
    let mut kld = 0.0;
    for (i, &value) in logits.iter().enumerate() {
        let logp = lp(i);
        if !logp.is_finite() {
            return Err("non-finite reference log-probability".into());
        }
        if logp > ref_max {
            ref_top = i;
            ref_max = logp;
        }
        // Match the Basalt / llama.cpp reference scorer's low-probability cutoff.
        if logp > -16.0 {
            kld += f64::from(logp.exp() * (logp - value + lse));
        }
    }
    Ok(Score {
        nll: f64::from(lse - logits[target]),
        nll_ref: -f64::from(lp(target)),
        kld,
        same_top: top == ref_top,
        target_at_ref_floor: code(target) == 0,
    })
}

pub fn run(args: &[String]) -> Result<()> {
    let mut weights = None;
    let mut reference = None;
    let mut limit = None;
    let mut start = 0usize;
    let mut args = args.iter();
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--weights" => weights = Some(value),
            "--ref" => reference = Some(Path::new(value)),
            "--start-chunk" => start = value.parse()?,
            "--chunks" => limit = Some(value.parse::<usize>()?),
            _ => return Err(format!("unknown eval-kld option {flag}").into()),
        }
    }
    let weights = weights.ok_or("--weights <directory> is required")?;
    let mut reference = Reference::open(reference.ok_or("--ref <file> is required")?)?;
    reference.seek_chunk(start)?;
    let remaining = reference.chunks - start;
    let chunks = limit.unwrap_or(remaining);
    if chunks == 0 || chunks > remaining {
        return Err(format!("--chunks must be between 1 and {remaining}").into());
    }
    let mut engine = Inkling2GpuEngine::load_from_dir_with_spec(weights, reference.ctx, true, 0)?;
    let mut row = vec![0; row_bytes(VOCAB)];
    let mut klds = Vec::new();
    let (mut sum_nll, mut sum_ref, mut matches) = (0.0, 0.0, 0usize);
    let mut ref_floor_targets = 0usize;
    let mut worst = (f64::NEG_INFINITY, start, 0);
    for chunk in start..start + chunks {
        let ids = reference.next_tokens()?;
        engine.reset()?;
        let first = reference.ctx / 2;
        engine.prefill_tokens(&ids[..=first])?;
        for pos in first..reference.ctx - 1 {
            let target = ids[pos + 1];
            reference.rows.read_exact(&mut row)?;
            let logits = engine.read_last_logits()?;
            let s = score_row(&logits, &row, target as usize)
                .map_err(|e| format!("chunk {chunk}, position {pos}: {e}"))?;
            if s.kld > worst.0 {
                worst = (s.kld, chunk, pos);
            }
            klds.push(s.kld);
            sum_nll += s.nll;
            sum_ref += s.nll_ref;
            matches += usize::from(s.same_top);
            ref_floor_targets += usize::from(s.target_at_ref_floor);
            if pos + 2 < reference.ctx {
                engine.forward_greedy_token(target)?;
            }
        }
    }
    let count = klds.len();
    let n = count as f64;
    let mean = klds.iter().sum::<f64>() / n;
    klds.sort_unstable_by(f64::total_cmp);
    let quantile = |p: f64| klds[((count - 1) as f64 * p).round_ties_even() as usize];
    let ppl = (sum_nll / n).exp();
    let ref_ppl = (sum_ref / n).exp();
    if ![ppl, ref_ppl, mean, klds[0], klds[count - 1]]
        .iter()
        .all(|v| v.is_finite())
    {
        return Err("non-finite aggregate score".into());
    }
    println!(
        concat!(
            "{{\"chunks\":{},\"context\":{},\"rows\":{},\"ppl\":{},\"ref_ppl\":{},",
            "\"mean_kld\":{},\"median_kld\":{},\"p99_kld\":{},\"max_kld\":{},",
            "\"start_chunk\":{},\"max_kld_chunk\":{},\"max_kld_position\":{},",
            "\"ref_floor_targets\":{},\"top1_matches\":{},\"top1_agreement\":{}}}"
        ),
        chunks,
        reference.ctx,
        count,
        ppl,
        ref_ppl,
        mean,
        quantile(0.5),
        quantile(0.99),
        worst.0,
        start,
        worst.1,
        worst.2,
        ref_floor_targets,
        matches,
        matches as f64 / n
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_row_with_odd_vocabulary_and_known_distribution() {
        let logp = 0.5f32.ln();
        let mut row = Vec::new();
        row.extend_from_slice(&(-logp).to_le_bytes());
        row.extend_from_slice(&(2.0 * logp).to_le_bytes());
        for code in [0u16, 0, 1, 65535] {
            row.extend_from_slice(&code.to_le_bytes());
        }
        assert_eq!(row.len(), row_bytes(3));
        let logits = [logp, logp, 0.0];
        let exact = score_row(&logits, &row, 0).unwrap();
        assert!(exact.kld.abs() < 1e-6 && exact.same_top && exact.target_at_ref_floor);
        assert!((exact.nll + 0.25f64.ln()).abs() < 1e-6);
        let swapped = score_row(&[0.0, logp, logp], &row, 2).unwrap();
        assert!(!swapped.same_top && !swapped.target_at_ref_floor);
        assert!((swapped.kld - 0.25 * 2f64.ln()).abs() < 1e-6);
        assert!(score_row(&[f32::NAN, logp, logp], &row, 2).is_err());
    }
}
