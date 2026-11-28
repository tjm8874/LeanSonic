//! `leansonic canary`: Deterministic constant-prompt & long-context retrieval gate (`#101`).
//!
//! Guards against silent numerical / attention / KV-cache regressions (such as FP16 Split-KV
//! partial overflow at long contexts or speculative verify divergence) by asserting:
//! 1. `short_tokyo` : Constant factual prompt (`"What is the capital of Japan?..."`) -> `"Tokyo"`,
//!    `0` non-finite (`NaN`/`Inf`) vocabulary logits, and (when a GPU drafter is present:
//!    `mtp.safetensors` or a DFlash 2 checkpoint, #103) `0-ULP` greedy token-sequence identity
//!    between `--spec 0` and `--spec 7`.
//! 2. `needle_16k`  : Deterministic `16,384`-token archive retrieval (`"RAVEN-731"` @ `50%` depth)
//!    + `0` non-finite logits.
//! 3. `needle_128k` : Deterministic `131,072`-token archive retrieval (`"MARSH-527"` @ `50%` depth
//!    and `"CORAL-841"` @ `90%` depth, enabled via `--long` / `--long-only`) + `0` non-finite logits.

use std::path::Path;
use std::time::Instant;

use leansonic_core::inkling2_gpu::Inkling2GpuEngine;
use leansonic_core::qwen38_gpu::Qwen38GpuEngine;
use leansonic_core::qwen38_mtp::{
    spec_generate_streaming as q38_spec_generate_streaming, Qwen38MtpDrafter,
    SpecStats as Q38SpecStats,
};
use leansonic_core::qwen_bpe::{
    QwenBpeTokenizer, QWEN_TOK_ENDOFTEXT, QWEN_TOK_IM_END, QWEN_TOK_THINK_CLOSE,
};
use leansonic_core::serving::json_escape;
use leansonic_core::tiktoken::{
    TiktokenBpe, TOK_CONTENT_TEXT, TOK_END_MESSAGE, TOK_END_SAMPLING, TOK_MSG_MODEL,
};

fn is_qwen38_dir(dir: &Path) -> bool {
    if let Ok(cfg) = std::fs::read_to_string(dir.join("config.json")) {
        if cfg.contains("qwen4_exp") || cfg.contains("Qwen4Exp") {
            return true;
        }
    }
    false
}

fn build_qwen_needle_prompt(
    tok: &QwenBpeTokenizer,
    len: usize,
    depth: usize,
    code: &str,
) -> Result<Vec<u32>, String> {
    if !(1..100).contains(&depth) || !(32..=1_048_576 - NEEDLE_MAX_NEW).contains(&len) {
        return Err("depth must be 1..99 and input plus output must fit context".into());
    }
    let prefix = tok.encode(
        "<|im_start|>user\nRead the archive and report the verification code for the Juniper cabinet.\n",
    );
    let marker = tok.encode(&format!(
        "\nArchive entry (Juniper cabinet): the Juniper cabinet was inspected in the registry; the verification code was {code}.\n"
    ));
    let suffix = tok.encode(
        "\nQuestion: What is the verification code for the Juniper cabinet? Reply with the code only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    );
    let at = len * depth / 100;
    if at < prefix.len() || at + marker.len() + suffix.len() > len {
        return Err("input is too short to place the marker at that depth".into());
    }
    let pre = at - prefix.len();
    let post = len - at - marker.len() - suffix.len();
    let objects = [
        "map case",
        "ceramic tray",
        "blue chest",
        "east shelf",
        "field notebook",
        "travel ledger",
        "green cabinet",
    ];
    let places = ["library", "registry", "museum", "office", "post room"];
    let mut filler = Vec::new();
    for i in 0.. {
        if filler.len() >= pre + post {
            break;
        }
        let object = objects[i % objects.len()];
        let place = places[i % places.len()];
        let entry = format!(
            "Archive entry {i}: the {object} was inspected in the {place}; the inventory count was {}.\n",
            17 + i % 41
        );
        filler.extend(tok.encode(&entry));
    }
    let mut ids = prefix;
    ids.extend_from_slice(&filler[..pre]);
    ids.extend(marker);
    ids.extend_from_slice(&filler[pre..pre + post]);
    ids.extend(suffix);
    Ok(ids)
}

fn decode_qwen_visible(tok: &QwenBpeTokenizer, tokens: &[u32]) -> (String, bool) {
    let start_idx = tokens
        .iter()
        .rposition(|&id| id == QWEN_TOK_THINK_CLOSE)
        .map(|p| p + 1)
        .unwrap_or(0);
    let mut bytes = Vec::new();
    let mut completed = false;
    for &id in &tokens[start_idx..] {
        if id == QWEN_TOK_IM_END || id == QWEN_TOK_ENDOFTEXT {
            completed = true;
            break;
        }
        if !tok.is_special_token(id) {
            bytes.extend_from_slice(tok.decode_token_bytes(id));
        }
    }
    (String::from_utf8_lossy(&bytes).into_owned(), completed)
}

fn run_qwen_canaries(
    weights_dir: &Path,
    short_only: bool,
    long: bool,
    long_only: bool,
    check_spec: bool,
    results: &mut Vec<CanaryCaseResult>,
) -> Result<(), String> {
    let tok = QwenBpeTokenizer::load_from_file(weights_dir.join("tokenizer.json"))
        .map_err(|e| format!("load Qwen tokenizer.json: {e}"))?;
    let max_ctx = if long {
        131_072 + NEEDLE_MAX_NEW
    } else if short_only {
        1024
    } else {
        16_384 + NEEDLE_MAX_NEW
    };
    let mut engine =
        Qwen38GpuEngine::load_from_dir_for_spec(weights_dir, max_ctx, true, check_spec)?;
    let mut mtp_opt = if check_spec
        && engine.supports_batched_verify()
        && Qwen38MtpDrafter::has_mtp_weights(&engine)
    {
        Some(Qwen38MtpDrafter::load_for_engine(&engine)?)
    } else {
        None
    };
    let stop_ids = [QWEN_TOK_IM_END, QWEN_TOK_ENDOFTEXT];

    if !long_only {
        {
            let t0 = Instant::now();
            let ids = tok.encode_chat_prompt(
                "What is the capital of Japan? Reply with the city name only.",
                "none",
            );
            engine.reset()?;
            let (first, _) = engine.prefill_tokens(&ids)?;
            let logits = engine.read_last_logits()?;
            let nonfinite = count_nonfinite(&logits);
            let gen = engine.generate_greedy_tokens(first, SHORT_MAX_NEW, &stop_ids, |_| {})?;
            let spec_0ulp = if let Some(mtp) = mtp_opt.as_mut() {
                engine.reset()?;
                mtp.reset();
                let refs = Qwen38MtpDrafter::target_q4_refs(&engine);
                let mut stats = Q38SpecStats::default();
                let spec_gen = q38_spec_generate_streaming(
                    &mut engine,
                    mtp,
                    refs,
                    &ids,
                    SHORT_MAX_NEW,
                    1,
                    &stop_ids,
                    &mut stats,
                    |_| {},
                )?;
                Some(spec_gen == gen)
            } else {
                None
            };
            let mut full = gen;
            if full.len() < SHORT_MAX_NEW {
                full.push(QWEN_TOK_IM_END);
            }
            let (decoded, completed) = decode_qwen_visible(&tok, &full);
            let spec_ok = spec_0ulp.unwrap_or(true);
            let exact = matches_expected_canary(&decoded, "Tokyo", true)
                && completed
                && nonfinite == 0
                && spec_ok;
            results.push(CanaryCaseResult {
                id: "short_tokyo",
                input_tokens: ids.len(),
                expected: "Tokyo",
                answer: decoded.trim().to_string(),
                nonfinite_logits: nonfinite,
                spec7_0ulp: spec_0ulp,
                elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
                pass: exact,
            });
        }

        if !short_only {
            let t0 = Instant::now();
            let ids = build_qwen_needle_prompt(&tok, 16_384, 50, "RAVEN-731")?;
            engine.reset()?;
            let (first, _) = engine.prefill_tokens(&ids)?;
            let logits = engine.read_last_logits()?;
            let nonfinite = count_nonfinite(&logits);
            let gen = engine.generate_greedy_tokens(first, NEEDLE_MAX_NEW, &stop_ids, |_| {})?;
            let mut full = gen;
            if full.len() < NEEDLE_MAX_NEW {
                full.push(QWEN_TOK_IM_END);
            }
            let (decoded, completed) = decode_qwen_visible(&tok, &full);
            let exact = matches_expected_canary(&decoded, "RAVEN-731", false)
                && completed
                && nonfinite == 0;
            results.push(CanaryCaseResult {
                id: "needle_16k_50",
                input_tokens: ids.len(),
                expected: "RAVEN-731",
                answer: decoded.trim().to_string(),
                nonfinite_logits: nonfinite,
                spec7_0ulp: None,
                elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
                pass: exact,
            });
        }
    }

    if long {
        for (case_id, depth, code) in [
            ("needle_128k_50", 50usize, "MARSH-527"),
            ("needle_128k_90", 90usize, "CORAL-841"),
        ] {
            let t0 = Instant::now();
            let ids = build_qwen_needle_prompt(&tok, 131_072, depth, code)?;
            engine.reset()?;
            let (first, _) = engine.prefill_tokens(&ids)?;
            let logits = engine.read_last_logits()?;
            let nonfinite = count_nonfinite(&logits);
            let gen = engine.generate_greedy_tokens(first, NEEDLE_MAX_NEW, &stop_ids, |_| {})?;
            let mut full = gen;
            if full.len() < NEEDLE_MAX_NEW {
                full.push(QWEN_TOK_IM_END);
            }
            let (decoded, completed) = decode_qwen_visible(&tok, &full);
            let exact =
                matches_expected_canary(&decoded, code, false) && completed && nonfinite == 0;
            results.push(CanaryCaseResult {
                id: case_id,
                input_tokens: ids.len(),
                expected: code,
                answer: decoded.trim().to_string(),
                nonfinite_logits: nonfinite,
                spec7_0ulp: None,
                elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
                pass: exact,
            });
        }
    }

    Ok(())
}

const SHORT_MAX_NEW: usize = 64;
const NEEDLE_MAX_NEW: usize = 1024;

#[derive(Debug, Clone)]
pub struct CanaryCaseResult {
    pub id: &'static str,
    pub input_tokens: usize,
    pub expected: &'static str,
    pub answer: String,
    pub nonfinite_logits: usize,
    pub spec7_0ulp: Option<bool>,
    pub elapsed_ms: f64,
    pub pass: bool,
}

fn build_needle_prompt(
    tok: &TiktokenBpe,
    len: usize,
    depth: usize,
    code: &str,
) -> Result<Vec<u32>, String> {
    if !(1..100).contains(&depth) || !(32..=1_048_576 - NEEDLE_MAX_NEW).contains(&len) {
        return Err("depth must be 1..99 and input plus output must fit context".into());
    }
    let mut ids = tok.format_inkling2_chat(
        "Read the archive and report the verification code for the Juniper cabinet.\n",
        "none",
    );
    if ids.pop() != Some(TOK_END_MESSAGE) {
        return Err("unexpected chat framing".into());
    }
    let marker = tok.encode_ordinary(&format!(
        "\nArchive entry (Juniper cabinet): the Juniper cabinet was inspected in the registry; the verification code was {code}.\n"
    ));
    let mut suffix = tok.encode_ordinary(
        "\nQuestion: What is the verification code for the Juniper cabinet? Reply with the code only.\n",
    );
    suffix.push(TOK_END_MESSAGE);
    let at = len * depth / 100;
    if at < ids.len() || at + marker.len() + suffix.len() > len {
        return Err("input is too short to place the marker at that depth".into());
    }
    let pre = at - ids.len();
    let post = len - at - marker.len() - suffix.len();
    let objects = [
        "map case",
        "ceramic tray",
        "blue chest",
        "east shelf",
        "field notebook",
        "travel ledger",
        "green cabinet",
    ];
    let places = ["library", "registry", "museum", "office", "post room"];
    let mut filler = Vec::new();
    for i in 0.. {
        if filler.len() >= pre + post {
            break;
        }
        let object = objects[i % objects.len()];
        let place = places[i % places.len()];
        let entry = format!(
            "Archive entry {i}: the {object} was inspected in the {place}; the inventory count was {}.\n",
            17 + i % 41
        );
        filler.extend(tok.encode_ordinary(&entry));
    }
    ids.extend_from_slice(&filler[..pre]);
    ids.extend(marker);
    ids.extend_from_slice(&filler[pre..pre + post]);
    ids.extend(suffix);
    Ok(ids)
}

pub fn extract_visible_answer_tokens(tokens: &[u32]) -> Vec<u32> {
    let mut model_message = false;
    let mut visible = false;
    let mut answer = Vec::new();
    for &id in tokens {
        match id {
            TOK_END_SAMPLING => break,
            TOK_MSG_MODEL => {
                model_message = true;
                visible = false;
            }
            TOK_CONTENT_TEXT => visible = model_message,
            TOK_END_MESSAGE => {
                model_message = false;
                visible = false;
            }
            0..=199997 if visible => answer.push(id),
            0..=199997 => {}
            _ => {
                model_message = false;
                visible = false;
            }
        }
    }
    answer
}

pub fn matches_expected_canary(answer: &str, expected: &str, strip_period: bool) -> bool {
    let trimmed = answer.trim();
    if strip_period {
        trimmed.trim_end_matches('.') == expected
    } else {
        trimmed == expected
    }
}

fn count_nonfinite(logits: &[f32]) -> usize {
    logits.iter().filter(|x| !x.is_finite()).count()
}

fn run_short_and_16k(
    weights_dir: &Path,
    tok: &TiktokenBpe,
    short_only: bool,
    check_spec: bool,
    results: &mut Vec<CanaryCaseResult>,
) -> Result<(), String> {
    // A GPU drafter (DSpark `mtp.safetensors` or DFlash 2) makes the spec-7 identity check
    // meaningful; DFlash 2-only packages have no `mtp.safetensors`.
    let has_mtp = check_spec
        && (weights_dir.join("mtp.safetensors").exists()
            || leansonic_core::inkling2_dflash2::find_dflash2(weights_dir).is_some());
    let init_spec = if has_mtp { 7 } else { 0 };
    let max_ctx = if short_only {
        1024
    } else {
        16_384 + NEEDLE_MAX_NEW
    };
    let mut engine =
        Inkling2GpuEngine::load_from_dir_with_spec(weights_dir, max_ctx, true, init_spec)?;

    // 1. Constant short factual canary (`"Tokyo"`) + finite logits + `--spec 0 == --spec 7` `0-ULP`
    {
        let t0 = Instant::now();
        let ids = tok.format_inkling2_chat(
            "What is the capital of Japan? Reply with the city name only.",
            "none",
        );
        engine.reset()?;
        engine.spec_k = 0;
        let (first0, _) = engine.prefill_tokens(&ids)?;
        let logits0 = engine.read_last_logits()?;
        let nonfinite = count_nonfinite(&logits0);
        let (gen0, _) = engine.generate_greedy_tokens(
            first0,
            SHORT_MAX_NEW,
            Some(TOK_END_SAMPLING),
            &ids,
            |_| {},
        )?;
        let decoded = tok.decode_tokens(&extract_visible_answer_tokens(&gen0));
        let completed = gen0.last() == Some(&TOK_END_MESSAGE) && gen0.len() < SHORT_MAX_NEW;

        let spec7_0ulp = if has_mtp {
            engine.reset()?;
            engine.spec_k = 7;
            let (first7, _) = engine.prefill_tokens(&ids)?;
            let (gen7, _) = engine.generate_greedy_tokens(
                first7,
                SHORT_MAX_NEW,
                Some(TOK_END_SAMPLING),
                &ids,
                |_| {},
            )?;
            Some(first0 == first7 && gen0 == gen7)
        } else {
            None
        };

        let exact = matches_expected_canary(&decoded, "Tokyo", true)
            && completed
            && nonfinite == 0
            && spec7_0ulp.unwrap_or(true);
        results.push(CanaryCaseResult {
            id: "short_tokyo",
            input_tokens: ids.len(),
            expected: "Tokyo",
            answer: decoded.trim().to_string(),
            nonfinite_logits: nonfinite,
            spec7_0ulp,
            elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
            pass: exact,
        });
    }

    // 2. Medium-context `16,384`-token archive retrieval (`"RAVEN-731"` @ 50% depth)
    if !short_only {
        let t0 = Instant::now();
        let ids = build_needle_prompt(tok, 16_384, 50, "RAVEN-731")?;
        engine.reset()?;
        engine.spec_k = 0;
        let (first, _) = engine.prefill_tokens(&ids)?;
        let logits = engine.read_last_logits()?;
        let nonfinite = count_nonfinite(&logits);
        let (generated, _) = engine.generate_greedy_tokens(
            first,
            NEEDLE_MAX_NEW,
            Some(TOK_END_SAMPLING),
            &ids,
            |_| {},
        )?;
        let decoded = tok.decode_tokens(&extract_visible_answer_tokens(&generated));
        let completed =
            generated.last() == Some(&TOK_END_MESSAGE) && generated.len() < NEEDLE_MAX_NEW;
        let exact =
            matches_expected_canary(&decoded, "RAVEN-731", false) && completed && nonfinite == 0;
        results.push(CanaryCaseResult {
            id: "needle_16k_50",
            input_tokens: ids.len(),
            expected: "RAVEN-731",
            answer: decoded.trim().to_string(),
            nonfinite_logits: nonfinite,
            spec7_0ulp: None,
            elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
            pass: exact,
        });
    }

    Ok(())
}

fn run_long_128k(
    weights_dir: &Path,
    tok: &TiktokenBpe,
    results: &mut Vec<CanaryCaseResult>,
) -> Result<(), String> {
    let mut engine =
        Inkling2GpuEngine::load_from_dir_with_spec(weights_dir, 131_072 + NEEDLE_MAX_NEW, true, 0)?;

    for (case_id, depth, code) in [
        ("needle_128k_50", 50usize, "MARSH-527"),
        ("needle_128k_90", 90usize, "CORAL-841"),
    ] {
        let t0 = Instant::now();
        let ids = build_needle_prompt(tok, 131_072, depth, code)?;
        engine.reset()?;
        let (first, _) = engine.prefill_tokens(&ids)?;
        let logits = engine.read_last_logits()?;
        let nonfinite = count_nonfinite(&logits);
        let (generated, _) = engine.generate_greedy_tokens(
            first,
            NEEDLE_MAX_NEW,
            Some(TOK_END_SAMPLING),
            &ids,
            |_| {},
        )?;
        let decoded = tok.decode_tokens(&extract_visible_answer_tokens(&generated));
        let completed =
            generated.last() == Some(&TOK_END_MESSAGE) && generated.len() < NEEDLE_MAX_NEW;
        let exact = matches_expected_canary(&decoded, code, false) && completed && nonfinite == 0;
        results.push(CanaryCaseResult {
            id: case_id,
            input_tokens: ids.len(),
            expected: code,
            answer: decoded.trim().to_string(),
            nonfinite_logits: nonfinite,
            spec7_0ulp: None,
            elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
            pass: exact,
        });
    }
    Ok(())
}

pub fn run(args: &[String]) -> Result<(), String> {
    let mut weights = String::new();
    let mut short_only = false;
    let mut long = false;
    let mut long_only = false;
    let mut check_spec = true;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-w" | "--weights" => {
                i += 1;
                weights = args
                    .get(i)
                    .cloned()
                    .ok_or("missing value after --weights")?;
            }
            "--short-only" => short_only = true,
            "--long" => long = true,
            "--long-only" => {
                long = true;
                long_only = true;
            }
            "--no-spec" => check_spec = false,
            other => return Err(format!("unknown canary option `{other}`")),
        }
        i += 1;
    }
    if weights.is_empty() {
        return Err(
            "canary requires --weights <dir> [--short-only] [--long] [--long-only] [--no-spec]"
                .into(),
        );
    }

    let weights_dir = Path::new(&weights);
    let mut results = Vec::new();

    if is_qwen38_dir(weights_dir) {
        run_qwen_canaries(
            weights_dir,
            short_only,
            long,
            long_only,
            check_spec,
            &mut results,
        )?;
    } else {
        let tok = TiktokenBpe::from_model_dir(weights_dir).map_err(|e| e.to_string())?;
        if !long_only {
            run_short_and_16k(weights_dir, &tok, short_only, check_spec, &mut results)?;
        }
        if long {
            run_long_128k(weights_dir, &tok, &mut results)?;
        }
    }

    let all_pass = !results.is_empty() && results.iter().all(|r| r.pass);
    let mut cases_json = Vec::with_capacity(results.len());
    for r in &results {
        let spec_field = match r.spec7_0ulp {
            Some(b) => b.to_string(),
            None => "null".to_string(),
        };
        cases_json.push(format!(
            "{{\"id\":\"{}\",\"input_tokens\":{},\"expected\":\"{}\",\"answer\":\"{}\",\"nonfinite_logits\":{},\"spec7_0ulp\":{},\"elapsed_ms\":{:.2},\"pass\":{}}}",
            r.id,
            r.input_tokens,
            json_escape(r.expected),
            json_escape(&r.answer),
            r.nonfinite_logits,
            spec_field,
            r.elapsed_ms,
            r.pass
        ));
    }
    println!(
        "{{\"canary_pass\":{all_pass},\"passed\":{},\"total\":{},\"cases\":[{}]}}",
        results.iter().filter(|r| r.pass).count(),
        results.len(),
        cases_json.join(",")
    );

    if !all_pass {
        let failed: Vec<&str> = results.iter().filter(|r| !r.pass).map(|r| r.id).collect();
        return Err(format!("canary failed on: {}", failed.join(", ")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use leansonic_core::tiktoken::{TOK_CONTENT_THINKING, TOK_MSG_USER};

    #[test]
    fn visible_answer_ignores_thinking_channel() {
        let tokens = [
            TOK_MSG_MODEL,
            TOK_CONTENT_THINKING,
            10,
            TOK_END_MESSAGE,
            TOK_MSG_USER,
            TOK_CONTENT_TEXT,
            11,
            TOK_END_MESSAGE,
            TOK_MSG_MODEL,
            TOK_CONTENT_TEXT,
            42,
            TOK_END_MESSAGE,
            TOK_END_SAMPLING,
        ];
        assert_eq!(extract_visible_answer_tokens(&tokens), vec![42]);
    }

    #[test]
    fn expected_canary_matching_rules() {
        assert!(matches_expected_canary("Tokyo", "Tokyo", true));
        assert!(matches_expected_canary("  Tokyo.\n", "Tokyo", true));
        assert!(!matches_expected_canary("tokyo", "Tokyo", true));
        assert!(!matches_expected_canary("Kyoto", "Tokyo", true));
        assert!(matches_expected_canary("  RAVEN-731\n", "RAVEN-731", false));
        assert!(!matches_expected_canary("RAVEN-731.", "RAVEN-731", false));
    }

    #[test]
    fn nonfinite_counter_catches_nan_and_inf() {
        assert_eq!(count_nonfinite(&[0.0, -1.5, 42.25]), 0);
        assert_eq!(
            count_nonfinite(&[0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY]),
            3
        );
    }
}
