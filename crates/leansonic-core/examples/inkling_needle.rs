//! Replay an exact-length retrieval case on the real Inkling-2-Mini target path.
//!
//! `cargo run --release -p leansonic-core --example inkling_needle -- MODEL_DIR 16384 50 RAVEN-731`
//!
//! Depth is measured in input-token IDs. Change the code and depth for independent
//! fixtures. The JSON result and exit status score only the completed text answer.
use leansonic_core::inkling2_gpu::Inkling2GpuEngine;
use leansonic_core::serving::json_escape;
use leansonic_core::tiktoken::{
    TiktokenBpe, TOK_CONTENT_TEXT, TOK_END_MESSAGE, TOK_END_SAMPLING, TOK_MSG_MODEL,
};

const MAX_NEW: usize = 192;

fn prompt(tok: &TiktokenBpe, len: usize, depth: usize, code: &str) -> Result<Vec<u32>, String> {
    if !(1..100).contains(&depth) || !(32..=1_048_576 - MAX_NEW).contains(&len) {
        return Err("depth must be 1..99 and input plus output must fit the model context".into());
    }
    if code.is_empty() || !code.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-') {
        return Err("use a nonempty alphanumeric code with optional hyphens".into());
    }
    let mut ids = tok.format_inkling2_chat(
        "Read the archive and report the verification code for the Juniper cabinet.\n",
        "none",
    );
    if ids.pop() != Some(TOK_END_MESSAGE) {
        return Err("unexpected chat framing".into());
    }
    let marker = tok.encode_ordinary(&format!(
        "\nThe verification code for the Juniper cabinet is {code}.\n"
    ));
    let mut suffix = tok.encode_ordinary("\nQuestion: What is the verification code for the Juniper cabinet? Reply with the code only.\n");
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
        let entry = format!("Archive entry {i}: the {object} was inspected in the {place}; the inventory count was {}.\n", 17+i%41);
        filler.extend(tok.encode_ordinary(&entry));
    }
    ids.extend_from_slice(&filler[..pre]);
    ids.extend(marker);
    ids.extend_from_slice(&filler[pre..pre + post]);
    ids.extend(suffix);
    Ok(ids)
}

fn answer_tokens(tokens: &[u32]) -> Vec<u32> {
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [model_dir, len, depth, code] = args.as_slice() else {
        return Err("usage: inkling_needle MODEL_DIR INPUT_TOKENS DEPTH_PERCENT CODE".into());
    };
    let len = len.parse::<usize>()?;
    let depth = depth.parse::<usize>()?;
    let tok = TiktokenBpe::from_model_dir(model_dir)?;
    let ids = prompt(&tok, len, depth, code)?;
    let mut engine = Inkling2GpuEngine::load_from_dir_with_spec(model_dir, len + MAX_NEW, true, 0)?;
    let (first, prefill_ms) = engine.prefill_tokens(&ids)?;
    let (generated, _) =
        engine.generate_greedy_tokens(first, MAX_NEW, Some(TOK_END_SAMPLING), &ids, |_| {})?;
    let decoded = tok.decode_tokens(&answer_tokens(&generated));
    let answer = decoded.trim();
    let exact =
        answer == code && generated.last() == Some(&TOK_END_MESSAGE) && generated.len() < MAX_NEW;
    println!("{{\"input_tokens\":{len},\"marker_position\":{},\"generated_tokens\":{},\"prefill_ms\":{prefill_ms:.2},\"expected\":\"{code}\",\"answer\":\"{}\",\"exact_match\":{exact}}}",
        len*depth/100, generated.len(), json_escape(answer));
    if !exact {
        return Err("answer incomplete or different from the planted code".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use leansonic_core::tiktoken::{TOK_CONTENT_THINKING, TOK_MSG_USER};

    #[test]
    fn ignores_reasoning_and_other_roles() {
        let tokens = [
            TOK_MSG_MODEL,
            TOK_CONTENT_THINKING,
            1,
            TOK_END_MESSAGE,
            TOK_MSG_USER,
            TOK_CONTENT_TEXT,
            2,
            TOK_END_MESSAGE,
            TOK_MSG_MODEL,
            TOK_CONTENT_TEXT,
            3,
            TOK_END_MESSAGE,
            TOK_END_SAMPLING,
            TOK_MSG_MODEL,
            TOK_CONTENT_TEXT,
            4,
        ];
        assert_eq!(answer_tokens(&tokens), vec![3]);
    }
}
