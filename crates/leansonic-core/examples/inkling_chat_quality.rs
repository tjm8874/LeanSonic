//! Tokenize a fixed chat pair or dump teacher-forced response logits after each engine reset.
use leansonic_core::{
    inkling2_gpu::Inkling2GpuEngine,
    tiktoken::{TiktokenBpe, TOK_CONTENT_TEXT, TOK_MSG_MODEL},
};
use std::{
    env,
    error::Error,
    fs,
    io::{BufWriter, Write},
    path::Path,
};

fn ids(path: &Path) -> Result<Vec<u32>, Box<dyn Error>> {
    let tokens = fs::read_to_string(path)?
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<Vec<u32>, _>>()?;
    if tokens.is_empty() || tokens.iter().any(|&t| t >= 200058) {
        return Err("empty or invalid token fixture".into());
    }
    Ok(tokens)
}

fn save_ids(path: &Path, tokens: &[u32]) -> std::io::Result<()> {
    fs::write(
        path,
        tokens
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(" ")
            + "\n",
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [mode, model, root, cases @ ..] = args.as_slice() else {
        return Err("expected tokenize|dump MODEL_DIR FIXTURE_DIR CASE ...".into());
    };
    if cases.is_empty() {
        return Err("no cases".into());
    }
    let root = Path::new(root);
    match mode.as_str() {
        "tokenize" => {
            let bpe = TiktokenBpe::from_model_dir(model)?;
            for case in cases {
                let dir = root.join(case);
                let prompt = fs::read_to_string(dir.join("prompt.txt"))?;
                let answer = fs::read_to_string(dir.join("answer.txt"))?;
                let mut prefix = bpe.format_inkling2_chat(&prompt, "none");
                prefix.extend([TOK_MSG_MODEL, TOK_CONTENT_TEXT]);
                let targets = bpe.encode_ordinary(&answer);
                if targets.is_empty()
                    || bpe.decode_tokens(&targets) != answer
                    || prefix.len() + targets.len() > 512
                {
                    return Err(format!("invalid answer or context for {case}").into());
                }
                save_ids(&dir.join("prefix.ids"), &prefix)?;
                save_ids(&dir.join("answer.ids"), &targets)?;
            }
        }
        "dump" => {
            let mut engine = Inkling2GpuEngine::load_from_dir_with_spec(model, 512, true, 0)?;
            for case in cases {
                let dir = root.join(case);
                let prompt = ids(&dir.join("prefix.ids"))?;
                let targets = ids(&dir.join("answer.ids"))?;
                if prompt.len() + targets.len() > 512 {
                    return Err("context exceeds 512".into());
                }
                engine.reset()?;
                let (mut next, _) = engine.prefill_tokens(&prompt)?;
                let mut out = BufWriter::new(fs::File::create_new(format!("{case}.f32"))?);
                let mut buffer = Vec::with_capacity(200058 * 4);
                for (i, &target) in targets.iter().enumerate() {
                    let row = engine.read_last_logits()?;
                    if row.len() != 200058
                        || row.iter().any(|x| !x.is_finite())
                        || row[next as usize]
                            != row.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                    {
                        return Err(format!("invalid logits at {case}/{i}").into());
                    }
                    buffer.clear();
                    for value in row {
                        buffer.extend(value.to_le_bytes());
                    }
                    out.write_all(&buffer)?;
                    if i + 1 < targets.len() {
                        next = engine.forward_greedy_token(target)?;
                    }
                }
                out.flush()?;
            }
        }
        _ => return Err("unknown mode".into()),
    }
    Ok(())
}
