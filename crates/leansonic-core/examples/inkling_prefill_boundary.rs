//! Save the two forced decode distributions after full or split prefill on one sequence.
use leansonic_core::inkling2_gpu::Inkling2GpuEngine;
use std::{env, error::Error, fs, io::Write, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [model, input, output] = args.as_slice() else {
        return Err("expected MODEL README.ids OUTPUT_DIR".into());
    };
    let ids = fs::read_to_string(input)?
        .trim()
        .split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()?;
    if ids.len() != 1024 || ids.iter().any(|&id| id >= 200058) {
        return Err("invalid 1024-token README fixture".into());
    }
    env::set_var("LEANSONIC_SEQS", "1");
    let mut engine = Inkling2GpuEngine::load_from_dir_with_spec(Path::new(model), 256, true, 0)?;
    fs::create_dir(output)?;
    for width in [256, 64] {
        let mut file = fs::File::create(Path::new(output).join(format!("w{width}.f32")))?;
        for slot in [31, 32, 33, 34, 39, 44, 50, 57, 59, 63] {
            engine.reset()?;
            for chunk in ids[3 * slot..3 * slot + 32 + slot].chunks(width) {
                engine.prefill_tokens(chunk)?;
            }
            for step in 0..2 {
                engine.forward_greedy_token(ids[512 + 2 * slot + step])?;
                let logits = engine.read_last_logits()?;
                if logits.len() != 200058 || logits.iter().any(|x| !x.is_finite()) {
                    return Err("invalid logits".into());
                }
                let bytes: Vec<u8> = logits.iter().flat_map(|x| x.to_le_bytes()).collect();
                file.write_all(&bytes)?;
            }
        }
    }
    Ok(())
}
