use leansonic_core::qwen38_gpu::Qwen38GpuEngine;
use std::{
    env,
    error::Error,
    fs,
    io::{BufWriter, Write},
};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [weights, fixtures, dest] = args.as_slice() else {
        return Err("expected WEIGHTS FIXTURES OUTPUT".into());
    };
    let seqs = fs::read_to_string(fixtures)?;
    let mut engine = Qwen38GpuEngine::load_from_dir(weights, 256)?;
    let mut out = BufWriter::new(fs::File::create_new(dest)?);
    for line in seqs.lines() {
        let ids: Vec<u32> = line
            .split_whitespace()
            .map(|s| s.parse())
            .collect::<Result<_, _>>()?;
        engine.reset()?;
        for &id in &ids[..ids.len() - 1] {
            engine.forward_greedy_token(id)?;
            for val in engine.diagnostic_head_input()? {
                out.write_all(&val.to_le_bytes())?;
            }
        }
    }
    out.flush()?;
    Ok(())
}
