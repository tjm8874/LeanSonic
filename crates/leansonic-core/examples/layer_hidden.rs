use leansonic_core::qwen38_gpu::{Qwen38GpuEngine, Q38_HC_D, Q38_LAYERS};
use std::{
    env,
    error::Error,
    fs,
    io::{BufWriter, Write},
};
fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<_> = env::args().skip(1).collect();
    let [weights, fixtures, destination, logits] = a.as_slice() else {
        return Err("expected WEIGHTS FIXTURES DEST LOGITS".into());
    };
    let mut engine = Qwen38GpuEngine::load_from_dir(weights, 256)?;
    let input = fs::read_to_string(fixtures)?;
    let seq: Vec<u32> = input
        .lines()
        .nth(1)
        .ok_or("Japanese fixture missing")?
        .split_whitespace()
        .take(9)
        .map(|s| s.parse())
        .collect::<Result<_, _>>()?;
    let mut trace = BufWriter::new(fs::File::create_new(destination)?);
    let stride = (Q38_LAYERS + 1) * Q38_HC_D * 4;
    let teacher = env::var_os("Q38_TRACE_TEACHER_STATES")
        .map(fs::read)
        .transpose()?;
    if teacher
        .as_ref()
        .is_some_and(|v| v.len() != seq.len() * stride)
    {
        return Err("unaligned teacher states".into());
    }
    for (i, id) in seq.into_iter().enumerate() {
        if let Some(bytes) = &teacher {
            engine.set_layer_teacher(&bytes[i * stride..(i + 1) * stride])?;
        }

        engine.forward_greedy_token(id)?;
        for val in engine.read_layer_trace()? {
            trace.write_all(&val.to_le_bytes())?;
        }
    }
    trace.flush()?;
    let mut lp = BufWriter::new(fs::File::create_new(logits)?);
    for value in engine.read_last_logits()? {
        lp.write_all(&value.to_le_bytes())?;
    }
    lp.flush()?;
    Ok(())
}
