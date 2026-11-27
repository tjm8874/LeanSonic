use leansonic_core::inkling2_gpu::Inkling2GpuEngine;
use std::{env, error::Error, fs};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [model, fixture, len] = args.as_slice() else {
        return Err("expected model, fixture, length".into());
    };
    let n = len.parse::<usize>()?;
    if ![66, 95].contains(&n) {
        return Err("expected 66 or 95".into());
    }
    let ids = fs::read_to_string(fixture)?
        .trim()
        .split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()?;
    if ids.len() != 1024 {
        return Err("bad fixture".into());
    }
    let mut eng = Inkling2GpuEngine::load_from_dir_with_spec(model, 2048, true, 0)?;
    eng.reset()?;
    eng.prefill_tokens(&ids[..n])?;
    eng.reset()?;
    env::set_var("INKLING_PROFILE_TAIL", "1");
    let (tok, _) = eng.prefill_tokens(&ids[..n])?;
    println!("final {tok}");
    Ok(())
}
