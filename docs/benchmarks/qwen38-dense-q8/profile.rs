//! Nsight capture of warmed prefill or fixed-history decode on an existing target.
use leansonic_core::qwen38_gpu::Qwen38GpuEngine;
use std::{env, error::Error, fs};

#[link(name = "cuda")]
unsafe extern "C" {
    fn cuProfilerStart() -> i32;
    fn cuProfilerStop() -> i32;
}

fn profile(mark: unsafe extern "C" fn() -> i32) -> Result<(), Box<dyn Error>> {
    let status = unsafe { mark() };
    if status != 0 {
        return Err(format!("CUDA profiler returned {status}").into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 4 || !matches!(args[3].as_str(), "prefill" | "decode") {
        return Err("expected MODEL TWO_LINE_IDS prefill|decode".into());
    }
    let text = fs::read_to_string(&args[2])?;
    let ids: Vec<Vec<u32>> = text
        .lines()
        .map(|line| line.split_whitespace().map(str::parse).collect())
        .collect::<Result<_, _>>()?;
    if ids.len() != 2 || ids.iter().any(Vec::is_empty) || ids[0].len() + ids[1].len() > 256 {
        return Err("expected a nonempty prompt and continuation within context 256".into());
    }
    let mut target = Qwen38GpuEngine::load_from_dir(&args[1], 256)?;
    for warm in [true, false] {
        target.reset()?;
        if !warm && args[3] == "prefill" {
            profile(cuProfilerStart)?;
        }
        let _ = target.prefill_tokens(&ids[0])?;
        if !warm && args[3] == "prefill" {
            profile(cuProfilerStop)?;
            break;
        }
        if !warm {
            profile(cuProfilerStart)?;
        }
        for &id in &ids[1] {
            let _ = target.forward_greedy_token(id)?;
        }
        if !warm {
            profile(cuProfilerStop)?;
        }
    }
    Ok(())
}
