//! Isolate a second, warm prefill in a CUDA profiler capture range.
use leansonic_core::inkling2_gpu::Inkling2GpuEngine;
use std::{env, error::Error, fs};

#[link(name = "cuda")]
unsafe extern "C" {
    fn cuProfilerStart() -> i32;
    fn cuProfilerStop() -> i32;
    fn cuCtxSynchronize() -> i32;
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [model, fixture] = args.as_slice() else {
        return Err("expected MODEL PREFIX.ids".into());
    };
    let ids = fs::read_to_string(fixture)?
        .trim()
        .split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(ids.len(), 1024);
    let mut engine = Inkling2GpuEngine::load_from_dir_with_spec(model, 2048, true, 0)?;
    let (first, _) = engine.prefill_tokens(&ids)?;
    let warm = engine.generate_greedy_tokens(first, 32, None, &ids, |_| {})?;
    engine.reset()?;
    let (again, _) = engine.prefill_tokens(&ids)?;
    assert_eq!(first, again);
    unsafe {
        assert_eq!(cuCtxSynchronize(), 0);
        assert_eq!(cuProfilerStart(), 0);
    }
    let measured = engine.generate_greedy_tokens(again, 32, None, &ids, |_| {})?;
    unsafe {
        assert_eq!(cuCtxSynchronize(), 0);
        assert_eq!(cuProfilerStop(), 0);
    }
    assert_eq!(warm, measured);
    println!("decode={} next={}", measured.0.len(), measured.1);
    Ok(())
}
