use leansonic_core::qwen38_gpu::Qwen38GpuEngine;
use std::{env, error::Error, fs};
fn read(s: &str) -> Result<Vec<u32>, Box<dyn Error>> {
    Ok(fs::read_to_string(s)?
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()?)
}
fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<_> = env::args().collect();
    let p = read(&a[2])?;
    let g = read(&a[3])?;
    let mut t = Qwen38GpuEngine::load_from_dir_for_spec(&a[1], 256, false, true)?;
    for offset in [0, 69] {
        for width in [0, 1, 2, 3] {
            env::remove_var("Q38_TRACE_OUTPUT");
            t.reset()?;
            let (first, _) = t.prefill_tokens(&p)?;
            assert_eq!(first, g[0]);
            for n in 0..offset {
                let next = t.forward_greedy_token(g[n])?;
                assert_eq!(next, g[n + 1]);
            }
            env::set_var(
                "Q38_TRACE_OUTPUT",
                format!("{}/o{}-b{}.bin", &a[4], offset, width),
            );
            let next = if width == 0 {
                t.forward_greedy_token(g[offset])?
            } else {
                t.forward_verify_tokens(&g[offset..offset + width])?[0]
            };
            println!("{offset}\t{width}\t{next}\t{}", g[offset + 1]);
        }
    }
    Ok(())
}
