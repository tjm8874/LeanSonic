use leansonic_core::qwen38_gpu::Qwen38GpuEngine;
use std::{env, error::Error, fs, path::Path};
fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<_> = env::args().collect();
    let ids = |s: &str| -> Result<Vec<u32>, Box<dyn Error>> {
        Ok(fs::read_to_string(s)?
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<Vec<_>, _>>()?)
    };
    let prompt = ids(&a[2])?;
    let generated = ids(&a[3])?;
    let mut eng = Qwen38GpuEngine::load_from_dir_with_opts(&a[1], 256, false)?;
    for arm in ["serial", "verify"] {
        env::remove_var("Q38_LAYER_TRACE");
        eng.reset()?;
        let (first, _) = eng.prefill_tokens(&prompt)?;
        let trace = Path::new(&a[4]).join(format!("{arm}.bin"));
        assert!(!trace.exists());
        env::set_var("Q38_LAYER_TRACE", trace);
        let mut observed = vec![first];
        for &t in &generated[..80] {
            observed.push(if arm == "serial" {
                eng.forward_greedy_token(t)?
            } else {
                eng.forward_verify_tokens(&[t])?[0]
            });
        }
        println!("{arm}\t{observed:?}");
    }
    Ok(())
}
