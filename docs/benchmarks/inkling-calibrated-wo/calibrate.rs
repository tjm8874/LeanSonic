use leansonic_core::inkling2_gpu::Inkling2GpuEngine;
use std::{env, error::Error, fs, path::Path};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [model, inputs] = args.as_slice() else {
        return Err("expected model and input windows".into());
    };
    if Path::new("activations.bin").exists() {
        return Err("use a fresh calibration output directory".into());
    }
    let mut engine = Inkling2GpuEngine::load_from_dir_with_spec(model, 512, true, 0)?;
    let source = fs::read_to_string(inputs)?;
    assert_eq!(source.lines().count(), 32);
    for line in source.lines() {
        let ids = line
            .split(',')
            .map(str::parse::<u32>)
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(ids.len(), 512);
        engine.reset()?;
        let (prediction, _) = engine.prefill_tokens(&ids)?;
        println!("{prediction}");
    }
    Ok(())
}
