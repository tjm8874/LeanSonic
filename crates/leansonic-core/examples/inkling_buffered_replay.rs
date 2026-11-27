use leansonic_core::inkling2_gpu::Inkling2GpuEngine;
use std::{collections::BTreeMap, env, error::Error, fs, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    let base: Vec<u32> = fs::read_to_string(&args[2])?
        .trim()
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    assert_eq!(base.len(), 1024);
    let mut e = Inkling2GpuEngine::load_from_dir_with_spec(&args[1], 2304, true, 0)?;
    let mut reference = BTreeMap::new();
    for cycle in 0..5 {
        let arms = if cycle % 2 == 0 {
            [false, true, true, false]
        } else {
            [true, false, false, true]
        };
        for (leg, buffered) in arms.into_iter().enumerate() {
            for (prefix, count) in [
                (32, 0),
                (32, 1),
                (32, 63),
                (32, 64),
                (32, 65),
                (1024, 129),
                (2046, 65),
                (2048, 64),
            ] {
                let prompt: Vec<u32> = base.iter().copied().cycle().take(prefix).collect();
                e.reset()?;
                let t = Instant::now();
                let (first, _) = e.prefill_tokens(&prompt)?;
                let pre = t.elapsed();
                if buffered && prefix == 32 && count == 0 {
                    assert!(e
                        .generate_fixed_greedy_tokens(first, usize::MAX, &prompt)
                        .is_err());
                }
                let t = Instant::now();
                let (ids, next) = if buffered {
                    e.generate_fixed_greedy_tokens(first, count, &prompt)?
                } else {
                    e.generate_greedy_tokens(first, count, None, &prompt, |_| {})?
                };
                let dec = t.elapsed();
                assert_eq!(ids.len(), count);
                let logits = e.read_last_logits()?;
                assert!(logits.iter().all(|x| x.is_finite()));
                let bits: Vec<_> = logits.iter().map(|x| x.to_bits()).collect();
                let after = e.forward_greedy_token(next)?;
                let sig = (ids.clone(), next, bits, after);
                if let Some(old) = reference.get(&(prefix, count)) {
                    if old != &sig {
                        return Err(format!("different token or FP32 logit at cycle={cycle} leg={leg}, {prefix}+{count}").into());
                    }
                } else {
                    reference.insert((prefix, count), sig);
                }
                let hash = logits.iter().fold(14695981039346656037u64, |v, x| {
                    (v ^ u64::from(x.to_bits())).wrapping_mul(1099511628211)
                });
                println!("{cycle},{leg},{buffered},{prefix},{count},{:.6},{:.6},{next},{after},{hash:016x},{}",pre.as_secs_f64()*1000.,dec.as_secs_f64()*1000.,ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" "));
            }
        }
    }
    Ok(())
}
