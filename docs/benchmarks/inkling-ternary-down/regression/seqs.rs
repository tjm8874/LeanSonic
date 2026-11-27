use leansonic_core::inkling2_gpu::Inkling2GpuEngine;
use std::{env, error::Error, fs};
type Engine = Inkling2GpuEngine;
fn prime(e: &mut Engine, ids: &[u32], slot: usize) -> Result<u32, String> {
    e.select_seq(slot)?;
    e.reset()?;
    let n = [31, 64, 129, 257][slot];
    let (v, _) = e.prefill_tokens(&ids[slot * 79..slot * 79 + n])?;
    Ok(v)
}
fn fingerprint(e: &Engine) -> Result<u64, String> {
    let v = e.read_last_logits()?;
    assert_eq!(v.len(), 200058);
    assert!(v.iter().all(|x| x.is_finite()));
    Ok(v.iter().fold(14695981039346656037u64, |h, x| {
        (h ^ u64::from(x.to_bits())).wrapping_mul(1099511628211)
    }))
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    let ids: Vec<u32> = fs::read_to_string(&args[2])?
        .trim()
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let mut e = Engine::load_from_dir_with_spec(&args[1], 512, true, 0)?;
    let order = [2, 0, 3, 1];
    let kept = [1, 3, 4, 2];
    for pass in 0..2 {
        let mut inputs = vec![];
        for s in 0..4 {
            let next = prime(&mut e, &ids, s)?;
            println!("prefix {pass} {s} {next} {:016x}", fingerprint(&e)?);
            let mut tokens = vec![next];
            tokens.extend_from_slice(&ids[620 + s * 9..625 + s * 9]);
            inputs.push(tokens);
        }
        let runs: Vec<_> = order.iter().map(|&s| (s, &inputs[s][..s + 3])).collect();
        let verified = e.verify_round(&runs)?;
        e.commit_verify_round(&order.map(|s| kept[s]))?;
        let step = order.map(|s| ids[920 + s]);
        let decoded = e.decode_round(&order, &step)?;
        for (j, &s) in order.iter().enumerate() {
            e.select_seq(s)?;
            let tok = e.forward_greedy_token(ids[960 + s])?;
            let h = fingerprint(&e)?;
            let _ = prime(&mut e, &ids, s)?;
            let mut serial = vec![];
            for &v in &inputs[s][..kept[s]] {
                serial.push(e.forward_greedy_token(v)?);
            }
            let follow = e.forward_greedy_token(step[j])?;
            let tail = e.forward_greedy_token(ids[960 + s])?;
            println!(
                "sequence {pass} {s} {:?} {} {} {h:016x} {:?} {} {} {:016x}",
                verified[j],
                decoded[j],
                tok,
                serial,
                follow,
                tail,
                fingerprint(&e)?
            );
        }
    }
    Ok(())
}
