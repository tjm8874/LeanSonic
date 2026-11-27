use leansonic_core::inkling2_gpu::Inkling2GpuEngine;
use std::{env, error::Error, fs};
#[link(name = "cuda")]
unsafe extern "C" {
    fn cuProfilerStart() -> i32;
    fn cuProfilerStop() -> i32;
    fn cuCtxSynchronize() -> i32;
}
fn read_ids(file: &str) -> Result<Vec<u32>, Box<dyn Error>> {
    let text = fs::read_to_string(file)?;
    Ok(text.trim().split(',').map(str::parse).collect::<Result<_, _>>()?)
}
fn enter() { unsafe { assert_eq!(cuCtxSynchronize(), 0); assert_eq!(cuProfilerStart(), 0); }}
fn leave() { unsafe { assert_eq!(cuCtxSynchronize(), 0); assert_eq!(cuProfilerStop(), 0); }}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [phase,model] = args.as_slice() else { return Err("expected PHASE MODEL".into()); };
    assert!(matches!(phase.as_str(), "prefill" | "decode"));
    let prompt = read_ids("prompt.ids")?;
    let anchor = read_ids("anchor.ids")?;
    assert_eq!((prompt.len(),anchor.len()),(1024,32));
    let mut engine=Inkling2GpuEngine::load_from_dir_with_spec(model,2048,true,0)?;
    let mut control=Vec::new();
    for rep in 0..2 {
        if rep>0 {engine.reset()?;}
        if rep==1 && phase=="prefill" {enter();}
        let (first,_)=engine.prefill_tokens(&prompt)?;
        if rep==1 && phase=="prefill" {leave();}
        let mut predicted=vec![first];
        if rep==1 && phase=="decode" {enter();}
        for &token in &anchor {predicted.push(engine.forward_greedy_token(token)?);}
        if rep==1 && phase=="decode" {leave();}
        if rep==0 {control=predicted;} else {assert_eq!(predicted, control);}
    }
    println!("{control:?}");
    Ok(())
}
