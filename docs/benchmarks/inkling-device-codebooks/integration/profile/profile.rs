use leansonic_core::inkling2_gpu::Inkling2GpuEngine;
use std::{env,error::Error,fs};
#[link(name="activity")]
unsafe extern "C" {
    fn capture_init();
    fn gpu_clock()->u64;
    fn capture_finish();
}
#[link(name="cuda")]
unsafe extern "C" { fn cuCtxSynchronize()->i32; }
fn timestamp()->u64 { unsafe { assert_eq!(cuCtxSynchronize(),0);gpu_clock() } }
fn ids(file:&str)->Result<Vec<u32>,Box<dyn Error>> {
    Ok(fs::read_to_string(file)?.trim().split(',').map(str::parse).collect::<Result<_,_>>()?)
}
fn main()->Result<(),Box<dyn Error>> {
    let args:Vec<_>=env::args().skip(1).collect();
    let [model, pre, cont]=args.as_slice() else {return Err("expected MODEL PROMPT CONTINUATION".into())};
    let pre=ids(pre)?;let cont=ids(cont)?;
    assert_eq!((pre.len(),cont.len()),(1024,32));
    let mut engine=Inkling2GpuEngine::load_from_dir_with_spec(model,8192,true,0)?;
    unsafe { capture_init(); }
    let mut control=Vec::new();
    for run in 0..3 {
        engine.reset()?;
        let start=timestamp();
        let (mut next,_)=engine.prefill_tokens(&pre)?;
        let middle=timestamp();
        let mut predictions=Vec::new();
        for &token in &cont { predictions.push(next);next=engine.forward_greedy_token(token)?; }
        let end=timestamp();
        if run==0 {control=predictions.clone();} else {assert_eq!(control,predictions);}
        println!("{run} {start} {middle} {end} {}",predictions.iter().map(u32::to_string).collect::<Vec<_>>().join(","));
    }
    unsafe {capture_finish();}
    Ok(())
}
