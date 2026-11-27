use leansonic_core::inkling2_gpu::Inkling2GpuEngine;
use std::{env,error::Error,fs,io::Write};
fn main()->Result<(),Box<dyn Error>>{
    let a: Vec<_>=env::args().collect();
    let mut e=Inkling2GpuEngine::load_from_dir_with_spec(&a[1],2048,true,a[2].parse()?)?;
    fs::create_dir(&a[3])?;
    for case in ["q00_code_merge","q11_json_schema","ja_reasoning","readme1024"] {
        let ids=fs::read_to_string(format!("{case}.ids"))?.split_whitespace().map(str::parse).collect::<Result<Vec<u32>,_>>()?;
        e.reset()?;let (first,_) =e.prefill_tokens(&ids)?;
        eprintln!("case,{case}");
        let (tokens,next)=e.generate_greedy_tokens(first,128,None,&ids,|_|{})?;
        let logits=e.read_last_logits()?;
        assert_eq!(logits.len(),200058);assert!(logits.iter().all(|x|x.is_finite()));
        let argmax=logits.iter().enumerate().max_by(|a,b|a.1.total_cmp(b.1)).unwrap().0 as u32;
        assert_eq!(argmax,next);
        let path=format!("{}/{case}.f32",&a[3]);
        let mut file=std::io::BufWriter::new(fs::File::create_new(path)?);
        for v in logits {file.write_all(&v.to_le_bytes())?;}
        file.flush()?;
        println!("{case},{next},{},{},{},{}",e.spec_drafted_total,e.spec_accepted_total,e.spec_verify_steps,tokens.iter().map(u32::to_string).collect::<Vec<_>>().join(" "));
    }
    Ok(())
}
