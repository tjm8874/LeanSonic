use leansonic_core::inkling2_gpu::Inkling2GpuEngine;
use std::{env,error::Error,fs,io::{BufWriter,Write}};
fn main() -> Result<(),Box<dyn Error>> {
    let arg: Vec<_> = env::args().skip(1).collect();
    let [model,fixture,output]=arg.as_slice() else{return Err("expected MODEL FIXTURE OUT".into());};
    let mut engine=Inkling2GpuEngine::load_from_dir_with_spec(model,2048,true,0)?;
    let mut writer=BufWriter::new(fs::File::create_new(output)?);
    for line in fs::read_to_string(fixture)?.lines() {
        let ids: Vec<u32>=line.split(',').map(str::parse).collect::<Result<_,_>>()?;
        assert_eq!(ids.len(),1056);
        engine.reset()?;
        let (mut next,_)=engine.prefill_tokens(&ids[..1024])?;
        for step in 0..=32 {
            let scores=engine.read_last_logits()?;
            assert_eq!(scores.len(),200058);
            assert!(scores.iter().all(|v|v.is_finite()));
            let top=scores.iter().enumerate().fold(0,|best,(i,&x)|if x>scores[best]{i}else{best});
            assert_eq!(top,next as usize);
            let mut bytes=Vec::with_capacity(scores.len()*4);
            for value in scores{bytes.extend_from_slice(&value.to_le_bytes());}
            writer.write_all(&bytes)?;
            if step<32 {next=engine.forward_greedy_token(ids[1024+step])?;}
        }
    }
    writer.flush()?;
    Ok(())
}
