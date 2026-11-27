use leansonic_core::inkling_embedding::{unpack_embedding,unpack_rank_embedding};
use std::{env,fs,mem::MaybeUninit,time::Instant};
fn main() {
    let args:Vec<_>=env::args().collect();
    let frame=fs::read(&args[2]).unwrap();
    let mut words=vec![MaybeUninit::uninit();201024*2304*2];
    let start=Instant::now();
    match args[1].as_str() {
        "rank"=>unpack_rank_embedding(&frame,&mut words),
        "prefix"=>unpack_embedding(&frame,&mut words),
        _=>panic!("expected rank or prefix")
    }.unwrap();
    println!("decode_ms={:.3}",start.elapsed().as_secs_f64()*1000.);
    // Both checked decoders initialize the entire destination on success.
    let raw=unsafe{std::slice::from_raw_parts(words.as_ptr().cast::<u8>(),words.len())};
    fs::write(&args[3],raw).unwrap();
}
