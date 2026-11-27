use leansonic_core::inkling_embedding::unpack_magnitude_embedding;
use std::{env,fs,mem::MaybeUninit,time::Instant};
fn main() {
    let args:Vec<_>=env::args().collect();
    let frame=fs::read(&args[1]).unwrap();
    let mut words=vec![MaybeUninit::uninit();201024*2304*2];
    let start=Instant::now();
    unpack_magnitude_embedding(&frame,&mut words).unwrap();
    println!("decode_ms={:.3}",start.elapsed().as_secs_f64()*1e3);
    // The checked decoder initialized the complete BF16 buffer.
    let raw=unsafe{std::slice::from_raw_parts(words.as_ptr().cast::<u8>(),words.len())};
    fs::write(&args[2],raw).unwrap();
}
