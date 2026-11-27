use leansonic_core::inkling_embedding::{pack_lut8_rans, unpack_lut8_embedding, unpack_lut8_rans};
use std::{env, fs, mem::MaybeUninit, time::Instant};
const COUNT: usize = 201024 * 2304;
fn main() {
    let args: Vec<_> = env::args().skip(1).collect();
    let raw = fs::read(&args[0]).unwrap();
    let start = Instant::now();
    let coded = pack_lut8_rans(&raw, COUNT).unwrap();
    println!("raw={} coded={} encode_ms={:.3}", raw.len(), coded.len(),start.elapsed().as_secs_f64()*1e3);
    fs::write(&args[1],&coded).unwrap();
    let mut plain_output = vec![MaybeUninit::uninit();COUNT*2];
    let mut compressed_output = plain_output.clone();
    for cycle in 0..5 {
        for (leg,trial) in (if cycle%2==0 {[false,true,true,false]} else {[true,false,false,true]}).into_iter().enumerate() {
            let start=Instant::now();
            if trial {
                unpack_lut8_rans(&coded,&mut compressed_output).unwrap();
            } else {
                unpack_lut8_embedding(&raw,&mut plain_output).unwrap();
            }
            println!("decode,{cycle},{leg},{trial},{:.3}",start.elapsed().as_secs_f64()*1e3);
        }
        // Each decoder initialized the entire allocation. Compare all bytes on every cycle.
        unsafe {
            let a=std::slice::from_raw_parts(plain_output.as_ptr().cast::<u8>(),COUNT*2);
            let b=std::slice::from_raw_parts(compressed_output.as_ptr().cast::<u8>(),COUNT*2);
            assert_eq!(a,b);
            if cycle==0 {fs::write(&args[2],b).unwrap();}
        }
    }
}
