use leansonic_core::inkling_mxfp8::{pack, unpack_nvfp4_codes};
use std::{env, fs::File, io::BufWriter};
fn main() {
    let args: Vec<_> = env::args().collect();
    assert_eq!(args.len(),4);
    assert_eq!(args[1],"pack");
    let data=std::fs::read(&args[2]).unwrap();
    let size=pack(&data,BufWriter::new(File::create_new(&args[3]).unwrap())).unwrap();
    let coded=std::fs::read(&args[3]).unwrap();
    assert_eq!(coded.len(),size);
    assert_eq!(unpack_nvfp4_codes(&coded,data.len(),false).unwrap(),data);
}
