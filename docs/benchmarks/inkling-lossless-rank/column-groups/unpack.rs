use leansonic_core::inkling_mxfp8::unpack_nvfp4_codes;
fn main() {
    let args: Vec<_> = std::env::args_os().collect();
    let encoded = std::fs::read(&args[1]).unwrap();
    let count = args[2].to_str().unwrap().parse::<usize>().unwrap();
    let decoded = unpack_nvfp4_codes(&encoded, count, false).unwrap();
    std::fs::write(&args[3], decoded).unwrap();
}
