use leansonic_core::inkling_embedding::{pack_lut8_rans, unpack_lut8_rans};
use std::{env, fs, mem::MaybeUninit};
fn main() {
    let args: Vec<_> = env::args().skip(1).collect();
    let count = 201024usize * 2304;
    if args[0] == "encode" {
        let coded=pack_lut8_rans(&fs::read(&args[1]).unwrap(), count).unwrap();
        fs::write(&args[2],coded).unwrap();
    } else {
        assert_eq!(args[0],"decode");
        let mut dst=vec![MaybeUninit::uninit();2*count];
        unpack_lut8_rans(&fs::read(&args[1]).unwrap(),&mut dst).unwrap();
        // Successful decode initializes every BF16 byte.
        let bytes=unsafe{std::slice::from_raw_parts(dst.as_ptr().cast::<u8>(),dst.len())};
        fs::write(&args[2],bytes).unwrap();
    }
}
