use std::{env, fs, mem::MaybeUninit};
fn main() {
    let a: Vec<_> = env::args().skip(1).collect();
    let frame = fs::read(&a[0]).unwrap();
    let mut dst = vec![MaybeUninit::uninit(); 201024 * 2304 * 2];
    leansonic_core::inkling_embedding::unpack_lut8_embedding(&frame, &mut dst).unwrap();
    // Successful decode initialized every output byte.
    let bytes = unsafe { std::slice::from_raw_parts(dst.as_ptr().cast::<u8>(), dst.len()) };
    fs::write(&a[1], bytes).unwrap();
}
