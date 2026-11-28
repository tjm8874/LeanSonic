use leansonic_core::{inkling_mxfp8, inkling_q8_scales};
use std::{env, error::Error, fs, io::Write};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [codes, scales, code_frame, scale_frame] = args.as_slice() else {
        return Err("expected CODES SCALES CODE_FRAME SCALE_FRAME".into());
    };
    let codes = fs::read(codes)?;
    let scales = fs::read(scales)?;
    let mut frame = Vec::new();
    inkling_mxfp8::pack(&codes, &mut frame)?;
    if inkling_mxfp8::unpack_nvfp4_codes(&frame, codes.len(), false)? != codes {
        return Err("Q8 code round-trip failed".into());
    }
    let table = inkling_q8_scales::pack(&scales)?;
    if inkling_q8_scales::unpack(&table, scales.len() / 4)? != scales {
        return Err("scale round-trip failed".into());
    }
    fs::File::create_new(code_frame)?.write_all(&frame)?;
    fs::File::create_new(scale_frame)?.write_all(&table)?;
    Ok(())
}
