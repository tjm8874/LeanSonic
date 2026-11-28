use leansonic_core::inkling_mxfp8::{pack_mxscale_codes, unpack_mxscale_codes, unpack_nvfp4_codes};
use std::{env, fs, io::BufWriter, path::Path, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let (codes, scales, dest) = match args.as_slice() {
        [codes, scales, dest] => (fs::read(codes)?, fs::read(scales)?, dest),
        [mode, count, code_frame, scale_frame, dest] if mode == "--from-rans" => {
            let count: usize = count.to_str().ok_or("invalid count")?.parse()?;
            if count == 0 || !count.is_multiple_of(32) {
                return Err("invalid block shape".into());
            }
            (
                unpack_nvfp4_codes(&fs::read(code_frame)?, count, false)?,
                unpack_nvfp4_codes(&fs::read(scale_frame)?, count / 32, false)?,
                dest,
            )
        }
        _ => {
            return Err(
                "expected CODES SCALES DEST or --from-rans COUNT CODE_FRAME SCALE_FRAME DEST"
                    .into(),
            )
        }
    };
    let output = Path::new(dest);
    let mut writer = BufWriter::new(fs::File::create_new(output)?);
    let encoded = pack_mxscale_codes(&codes, &scales, &mut writer)?;
    writer.into_inner()?;
    let frame = fs::read(output)?;
    assert_eq!(encoded, frame.len());
    let start = Instant::now();
    let recovered = unpack_mxscale_codes(&frame, scales)?;
    let elapsed = start.elapsed().as_secs_f64();
    if recovered != codes {
        return Err("reconstructed MXFP8 codes differ from input".into());
    }
    println!("{encoded} {elapsed:.6}");
    Ok(())
}
