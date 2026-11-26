// The packer seeks to the validated BF16 tensor and connects its file descriptors
// to this CPU-only helper. All codes precede all scales in the output shard.
use leansonic_core::inkling2_gpu::pack_q8_rows;
use std::io::{self, Read, Write};

fn main() -> io::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if let [mode, size] = args.as_slice() {
        if matches!(mode.as_str(), "scale-size" | "scale-pack") {
            let n: usize = size.parse().map_err(io::Error::other)?;
            if n > 128 * 2304 * 144 {
                return Err(io::Error::other("invalid scale tensor length"));
            }
            let mut values = vec![0; n];
            io::stdin().lock().read_exact(&mut values)?;
            let packing = leansonic_core::inkling_scales::ScalePacking::choose(&values)
                .map_err(io::Error::other)?;
            if mode == "scale-size" {
                println!("{}", packing.encoded_len());
                return Ok(());
            }
            let mut dst = io::BufWriter::new(io::stdout().lock());
            packing.write(&values, &mut dst)?;
            return dst.flush();
        }
    }
    let dimensions = args.as_slice();
    let (rows, cols) = match dimensions {
        [rows, cols] => (rows.parse::<usize>().ok(), cols.parse::<usize>().ok()),
        _ => (None, None),
    };
    let (Some(rows), Some(cols)) = (rows, cols) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "expected row and column counts",
        ));
    };
    if rows == 0 || rows > 200_058 || !matches!(cols, 2304 | 3072) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid Inkling projection dimensions",
        ));
    }
    let mut src = io::stdin().lock();
    let mut dst = io::stdout().lock();
    let mut bf16 = vec![0; 1024.min(rows) * cols * 2];
    let mut codes = vec![0; 1024.min(rows) * cols];
    let mut scales = vec![0; rows * (cols / 64) * 4];
    for first in (0..rows).step_by(1024) {
        let elements = (rows - first).min(1024) * cols;
        let src_tile = &mut bf16[..elements * 2];
        let codes_tile = &mut codes[..elements];
        let offset = first * (cols / 64) * 4;
        src.read_exact(src_tile)?;
        pack_q8_rows(
            src_tile,
            cols,
            codes_tile,
            &mut scales[offset..offset + elements / 64 * 4],
        );
        dst.write_all(codes_tile)?;
    }
    dst.write_all(&scales)?;
    dst.flush()
}
