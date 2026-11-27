// The packer seeks to the validated BF16 tensor and connects its file descriptors
// to this CPU-only helper. All codes precede all scales in the output shard.
use leansonic_core::inkling2_gpu::pack_q8_rows;
use std::io::{self, Read, Write};

fn main() -> io::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if let [mode, size, width] = args.as_slice() {
        if matches!(
            mode.as_str(),
            "neighbor-size" | "neighbor-pack" | "mean8-size" | "mean8-pack"
        ) {
            let n: usize = size.parse().map_err(io::Error::other)?;
            let width = width.parse().map_err(io::Error::other)?;
            if n > 128 * 2304 * 144 {
                return Err(io::Error::other("invalid scale matrix size"));
            }
            let mut raw = vec![0; n];
            io::stdin().lock().read_exact(&mut raw)?;
            let sizing = mode.ends_with("-size");
            let mut stdout = io::BufWriter::new(io::stdout().lock());
            let mut sink = io::sink();
            let dst: &mut dyn Write = if sizing { &mut sink } else { &mut stdout };
            let bytes = if mode.starts_with("mean8-") {
                leansonic_core::inkling_mxfp8::pack_mean_scales(&raw, width, dst)?
            } else {
                leansonic_core::inkling_mxfp8::pack_scale_context(&raw, width, dst)?
            };
            if sizing {
                writeln!(stdout, "{bytes}")?;
            }
            stdout.flush()?;
            return Ok(());
        }
    }
    if let [mode, size] = args.as_slice() {
        if matches!(mode.as_str(), "bf16planes-size" | "bf16planes-pack") {
            let n: usize = size.parse().map_err(io::Error::other)?;
            if n > 128 * 2304 * 2 {
                return Err(io::Error::other("unexpected router/controller length"));
            }
            let mut raw = vec![0; n];
            io::stdin().lock().read_exact(&mut raw)?;
            let frame = leansonic_core::inkling_bf16_planes::pack(&raw)?;
            if mode == "bf16planes-size" {
                println!("{}", frame.len());
            } else {
                io::stdout().lock().write_all(&frame)?;
            }
            return Ok(());
        }
        if matches!(mode.as_str(), "q8dict-size" | "q8dict-pack") {
            let n: usize = size.parse().map_err(io::Error::other)?;
            if n > 200_058 * 36 * 4 {
                return Err(io::Error::other("invalid Q8 scale length"));
            }
            let mut raw = vec![0; n];
            io::stdin().lock().read_exact(&mut raw)?;
            let packed = leansonic_core::inkling_q8_scales::pack(&raw).map_err(io::Error::other)?;
            if mode == "q8dict-size" {
                println!("{}", packed.len());
            } else {
                io::stdout().lock().write_all(&packed)?;
            }
            return Ok(());
        }
        if matches!(
            mode.as_str(),
            "mxfp8-size" | "mxfp8-pack" | "rans-size" | "rans-pack" | "nvctx-size" | "nvctx-pack"
        ) {
            let n: usize = size.parse().map_err(io::Error::other)?;
            if n == 0 || n > 128 * 2304 * 2304 {
                return Err(io::Error::other("invalid MXFP8 matrix size"));
            }
            let mut raw = vec![0; n];
            io::stdin().lock().read_exact(&mut raw)?;
            let mut dst = io::BufWriter::new(io::stdout().lock());
            let sizing = mode.ends_with("-size");
            let mut sink = io::sink();
            let output: &mut dyn Write = if sizing { &mut sink } else { &mut dst };
            let size = if mode.starts_with("nvctx-") {
                leansonic_core::inkling_mxfp8::pack_nvfp4_context(&raw, output)?
            } else {
                leansonic_core::inkling_mxfp8::pack(&raw, output)?
            };
            if sizing {
                writeln!(dst, "{size}")?;
            }
            dst.flush()?;
            return Ok(());
        }
        if matches!(mode.as_str(), "embedding-size" | "embedding-pack") {
            let n: usize = size.parse().map_err(io::Error::other)?;
            if n > 201_024 * 2304 * 2 {
                return Err(io::Error::other("invalid embedding length"));
            }
            let src = io::stdin().lock();
            if mode == "embedding-size" {
                let size = leansonic_core::inkling_embedding::pack_embedding(src, n, io::sink())?;
                println!("{size}");
            } else {
                let mut dst = io::BufWriter::new(io::stdout().lock());
                leansonic_core::inkling_embedding::pack_embedding(src, n, &mut dst)?;
                dst.flush()?;
            }
            return Ok(());
        }
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
    let (mode, encode_codes, encode_scales, dimensions) = match args.as_slice() {
        [mode, codes, scales, dims @ ..]
            if matches!(mode.as_str(), "matrix-size" | "matrix-pack")
                && matches!(codes.as_str(), "raw" | "rans")
                && matches!(scales.as_str(), "raw" | "dict") =>
        {
            (mode.as_str(), codes == "rans", scales == "dict", dims)
        }
        [mode, dims @ ..] if matches!(mode.as_str(), "matrix-dict-size" | "matrix-dict-pack") => {
            (mode.as_str(), false, true, dims)
        }
        dims => ("raw", false, false, dims),
    };
    let sizing = mode.ends_with("-size");
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
    // The encoder needs a histogram for the whole code plane. Only offline
    // entropy packing collects it; raw conversion continues to stream tiles.
    let mut all_codes = Vec::with_capacity(if encode_codes { rows * cols } else { 0 });
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
        if encode_codes {
            all_codes.extend_from_slice(codes_tile);
        } else if !sizing {
            dst.write_all(codes_tile)?;
        }
    }
    let code_len = if !encode_codes {
        rows * cols
    } else if sizing {
        leansonic_core::inkling_mxfp8::pack(&all_codes, io::sink())?
    } else {
        leansonic_core::inkling_mxfp8::pack(&all_codes, &mut dst)?
    };
    let scale_plane = if encode_scales {
        leansonic_core::inkling_q8_scales::pack(&scales).map_err(io::Error::other)?
    } else {
        scales
    };
    match mode {
        "matrix-size" => writeln!(dst, "{} {}", code_len, scale_plane.len())?,
        "matrix-dict-size" => writeln!(dst, "{}", scale_plane.len())?,
        _ => dst.write_all(&scale_plane)?,
    }
    dst.flush()
}
