// The packer seeks to the validated BF16 tensor and connects its file descriptors
// to this CPU-only helper. All codes precede all scales in the output shard.
use leansonic_core::inkling2_gpu::{pack_q8_rows, pack_q8_rows_refined};
use std::io::{self, Read, Write};

fn main() -> io::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if let [mode, size] = args.as_slice() {
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
            "mxfp8-size" | "mxfp8-pack" | "rans-size" | "rans-pack"
        ) {
            let n: usize = size.parse().map_err(io::Error::other)?;
            if n == 0 || n > 128 * 2304 * 2304 {
                return Err(io::Error::other("invalid MXFP8 matrix size"));
            }
            let mut raw = vec![0; n];
            io::stdin().lock().read_exact(&mut raw)?;
            if mode.ends_with("-size") {
                println!("{}", leansonic_core::inkling_mxfp8::pack(&raw, io::sink())?);
            } else {
                let mut dst = io::BufWriter::new(io::stdout().lock());
                leansonic_core::inkling_mxfp8::pack(&raw, &mut dst)?;
                dst.flush()?;
            }
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
    let (refine, args) = match args.as_slice() {
        [flag, args @ ..] if flag == "refine-output" => (true, args),
        args => (false, args),
    };
    let pack_rows = if refine {
        pack_q8_rows_refined
    } else {
        pack_q8_rows
    };
    let (mode, dimensions) = match args {
        [mode, dims @ ..] if matches!(mode.as_str(), "matrix-dict-size" | "matrix-dict-pack") => {
            (mode.as_str(), dims)
        }
        dims => ("raw", dims),
    };
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
        pack_rows(
            src_tile,
            cols,
            codes_tile,
            &mut scales[offset..offset + elements / 64 * 4],
        );
        if mode != "matrix-dict-size" {
            dst.write_all(codes_tile)?;
        }
    }
    if mode == "raw" {
        dst.write_all(&scales)?;
    } else {
        let packed = leansonic_core::inkling_q8_scales::pack(&scales).map_err(io::Error::other)?;
        if mode == "matrix-dict-size" {
            writeln!(dst, "{}", packed.len())?;
        } else {
            dst.write_all(&packed)?;
        }
    }
    dst.flush()
}
