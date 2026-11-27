use leansonic_core::inkling_mxfp8::{
    pack_four_scale_boundaries, unpack_four_scale_banks, unpack_mean8_scale_codes,
};
use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
};
fn slice(f: &mut File, offset: &str, len: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    f.seek(SeekFrom::Start(offset.parse()?))?;
    let mut x = vec![0; len.parse()?];
    f.read_exact(&mut x)?;
    Ok(x)
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [path, w, n, s, m, width, ids, output] = args.as_slice() else {
        return Err("expected source, offsets, width, restored IDs, output".into());
    };
    let width: usize = width.parse()?;
    let mut file = File::open(path)?;
    let original = slice(&mut file, w, n)?;
    let packed_scales = slice(&mut file, s, m)?;
    let scales = unpack_mean8_scale_codes(&packed_scales, 128 * 2304 * width, width)?;
    let raw = unpack_four_scale_banks(&original, scales.clone())?;
    let base: &[u8; 384] = original[..384].try_into()?;
    let mut restored = [false; 128];
    for id in ids.split(',') {
        let e: usize = id.parse()?;
        assert!(e < 128 && !restored[e]);
        restored[e] = true;
    }
    let mut best = original.len();
    let mut best_mode = "original";
    println!("original {best}");
    for mode in ["2+2", "3+1-low", "3+1-high", "1+3-low", "1+3-high"] {
        let mut cuts = [0u8; 384];
        for e in 0..128 {
            let lo = base[128 + e];
            let mid = base[e];
            let hi = base[256 + e];
            let (l, c, h) = match (mode, restored[e]) {
                ("2+2", false) => (mid, 127, 127),
                ("2+2", true) => (0, 0, mid),
                ("3+1-low", false) => (lo, mid, 127),
                ("3+1-high", false) => (mid, hi, 127),
                ("3+1-low" | "3+1-high", true) => (0, 0, 0),
                ("1+3-low" | "1+3-high", false) => (127, 127, 127),
                ("1+3-low", true) => (0, lo, mid),
                ("1+3-high", true) => (0, mid, hi),
                _ => unreachable!(),
            };
            cuts[e] = c;
            cuts[128 + e] = l;
            cuts[256 + e] = h;
        }
        let mut frame = Vec::new();
        let size = pack_four_scale_boundaries(&raw, scales.clone(), &cuts, &mut frame)?;
        assert_eq!(frame.len(), size);
        println!("{mode} {size}");
        if size < best {
            let decoded = unpack_four_scale_banks(&frame, scales.clone())?;
            assert!(decoded == raw);
            fs::write(output, &frame)?;
            best = size;
            best_mode = mode;
        }
    }
    println!("selected {best_mode} {best} raw {}", raw.len());
    Ok(())
}
