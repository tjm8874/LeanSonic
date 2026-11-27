use leansonic_core::inkling_mxfp8::{
    canonicalize_nvfp4_zeros, pack_nvfp4_trim_context, unpack_nvfp4_trim_context,
};
use std::{
    env,
    error::Error,
    fs,
    io::{BufWriter, Read, Seek, SeekFrom},
};
fn read_range(f: &mut fs::File, offset: u64, n: usize) -> std::io::Result<Vec<u8>> {
    let mut v = vec![0; n];
    f.seek(SeekFrom::Start(offset))?;
    f.read_exact(&mut v)?;
    Ok(v)
}
// All four choices have the same elementwise squared quantization error.
// The signed block-sum error controls the output error for a constant input;
// preserving the sum of squares is a secondary criterion, with the old map
// winning exact ties. Doubled E2M1 units make both comparisons exact.
#[derive(Copy, Clone, Default)]
struct Choice {
    byte: u8,
    sum: i16,
    energy: i16,
    changes: u8,
}

fn build_choices() -> [[Choice; 256]; 4] {
    let doubled = [0i16, 1, 2, 3, 4, 6, 8, 12];
    std::array::from_fn(|mode| {
        std::array::from_fn(|code| {
            let mut entry = Choice::default();
            for shift in [0, 4] {
                let original = ((code >> shift) & 15) as u8;
                let m = original & 7;
                let q = match m {
                    1 if mode & 1 != 0 => 2,
                    1 => 0,
                    3 if mode & 2 != 0 => 2,
                    3 => 4,
                    _ => m,
                };
                let sign = if (original & 8) == 0 { 1 } else { -1 };
                let v = doubled[usize::from(m)];
                let y = doubled[usize::from(q)];
                entry.sum += sign * (y - v);
                entry.energy += y * y - v * v;
                entry.changes += u8::from(q != m);
                entry.byte |= (if q == 0 { 0 } else { q | (original & 8) }) << shift;
            }
            entry
        })
    })
}

fn closest_sum(block: &[u8; 8], tables: &[[Choice; 256]; 4]) -> usize {
    let mut best = (i16::MAX, i16::MAX);
    let mut selected = 0;
    for (mode, table) in tables.iter().enumerate() {
        let (mut sum, mut energy) = (0i16, 0i16);
        for &code in block {
            let e = table[usize::from(code)];
            sum += e.sum;
            energy += e.energy;
        }
        let error = (sum.abs(), energy.abs());
        if error < best {
            best = error;
            selected = mode;
        }
    }
    selected
}

fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<_> = env::args().skip(1).collect();
    let [path, wo, so, n, cut, out] = a.as_slice() else {
        return Err("usage: official weight_offset scale_offset byte_count cutoffs output".into());
    };
    let mut f = fs::File::open(path)?;
    let n: usize = n.parse()?;
    if ![128 * 2304 * 1152, 128 * 2304 * 576].contains(&n) {
        return Err("invalid plane".into());
    }
    let mut v = read_range(&mut f, wo.parse()?, n)?;
    let scales = read_range(&mut f, so.parse()?, n / 8)?;
    let limits = fs::read(cut)?;
    if limits.len() != 128 {
        return Err("128 cutoffs required".into());
    }
    canonicalize_nvfp4_zeros(&mut v);
    let maps = build_choices();
    let control = env::var_os("INKLING_BASELINE_ROUND").is_some();
    let mut modes = [0u64; 4];
    let mut original_signature_error = 0u64;
    let mut selected_signature_error = 0u64;
    let mut changed = 0u64;
    for (expert, groups) in v.chunks_exact_mut(n / 128).enumerate() {
        let base = expert * (n / 128 / 8);
        let limit = limits[expert];
        for (i, w) in groups.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            if limit != 255 && scales[base + i] <= limit {
                let mode = if control { 0 } else { closest_sum(w, &maps) };
                modes[mode] += 1;
                let (mut before, mut after) = (0i16, 0i16);
                for x in w {
                    let i = usize::from(*x);
                    before += maps[0][i].sum;
                    after += maps[mode][i].sum;
                    changed += u64::from(maps[mode][i].changes);
                    *x = maps[mode][i].byte;
                }
                original_signature_error += u64::from(before.unsigned_abs());
                selected_signature_error += u64::from(after.unsigned_abs());
            }
        }
    }
    let len = pack_nvfp4_trim_context(&v, BufWriter::new(fs::File::create_new(out)?))?;
    let data = fs::read(out)?;
    if data.len() != len || unpack_nvfp4_trim_context(&data, n)? != v {
        return Err("frame verification failed".into());
    }
    assert!(selected_signature_error <= original_signature_error);
    eprintln!(
        "maps {modes:?} original_absolute_sum_error={original_signature_error} new_absolute_sum_error={selected_signature_error}"
    );
    println!("{len} {changed}");
    Ok(())
}
