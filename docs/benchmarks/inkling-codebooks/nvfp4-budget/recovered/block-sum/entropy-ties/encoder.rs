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
// coded cost and then energy change break signed-error ties, with the old map
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

// Use the same byte position and largest prior magnitude as the direct codec.
fn table_index(column: usize, largest: usize) -> usize {
    column * 8 + largest
}
fn largest_after(largest: usize, value: u8) -> usize {
    largest
        .max(usize::from(value & 7))
        .max(usize::from((value >> 4) & 7))
}
type BitCosts = Vec<[u16; 256]>;

fn fit_baseline(raw: &[u8], scales: &[u8], limits: &[u8], baseline: &[Choice; 256]) -> BitCosts {
    let mut counts = vec![[1u64; 256]; 64];
    for (expert, bytes) in raw.chunks_exact(raw.len() / 128).enumerate() {
        let first = expert * bytes.len() / 8;
        for (i, block) in bytes.as_chunks::<8>().0.iter().enumerate() {
            if limits[expert] == 255 || scales[first + i] > limits[expert] {
                continue;
            }
            let mut largest = 0;
            for (col, &original) in block.iter().enumerate() {
                let value = baseline[usize::from(original)].byte;
                counts[table_index(col, largest)][usize::from(value)] += 1;
                largest = largest_after(largest, value);
            }
        }
    }
    counts
        .into_iter()
        .map(|row| {
            let total: u64 = row.iter().sum();
            row.map(|count| (256. * (total as f64 / count as f64).log2()).round() as u16)
        })
        .collect()
}

fn closest_sum(block: &[u8; 8], tables: &[[Choice; 256]; 4], costs: &BitCosts) -> usize {
    let mut best = (i16::MAX, u32::MAX, i16::MAX);
    let mut selected = 0;
    for (mode, table) in tables.iter().enumerate() {
        let (mut sum, mut energy, mut cost, mut largest) = (0i16, 0i16, 0u32, 0usize);
        for (col, &original) in block.iter().enumerate() {
            let e = table[usize::from(original)];
            sum += e.sum;
            energy += e.energy;
            cost += u32::from(costs[table_index(col, largest)][usize::from(e.byte)]);
            largest = largest_after(largest, e.byte);
        }
        let error = (sum.abs(), cost, energy.abs());
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
    let costs = fit_baseline(&v, &scales, &limits, &maps[0]);
    let mut modes = [0u64; 4];
    let mut original_signature_error = 0u64;
    let mut selected_signature_error = 0u64;
    let mut changed = 0u64;
    for (expert, groups) in v.chunks_exact_mut(n / 128).enumerate() {
        let base = expert * (n / 128 / 8);
        let limit = limits[expert];
        for (i, w) in groups.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            if limit != 255 && scales[base + i] <= limit {
                let mode = closest_sum(w, &maps, &costs);
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
