//! Offline fit of expert-specific scale boundaries against the preceding frame's rANS tables.
const EXPERTS: usize = 128;
const CONTEXTS: usize = 64;
const SYMBOLS: usize = 256;

// For each symbol store the estimated cost of the lower bank minus the upper bank,
// in 1/256 bits. A missing symbol gets half a frequency count for fitting only.
// The encoder rebuilds its normalized tables from the complete reassigned weights.
fn cost_differences(tables: &[u8]) -> Result<Vec<i16>, &'static str> {
    let n = CONTEXTS * SYMBOLS;
    let lookup = (0..=4096)
        .map(|f| (-f64::from(f).max(0.5).log2() * 256.).round() as i16)
        .collect::<Vec<_>>();
    let mut costs = Vec::with_capacity(2 * n);
    for pair in tables.as_chunks::<{ 4 * CONTEXTS * SYMBOLS }>().0 {
        let (low, high) = pair.split_at(2 * n);
        for (&a, &b) in low.as_chunks::<2>().0.iter().zip(high.as_chunks::<2>().0) {
            let get = |bytes| {
                lookup
                    .get(usize::from(u16::from_le_bytes(bytes)))
                    .ok_or("invalid rANS frequency")
            };
            costs.push(*get(a)? - *get(b)?);
        }
    }
    Ok(costs)
}

fn boundary(diff: &[i64; 128], start: u8, end: u8, old: u8) -> u8 {
    let mut delta = 0;
    let mut best = (0, old.abs_diff(start), start);
    for c in start..end {
        delta += diff[usize::from(c)];
        let cut = c + 1;
        best = best.min((delta, old.abs_diff(cut), cut));
    }
    best.2
}

pub(super) fn fit(raw: &[u8], scales: &[u8], frame: &[u8]) -> Result<[u8; 384], &'static str> {
    let (initial, payload) = frame
        .split_first_chunk::<384>()
        .ok_or("missing four-bank boundaries")?;
    let tables = payload
        .get(..4 * CONTEXTS * SYMBOLS * 2)
        .ok_or("missing four-bank frequency tables")?;
    if scales.is_empty()
        || !scales.len().is_multiple_of(EXPERTS)
        || scales.len().checked_mul(8) != Some(raw.len())
    {
        return Err("invalid expert weight or scale dimensions");
    }
    let costs = cost_differences(tables)?;
    let mut cuts = *initial;
    let per_expert = scales.len() / EXPERTS;
    for (e, (expert, values)) in raw
        .chunks_exact(per_expert * 8)
        .zip(scales.chunks_exact(per_expert))
        .enumerate()
    {
        let cut = cuts[e];
        let low = cuts[EXPERTS + e];
        let high = cuts[2 * EXPERTS + e];
        if low > cut || high < cut || high > 127 {
            return Err("invalid source boundaries");
        }
        let mut diff = [0i64; 128];
        for (block, &s) in expert.as_chunks::<8>().0.iter().zip(values) {
            let pair = usize::from(s >= cut) * CONTEXTS * SYMBOLS;
            let mut prev = 0usize;
            let mut delta = 0i32;
            for (col, &v) in block.iter().enumerate() {
                delta += i32::from(costs[pair + (col * 8 + prev) * SYMBOLS + usize::from(v)]);
                prev = prev.max(usize::from((v & 7).max((v >> 4) & 7)));
            }
            *diff.get_mut(usize::from(s)).ok_or("invalid scale code")? += i64::from(delta);
        }
        cuts[EXPERTS + e] = boundary(&diff, 0, cut, low);
        cuts[2 * EXPERTS + e] = boundary(&diff, cut, 127, high);
    }
    Ok(cuts)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fit_minimizes_cost_and_preserves_old_cut_on_ties() {
        let mut costs = [0; 128];
        costs[64] = -8;
        costs[65] = 2;
        costs[66] = 7;
        assert_eq!(boundary(&costs, 64, 70, 68), 65);
        assert_eq!(boundary(&costs, 70, 127, 80), 80);
        assert_eq!(boundary(&costs, 0, 0, 0), 0);
    }
    #[test]
    fn learns_expert_specific_splits_and_preserves_weight_bytes() {
        use leansonic_core::inkling_mxfp8::{pack_four_scale_boundaries, unpack_four_scale_banks};
        let mut raw = Vec::new();
        let mut scales = Vec::new();
        let mut cuts = [0; 384];
        for expert in 0..128 {
            cuts[expert] = 64;
            cuts[128 + expert] = 32;
            cuts[256 + expert] = 96;
            let shift = if expert % 2 == 0 { 4 } else { -4 };
            for scale in 0..128 {
                scales.push(scale as u8);
                let symbol = if scale < 32 + shift {
                    0x77
                } else if scale < 64 {
                    0x22
                } else if scale < 96 + shift {
                    0xff
                } else {
                    0x44
                };
                raw.extend([symbol; 8]);
            }
        }
        let mut initial = Vec::new();
        pack_four_scale_boundaries(&raw, scales.clone(), &cuts, &mut initial).unwrap();
        let fitted = fit(&raw, &scales, &initial).unwrap();
        assert_eq!(fitted[..128], cuts[..128]);
        for expert in 0..128 {
            let delta = if expert % 2 == 0 { 4 } else { -4 };
            assert_eq!(i32::from(fitted[128 + expert]), 32 + delta);
            assert_eq!(i32::from(fitted[256 + expert]), 96 + delta);
        }
        let mut output = Vec::new();
        pack_four_scale_boundaries(&raw, scales.clone(), &fitted, &mut output).unwrap();
        assert!(output.len() < initial.len());
        assert_eq!(unpack_four_scale_banks(&output, scales).unwrap(), raw);
    }
}
