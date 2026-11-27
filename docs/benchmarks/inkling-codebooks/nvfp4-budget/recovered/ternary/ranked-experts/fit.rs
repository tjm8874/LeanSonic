//! Restore complete ranked experts from the higher-precision frame under an exact disk budget.
use leansonic_core::inkling_mxfp8::{
    pack_mean_scales, pack_scale_direct, unpack_mean8_scale_codes, unpack_scale_direct,
};
use std::{env, error::Error, fs, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [folder, width, cap] = args.as_slice() else {
        return Err("expected work folder, scale width, maximum combined frame bytes".into());
    };
    let dir = Path::new(folder);
    let width: usize = width.parse()?;
    let cap: usize = cap.parse()?;
    if !matches!(width, 72 | 144) {
        return Err("unsupported expert scale width".into());
    }
    let rank = fs::read(dir.join("rank.bin"))?;
    let mut check = rank.clone();
    check.sort_unstable();
    assert_eq!(check, (0u8..128).collect::<Vec<_>>());
    let stride = 2304 * width;
    let load = |name: &str| -> Result<_, Box<dyn Error>> {
        let frame = fs::read(dir.join(format!("{name}.weights")))?;
        let cuts: [u8; 128] = frame.get(..128).ok_or("missing cutoffs")?.try_into()?;
        let scales = unpack_mean8_scale_codes(
            &fs::read(dir.join(format!("{name}.scales")))?,
            128 * stride,
            width,
        )?;
        let weights = unpack_scale_direct(&frame, scales.clone())?;
        assert_eq!(weights.len(), 128 * stride * 8);
        Ok((cuts, scales, weights))
    };
    let (mut cuts, mut scales, mut weights) = load("ternary")?;
    let (base_cuts, base_scales, base_weights) = load("control")?;
    assert_eq!(cuts, base_cuts);
    let original_scales = scales.clone();
    let original_weights = weights.clone();
    let (mut low, mut high, mut selected) = (0usize, 128usize, 0usize);
    // Feasibility is checked on complete physical frames at every trial. Search
    // only prefixes in the predetermined training rank; keep the best tried fit.
    while low + 1 < high {
        let mid = (low + high) / 2;
        for (i, &expert) in rank.iter().enumerate() {
            let e = usize::from(expert);
            let ss = e * stride..(e + 1) * stride;
            let ws = ss.start * 8..ss.end * 8;
            let (s, w) = if i < mid {
                (&base_scales, &base_weights)
            } else {
                (&original_scales, &original_weights)
            };
            scales[ss.clone()].copy_from_slice(&s[ss]);
            weights[ws.clone()].copy_from_slice(&w[ws]);
            cuts[e] = base_cuts[e];
        }
        let (mut frame, mut sf) = (Vec::new(), Vec::new());
        pack_scale_direct(&weights, scales.clone(), &cuts, &mut frame)?;
        pack_mean_scales(&scales, width, &mut sf)?;
        println!("trial {mid} {} {}", frame.len(), sf.len());
        if frame.len() + sf.len() <= cap {
            let decoded = unpack_mean8_scale_codes(&sf, 128 * stride, width)?;
            assert_eq!(decoded, scales);
            assert_eq!(unpack_scale_direct(&frame, decoded)?, weights);
            fs::write(dir.join("fit.weights"), frame)?;
            fs::write(dir.join("fit.scales"), sf)?;
            selected = mid;
            low = mid;
        } else {
            high = mid;
        }
    }
    if selected == 0 {
        return Err("no nonempty restoration fit the budget".into());
    }
    println!("selected {selected}");
    Ok(())
}
