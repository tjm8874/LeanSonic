const LEVELS: [i64; 8] = [0, 1, 2, 3, 4, 6, 8, 12];
const TRIM: [u8; 8] = [0, 0, 2, 4, 4, 5, 6, 7];
fn alphabet() -> &'static [u8] {
    match std::env::var("INKLING_OFFLINE_PALETTE").as_deref() {
        Ok("0124") => &[0, 2, 4, 6],
        Ok("01") => &[0, 2],
        Ok("02") => &[0, 4],
        Ok("03") => &[0, 5],
        Ok("04") => &[0, 6],
        Ok("06") => &[0, 7],
        Ok("024") => &[0, 4, 6],
        Ok("012") => &[0, 2, 4],
        Ok("0h124") => &[0, 1, 2, 4, 6],
        Ok("01234") => &[0, 2, 4, 5, 6],
        Ok("01246") => &[0, 2, 4, 6, 7],
        Ok("012346") => &[0, 2, 4, 5, 6, 7],
        _ => panic!("unrecognized offline palette"),
    }
}
fn e4(s: u8) -> i64 {
    (8 + i64::from(s & 7)) << ((s >> 3) - 1)
}
fn sign(original: u8, mag: u8) -> u8 {
    if mag == 0 {
        0
    } else {
        (original & 8) | mag
    }
}
struct Trial {
    code: u8,
    mag: [u8; 8],
    err: [i64; 8],
}
fn trials(s: u8) -> Vec<Trial> {
    let reduced = alphabet();
    let penalty: i64 = std::env::var("TERNARY_NONZERO_COST").ok().and_then(|v| v.parse().ok()).unwrap_or(-1);
    let mut candidates: Vec<_> = if penalty < 0 {
        ((s - 2).max(64)..=s.saturating_add(6).min(126)).collect()
    } else {
        (64..=126).collect()
    };
    candidates.sort_by_key(|&x| (s.abs_diff(x), x));
    candidates
        .into_iter()
        .map(|code| {
            let err_fn =
                |m: usize, q: u8| (LEVELS[m] * e4(s) - LEVELS[usize::from(q)] * e4(code)).pow(2);
            let cost = |m: usize, q: u8| err_fn(m, q) + if q > 0 { penalty.max(0) * e4(s).pow(2) } else { 0 };
            let mag =
                std::array::from_fn(|m| *reduced.iter().min_by_key(|&&q| cost(m, q)).unwrap());
            Trial {
                code,
                mag,
                err: std::array::from_fn(|m| cost(m, mag[m])),
            }
        })
        .collect()
}

fn weight_sse(hist: &[i64; 8], s: u8, mag: &[u8; 8], code: u8) -> i64 {
    (0..8).map(|m| hist[m] * (LEVELS[m] * e4(s) - LEVELS[usize::from(mag[m])] * e4(code)).pow(2)).sum()
}
// Reproduce the fixed-support W196 fit within its original entropy bank.
fn refit(hist: &[i64; 8], s: u8, mag: &[u8; 8], old: u8, cut: u8) -> u8 {
    let (lo, hi) = if cut == 255 { (64,126) } else if old <= cut { (64,cut) } else { (cut.saturating_add(1).max(64),126) };
    (lo..=hi).min_by_key(|&c| (weight_sse(hist,s,mag,c), old.abs_diff(c),c)).unwrap()
}
