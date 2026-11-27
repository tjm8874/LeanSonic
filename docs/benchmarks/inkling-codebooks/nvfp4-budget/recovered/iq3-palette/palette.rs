const LEVELS: [i64; 8] = [0, 1, 2, 3, 4, 6, 8, 12];
const TRIM: [u8; 8] = [0, 0, 2, 4, 4, 5, 6, 7];
const REDUCED: [u8; 4] = [0, 2, 4, 6]; // 0, 1, 2, 4 in E2M1 units.
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
    let mut candidates: Vec<_> = ((s - 2).max(64)..=s.saturating_add(6).min(126)).collect();
    candidates.sort_by_key(|&x| (s.abs_diff(x), x));
    candidates
        .into_iter()
        .map(|code| {
            let err_fn =
                |m: usize, q: u8| (LEVELS[m] * e4(s) - LEVELS[usize::from(q)] * e4(code)).pow(2);
            let mag =
                std::array::from_fn(|m| *REDUCED.iter().min_by_key(|&&q| err_fn(m, q)).unwrap());
            Trial {
                code,
                mag,
                err: std::array::from_fn(|m| err_fn(m, mag[m])),
            }
        })
        .collect()
}
