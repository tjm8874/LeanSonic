use leansonic_core::inkling_mxfp8::{
    pack_mean_scales, pack_scale_direct, unpack_mean8_scale_codes, unpack_scale_direct,
};
use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
};
include!("palette.rs");
struct Sample {
    values: Vec<u8>,
    scales: Vec<u8>,
    err: u128,
}
impl Sample {
    fn new() -> Self {
        Self {
            values: vec![],
            scales: vec![],
            err: 0,
        }
    }
    fn push(&mut self, block: &[u8], map: &[u8; 8], s: u8, err: i64) {
        for &x in block {
            self.values.push(
                sign(x, map[usize::from(x & 7)])
                    | (sign(x >> 4, map[usize::from((x >> 4) & 7)]) << 4),
            );
        }
        self.scales.push(s);
        self.err += err as u128;
    }
    fn measure(&self, limits: &[u8; 128], width: usize) -> Result<(usize, usize), Box<dyn Error>> {
        let (mut w, mut s) = (vec![], vec![]);
        pack_scale_direct(&self.values, self.scales.clone(), limits, &mut w)?;
        pack_mean_scales(&self.scales, width, &mut s)?;
        let actual = unpack_mean8_scale_codes(&s, self.scales.len(), width)?;
        if actual != self.scales || unpack_scale_direct(&w, actual)? != self.values {
            return Err("roundtrip failure".into());
        }
        Ok((w.len(), s.len()))
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [file, wo, so, width, cut, seed] = args.as_slice() else {
        return Err(
            "expected official shard, weight offset, scale offset, row width, cutoff file and seed"
                .into(),
        );
    };
    let wo: u64 = wo.parse()?;
    let so: u64 = so.parse()?;
    let width: usize = width.parse()?;
    let seed: u64 = seed.parse()?;
    assert!(matches!(width, 72 | 144));
    let limits: [u8; 128] = fs::read(cut)?
        .try_into()
        .map_err(|_| "128 cutoff bytes required")?;
    let banks = limits.map(|x| if x == 255 { 0 } else { x + 1 });
    let identity = [0, 1, 2, 3, 4, 5, 6, 7];
    let mut source = File::open(file)?;
    let mut bytes = vec![0u8; 8 * width];
    let mut ss = vec![0u8; width];
    let tr: Vec<_> = (64..=126).map(trials).collect();
    let (mut control, mut low, mut all) = (Sample::new(), Sample::new(), Sample::new());
    let refit_parent = env::var("TERNARY_NONZERO_COST").as_deref() == Ok("-1");
    let mut eligible = 0usize;
    for expert in 0..128usize {
        let old = [control.err, low.err, all.err];
        let mut signal = 0u128;
        for group in 0..18usize {
            // One fixed row in each consecutive group of 128 rows.
            let mut key = seed
                .wrapping_add((expert * 18 + group) as u64)
                .wrapping_mul(0x9e3779b97f4a7c15);
            key = (key ^ (key >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            key = (key ^ (key >> 27)).wrapping_mul(0x94d049bb133111eb);
            let row = expert * 2304 + group * 128 + ((key ^ (key >> 31)) % 128) as usize;
            source.seek(SeekFrom::Start(wo + (row * width * 8) as u64))?;
            source.read_exact(&mut bytes)?;
            source.seek(SeekFrom::Start(so + (row * width) as u64))?;
            source.read_exact(&mut ss)?;
            for (block, &s) in bytes.as_chunks::<8>().0.iter().zip(&ss) {
                if !(64..=126).contains(&s) {
                    return Err("unexpected scale".into());
                }
                let mut hist = [0i64; 8];
                for &x in block {
                    hist[usize::from(x & 7)] += 1;
                    hist[usize::from((x >> 4) & 7)] += 1;
                }
                signal += (0..8)
                    .map(|i| hist[i] * (LEVELS[i] * e4(s)).pow(2))
                    .sum::<i64>() as u128;
                let selected = limits[expert] != 255 && s <= limits[expert];
                eligible += usize::from(selected);
                let base = if selected { &TRIM } else { &identity };
                let loss: i64 = (0..8)
                    .map(|i| hist[i] * ((LEVELS[i] - LEVELS[usize::from(base[i])]) * e4(s)).pow(2))
                    .sum();
                control.push(block, base, s, loss);
                let (_, t) = tr[usize::from(s - 64)]
                    .iter()
                    .map(|t| {
                        let loss: i64 = (0..8).map(|i| hist[i] * t.err[i]).sum();
                        (loss, t)
                    })
                    .min_by_key(|x| x.0)
                    .unwrap();
                let code = if refit_parent { refit(&hist, s, &t.mag, t.code, limits[expert]) } else { t.code };
                let best = weight_sse(&hist,s,&t.mag,code);
                all.push(block, &t.mag, code, best);
                if selected {
                    low.push(block, &t.mag, code, best)
                } else {
                    low.push(block, &identity, s, 0)
                }
            }
        }
        println!(
            "expert {expert} {signal} {} {} {}",
            control.err - old[0],
            low.err - old[1],
            all.err - old[2]
        );
    }
    for (name, x) in [("control", control), ("low", low), ("all", all)] {
        let (w, s) = x.measure(&banks, width)?;
        println!("{name} {w} {s} {} {eligible} {}", x.scales.len(), x.err);
    }
    Ok(())
}
