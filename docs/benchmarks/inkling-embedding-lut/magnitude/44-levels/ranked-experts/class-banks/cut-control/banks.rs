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
// Fit physical scale boundaries on a deterministic subsample of retained experts.
// Byte-histogram entropy proposes cuts; only the complete encoded length selects them.
fn proposed(raw: &[u8], scales: &[u8], base: &[u8;384], restored: &[bool;128], relative: bool) -> (i32,i32) {
    let mut hist = vec![[0u32;256];256];
    let stride = scales.len()/128;
    for e in 0..128 {
        if restored[e] {continue}
        for i in ((e*19)%31..stride).step_by(31) {
            let block = e*stride+i;
            let key = if relative {i32::from(scales[block])-i32::from(base[128+e])+128} else {i32::from(scales[block])};
            for &v in &raw[block*8..block*8+8] { hist[key as usize][v as usize]+=1; }
        }
    }
    let keys:Vec<_>=(0..256).filter(|&i|hist[i].iter().any(|&v|v!=0)).collect();
    assert!(keys.len()>=3);
    let n=keys.len();
    let mut cost=vec![vec![0.;n+1];n];
    for lo in 0..n {
        let mut sums=[0u32;256]; let mut total=0;
        for hi in lo..n {
            for (count,&x) in sums.iter_mut().zip(&hist[keys[hi]]) { *count+=x; total+=x; }
            let v=f64::from(total);cost[lo][hi+1]=v*v.log2()-sums.iter().filter(|&&m|m>0).map(|&m| f64::from(m)*f64::from(m).log2()).sum::<f64>();
        }
    }
    let mut best=(f64::INFINITY,0,0);
    for a in 1..n-1 { for b in a+1..n { let v=cost[0][a]+cost[a][b]+cost[b][n];if v<best.0 {best=(v,keys[a],keys[b]);} } }
    let shift=if relative {128} else {0};(best.1 as i32-shift,best.2 as i32-shift)
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
    for (mode,relative) in [("absolute",false),("relative",true)] {
        let (lo,hi)=proposed(&raw,&scales,base,&restored,relative);
        let mut cuts=[0u8;384];
        for e in 0..128 {
            if restored[e] {continue}
            let offset=if relative {i32::from(base[128+e])} else {0};
            cuts[128+e]=(lo+offset).clamp(0,127) as u8;
            cuts[e]=(hi+offset).clamp(0,127) as u8;
            cuts[256+e]=127;
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
