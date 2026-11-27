//! Rebuild one four-bank expert weight/scale pair from a specified set of complete experts.
use leansonic_core::inkling_mxfp8::{
    pack_four_scale_banks, pack_mean_scales, unpack_four_scale_banks, unpack_mean8_scale_codes,
};
use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::Path,
};

fn read(file: &mut File, offset: &str, size: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    file.seek(SeekFrom::Start(offset.parse()?))?;
    let mut bytes = vec![0; size.parse()?];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn load(args: &[String], width: usize) -> Result<([u8; 128], Vec<u8>, Vec<u8>), Box<dyn Error>> {
    let mut shard = File::open(&args[0])?;
    let weights = read(&mut shard, &args[1], &args[2])?;
    let scale_frame = read(&mut shard, &args[3], &args[4])?;
    let scales = unpack_mean8_scale_codes(&scale_frame, 128 * 2304 * width, width)?;
    let cuts: [u8; 128] = weights[..128].try_into()?;
    Ok((
        cuts,
        unpack_four_scale_banks(&weights, scales.clone())?,
        scales,
    ))
}

fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<_> = env::args().skip(1).collect();
    if a.len() != 13 {
        return Err(
            "width, expert IDs, output prefix and two shard/offset/size pairs required".into(),
        );
    }
    let width: usize = a[0].parse()?;
    assert!(matches!(width, 72 | 144));
    let ids = a[1]
        .split(',')
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()?;
    let mut selected = [false; 128];
    for &id in &ids {
        assert!(id < 128 && !selected[id]);
        selected[id] = true;
    }
    let (cuts, mut data, mut scales) = load(&a[3..8], width)?;
    let (other, high, high_scales) = load(&a[8..13], width)?;
    assert_eq!(cuts, other);
    let stride = 2304 * width;
    for &e in &ids {
        let sc = e * stride..(e + 1) * stride;
        let w = sc.start * 8..sc.end * 8;
        scales[sc.clone()].copy_from_slice(&high_scales[sc]);
        data[w.clone()].copy_from_slice(&high[w]);
    }
    let mut new = vec![];
    let mut sf = vec![];
    pack_four_scale_banks(&data, scales.clone(), &cuts, &mut new)?;
    pack_mean_scales(&scales, width, &mut sf)?;
    if a[2] != "-" {
        let decoded = unpack_mean8_scale_codes(&sf, 128 * stride, width)?;
        assert!(decoded == scales && unpack_four_scale_banks(&new, decoded)? == data);
        let path = Path::new(&a[2]);
        fs::write(path.with_extension("weights"), &new)?;
        fs::write(path.with_extension("scales"), &sf)?;
    }
    println!("{} {}", new.len(), sf.len());
    Ok(())
}
