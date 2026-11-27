//! Compare decoded expert blocks read directly from the three shipped safetensors.
use leansonic_core::inkling_mxfp8::{unpack_mean8_scale_codes, unpack_scale_direct};
use std::{
    env,
    error::Error,
    fs::File,
    io::{Read, Seek, SeekFrom},
};

fn frame(file: &mut File, offset: u64, size: usize) -> std::io::Result<Vec<u8>> {
    file.seek(SeekFrom::Start(offset))?;
    let mut data = vec![0; size];
    file.read_exact(&mut data)?;
    Ok(data)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 17 {
        return Err(
            "expected width, selected IDs and three sets of shard/offset/size/offset/size".into(),
        );
    }
    let width: usize = args[0].parse()?;
    assert!(matches!(width, 72 | 144));
    let selected: std::collections::BTreeSet<_> = args[1]
        .split(',')
        .map(str::parse::<usize>)
        .collect::<Result<_, _>>()?;
    assert!(selected.iter().all(|&id| id < 128));
    let stride = 2304 * width;
    let mut decoded = Vec::new();
    for group in args[2..].chunks_exact(5) {
        let mut file = File::open(&group[0])?;
        let w = frame(&mut file, group[1].parse()?, group[2].parse()?)?;
        let s = frame(&mut file, group[3].parse()?, group[4].parse()?)?;
        let scales = unpack_mean8_scale_codes(&s, 128 * stride, width)?;
        let weights = unpack_scale_direct(&w, scales.clone())?;
        assert_eq!(weights.len(), 128 * stride * 8);
        assert_eq!(scales.len(), 128 * stride);
        decoded.push((weights, scales));
    }
    let mut distinguishable = 0;
    for id in 0..128 {
        let s = id * stride..(id + 1) * stride;
        let w = s.start * 8..s.end * 8;
        let parent = &decoded[usize::from(selected.contains(&id))];
        assert!(
            decoded[2].0[w.clone()] == parent.0[w.clone()],
            "expert {id} weights differ"
        );
        assert!(
            decoded[2].1[s.clone()] == parent.1[s.clone()],
            "expert {id} scales differ"
        );
        if decoded[0].0[w.clone()] != decoded[1].0[w] || decoded[0].1[s.clone()] != decoded[1].1[s]
        {
            distinguishable += 1;
        }
    }
    println!(
        "experts=128 restored={} retained={} distinguishable={} weight_bytes={} scale_codes={}",
        selected.len(),
        128 - selected.len(),
        distinguishable,
        128 * stride * 8,
        128 * stride
    );
    Ok(())
}
