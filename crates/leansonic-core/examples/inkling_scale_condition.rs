//! Test whether the existing lossless scales can select smaller NVFP4 streams.
use leansonic_core::inkling_mxfp8::{
    pack_nvfp4_context, unpack_mean8_scale_codes, unpack_nvfp4_codes, unpack_nvfp4_trim_context,
};
use std::error::Error;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

fn section(file: &mut File, offset: &str, length: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    file.seek(SeekFrom::Start(offset.parse()?))?;
    let mut bytes = vec![0; length.parse()?];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn arm(scale: u8, cutoff_plus_one: u8) -> usize {
    usize::from(scale < cutoff_plus_one)
}

fn pack(raw: &[u8], scales: &[u8], cuts: &[u8; 128]) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut parts = [Vec::new(), Vec::new()];
    let groups_per_expert = scales.len() / cuts.len();
    for (i, bytes) in raw.chunks_exact(8).enumerate() {
        parts[arm(scales[i], cuts[i / groups_per_expert])].extend_from_slice(bytes);
    }
    let mut frame = cuts.to_vec();
    frame.resize(144, 0);
    for (j, part) in parts.iter_mut().enumerate() {
        if part.is_empty() {
            continue;
        }
        part.resize(part.len().next_multiple_of(32), 0);
        let bytes = pack_nvfp4_context(part, &mut frame)?;
        frame[128 + 8 * j..136 + 8 * j].copy_from_slice(&(bytes as u64).to_le_bytes());
    }
    Ok(frame)
}

fn decode(frame: &[u8], scales: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    let cuts = frame.get(..128).ok_or("missing cutoffs")?;
    let header = frame.get(128..144).ok_or("missing sizes")?;
    let mut sizes = [0usize; 2];
    let groups_per_expert = scales.len() / 128;
    for (i, &scale) in scales.iter().enumerate() {
        sizes[arm(scale, cuts[i / groups_per_expert])] += 8;
    }
    let mut data = &frame[144..];
    let mut parts = [Vec::new(), Vec::new()];
    for (j, field) in header.chunks_exact(8).enumerate() {
        let bytes = u64::from_le_bytes(field.try_into()?) as usize;
        let (encoded, rest) = data.split_at_checked(bytes).ok_or("truncated stream")?;
        if sizes[j] > 0 {
            parts[j] = unpack_nvfp4_codes(encoded, sizes[j].next_multiple_of(32), true)?;
            if parts[j][sizes[j]..].iter().any(|&b| b != 0) {
                return Err("nonzero decoded padding".into());
            }
        } else if bytes != 0 {
            return Err("nonempty stream for empty group".into());
        }
        data = rest;
    }
    if !data.is_empty() {
        return Err("trailing bytes".into());
    }
    let mut pos = [0usize; 2];
    let mut out = Vec::with_capacity(scales.len() * 8);
    for (i, &scale) in scales.iter().enumerate() {
        let j = arm(scale, cuts[i / groups_per_expert]);
        out.extend_from_slice(&parts[j][pos[j]..pos[j] + 8]);
        pos[j] += 8;
    }
    Ok(out)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [path, weight_off, weight_len, scale_off, scale_len, raw_len, cutoff_file] =
        args.as_slice()
    else {
        return Err("expected shard, weight and scale slices, raw length and cutoff file".into());
    };
    let count: usize = raw_len.parse()?;
    if count == 0 || !count.is_multiple_of(8 * 128) {
        return Err("invalid original NVFP4 plane length".into());
    }
    let mut file = File::open(path)?;
    let scale = unpack_mean8_scale_codes(&section(&mut file, scale_off, scale_len)?, count / 8)?;
    let raw = unpack_nvfp4_trim_context(&section(&mut file, weight_off, weight_len)?, count)?;
    let cuts: [u8; 128] = fs::read(cutoff_file)?
        .try_into()
        .map_err(|_| "expected 128 cutoff bytes")?;
    let frame = pack(&raw, &scale, &cuts)?;
    let start = Instant::now();
    let roundtrip = decode(&frame, &scale)?;
    let elapsed = start.elapsed().as_secs_f64();
    if roundtrip != raw {
        return Err("changed a reconstructed byte".into());
    }
    println!("{} {} {:.6}", frame.len(), roundtrip.len(), elapsed);
    Ok(())
}
