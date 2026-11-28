#[allow(dead_code)]
#[path = "../../../crates/leansonic-core/src/qwen38_pack.rs"]
mod qwen38_pack;
use std::{
    env,
    error::Error,
    fs::File,
    hint::black_box,
    io::{Read, Seek, SeekFrom},
    time::Instant,
};

fn read_range(path: &str, offset: &str, size: usize) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut input = File::open(path)?;
    input.seek(SeekFrom::Start(offset.parse()?))?;
    let mut bytes = vec![0; size];
    input.read_exact(&mut bytes)?;
    Ok(bytes)
}
fn raw_row(raw: &[u8], row: usize) -> [u16; 5] {
    let bytes = &raw[row * 10..row * 10 + 10];
    std::array::from_fn(|i| u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [original, original_offset, compact, frame_offset, frame_bytes, rows] = args.as_slice()
    else {
        return Err("expected ORIGINAL OFFSET PACKED FRAME_OFFSET FRAME_BYTES ROWS".into());
    };
    let rows: usize = rows.parse()?;
    let raw = read_range(
        original,
        original_offset,
        rows.checked_mul(10).ok_or("overflow")?,
    )?;
    let packed = read_range(compact, frame_offset, frame_bytes.parse()?)?;
    let view = qwen38_pack::Frame::open(&packed, rows * 5)?;
    let mut seed = 128u64;
    let indices: Vec<_> = (0..65536)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 32) as usize % rows
        })
        .collect();
    for &row in indices.iter().chain(&[0, rows - 1]) {
        if view.row5(row)? != raw_row(&raw, row) {
            return Err("row mismatch".into());
        }
    }
    let mut expected = None;
    for cycle in 0..6 {
        let order = if cycle % 2 == 0 {
            [false, true, true, false]
        } else {
            [true, false, false, true]
        };
        for coded in order {
            let mut sum = 0u64;
            let start = Instant::now();
            if coded {
                for &i in &indices {
                    sum += black_box(view.row5(i)?)
                        .into_iter()
                        .map(u64::from)
                        .sum::<u64>();
                }
            } else {
                for &i in &indices {
                    sum += black_box(raw_row(&raw, i))
                        .into_iter()
                        .map(u64::from)
                        .sum::<u64>();
                }
            }
            let ns = start.elapsed().as_nanos();
            assert_eq!(*expected.get_or_insert(sum), sum);
            if cycle != 0 {
                println!("{cycle},{coded},{ns},{},{sum}", indices.len());
            }
        }
    }
    Ok(())
}
