use leansonic_core::inkling2_gpu::mxscale_load_benchmark;
use std::{
    env, fs,
    hint::black_box,
    io::{self, Write},
    time::Instant,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [format, codes, scales, count, action] = args.as_slice() else {
        return Err("expected FORMAT CODES SCALES COUNT ACTION".into());
    };
    let count: usize = count.parse()?;
    let codes = fs::read(codes)?;
    let scales = fs::read(scales)?;
    let start = Instant::now();
    let packed = mxscale_load_benchmark(format == "mx", &codes, &scales, count)?;
    let seconds = start.elapsed().as_secs_f64();
    assert_eq!(black_box(&packed).len(), 16 + count / 2);
    if action == "dump" {
        io::stdout().lock().write_all(&packed)?;
    } else {
        println!("{seconds:.9}");
    }
    Ok(())
}
