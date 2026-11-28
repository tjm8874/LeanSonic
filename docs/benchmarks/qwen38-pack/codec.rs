#[allow(dead_code)]
#[path = "../../../crates/leansonic-core/src/qwen38_pack.rs"]
mod qwen38_pack;
use std::{
    env,
    error::Error,
    io::{self, Read, Write},
};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let mut data = Vec::new();
    io::stdin().read_to_end(&mut data)?;
    let out = io::stdout();
    let mut out = io::BufWriter::new(out.lock());
    match args.as_slice() {
        [op] if op == "pack" => out.write_all(&qwen38_pack::encode(&data)?)?,
        [op, count] if op == "restore" => {
            let count = count.parse()?;
            let view = qwen38_pack::Frame::open(&data, count)?;
            for i in 0..count {
                out.write_all(&view.word(i)?.to_le_bytes())?;
            }
        }
        _ => return Err("expected pack or restore WORDS".into()),
    }
    out.flush()?;
    Ok(())
}
