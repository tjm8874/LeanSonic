// The packer seeks to the validated BF16 tensor and connects its file descriptors
// to this CPU-only helper. All codes precede all scales in the output shard.
use leansonic_core::inkling2_gpu::{pack_q8_rows, D, VOCAB};
use std::io::{self, Read, Write};

fn main() -> io::Result<()> {
    let mut src = io::stdin().lock();
    let mut dst = io::stdout().lock();
    let mut bf16 = vec![0; 1024 * D * 2];
    let mut codes = vec![0; 1024 * D];
    let mut scales = vec![0; VOCAB * (D / 64) * 4];
    for first in (0..VOCAB).step_by(1024) {
        let elements = (VOCAB - first).min(1024) * D;
        let src_tile = &mut bf16[..elements * 2];
        let codes_tile = &mut codes[..elements];
        let offset = first * (D / 64) * 4;
        src.read_exact(src_tile)?;
        pack_q8_rows(
            src_tile,
            D,
            codes_tile,
            &mut scales[offset..offset + elements / 64 * 4],
        );
        dst.write_all(codes_tile)?;
    }
    dst.write_all(&scales)?;
    dst.flush()
}
