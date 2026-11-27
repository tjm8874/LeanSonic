# Storage and accuracy experiments after canonical NVFP4 zero packing

The two experiments use the 15,548,157,542-byte canonical package (#125) as the parent. All sizes below count the tokenizer.

## Retaining every normalized code

We encoded and reopened all 40 NVFP4 expert planes with two offline rANS variants. The context combines the position, the greatest magnitude already decoded in the current 16-value block, and the running average magnitude (thresholds 2, 3 and 4). Each decoded plane was compared byte-for-byte against the normalized official codes. The reference encoder reproduced the existing 40 complete frame lengths.

| Frame encoding | Total encoded frame bytes | Bytes saved over canonical | Sum of decoder-check durations (s) |
|---|---:|---:|---:|
| Canonical byte-context | 9,887,406,872 | 0 | 30.75 |
| Byte symbol with running mean | 9,884,118,088 | 3,288,784 | 47.01 |
| Nibble symbol with running mean | 9,879,709,056 | 7,697,816 | 99.15 |

The last column sums per-tensor times, including reading and validating the frames, from a single three-worker CPU run. It is not end-to-end startup latency. The smallest candidate saves only 0.050% of the complete package and takes substantially longer to decode in this experiment. The production reader retains its existing format. [Per-plane measurements](inkling-code-prefix-probe.json).

Reproduce with `c++ -O3 -std=c++17 tools/inkling_storage_probe/prefix.cpp -o prefix`. For each raw NVFP4 expert tensor in layers 0–19, call `./prefix SHARD ABSOLUTE_OFFSET RAW_BYTES NEW_FRAME nibble` (or `byte`). The absolute offset is 8 plus the little-endian safetensors header length plus the tensor's starting data offset. The program normalizes E2M1 zero signs, encodes the entire selected plane, reopens the file, and checks every decoded code and the final states.

## Reducing the precision of the MXFP8 expert weights

For each selected gate/up and down tensor, we round each finite E4M3 weight code to the nearest code with an even magnitude index. Exact midpoints select the even retained index; the original sign is preserved, including signed zero. This removes one mantissa bit while retaining the full exponent range. Each output uses the existing byte-rANS runtime format. Scales, other expert weights, the tokenizer and execution kernels stay the same. We decoded the changed frames and compared them with the rounded source, then checked the SHA-256 of every published tensor payload against its intended source.

| Research package | Total bytes | Saved vs canonical | Changed tensors |
|---|---:|---:|---:|
| Parent | 15,548,157,542 | 0 | 0 |
| Layer 23 only | 15,396,463,478 | 151,694,064 | 2 |
| Layers 20–23 | 14,941,300,742 | 606,856,800 | 8 |

These two packages change the numerical weights. The sub-15-GB package is a research candidate pending BF16 corpus validation. [Tensor receipts and shard hashes](inkling-mxfp8-rounding.json).

Reproduce from the official checkpoint and the parent from #125. Reserve space for both complete output folders plus temporary expert frames.

```sh
cargo build --release -p leansonic-core --example inkling_mxfp8_round
python3 tools/inkling_storage_probe/pack_mxfp8.py "$OFFICIAL" "$CANONICAL" "$OUTPUT"
```
