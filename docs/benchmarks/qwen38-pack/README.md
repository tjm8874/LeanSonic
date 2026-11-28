# Lossless Qwen3.8-Flash-Next distribution and regression test

The original MLX Q4 `flashnext-norm-zero` checkpoint contains 22 shards. This tool preserves every BF16 bit in the scale and bias planes of the 48 routed-expert layers and 128 Engram partitions. Each eligible plane uses the smaller of its raw representation and a dictionary with fixed-width indices. Other tensor payloads, the original safetensors headers, and auxiliary files pass through unchanged. The packed directory is a distribution format. Restore it before opening it with the stock runtime.

## Measured results

The [run record](results.json) includes the pinned engine commit, the SHA-256 of each original shard, size totals, and GPU results. GB here means 10⁹ bytes.

| Measurement | Original | Packed |
| --- | ---: | ---: |
| Complete distribution, including auxiliary files and pack manifest | 113.233 GB | 105.965 GB |
| All 22 shards | 113.210 GB | 105.941 GB |
| Engram BF16 scales and biases, 256 planes | 6.400 GB | 4.000 GB |
| Expert BF16 scales and biases, 288 planes | 15.099 GB | 10.231 GB |

All 22 restored files matched their original SHA-256 when reread from disk. All 12 auxiliary files matched. On an RTX PRO 5000 72GB, the original and restored checkpoint produced identical FP32 logits at all 49 teacher-forced positions over 248,320 vocabulary entries. Forward KL was zero at every position and all four eight-token greedy continuations matched. The reference is the original quantized checkpoint. These fixtures cover English, Japanese, code and repetition. This run measures preservation of the checkpoint's predictions. Both loads used 45/48 expert layers in VRAM on this engine revision. A second pair of runs on the RTX PRO 4500 32GB produced the same logits hash, zero KL and 4/4 matching continuations. Both 32GB loads used 18/48 expert layers in VRAM and 30/48 in pinned host memory.

For all 256 Engram scale/bias planes, the random-row test compared five BF16 words per row at 65,536 deterministic draws plus the first and last rows. Fetching a five-word row from the resident dictionary frame took 8.40 ns for scales and 8.30 ns for biases, versus 2.44 ns and 2.42 ns for the raw planes. Figures are medians of per-plane medians from alternating runs after warmup. They were collected on an AMD Ryzen 9 9900X and exclude disk reads, PCIe transfers and model execution. Integrating the packed Engram path into the inference loader remains separate work.

## Reproduce the pack and row test

Start at the repository root with a recent Rust compiler and Python 3.11 or later. Use new destination directories on a filesystem with space for both outputs.

```sh
d=docs/benchmarks/qwen38-pack
rustc --edition 2021 --test crates/leansonic-core/src/qwen38_pack.rs -o /tmp/q38-test
/tmp/q38-test
rustc --edition 2021 -O "$d/codec.rs" -o /tmp/q38-codec
python3 "$d/transport.py" pack /tmp/q38-codec "$MODEL" "$PACKED"
python3 "$d/transport.py" restore /tmp/q38-codec "$PACKED" "$RESTORED"
rustc --edition 2021 -O "$d/row_bench.rs" -o /tmp/q38-rows
python3 "$d/bench_rows.py" "$MODEL" "$PACKED" /tmp/q38-rows > rows.jsonl
```

The packed shard starts with `Q38DIST1`, followed by the complete original safetensors header. Payloads follow the header's original offset order. An eligible plane starts with a little-endian u64 stored-frame length, with zero indicating raw bytes. A `Q38DICT1` frame stores the u64 word count, u32 dictionary length, BF16 dictionary words, densely packed little-endian indices and two zero padding bytes. A single-entry dictionary uses zero index bits. Reading one word or five-word row only needs the dictionary and a bounded window of the index stream. The restore tool checks the digest before publishing the output directory.

## Reproduce the GPU regression

In a scratch checkout of the engine revision in `results.json`, copy this directory's `teacher.rs` to `crates/leansonic-core/examples/q38_teacher.rs`. The included `fixtures.ids` were generated from `fixtures.txt` with the checkpoint's tokenizer, with special tokens disabled. Reserve an idle GPU and use the same GPU, cubin and engine settings in both commands. The scoring step requires NumPy.

```sh
d=docs/benchmarks/qwen38-pack
cargo build --release -p leansonic-core --example q38_teacher
nvcc -arch=sm_120a -O3 -std=c++17 -Ikernels --cubin kernels/qwen38_sm120.cu -o /tmp/q38-teacher.cubin
export LEANSONIC_QWEN38_CUBIN_PATH=/tmp/q38-teacher.cubin
target/release/examples/q38_teacher "$MODEL" "$d/fixtures.ids" score-original
target/release/examples/q38_teacher "$RESTORED" "$d/fixtures.ids" score-restored
OPENBLAS_NUM_THREADS=1 python3 "$d/score.py" score-original score-restored --require-exact
```

Each arm writes all finite FP32 logits after every teacher-forced input token and independently generates eight greedy tokens for each fixture. The scorer verifies prefix and target alignment, compares the logged top tokens against argmax, and computes KL and target-token perplexity in FP64. Omit `--require-exact` when measuring a lossy candidate.

## Q4 weight-block follow-up

The independent `weight_blocks.py` experiment samples three 256 KiB windows per tensor to screen Zstandard levels 1, 3 and 6 at 16, 64 and 256 KiB block sizes. It then encodes the winning configuration for each group across every Q4 weight, decodes and verifies every compressed block, and counts an eight-byte index entry per block with raw fallback. Both groups selected 256 KiB and level 1. The measured expert payload plus index was 57,997,031,440 bytes (originally 60,397,977,600), and Engram was 24,650,963,540 bytes (originally 25,600,122,880). These figures estimate a separate seekable format. The published distribution totals above include only the dictionary pack. A single row lookup in this experimental design could require decompressing a full 256 KiB block.

```sh
# Requires the Python zstandard package (tested with 0.25.0).
python3 docs/benchmarks/qwen38-pack/weight_blocks.py "$MODEL" > weight-blocks.jsonl
```

### Does parallel decompression help load the experts?

On the Ryzen 9 9900X host we compared warm reads of the three expert Q4 projections in each of layers 0, 23 and 47. Each projection was 400 MiB. The native reader uses one contiguous raw read per worker or decompresses independently indexed 256 KiB blocks directly into the destination buffer. Timing includes worker startup and excludes allocation of the output buffer, packing, checksum calculation and the final byte comparison. Each arm completed a warmup and four alternating timed reads. Every output matched its source in full.

| CPU threads | Raw projection (ms) | Packed projection (ms) |
| ---: | ---: | ---: |
| 1 | 41.56 | 228.80 |
| 2 | 30.21 | 118.21 |
| 4 | 25.72 | 61.32 |
| 8 | 25.34 | 34.76 |

The table gives the median of nine per-projection medians. The [per-run samples](block-read-results.json) include physical plane sizes with their index. Parallel workers reduced decompression latency, but raw reads were faster at every matched thread count. Keep the blocked Q4 weights as a distribution experiment. GPU expert residency, PCIe traffic, cold I/O and token latency need separate measurements.

On Linux with a recent C++ compiler, libzstd development files and the Python zstandard package, run the following commands from the repository root. The scratch directory must have room for one packed projection.

```sh
d=docs/benchmarks/qwen38-pack
c++ -std=c++17 -O3 -pthread "$d/block_read.cpp" -lzstd -o /tmp/q38-block-read
python3 "$d/bench_weight_reads.py" "$MODEL" /tmp/q38-block-read "$SCRATCH" > block-read.jsonl
```
