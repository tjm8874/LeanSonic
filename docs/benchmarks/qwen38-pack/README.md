# Lossless Qwen3.8-Flash-Next distribution and regression test

The original MLX Q4 `flashnext-norm-zero` checkpoint contains 22 shards. This tool preserves every BF16 bit in the scale and bias planes of the 48 routed-expert layers and 128 Engram partitions. Each eligible plane uses the smaller of its raw representation and a dictionary with fixed-width indices. Other tensor payloads, the original safetensors headers, and auxiliary files pass through unchanged. The Q38DIST1 transport archive requires restoration. A second format keeps the 22 safetensors shards directly loadable by LeanSonic and compresses only the Engram scales and biases.

## Measured results

The [run record](results.json) includes the pinned engine commit, the SHA-256 of each original shard, size totals, and GPU results. GB here means 10⁹ bytes.

| Measurement | Original | Packed |
| --- | ---: | ---: |
| Complete distribution, including auxiliary files and pack manifest | 113.233 GB | 105.965 GB |
| All 22 shards | 113.210 GB | 105.941 GB |
| Engram BF16 scales and biases, 256 planes | 6.400 GB | 4.000 GB |
| Expert BF16 scales and biases, 288 planes | 15.099 GB | 10.231 GB |

All 22 restored files matched their original SHA-256 when reread from disk. All 12 auxiliary files matched. On an RTX PRO 5000 72GB, the original and restored checkpoint produced identical FP32 logits at all 49 teacher-forced positions over 248,320 vocabulary entries. Forward KL was zero at every position and all four eight-token greedy continuations matched. The reference is the original quantized checkpoint. These fixtures cover English, Japanese, code and repetition. This run measures preservation of the checkpoint's predictions. Both loads used 45/48 expert layers in VRAM on this engine revision. A second pair of runs on the RTX PRO 4500 32GB produced the same logits hash, zero KL and 4/4 matching continuations. Both 32GB loads used 18/48 expert layers in VRAM and 30/48 in pinned host memory. The RTX 5070 Ti 16 GB and RTX 5060 Ti 16 GB also ran the SHA-verified, transport-restored checkpoint on a second host. Both matched the original PRO 5000 logit hash, all 49 top tokens and all four greedy canaries, with 7/48 expert layers in VRAM and 41/48 pinned. These 16 GB runs use the earlier engine commit listed in `results.json`. [Cross-device records](transport-16gb.json)

For all 256 Engram scale/bias planes, the random-row test compared five BF16 words per row at 65,536 deterministic draws plus the first and last rows. Fetching a five-word row from the resident dictionary frame took 8.40 ns for scales and 8.30 ns for biases, versus 2.44 ns and 2.42 ns for the raw planes. Figures are medians of per-plane medians from alternating runs after warmup. They were collected on an AMD Ryzen 9 9900X and exclude disk reads, PCIe transfers and model execution. The native inference measurements below include the row decoder in the running engine.

## Direct Engram inference on Wave 1

The directly loadable checkpoint occupies **110,833,479,038 bytes**, down from 113,233,127,335 bytes (2,399,648,297 bytes saved). The converter verified 256 dictionary frames by decoding them. A separate audit checked all 3,491 remaining tensor payloads and all 256 decoded frames against the source. The host-side addressable Engram payload fell from 32,000,153,600 to 29,600,516,096 bytes. These byte counts describe mapped tensor spans. Resident host memory depends on the operating system's page cache.

The table gives medians of warmed greedy requests at context 256 and 32 generated tokens. TTFT includes prefill and the first output token. Prefill throughput is input tokens divided by TTFT. Decode throughput covers the remaining 31 tokens. Each process performed one warmup and three timed requests per input. On 72 GB the process order was raw, direct, direct, raw. On 32 GB it was raw, direct. The NVML process sampler polled at 25 ms. All repetitions produced the same token IDs in both formats on each card. [Individual timings, token IDs, quality metrics and commits](direct-results.json)

| GPU | Input tokens | TTFT raw / direct (ms) | Prefill raw / direct (tok/s) | Decode raw / direct (tok/s) | Peak raw / direct (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX PRO 5000 72 GB | 16 | 92.47 / 92.47 | 173.04 / 173.02 | 162.94 / 162.91 | 70.26 / 70.26 |
| RTX PRO 5000 72 GB | 128 | 745.36 / 745.18 | 171.73 / 171.77 | 158.89 / 158.84 | 70.26 / 70.26 |
| RTX PRO 4500 32 GB | 16 | 722.38 / 721.77 | 22.15 / 22.17 | 21.90 / 21.88 | 29.79 / 29.79 |
| RTX PRO 4500 32 GB | 128 | 5781.17 / 5779.53 | 22.14 / 22.15 | 21.83 / 21.84 | 29.79 / 29.79 |

Wave 1 kept 48/48 expert layers in VRAM on 72 GB and 19/48 on 32 GB. On both cards, the raw Q4 and direct Q4 models produced byte-identical full-vocabulary logits at all 49 positions, mean and worst KL 0, top-1 agreement 49/49, PPL 52.9910 for both, and 4/4 matching greedy continuations. Comparing the older Q4 engine revision with Wave 1 changes floating-point logits slightly, so comparisons across engine revisions are recorded separately.

To reproduce, use the source Q4 model and a new destination directory outside it. The converter preserves raw BF16 for planes that do not shrink. In the current branch, build the codec and the GPU runners in a scratch checkout.

```sh
d=docs/benchmarks/qwen38-pack
s=docs/benchmarks/qwen38-engine-comparison
rustc --edition 2021 -O "$d/codec.rs" -o /tmp/q38-codec
python3 "$d/pack_engram.py" "$MODEL" "$DIRECT" /tmp/q38-codec
cp "$d/teacher.rs" crates/leansonic-core/examples/q38_teacher.rs
cp "$s/q38_stream.rs" crates/leansonic-core/examples/
cargo build --release -p leansonic-core --example q38_teacher --example q38_stream
nvcc -arch=sm_120a -O3 -std=c++17 -Ikernels --cubin kernels/qwen38_sm120.cu -o /tmp/q38-wave1.cubin
export LEANSONIC_QWEN38_CUBIN_PATH=/tmp/q38-wave1.cubin
# Set CUDA_VISIBLE_DEVICES to the reserved GPU UUID for NVML and the runner.
target/release/examples/q38_teacher "$MODEL" "$d/fixtures.ids" raw-score
target/release/examples/q38_teacher "$DIRECT" "$d/fixtures.ids" direct-score
OPENBLAS_NUM_THREADS=1 python3 "$d/score.py" raw-score direct-score --require-exact
python3 "$s/compare_stream.py" "$d/fixtures.ids" raw.log raw.json target/release/examples/q38_stream "$MODEL" 256
python3 "$s/compare_stream.py" "$d/fixtures.ids" direct.log direct.json target/release/examples/q38_stream "$DIRECT" 256
```

For accuracy against the official BF16 weights, use the [shared quality reference and native KL results](reference/README.md).

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
