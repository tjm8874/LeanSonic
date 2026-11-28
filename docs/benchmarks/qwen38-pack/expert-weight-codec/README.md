# Directly loadable Zstandard frames for routed Q4 weights

On Qwen3.8-Flash-Next MLX Q4, the complete 22-shard package measured **103,556,413,821 bytes**. Its input was the [105,964,828,539-byte directly loadable checkpoint](../direct-experts/README.md) from #265. The additional reduction is **2,408,414,718 bytes (2.27%)**. This preserves the stock Q4 quantization. Relative to the original 113,233,127,335-byte distribution the two changes together saved 9,676,713,514 bytes.

The converter encodes the 512 experts in each of the 144 routed gate/up/down planes independently. Each 819,200-byte source record is stored either as a checksummed Zstandard level-1 frame or verbatim when compression would be larger. An eight-byte magic, expert count, source record length, and 513 little-endian u32 payload offsets precede each plane. Raw tensors and encoded planes may coexist in the same checkpoint. An independent audit reopened both packages and compared all 73,728 expert records (60,397,977,600 source bytes), 3,603 other tensors and 11 auxiliary files. The converter verified all compressed records on write as well.

## RTX PRO 5000 72 GB

With all 48 layers in VRAM, 494 full-vocabulary FP32 logit rows and 16 eight-token continuations matched stock Q4 exactly. Against official BF16 the stock mean/worst KL was 0.269670/2.499328, top-1 agreement 406/494 and PPL 15.9460 (reference 11.3821). With 47 resident layers and one mmap-staged layer, all 49 original logit rows and all four greedy continuations matched the same-placement control. In the mmap path only requested experts are decoded. Resident Z24 placement decodes one weight plane at a time before copying it into the normal inference layout.

| Context 256, 32 greedy output tokens | #265 prefill / decode tok/s | With Q4 frames prefill / decode tok/s |
| --- | ---: | ---: |
| 16-token prompt | 383.70 / 162.87 | 384.84 / 162.80 |
| 128-token prompt | 469.71 / 158.91 | 469.93 / 158.91 |

Same binary at `b2f5a46`, GPU, cubin, prompts and IDs. Launch order was base / frames / frames / base. Each launch used one warmup and three timed repetitions per prompt. All 32 generated sequences including warmups matched their same-prompt controls. Prefill includes time to first token. Decode uses the remaining 31 token intervals. Peak sampled process VRAM was 75,486,986,240 bytes for both. Median launch-to-ready time rose from **13.68 s to 17.64 s** (individual launches 14.33/13.03 s and 19.42/15.86 s). Filesystem cache was not cleared. Keep the smaller layout optional where startup latency matters. [Run records, exact comparisons and independent disk audit](pro5000.json)

## Reproduce

Start with the #265 Engram and routed parameter dictionary checkpoint. The converter and audit use Python with the `zstandard` module. The native feature links libzstd (install its development library when building). Ordinary Q4 checkpoints work with the default build.

```sh
d=docs/benchmarks/qwen38-pack
python3 "$d/expert-weight-codec/pack_weights.py" "$DIRECT_EXPERTS" "$NEW_DIR"
python3 "$d/expert-weight-codec/audit_weights.py" "$DIRECT_EXPERTS" "$NEW_DIR"
cargo test -p leansonic-core --features q38-zstd-weights expert_weight
mkdir -p crates/leansonic-core/examples
cp "$d/teacher.rs" crates/leansonic-core/examples/q38_teacher.rs
cp docs/benchmarks/qwen38-engine-comparison/q38_stream.rs crates/leansonic-core/examples/
cargo build -p leansonic-core --features q38-zstd-weights --release --example q38_teacher --example q38_stream
nvcc -arch=sm_120a -O3 -std=c++17 -Ikernels --cubin kernels/qwen38_sm120.cu -o wave3.cubin
export LEANSONIC_QWEN38_CUBIN_PATH=$PWD/wave3.cubin
```

Use the [frozen 16-input fixture](https://github.com/tjm8874/LeanSonic/blob/492b72b/docs/benchmarks/qwen38-pack/input-embedding/extended.ids) with the native teacher for each checkpoint. Run `score.py BASE_TEACHER FRAMED_TEACHER --require-exact`. Use the existing compare_stream.py script and four-fixture input from the [engine comparison](../../qwen38-engine-comparison/README.md) to reproduce the timing runs. For the mmap control, set `LEANSONIC_MAX_RESIDENT_LAYERS=47 LEANSONIC_SPILLED_MODE=compact` for each teacher launch.
