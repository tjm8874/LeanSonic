# Speculative decoding with row-relative Markov exponents

Both arms use the same fixed seven-level target, tokenizer, spec7 loader, cubin, and compressed Q8 drafter projections. The control sidecar stores its Markov head in low/high BF16 byte planes (333,208,152 bytes). The candidate stores row-relative exponent, sign/fraction and mean-exponent frames (326,519,568 bytes). Including the target and tokenizer, the respective totals are 11,942,572,054 and 11,935,883,470 bytes.

Each board runs five cycles with four fresh processes per cycle, alternating ABBA and BAAB. Each process runs code, JSON, and Japanese fixtures twice with a reset before every request. Timing uses the second pass. A 4 ms NVML sampler observes process memory throughout startup and generation. Loader-reported startup uses the operating system’s existing file cache. All runs must activate DSpark with FP16 KV and 24/24 resident expert layers.

## 5070Ti

Values are control → candidate. Decode uses 128 generated tokens per request. The interval is the paired, five-cycle bootstrap for candidate minus control decode time in milliseconds.

| Fixture | Input tokens | Accepted / proposed | Prefill ms | Decode tok/s | Decode difference, 95% interval (ms) |
|---|---:|---:|---:|---:|---:|
| q00_code_merge | 29 | 97/142 | 34.140 → 34.134 | 257.61 → 257.61 | +0.00 [-0.17, +0.15] |
| q11_json_schema | 32 | 99/159 | 38.072 → 38.068 | 246.90 → 246.96 | -0.14 [-0.29, +0.03] |
| ja_reasoning | 49 | 59/113 | 43.776 → 43.772 | 184.61 → 184.61 | +0.00 [-0.17, +0.21] |

Peak observed process VRAM was 15,546,187,776 → 15,546,187,776 bytes. Mean loader-reported startup was 15.607 → 15.455 seconds (paired difference -0.152 seconds, 95% interval [-0.240, -0.090]). All 120 requests agreed per fixture on generated IDs, next token, final FP32 logit hash, and proposed/accepted/verification counts. These signatures also match [the official-sidecar replay on the RTX 5070 Ti](https://github.com/tjm8874/LeanSonic/blob/cdb13cda66cc3494f20356e8a20b1c7c88e03baf/docs/benchmarks/inkling-dspark-package/entropy/spec7/5070Ti-verified.json). Per-round proposal sequences were recorded separately for the earlier three-arm replay.

[Individual timings, generated IDs, hashes, loader reports and paired statistics](5070Ti-verified.json).

## Reproduction

Build [the integrated benchmark revision](https://github.com/tjm8874/LeanSonic/commit/8d3b2aa2eef180c7e0de3e29b5fdb9bcbde1a36a) with the example below. The [packing recipe](../) produces the candidate. Give both model directories the same target shard and tokenizer, and put the corresponding sidecar in each as mtp.safetensors. Use the three adjacent input ID files in the working directory. Set CUDA_VISIBLE_DEVICES to the chosen card’s UUID for the NVML sampler.

```sh
cp "$RECIPE/bench.rs" crates/leansonic-core/examples/dspark_three.rs
cargo build --release -p leansonic-core --example dspark_three
cp target/release/examples/dspark_three "$RECIPE/bench"
nvcc -Ikernels -arch=sm_120a -std=c++17 -O3 --cubin kernels/inkling2_sm120.cu -o "$RECIPE/engine.cubin"
cd "$RECIPE"
export LEANSONIC_KV=fp16 LEANSONIC_PDL=1 LEANSONIC_PDL_ALL=1 LEANSONIC_SEQS=1
export LEANSONIC_MXFP8_DEVICE_U4=1 LEANSONIC_MXFP8_DIRECT_ALL=1 LEANSONIC_CUBIN_PATH="$PWD/engine.cubin"
python3 leg.py first-control "$CONTROL_MODEL_DIR"
python3 leg.py first-candidate "$CANDIDATE_MODEL_DIR"
```
