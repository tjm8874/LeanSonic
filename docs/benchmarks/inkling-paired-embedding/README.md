# Interleave two lossless embedding prefix streams per worker

The existing input tensor is 618,590,352 bytes on disk and 926,318,592 bytes when expanded. Each worker used to finish one 128 KiB tile before starting the next. The change alternates groups of four BF16 values between two tiles, so their prefix lookups have independent bit offsets. It keeps the eight-worker limit, file format, output allocation and malformed-frame checks.

## Full-tensor CPU benchmark

The same Linux release binary calls the baseline function from `00052c8384ac47dea1bc4b70f06824382cfe7425` and the normal public loader function from `096c1a9cb3b303809d370c8d970f4d09418d4879`. Both arms write into the same buffer. Each leg poisons the destination before the timer starts and compares every decoded byte against the official tensor after it stops. Ten alternating ABBA/BAAB cycles follow a four-leg warm-up. The interval resamples whole cycles, seed 8874, 20,000 draws. This is CPU unpack time with resident input bytes.

| CPU | Prior | Interleaved | Elapsed change, 95% paired interval | Complete byte comparisons |
|---|---:|---:|---:|---:|
| Ryzen 9 9900X | 56.194 ms | 51.536 ms | −8.29% [−10.06%, −6.06%] | 44/44 |
| Core i5-14400 | 84.180 ms | 71.318 ms | −15.28% [−15.67%, −14.91%] | 44/44 |

The Ryzen run used normal host scheduling at nice +10, with other GPU work on the machine. Read the [raw Ryzen timings](ryzen.csv), [raw Intel timings](intel.csv) and [summary](stats.json). The Intel run held the benchmark host lock, with the identical executable and tensor digests. On the original Linux build, the executable SHA-256 was `8d94959a8573a8c79b981e04e75f44eb78a053eda94728699a94b747bb3c2c81`. The extraction script verifies both model tensor hashes before timing.

From a fresh checkout of the candidate, set RECIPE to this directory and pass the official checkpoint and a compact shard containing `model.llm.embed.weight.prefix_v1`. Run extraction in an empty scratch directory.

```sh
(cd "$SCRATCH" && python3 "$RECIPE/extract.py" "$OFFICIAL_SHARD" "$COMPACT_SHARD")
ex=crates/leansonic-core/examples/pair_bench
mkdir -p "$ex"
git show 00052c8384ac47dea1bc4b70f06824382cfe7425:crates/leansonic-core/src/inkling_embedding.rs >"$ex/baseline.rs"
cp "$RECIPE/bench.rs" "$ex/main.rs"
cargo build --release -p leansonic-core --example pair_bench
target/release/examples/pair_bench "$SCRATCH/official.bf16" "$SCRATCH/prefix.bin" >trial.csv
python3 "$RECIPE/summarize.py" trial.csv
```

## GPU loader and generated output

Both source revisions loaded the same [compact codebook model](../inkling-codebooks/README.md) on an RTX PRO 4500, driver 595.99.02. Shard SHA-256 was `a60c22b6626a50e1380abbfb80b69a73ff96d6404fb5b60c750d5bff7ff4cd02`. For 32 and 1,024 input tokens, we compared the 64 generated IDs and the successor plus all 200,058 FP32 logits immediately after prefill and at the end of generation. The four logit rows match byte-for-byte (3,200,928 bytes), as do the output IDs. Each arm ran one model load, which does not establish an end-to-end startup effect. [Output digests](smoke.sha), [generated IDs](src-out.ids), [binaries and prompt](smoke-inputs.sha), [device](smoke-gpu.csv).

In separate checkouts of the baseline and candidate, use the same cubin, compact model and prompt. Set ARM to a separate output path for each checkout.

```sh
cp "$RECIPE/smoke.rs" crates/leansonic-core/examples/pair_smoke.rs
cargo build --release -p leansonic-core --example pair_smoke
nvcc -arch=sm_120a -std=c++17 -O3 --cubin kernels/inkling2_sm120.cu -o engine.cubin
export LEANSONIC_CUBIN_PATH="$PWD/engine.cubin"
export LEANSONIC_KV=fp16 LEANSONIC_SEQS=1 LEANSONIC_PDL=1 LEANSONIC_PDL_ALL=1
export LEANSONIC_PREFILL_CHUNK=512 LEANSONIC_MXFP8_DEVICE_U4=1 LEANSONIC_MXFP8_DIRECT_ALL=1
target/release/examples/pair_smoke "$MODEL" docs/benchmarks/inputs/inkling-readme1024.ids "$ARM"
```

Compare the two .f32 files and the two .ids files using `cmp`.
