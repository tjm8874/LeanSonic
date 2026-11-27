# Four-bank Inkling load and inference on the RTX PRO 5000

Both stores load all 24 expert layers on a 72 GB RTX PRO 5000, driver 595.99.02. The two-bank package is 13,479,710,294 bytes and the four-bank package is 13,455,139,382 bytes, including the tokenizer. Five cycles ran four independent processes each. Odd cycles used two/four/four/two banks and even cycles reversed the order. Each process ran four passes with 32, 1,024 and 2,048 input tokens and 64 generated tokens. Across all 240 requests, both packages gave identical 64 generated IDs, terminal prediction and terminal-logit FNV fingerprint at each length. The first pass was excluded from the inference timings. Read the [full per-process and per-request results](results.json).

| Metric | Two banks | Four banks | Paired delta, 95% cycle-bootstrap interval |
|---|---:|---:|---:|
| Loader-reported elapsed time | 8.470 s | 8.325 s | −0.145 s [−0.898, +0.319] |
| Process peak host RSS | 6,651,049.6 KiB | 6,651,060.8 KiB | +11.2 KiB [−411.2, +463.6] |
| Prefill, 1,024 input tokens | 136.796 ms | 136.763 ms | −0.033 ms [−0.060, −0.011] |
| Decode, 64 tokens after 1,024 input tokens | 451.498 ms | 451.433 ms | −0.065 ms [−0.127, −0.009] |

The package saves 24,570,912 bytes. The 20 process launches leave uncertainty about the size and direction of the startup effect. The loader reported 14.55 GiB resident VRAM in each arm. This test collected process peak **host** RSS. See the [paired BF16 scores](../paired-quality.md) and the [fixed-input engine comparison](../fixed-q3.md) for their respective measurements.

Use source `5470f092e1695d660be8aa6dd58f6b9aab02acce`, the two packages from the [conversion recipe](../README.md), and the cubin and prompt pinned in the fixed-input comparison. The driver repeats the 1,024-token input for the long case. Benchmark binary SHA-256 was `950c338456de7b15aa401ee3b4aac4347270b61e17443c18c0323ac5ffd5f11b`, cubin `bb9b28575ae507330bb3d10d3235c33524d31b1cb48fbfeda9c6b95ea93d36e6` and prompt `e93129d009d923541cd0b1ee39a6f121c11e000638ceb868258665ec974ed843`.

```sh
cp "$RECIPE/bench.rs" crates/leansonic-core/examples/four_scale_engine.rs
cargo build --release -p leansonic-core --example four_scale_engine
export LEANSONIC_KV=fp16 LEANSONIC_PDL=1 LEANSONIC_PDL_ALL=1 LEANSONIC_SEQS=1
export LEANSONIC_PREFILL_CHUNK=512 LEANSONIC_MXFP8_DEVICE_U4=1 LEANSONIC_MXFP8_DIRECT_ALL=1
export LEANSONIC_CUBIN_PATH="$MATCHED_CUBIN"
for cycle in 1 2 3 4 5; do
  arms=("$TWO" "$FOUR" "$FOUR" "$TWO")
  if ((cycle % 2 == 0)); then arms=("$FOUR" "$TWO" "$TWO" "$FOUR"); fi
  for leg in 0 1 2 3; do
    /usr/bin/time -f '%M' -o "$cycle-$leg.rss-kib" target/release/examples/four_scale_engine \
      "${arms[$leg]}" "$PROMPT_IDS" >"$cycle-$leg.csv" 2>"$cycle-$leg.log"
  done
done
```
