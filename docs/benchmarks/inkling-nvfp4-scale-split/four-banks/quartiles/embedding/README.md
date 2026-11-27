# Lossless input embedding size versus host decode time

Both experiments use the full official Inkling-2-Mini input embedding, 201,024 rows by 2,304 BF16 values (926,318,592 bytes). The control encoder reproduced the 618,590,352-byte prefix tensor in the four-bank package byte-for-byte. Every timed decode, including all controls, checked the full output against the official bytes. Each comparison ran six alternating ABBA/BAAB cycles on an AMD Ryzen 9 9900X with up to eight decoder threads. The first cycle is warm-up. The reported interval resamples five whole cycles. The encoded tensor and destination were resident in host memory. These times measure only decompression.

| Trial | Control bytes | Trial bytes | Saved bytes | Control → trial decode (ms) | Paired delta in ms (95% interval) |
|---|---:|---:|---:|---:|---:|
| rans | 618,590,352 | 609,686,616 | 8,903,736 | 55.59 → 218.97 | +163.38 [157.74, 167.50] |
| prefix | 618,590,352 | 613,341,696 | 5,248,656 | 54.02 → 91.31 | +37.29 [36.39, 37.99] |

The rANS trial stores the sign and seven significand bits as one byte and the full exponent as another. Each plane in a 1,024-row tile is entropy coded. On seven disjoint 1,024-row windows, subtracting row or column means cost more bytes than coding the full exponents directly. The prefix trial uses raw sign/fraction bytes with a fixed, canonical prefix code for twelve frequent exponents and an escape. It uses the same 128 KiB raw tile size as the control. The codebook was selected from the complete weight histogram. Every BF16 pattern is representable. Neither prototype is enabled in the loader. The faster control remains in the package.

## Reproduction

Use the official BF16 tensor model.llm.embed.weight and the [existing packed tensor](../) model.llm.embed.weight.prefix_v1, copied verbatim from their safetensors offsets. Name the extracted tensors official.bf16 and prefix.bin. Use the crate at revision 3ba1277. Run the following in separate fresh directories for each trial. The prefix trial includes its adjacent module. Each binary opens a new output file for its encoded weights.

```sh
mkdir -p crates/leansonic-core/examples/embed_trial
cp "$RECIPE/full.rs" crates/leansonic-core/examples/embed_trial/main.rs
# Prefix trial only:
test ! -e "$RECIPE/embed_prefix_trial.rs" || cp "$RECIPE/embed_prefix_trial.rs" crates/leansonic-core/examples/embed_trial/
cargo build --release -p leansonic-core --example embed_trial
target/release/examples/embed_trial "$OFFICIAL_BF16" "$PREFIX_FRAME"
```

[rANS driver, seven-window results and digests](rans/) · [Prefix driver, timings and digest](prefix/) · [Paired statistics](summary.json).
