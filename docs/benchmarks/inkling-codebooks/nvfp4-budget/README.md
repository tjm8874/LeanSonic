# Reallocating precision between NVFP4 and MXFP8 experts

Both candidates retain the middle package’s MXFP8 codebooks for the last four layers. In the first 20 layers, low-scale NVFP4 blocks replace magnitudes 0.5 with 0 and 1.5 with 2. Nonzero results retain their sign. A separate scale cutoff is chosen for each expert and plane. The fitter uses checkpoint weights and block scales, with triple weight on the 12 recurrent layers. It samples one block in 128 per expert and estimates the encoded bit rate using the original prefix-maximum contexts. The encoder checks that decoding each of the 40 complete frames reproduces its intended NVFP4 bytes.

| Distribution | Actual bytes, including tokenizer | Difference from original Q3 | Changed NVFP4 weights |
|---|---:|---:|---:|
| first | 13,635,536,582 | +3,758,566 | 4,090,341,181 |
| recalibrated | 13,623,652,382 | -8,125,634 | 4,123,963,140 |

The first fit underestimated the full package by 15,795,030 B. The second fit corrected for the observed error and reserved another 3,000,000 B. Both packages use the stock engine’s existing storage format. File sizes are measured after encoding all tensors and copying the tokenizer.

## WikiText-2 against BF16

| Split | Package | Mean KL | Worst KL (chunk/position) | Top-1 matches | PPL |
|---|---|---:|---:|---:|---:|
| test | middle | 0.0037406 | 0.6201534 (69/353) | 24,828/25,500 | 7.078202 |
| test | compact | 0.0137735 | 0.7519345 (69/353) | 24,289/25,500 | 7.131377 |
| test | first | 0.0100459 | 1.8819803 (99/398) | 24,338/25,500 | 7.092837 |
| test | recalibrated | 0.0100863 | 1.8988259 (99/398) | 24,362/25,500 | 7.091487 |
| valid | middle | 0.0033250 | 0.3668402 (46/379) | 24,893/25,500 | 6.506839 |
| valid | compact | 0.0131739 | 0.9835867 (7/467) | 24,341/25,500 | 6.564200 |
| valid | recalibrated | 0.0088730 | 2.3469483 (46/379) | 24,476/25,500 | 6.522603 |

On test, the first candidate improves mean KL, top-1 and PPL over compact, but worst KL increases at chunk 99, position 398. Keep the existing production model.

The recalibrated model saves 3,093,592 B against compact. It gains 73 matching BF16 predictions on test, but reaches worst KL 1.8988259 at 99/398. Both independently built packages have the same worst position on this split.

Bilingual (first, PRO 5000): mean KL 0.0093348, worst 0.1847861, top-1 599/613, PPL 3.857292.

## Reproduce

Start from the official Inkling-2-Mini shard and the [middle package](../README.md). Use the same official model and tokenizer digests as the parent experiment. Python 3.11+, NumPy and the repository Rust toolchain are required. Select an absent output directory for each build. The saved cutoff files reconstruct each model without repeating calibration.

```sh
tools=tools/inkling_storage_probe
export PYTHONPATH="$PWD/$tools" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
dir=docs/benchmarks/inkling-codebooks/nvfp4-budget
cp "$tools/nvfp4_budget/encode.rs" crates/leansonic-core/examples/nvfp4_scale_budget.rs
cargo build --release -p leansonic-core --example nvfp4_scale_budget
python3 "$tools/nvfp4_budget/assemble.py" "$OFFICIAL_SHARD" "$MIDDLE" "$dir/recalibrated-cutoffs.json" "$PWD/target/release/examples/nvfp4_scale_budget" "$NEW_MODEL"
LEANSONIC_KV=fp16 LEANSONIC_PDL=1 leansonic eval-kld --weights "$NEW_MODEL" --ref "$TEST_KLD" --chunks 100
```

To refit from scratch, run `select_cutoffs.py OFFICIAL_SHARD MIDDLE output.json target` with target 13619778016 for the first fit or 13607950944 for the recalibrated fit. The corresponding SHA-256 of each resulting shard, per-plane sizes, hashes and quality scores are in the [receipt](receipt.json).
