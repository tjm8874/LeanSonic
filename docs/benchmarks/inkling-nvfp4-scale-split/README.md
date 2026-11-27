# Reuse the scale tensor to compress Inkling NVFP4 weights

The input is the [recovered-precision Inkling package](https://github.com/tjm8874/LeanSonic/tree/ebb434f/docs/benchmarks/inkling-codebooks/nvfp4-budget/recovered). For each of its 128 experts per plane, an existing E4M3 scale and one cutoff byte determine which of two NVFP4 streams supplies the next eight-byte block. The scale bytes come from the original lossless mean-context tensor shipped inside the shard. No side bitmap is needed. Both streams use the existing contextual rANS codec and retain original block order within each stream.

| Recovered-precision package | Bytes including tokenizer |
|---|---:|
| Prefix-context frames | 13,608,770,734 |
| Scale-selected frames | 13,479,712,078 |

All 40 expert planes became smaller and each decoded exactly to its input. The 129,058,656 B distribution saving includes the changed safetensors header and every cutoff. The 3,386,494-byte tokenizer was byte-verified. The final 13,476,325,584-byte shard has SHA-256 `350399eda8697f36bb99ad7d58b1b250aead22cdbb36d3ebf9a6ace8d36fb661`. Independent per-plane measurements and the final repacker agreed on every frame size. The repacker checked all 999 assembled tensor payloads.

## Reproduce

Build the input and obtain the fixed selection JSON from the linked recipe. The official shard supplies the original shapes. The decoder in this implementation reads all weight and scale data from the new package. Use a new destination directory.

```sh
cargo test --workspace --offline
cargo clippy --workspace --all-targets --offline -- -D warnings
cargo build --release --offline -p leansonic-core --example inkling_scale_condition
python3 tools/inkling_storage_probe/trim_context/repack.py \
  "$RECOVERED_MODEL" "$OFFICIAL_SHARD" target/release/examples/inkling_scale_condition \
  "$OUTPUT_DIR" --scale-cutoffs "$RECOVERED_SELECTION"
```

The loader materializes the two streams, scale codes and one decoded CPU weight plane before upload. Resident memory, initialization delay and end-to-end output checks are separate adoption criteria. [All packed file sizes and hashes](storage.json) · [Independent per-plane round trips](planes.json).

[GPU free-generation parity and the 25,500-position BF16 test](runtime.md) were verified with the fully assembled distribution.

[Fresh-process loader time and peak host RSS](load.md) quantify the 8.131 s startup cost.

[Direct decoding in model order](direct/README.md) builds the alternative 13,479,710,294-byte package and includes its BF16 checks.

On upstream main 66a388a, the pre-existing 13,445,948,726-byte trim package and its source distribution produced the same 64 greedy IDs, next ID and final-logit hash for all eight requests on each of RTX 5060 Ti and 5070 Ti. [Integrated-main build digests and individual traces](main-parity.json).
