# Lossless storage of Inkling-2-Mini's rounded NVFP4 experts

The input is the [15-symbol MXFP8, routing-weighted NVFP4 package and recipe](https://github.com/tjm8874/LeanSonic/tree/c9a6a63/docs/benchmarks/inkling-codebooks/nvfp4-budget/capped). Its shard SHA-256 is `d4bdf49edd6ae8f891367a96eedc1dea2cef7eb341cd4d9c60137abfb6f83d4c`. The experiment changes the storage of the 40 NVFP4 expert planes only. In each eight-byte group the rANS context tracks whether a magnitude of 1 or 3 has already occurred. The encoder and decoder compute the extra context bit from the same preceding symbols. The loader recognizes the new tensor suffix and also accepts existing distributions.

| Distribution including tokenizer | Actual bytes |
|---|---:|
| Capped input | 13,624,461,934 |
| Capped input with the new lossless frames | 13,445,948,726 |
| Public Unsloth Q3_K_XL GGUF | 13,631,778,016 |

All 40 trial frames were smaller. The encoder compared all 10,192,158,720 decoded bytes against the input planes. After assembly, the repacker verified all 999 tensor payloads and the tokenizer. The [receipt](storage.json) includes per-plane sizes and both output file hashes. The saving is 178,513,208 bytes against the exact input weights. The Q3 reference uses a different quantization.

## Reproduce

Build the input from the linked recipe. Give the official Inkling-2-Mini shard as the shape reference. The final argument must point to a new directory whose parent exists and has enough space for the new shard and temporary frames.

```sh
cargo test --workspace --offline
cargo clippy --workspace --all-targets --offline -- -D warnings
cargo build --release --offline -p leansonic-core --example inkling_trim_context
python3 tools/inkling_storage_probe/trim_context/repack.py \
  "$CAPPED_MODEL_DIR" "$OFFICIAL_SAFETENSORS" \
  target/release/examples/inkling_trim_context "$OUTPUT_DIR"
```

On-disk size and decoded-weight identity are verified here. Startup cost, end-to-end logits and matched inference speed require runtime measurements.
