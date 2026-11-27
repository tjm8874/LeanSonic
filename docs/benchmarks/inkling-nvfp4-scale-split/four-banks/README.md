# Four lossless scale bands for Inkling NVFP4 weights

The [two-bank direct model](../direct/) is 13,479,710,294 bytes with its tokenizer. The four-bank package is **13,455,139,382 bytes**, saving **24,570,912 bytes**. Both use the same quantized weights and scales. The initial full build and a separate run through the existing repacker produced identical files, including model shard SHA-256 `a77c9b258cd0faf54d95b86a0a27af6ccf7db62259311a6dbbb59644fb260faf`.

The primary cutoff of each expert is unchanged. The encoder divides the scales on either side at their closest population split and stores three boundaries per expert. The original 64 position-and-prefix tables are trained independently in each of four bands. The existing GPU kernels consume the decoded weights. The loader recognizes the new tensor suffix `.nvscale_four_v1` and streams each tile to its normal location. Conversion retains the original two-bank frame for any plane whose four-bank representation is larger. Here every one of the 40 planes shrank. Before writing a candidate, the encoder compared every byte with the original decoded plane (10,192,158,720 bytes total). The repacker hashed and reread all 999 final tensor payloads.

Each table uses a 4,096-entry decoder lookup. Four banks increase the transient CPU lookup for a plane from 2 to 4 MiB. Startup and BF16/greedy inference comparisons with the same host executable and cubin run separately from the file roundtrip.

## Reproduce

Build the example from 0b6793e. Start with the two-bank direct package. Use the same original official shard and cutoff selection as in the parent recipe. Set NEW_MODEL_DIR to an unused directory.

```sh
cargo build --release -p leansonic-core --example inkling_scale_condition
python3 tools/inkling_storage_probe/trim_context/repack.py "$DIRECT_MODEL" "$OFFICIAL_SHARD" target/release/examples/inkling_scale_condition "$NEW_MODEL_DIR" --scale-cutoffs "$SELECTION" --four-banks > release.json
cmp release.json "$RECIPE/release.json"
```

[Production conversion, per-plane byte counts and hashes](release.json) · [First full packaging run](receipt.json).
