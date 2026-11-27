# Lossless four-bank encoding after restoring 42 experts

All 40 NVFP4 expert weight planes became smaller. The shipped shard and tokenizer dropped from 7,764,131,886 to **7,747,846,286 bytes**, saving 16,285,600 bytes including safetensors header changes. The 20 gate/up frames saved 10,998,096 bytes and the 20 down frames saved 5,287,424 bytes. The remaining 80 bytes came from the header. This leaves 16,445,490 bytes below the public IQ1_S size.

An independent decoder reread the parent and final shards, compared every one of the 10,192,158,720 decoded NVFP4 weight bytes, and matched all 959 other tensor payloads and the tokenizer. The restored experts, their scales, and the higher-precision layer 17 down weights are identical to the [ranked direct-frame parent](../). The deployed shard SHA-256 is `3b73eb1dbec91724dcb05138ed882b326a504b411ecd005f247bc4fc5a53e162`. [Encoded frame sizes and hashes](release.json) · [Independent per-plane decode](verified.json)

## Reproduce

Use the [published four-bank repacker](https://github.com/tjm8874/LeanSonic/tree/16ee825/docs/benchmarks/inkling-nvfp4-scale-split/four-banks) with the ranked direct-frame parent and the same original cutoff list. The packer revision was 3febbfb. Build the separate verifier in a 238b46c checkout. Set RECIPE to this directory, PARENT to the ranked model, and OUTPUT to a fresh destination.

```sh
cargo build --release -p leansonic-core --example inkling_scale_condition
python3 tools/inkling_storage_probe/trim_context/repack.py "$PARENT" "$OFFICIAL_SHARD" "$PWD/target/release/examples/inkling_scale_condition" "$OUTPUT" --scale-cutoffs "$SELECTION" --four-banks >release.json
cp "$RECIPE/verify_plane.rs" "$DECODER/crates/leansonic-core/examples/verify_ranked_banks.rs"
(cd "$DECODER" && cargo build --release -p leansonic-core --example verify_ranked_banks)
python3 "$RECIPE/verify.py" "$PARENT" "$OUTPUT" "$DECODER/target/release/examples/verify_ranked_banks" release.json
```
