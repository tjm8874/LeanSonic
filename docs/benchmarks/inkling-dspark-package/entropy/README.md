# Lossless entropy coding of the prepacked DSpark sidecar

The official BF16 sidecar is 635,801,416 bytes. The [prepacked Q8_64 sidecar](../) is 386,970,072 bytes. Its losslessly compressed version is **361,641,136 bytes**, saving another **25,328,936 bytes** (274,160,280 total). The complete fixed seven-level target, tokenizer and compact drafter occupy **11,971,005,038 bytes**. This is a speculative configuration with a separate drafter.

All 16 code planes use the existing rANS format (12,928,720 payload bytes saved), and all 16 scale planes use the existing FP32 dictionary (12,400,408 payload bytes saved). Header changes account for the remaining difference. Every frame was decompressed and checked byte-for-byte against the verified raw Q8_64 sidecar. All 59 tensors passed and included all 265,420,800 codes and 4,147,200 float scales. The retained BF16 Markov head and other BF16 tensors are copied unchanged. SHA-256 for the final file is `ef18c68c6a2a195ae0e5c1a1889cad57db39c445c73b5c5f8b99ba11359c6bfe`.

Use the loader from [PR #180](https://github.com/tjm8874/LeanSonic/pull/180). Passing `--compact` only changes the disk representation. If the encoded representation of an individual plane is larger, the packer retains its raw form. The raw sidecar already passed Claude's official-versus-Q8 spec7 integration check. The entropy-coded sidecar is undergoing a separate comparison.

```sh
python3 tools/pack_inkling_drafter.py "$OFFICIAL_MTP" "$RAW_MTP"
python3 tools/pack_inkling_drafter.py "$OFFICIAL_MTP" "$COMPACT_MTP" --compact
cp "$RECIPE/decode.rs" crates/leansonic-core/examples/verify_sidecar_plane.rs
cargo build --release -p leansonic-core --example verify_sidecar_plane
python3 "$RECIPE/verify.py" "$RAW_MTP" "$COMPACT_MTP" "$PWD/target/release/examples/verify_sidecar_plane" > verify.json
cmp verify.json "$RECIPE/verify.json"
```

[Per-tensor decoded SHA-256, sizes and file hashes](verify.json).
