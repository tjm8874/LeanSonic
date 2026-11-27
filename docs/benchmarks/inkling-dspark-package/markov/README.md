# Store the DSpark Markov head in two lossless BF16 byte planes

The official Markov head is BF16 [201024, 256] and the inference kernels consume its original BF16 values. On that actual 102,924,288-byte tensor, the existing embedding prefix codec produced 84,047,912 bytes. Coding its low and high bytes as two rANS planes produced **74,491,296 bytes**. Both trials restored every BF16 bit. Reuse the target's BF16 weight reader for this single drafter tensor. It accepts either its historical BF16 matrix or `markov_head.markov_w1.weight.bf16planes_v1` in the existing two-plane format. The original VRAM versus pinned-host placement policy still runs after decoding.

With both Q8 entropy coding and BF16 Markov coding the full sidecar is **333,208,152 bytes**, SHA-256 `9ba510a93982f070c85fc172713717cfc568f7430643cd2e3ca950b2399e3eb3`. This saves 28,432,984 bytes versus the 361,641,136-byte compact Q8 sidecar, or 302,593,264 bytes versus the 635,801,416-byte official sidecar. With the fixed seven-level target and tokenizer, the complete distribution occupies **11,942,572,054 bytes**. All 59 decompressed tensors matched the verified raw Q8 sidecar through a separate reader, including the 51,462,144 BF16 Markov coefficients. The encoder requires the source head to have its official shape and type.

Use the [Q8 entropy recipe](../entropy/) and pass both switches for the Markov version. Use this loader together with #180 for native Q8 planes. GPU generation, draft acceptance and sampled peak memory use a separate end-to-end comparison.

```sh
python3 tools/pack_inkling_drafter.py "$OFFICIAL_MTP" "$COMPACT_MARKOV_MTP" --compact --compact-markov
cp "$RECIPE/decode.rs" crates/leansonic-core/examples/verify_markov_plane.rs
cargo build --release -p leansonic-core --example verify_markov_plane
python3 "$RECIPE/verify.py" "$RAW_MTP" "$COMPACT_MARKOV_MTP" "$PWD/target/release/examples/verify_markov_plane" > verify.json
cmp verify.json "$RECIPE/verify.json"
```

[Per-tensor sizes, SHA-256 and independent decode checks](verify.json).
