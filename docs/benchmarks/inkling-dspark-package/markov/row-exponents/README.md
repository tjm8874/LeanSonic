# Compress the Markov head using exponent differences within each row

The BF16 Markov matrix contains 201,024 rows of 256 values. A BF16 exponent straddles the two bytes. For each row the encoder stores the nearest integer to its mean exponent (ties rounded up), then codes sign plus fraction and the eight-bit wrapping difference from the row mean. Three existing lossless rANS streams hold these fields. BF16 values including zero, subnormals and NaNs retain their exact bit patterns.

| Encoding of the full Markov matrix | Stored bytes including frame lengths |
|---|---:|
| bytes | 74,491,296 |
| fields | 74,217,136 |
| column | 74,534,888 |
| column-fields | 74,318,768 |
| row-fields | 67,802,720 |

The bytes control is the original low/high-byte frame. The fields trial splits exponent from sign and fraction. The column variants give each column a separate high-byte or exponent frame. Every trial encoded and decoded the same 102,924,288 actual bytes, then compared all 51,462,144 words to the input. Only row-fields enters the loader in this PR.

The complete sidecar produced by the release converter is **326,519,568 bytes** (SHA-256 `b808a6e18be1c08bf2a67d6167a835b43b11c51a706c46dbe9e29c87a14f9aee`). This saves 6,688,584 bytes compared with the previous Markov codec and 309,281,848 bytes compared with the official drafter. With the same fixed seven-level target and tokenizer, the total is **11,935,883,470 bytes**. The separate verifier unpacked all 59 tensors of the new sidecar and compared them byte for byte with the raw Q8 reference. The independently restored Markov matrix has SHA-256 `ff765d557ddb8972cda57fc452ee7835ff32082f600d143db91bff0adcca184a`. GPU throughput, peak memory and actual speculative acceptance require an end-to-end replay.

## Reproduction

First build the raw Q8 sidecar using the [base recipe](../../). The alternative uses the same official input and only changes storage. The verification below invokes the underlying byte rANS decoder and recombines the three streams in Python independently of the inference loader.

```sh
python3 tools/pack_inkling_drafter.py "$OFFICIAL_MTP" "$ROW_MTP" --compact --markov-row-exponents
cp "$RECIPE/decode.rs" crates/leansonic-core/examples/verify_row_plane.rs
cargo build --release -p leansonic-core --example verify_row_plane
python3 "$RECIPE/verify.py" "$RAW_MTP" "$ROW_MTP" "$PWD/target/release/examples/verify_row_plane" > measured.json
cmp measured.json "$RECIPE/verify.json"
```

[Per-tensor receipt](verify.json) · [Independent transform experiment](probe.rs) · [Raw frame sizes and input digest](probe-results.txt).
