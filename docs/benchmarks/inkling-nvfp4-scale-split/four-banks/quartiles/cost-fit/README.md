# Lossless expert-specific boundaries for the four-bank model

Two offline encoding passes reduced the Inkling-2-Mini target and tokenizer by **5,205,352 bytes**. The original four-bank runtime can load all three packages. The tokenizer accounts for 3,386,494 bytes in each.

| Package | Total bytes | Bytes saved in this pass | Boundaries moved | Smaller frames / 40 |
|---|---:|---:|---:|---:|
| [Quartile control](../) | 13,454,179,366 | — | — | — |
| [First fit](first.json) | 13,449,275,278 | 4,904,088 | 6,147 | 40 |
| [Second fit](second.json) | 13,448,974,014 | 301,264 | 1,003 | 40 |

Each expert retains the original boundary between trimmed and original weights. Within each side, fit its secondary boundary against the preceding frame's normalized rANS frequencies. We accumulate low-bank minus high-bank cost by scale code in 1/256 bits, use half a count for previously unseen symbols, and break ties by proximity to the previous boundary. The encoder rebuilds the tables over all assigned bytes. The repacker keeps a new frame only when it occupies fewer actual bytes.

The loader from 238b46c reconstructed the same 10,192,158,720 expert-weight bytes on each pass. The verifier also checked the other 959 tensor payloads, identifiers, dtypes, metadata and tokenizer against the preceding package. [First decode](first-decode.json) · [Second decode](second-decode.json). The final shard SHA-256 is `20058b671e1eea2ffb0c54acbb2f9752f4355afb12d14388433733cae3a2f61e`.

## Reproduce both passes

Use the [quartile control and its selection file](../) with the matching official original shard. Set QUARTILE, OFFICIAL_SHARD and SELECTION to absolute paths. Start in the checkout containing this directory. FIRST and SECOND must name new destinations. DECODER is a separate checkout of 238b46c. The measured packer source was 3febbfb.

```sh
check="$PWD/docs/benchmarks/inkling-nvfp4-scale-split/four-banks/quartiles/cost-fit"
cargo build --release -p leansonic-core --example inkling_scale_condition
encoder="$PWD/target/release/examples/inkling_scale_condition"
cp "$check/verify_plane.rs" "$DECODER/crates/leansonic-core/examples/cost_fit_reference_decode.rs"
(cd "$DECODER" && cargo build --release -p leansonic-core --example cost_fit_reference_decode)
decoder="$DECODER/target/release/examples/cost_fit_reference_decode"
python3 tools/inkling_storage_probe/trim_context/repack.py "$QUARTILE" "$OFFICIAL_SHARD" "$encoder" "$FIRST" --scale-cutoffs "$SELECTION" --refit-four-banks > first.json
python3 "$check/verify.py" "$QUARTILE" "$FIRST" "$decoder" first.json > first-decode.json
python3 tools/inkling_storage_probe/trim_context/repack.py "$FIRST" "$OFFICIAL_SHARD" "$encoder" "$SECOND" --scale-cutoffs "$SELECTION" --refit-four-banks > second.json
python3 "$check/verify.py" "$FIRST" "$SECOND" "$decoder" second.json > second-decode.json
cmp first-decode.json "$check/first-decode.json"
cmp second-decode.json "$check/second-decode.json"
```
