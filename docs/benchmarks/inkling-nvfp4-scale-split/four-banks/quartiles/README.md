# Select scale boundaries by measured frame size

The four-bank Inkling-2-Mini target plus tokenizer occupies **13,454,179,366 bytes**, saving **960,016 bytes** against the 13,455,139,382-byte median-cut file. The same loader reads both. All 40 expert planes shrank and a full decode matched all **10,192,158,720** original expert-weight bytes. The unchanged tokenizer is 3,386,494 bytes. The assembler reread the final payloads of all 999 tensors and verified their hashes.

For each expert, keep the existing split between trimmed and original scales. Try the first, second and third quartiles on each side of that split, nine combinations per plane. Select the combination producing the smallest actual rANS frame on a deterministic sample of one block in every 64. Prefer the original median rule on ties. Recompute its boundaries over the full scale histogram, encode and decode the whole plane, then retain the candidate only if it is smaller. Thirty-eight planes chose the second and third quartiles, two chose the first and third. The new file uses the same number of decoder tables and the same frame layout as the previous package.

## Reproduction

Start with the [four-bank package](../), the matching official shard and the same per-expert cutoff selection. Build the encoder from 585c2be. The destination must be a new directory.

```sh
cargo build --release -p leansonic-core --example inkling_scale_condition
python3 tools/inkling_storage_probe/trim_context/repack.py "$FOUR_BANK_MODEL" "$OFFICIAL_SHARD" target/release/examples/inkling_scale_condition "$DEST" --scale-cutoffs "$SELECTION" --retune-four-banks > release.json
```

The expected model SHA-256 is `203593d3f83fb50c7e2019686ad53ccfb021ffd0f500f7d3e63905d9db93c1bc`. The independent assembly at e81a9ac records all nine trial sizes, chosen quartiles and full-plane byte counts. [Assembly receipt](receipt.json).

The normal repacker at 585c2be independently rebuilt the package. Both complete files matched the initial assembly by length and SHA-256. [Verified repacker output](release.json).
