# Lossless BF16 embedding ranks on Inkling-2-Mini

The new package is **13,438,517,744 bytes**, including the tokenizer, down **10,456,270 bytes** from the 13,448,974,014-byte four-bank parent. All 463,159,296 BF16 embedding words match both the original publisher tensor and the existing prefix decoder byte for byte. The other 998 tensor payloads and the tokenizer match the parent. The model requires the new reader at load time. The runtime still reads the same pinned BF16 weights.

| Measurement | Prefix | Rank |
|---|---:|---:|
| Embedding frame bytes | 618,590,352 | 608,134,954 |
| Complete distribution bytes | 13,448,974,014 | 13,438,517,744 |
| One release CPU decode, ms | 79.537 | 580.092 |

The isolated CPU timings exclude file I/O. This first rank implementation takes about 501 ms longer to unpack the tensor in that run. The measured saving is 0.078% of the model. Full loader, inference and other-model results are separate measurements. [All sizes and hashes](receipt.json).

The encoder ranks all 6,001 distinct BF16 bit patterns by frequency, with ties in unsigned bit-pattern order. It measures the empirical entropy of the upper ranks for each feasible low-bit width and chooses five bits on this model. The frame holds a 16-byte header, the exact little-endian u16 table, five bytes of packed low bits per eight ranks, and a tiled rANS stream of the upper ranks. The decoder uses the host loader's existing rANS implementation to stream into the final pinned buffer. Raw BF16 and prefix packages retain their original readers.

## Reproduce

Use the [parent build and package hashes](https://github.com/tjm8874/LeanSonic/tree/588e3cd/docs/benchmarks/inkling-nvfp4-scale-split/four-banks/quartiles/cost-fit). Use publisher revision b85c3cfa of Inkling-2-Mini. Copy pack.rs and decode.rs from this directory into crates/leansonic-core/examples as rank_pack.rs and rank_decode.rs. With the repo toolchain, Python 3.11+ and NumPy, run in a fresh working directory:

```sh
cargo build --release -p leansonic-core --example rank_pack --example rank_decode
python3 encode.py OFFICIAL_SHARD /absolute/path/to/rank_pack FIT_DIR > encoding.json
rank_decode rank FIT_DIR/embedding.frame rank.bf16
sha256sum rank.bf16
python3 assemble.py FOUR_BANK_PARENT FIT_DIR/embedding.frame NEW_MODEL > model.json
```

The expected decoded SHA-256 is 86afe04b91893883cf823392f1c954bbf0881988e0b6c5f34ed7b2f938fe0a30. The original prefix frame, rank frame and official source were independently checked against it. Verify the resulting shard and tokenizer against receipt.json. For a GPU model comparison use identical cubins and scorer flags for both model directories and check full-vocabulary output fingerprints.
