# Lossless BF16 embedding ranks on Inkling-2-Mini

The new package is **13,438,517,744 bytes**, including the tokenizer, down **10,456,270 bytes** from the 13,448,974,014-byte four-bank parent. All 463,159,296 BF16 embedding words match both the original publisher tensor and the existing prefix decoder byte for byte. The other 998 tensor payloads and the tokenizer match the parent. The model requires the new reader at load time. The runtime still reads the same pinned BF16 weights.

| Measurement | Prefix | Rank |
|---|---:|---:|
| Embedding frame bytes | 618,590,352 | 608,134,954 |
| Complete distribution bytes | 13,448,974,014 | 13,438,517,744 |
| One release CPU decode, ms | 79.537 | 580.092 |

The single-run timings above describe the first rank reader. The revised reader assigns disjoint output tiles to CPU workers and consumes each decoded upper-rank tile immediately. In five mirrored cycles on the same host the original rank reader took **620.33 ms**, the revised reader **97.95 ms**, and the prefix reader **75.26 ms**. The new reader cuts mean rank time by **84.21%** with the same output hash for all 926,318,592 bytes. Its residual cost versus prefix was **22.69 ms**. The 30 fresh processes rotated prefix, original rank and revised rank. Timers covered CPU decoding and excluded file I/O. [All samples, checksums and paired cycle interval](cpu.json). The measured saving is 0.078% of the model. Full loader, inference and other-model results are separate measurements. [All sizes and hashes](receipt.json).

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

## GPU replay

The PR's fused loader and its normal CLI at 5148971 loaded both the prefix and rank four-bank packages on RTX PRO 4500. Both runs used the exact same SM120a cubin (repository SASS audit passed), FP16 KV, PDL, 512-token prefill chunks and all 24 resident expert layers. The output row files match byte for byte across **1,530 positions**. Each row includes the score, next token and a fingerprint of the complete FP32 vocabulary. Both packages produced mean KL **0.0085970**, worst KL **0.6397660**, BF16 top-1 **1,469/1,530** and PPL **6.564087**. [GPU, executable, oracle and row hashes](gpu.json).

Use [fixture.py](fixture.py) with the full BF16 test oracle to create the identical six-chunk reference. Apply [observe.py](observe.py) to crates/leansonic-cli/src/eval_kld.rs to emit score rows, build the release CLI, and compile kernels/inkling2_sm120.cu with nvcc -Ikernels -arch=sm_120a -O3 -std=c++17 --cubin. Enable LEANSONIC_PDL, LEANSONIC_PDL_ALL, LEANSONIC_MXFP8_DEVICE_U4 and LEANSONIC_MXFP8_DIRECT_ALL. Set LEANSONIC_SEQS=1, LEANSONIC_CTX=8192, LEANSONIC_KV=fp16, LEANSONIC_PREFILL_CHUNK=512 and LEANSONIC_CUBIN_PATH to that cubin. Run `leansonic eval-kld --weights MODEL --ref SUBSET --chunks 6` twice with separate DIAG_ROWS paths. Compare the complete files and scorer JSON.
