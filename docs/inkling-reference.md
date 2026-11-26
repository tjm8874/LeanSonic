# Reproduce the Inkling-2-Mini quality reference

The reference for GPU text inference uses the official Inkling-2-Mini language weights. The llama.cpp converter expands the release’s NVFP4 and MXFP8 expert weights to BF16. A full-offload run on an RTX PRO 5000 72 GB Blackwell with CUDA 13.0.88 and driver 595.99.02 reproduced the published reference byte-for-byte. The following commands use an existing official model directory, the WikiText-2 raw test split, and the converter’s Python dependencies. Run them from a new directory with space for the 52.23 GB GGUF and the 10.20 GB logits file.

```bash
git init llama.cpp
git -C llama.cpp fetch --depth=1 https://github.com/ggml-org/llama.cpp.git 29b52d398b294509746ec0a3f28948a3107c0213
git -C llama.cpp checkout --detach FETCH_HEAD
cmake -S llama.cpp -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --target llama-perplexity -j
python3 llama.cpp/convert_hf_to_gguf.py "$MODEL_DIR" --outtype bf16 --outfile official-bf16.gguf
build/bin/llama-perplexity -m official-bf16.gguf -f "$WIKITEXT_DIR/wiki.test.raw" \
  -c 512 -b 8192 -ub 8192 -ngl 99 -t 8 --no-warmup --chunks 100 --save-all-logits inkling2-bf16-base.kld
sha256sum inkling2-bf16-base.kld
```

| File | SHA-256 |
|---|---|
| Official model-00001-of-00001.safetensors | `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1` |
| wiki.test.raw | `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08` |
| official-bf16.gguf (52,226,509,568 bytes in this build) | `d4b5181aeb49acf2b08acaa2ad5e0f9f9710e8030d589b3c6e9f43150cf10804` |
| inkling2-bf16-base.kld (10,203,366,820 bytes) | `895fac2136c06dce1854568c83eb76a0225a05fab3c37897a12fec1d56949b97` |

The file contains 100 independently positioned chunks, 512 token IDs per chunk, and 255 scored positions per chunk. At prediction position 256 the target is token 257. Token IDs and positions are zero-based. The generator runs 16 sequences per batch. Changing the batch shape can change rounding and expert selection.

The generator reported PPL 7.0724 from raw logits. Decoding the stored probabilities gives PPL 7.06440819. The writer clips each logit to at most 16 below its row maximum, then quantizes the resulting range to 16-bit codes. On this corpus 14 of the 25,500 next-token targets received code zero. The file cannot recover their original logits. LeanSonic reports the stored value as `ref_ppl` and its own raw-logit perplexity as `ppl`. The scorer uses reference log-probabilities above −16 for KL.

The public model, corpus and original evaluation setup are available from the [model publisher](https://huggingface.co/thinkingmachines/Inkling-2-Mini), the [WikiText-2 archive used by llama.cpp](https://huggingface.co/datasets/ggml-org/ci/blob/main/wikitext-2-raw-v1.zip) and the [Basalt precision notes](https://github.com/jesdga95/basalt/blob/5b8a82c6b/docs/precision/inkling2.md).
