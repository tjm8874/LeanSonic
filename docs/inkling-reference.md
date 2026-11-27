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

## Measure how batching changes the BF16 predictions

The optional probe runs the pinned BF16 model twice on the same GPU with the same token IDs and a fresh KV cache for each run. In `full` mode it processes one 512-token sequence in a single call. In `stream` mode it processes the first 256 tokens in four calls of 64, then feeds one corpus token per call, following the schedule in LeanSonic’s `eval-kld`. The original oracle batches 16 sequences. Each CSV row gives the zero-based chunk and prediction position, KL against that oracle, and the leading token ID from each distribution.

Using the llama.cpp checkout, build directory and files from above, set `LEANSONIC_DIR` to the LeanSonic checkout and select one GPU that can hold the BF16 model. The probe uses the existing llama.cpp libraries.

```bash
c++ -O2 -std=c++17 -ffp-contract=off \
  -Illama.cpp/include -Illama.cpp/ggml/include "$LEANSONIC_DIR/tools/replay_inkling_bf16.cpp" \
  -Lbuild/bin -Wl,-rpath,"$PWD/build/bin" -lllama -lggml -lggml-base -o replay_inkling_bf16
./replay_inkling_bf16 official-bf16.gguf inkling2-bf16-base.kld 2 29 44 69 > schedules.csv
```

On PRO 5000 the two schedules chose different next tokens at 18 of the 1,020 positions in these four chunks. At chunk 69, position 353, the 512-token call predicted token 1057. The streaming schedule and the published oracle both predicted 2952. These chunks were selected to investigate known errors. Use the original 100-chunk oracle for the project’s quality gate. [Experiment and per-schedule results](https://github.com/tjm8874/LeanSonic/issues/35#issuecomment-7201276035)

## Compare single-request latency with the BF16 reference

On an RTX PRO 5000 72 GB, LeanSonic at `f5bda42` processed the 1,024-token README prompt in 455.65 ms and the 32 subsequent steps in 340.91 ms. The pinned llama.cpp BF16 build took 598.19 ms and 566.04 ms, respectively. Both predicted the same 32 token IDs on every request. Across the separate WikiText evaluation, LeanSonic agreed with the BF16 oracle at 24,880 of 25,500 positions. [Quality receipt](benchmarks/inkling-adaptive-prefill.json)

| Measurement | LeanSonic native SM120a | llama.cpp BF16 |
| --- | ---: | ---: |
| Prefill, 1,024 tokens | 455.65 ms | 598.19 ms |
| Decode, 32 fixed tokens | 340.91 ms | 566.04 ms |
| Highest sampled process GPU memory | 17,658 MiB | 53,004 MiB |
| Model file | 19,526,019,168 B | 52,226,509,568 B |

Use the official weights and the BF16 expansion built above. The following commands assume the LeanSonic repository is the current directory and the three environment variables point to the official model, pinned llama.cpp checkout, and its CUDA build directory.

```bash
cargo build --release -p leansonic-core --example inkling_fixed_decode
nvcc -O3 -arch=sm_120a --cubin kernels/inkling2_sm120.cu -o /tmp/inkling-bench.cubin
c++ -std=c++17 -O2 tools/benchmark_inkling_bf16.cpp \
  -I"$LLAMA_SRC/include" -I"$LLAMA_SRC/ggml/include" -L"$LLAMA_BUILD/bin" \
  -Wl,-rpath,"$LLAMA_BUILD/bin" -lllama -lggml -lggml-base -o /tmp/benchmark_inkling_bf16
inputs=docs/benchmarks/inputs
export LEANSONIC_CUBIN_PATH=/tmp/inkling-bench.cubin
target/release/examples/inkling_fixed_decode "$MODEL_DIR" \
  "$inputs/inkling-readme1024.ids" "$inputs/inkling-continuation32.ids"
/tmp/benchmark_inkling_bf16 official-bf16.gguf \
  "$inputs/inkling-readme1024.ids" "$inputs/inkling-continuation32.ids"
```

Each executable prints the request number, prefill and decode milliseconds, count of matching greedy predictions and first predicted ID. A fresh process runs one warm-up and four measured requests. For the table we ran five LeanSonic–llama.cpp–llama.cpp–LeanSonic cycles on the same idle GPU, with an 8,192-token context and drafting disabled. Each process used the fixed continuation even while checking its own greedy predictions. The paired reduction was 23.83% for prefill (95% bootstrap interval 23.77–23.89%) and 39.77% for decode (39.75–39.79%). The raw samples and aggregation method are in the [comparison receipt](benchmarks/inkling-llama-bf16.json).

## Score another engine against the same reference

`eval-kld --logits-f32` reads raw teacher-forced logits from a file or standard input and runs on the CPU. For example, to score chunk 69 alone, run:

```bash
leansonic eval-kld --ref inkling2-bf16-base.kld --start-chunk 69 --chunks 1 --logits-f32 chunk69.f32
```

Feed the reference token IDs to the engine in their original order, resetting its state at each chunk. For this 512-token oracle, export the distribution immediately after each input position from 256 through 510 inclusive. The target at position 510 is token 511. Each row contains 200,058 little-endian FP32 logits in vocabulary-ID order. Concatenate the 255 rows per chunk, then concatenate the requested chunks in increasing order, without headers or padding. A single chunk occupies 204,059,160 bytes. Pass `--logits-f32 -` to stream rows from a producer. The JSON fields and reference scoring rules are shared with GPU evaluation. Record the producer's model, vocabulary, prefill schedule and KV format alongside the result.

## V12 with native Q8 scale dictionaries

These runs use LeanSonic `7a02b538` (default Tensor Core prefill), a text-only Basalt `5fe1928d` package, and the pinned llama.cpp BF16 reference. The LeanSonic text shard plus tokenizer occupies 16,038,568,990 bytes. Basalt’s text-only package occupies 18,327,104,737 bytes. The BF16 reference GGUF occupies 52,226,509,568 bytes.

| GPU / engine | Prompt tokens | Prefill (ms) | Decode steps | Decode (ms) | Highest sampled process VRAM (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX 5060 Ti / LeanSonic | 32 | 246.34 | 16 | 335.72 | 15.422 |
| RTX 5060 Ti / Basalt | 32 | 293.60 | 16 | 377.30 | 14.459 |
| RTX 5060 Ti / LeanSonic | 1,024 | 388.83 | 16 | 392.60 | 15.418 |
| RTX 5060 Ti / Basalt | 1,024 | 1,284.50 | 16 | 393.60 | 14.461 |
| RTX PRO 4500 / LeanSonic | 32 | 38.51 | 16 | 115.87 | 16.369 |
| RTX PRO 4500 / Basalt | 32 | 55.90 | 16 | 115.70 | 18.096 |
| RTX PRO 4500 / LeanSonic | 1,024 | 134.86 | 16 | 118.68 | 16.895 |
| RTX PRO 4500 / Basalt | 1,024 | 131.10 | 16 | 121.10 | 18.096 |
| RTX 5070 Ti / LeanSonic | 32 | 107.74 | 16 | 168.76 | 15.412 |
| RTX 5070 Ti / Basalt | 32 | 265.80 | 16 | 312.60 | 14.453 |
| RTX 5070 Ti / LeanSonic | 1,024 | 164.99 | 16 | 190.29 | 15.410 |
| RTX 5070 Ti / Basalt | 1,024 | 450.10 | 16 | 324.10 | 14.455 |
| RTX PRO 5000 / LeanSonic | 32 | 33.18 | 16 | 96.22 | 16.447 |
| RTX PRO 5000 / Basalt | 32 | 50.30 | 16 | 87.00 | 18.174 |
| RTX PRO 5000 / LeanSonic | 1,024 | 102.63 | 16 | 98.34 | 16.953 |
| RTX PRO 5000 / Basalt | 1,024 | 106.60 | 16 | 93.00 | 18.174 |
| RTX PRO 5000 / LeanSonic | 1,024 | 94.65 | 32 | 196.62 | 17.004 |
| RTX PRO 5000 / llama.cpp BF16 | 1,024 | 598.69 | 32 | 566.38 | 51.762 |

The RTX 5060 Ti comparison uses the README prompt and its first 32 IDs, FP16 KV, context 8,192, greedy selection and speculation disabled. Each prompt ran once per fresh process. Five balanced cycles compared native Q8 scales, dictionary Q8 scales and Basalt in mirrored, rotating order. All three produced the same first 16 IDs at both lengths. For the 1,024-token prompt the dictionary package saved 69.73% of Basalt's prefill time. Its 16-step decode difference was within run-to-run variation (paired 95% interval for time saved −0.18% to +0.58%). LeanSonic used about 0.96 GiB more peak VRAM. Basalt's CLI reports its phase times to whole milliseconds.

On RTX 5070 Ti at 1,024 input tokens, LeanSonic took 164.99 ms for prefill and 190.29 ms for 16 decode steps. Basalt took 450.10 ms and 324.10 ms. LeanSonic used 0.955 GiB more sampled peak VRAM. On PRO 5000 the ordering split by phase: prefill took 102.63 vs 106.60 ms, while decode took 98.34 vs 93.00 ms. The paired 95% interval for the PRO 5000 decode time saved by LeanSonic was −6.32% to −5.25%. All 240 processes across four cards returned the same first 16 IDs for their prompt.

On PRO 4500 the same balanced comparison returned identical first-16 token IDs in every run. At 1,024 input tokens the paired 95% interval for prefill time saved against Basalt ranged from −8.23% to +0.43%. LeanSonic's 16-step decode time was lower (paired interval for time saved 1.81% to 2.21%). At 32 input tokens the prefill time reduction was 31.1% and the decode interval included zero.

On PRO 5000, use the two fixed-continuation harnesses described above with the V12 revision and dictionary package. Each process ran five requests. All 100 requests predicted the 32 supplied IDs, and the timing excludes each process's first request. The two workloads in the table have different decode lengths and warm-up protocols.

For the Basalt comparison, prepare the [text-only Basalt container](https://github.com/tjm8874/LeanSonic/tree/87e016b/docs/benchmarks/inkling-basalt-text), create the short prompt from the first 32 comma-separated IDs, and substitute either token file for `$PROMPT`. These settings run 16 decode forwards in both engines. Basalt counts the prefill prediction in its output limit.

```bash
export LEANSONIC_CTX=8192 LEANSONIC_KV=fp16 LEANSONIC_CUBIN_PATH="$PWD/inkling.cubin"
target/release/leansonic bench --weights "$DICT" --tokens "@$PROMPT" --gen 16 --spec 0 --repeat 1
"$BASALT" --model "$TEXT_BASALT" --tokens-file "$PROMPT" --gpu 0 --kv fp16 --max-context 8192 --spec 0 --temperature 0 --max-new 17
```

On the separate 25,500-position WikiText BF16 evaluation, V12 had mean KL 0.003176, worst KL 0.674716, 24,879 top-1 matches and PPL 7.072582. Basalt had mean KL 0.002984, worst KL 0.330612, 24,948 matches and PPL 7.068542. The raw BF16 generator reported PPL 7.0724; the clipped and quantized reference file gives 7.064408. The [measurement file](benchmarks/inkling-v12-matched-engines.json) includes the run-level timings, sampled VRAM, token IDs, input hashes and quality results.
