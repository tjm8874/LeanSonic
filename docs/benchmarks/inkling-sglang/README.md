# Original SGLang on Inkling-2-Mini

Single-sequence inference on RTX PRO 5000 72 GB Blackwell, driver 595.99.02. Both configurations use one 2,048-token context and an identical 1,024-token README prefix. SGLang greedily generates 33 IDs. LeanSonic replays the first 32 as fixed input to keep the history identical even if a prediction differs. Both produce 33 predictions. The table counts 1,024 prefill inputs and 32 subsequent decode steps. Warm phase figures are medians over seven requests after excluding the first. Each engine has one process lifetime in this comparison.

| Engine and distribution | Prefill ms | Prefill tok/s | Decode 32 ms | Decode tok/s | Sampled peak GPU bytes | Distribution bytes |
|---|---:|---:|---:|---:|---:|---:|
| SGLang, original checkpoint | 84.707 | 12,088.7 | 245.663 | 130.26 | 24,012,390,400 | 20,191,643,333 |
| LeanSonic, canonical | 89.449 | 11,447.9 | 172.226 | 185.80 | 17,882,415,104 | 15,548,157,542 |

The engines agreed on 32/33 greedy predictions along the fixed replay.

SGLang uses the original official checkpoint, BF16 compute/KV configuration, FA4 attention, single GPU, page size 128, no radix cache, one active request and a static memory fraction of 0.85. Its prefill phase is the engine-reported prefill latency. Decode time uses the reported decode throughput. LeanSonic uses FP16 KV, PDL and its native phase timers. The measurements therefore include implementation and precision choices. NVML samples the target process and descendants across loading and inference. SGLang also ran the bilingual suite during that process lifetime. The recorded SGLang distribution is the complete pinned snapshot, including a 635,801,416-byte MTP file. The timing run used no speculative decoder. The total describes the downloaded distribution, and a smaller text-only installation may be possible.

## Free generation in English and Japanese

SGLang answered 7/12 questions correctly at 128 tokens and all 12 at 384. The shorter cap is scored by truncating the same greedy stream. A visible answer must be correct and terminate within budget. The repeat of the first question reproduced the entire token sequence. For reference, canonical LeanSonic scored 8/12 and 12/12 in the [independent 512-context suite](../inkling-functional-chat.md). Those earlier results used the same IDs and scoring rules with a different configured context capacity.

## Reproduce

Use fresh directories and pin one GPU. Install [SGLang at the measured revision](https://github.com/sgl-project/sglang/tree/9c64454bba553e475ce7b7b790015d2f941e2807) and download the [official checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/b85c3cfa615fc413cc9af3379f491b963f3b586a). The [distribution manifest](distribution.json) contains all file sizes and SHA-256 digests. Set REPO to the LeanSonic checkout with these fixtures, OFFICIAL to the checkpoint, and SGLANG_PREPARED to the absolute path of distribution.json.

```sh
tools="$REPO/tools/inkling_chat_tasks"
bench="$REPO/docs/benchmarks/inkling-sglang"
cp "$tools/fixtures.json" "$REPO/docs/benchmarks/inputs/inkling-readme1024.ids" .
python3 "$tools/prepare.py" "$OFFICIAL" ids
python3 "$bench/measure.py"
python3 "$tools/score_native.py" "$OFFICIAL" traces/sglang.ids >scores.json
```

For the matched LeanSonic leg, build the driver and native cubin from engine commit `1f70908`. Set PINNED to this checkout and MODEL to the [canonical package](../inkling-storage-candidates.md). The fixture continuation was extracted from the first SGLang request. To measure per-process VRAM, launch each command in the background on an idle GPU, save its PID in a file and immediately run `python3 peak.py PID_FILE PEAK_JSON GPU_UUID`. The sampler selects descendants and checks Linux process start time.

```sh
cp "$bench/replay.rs" "$PINNED/crates/leansonic-core/examples/inkling_sglang_replay.rs"
(cd "$PINNED" && cargo build --release -p leansonic-core --example inkling_sglang_replay && nvcc -arch=sm_120a -std=c++17 -O3 --cubin kernels/inkling2_sm120.cu -o matched.cubin)
LEANSONIC_KV=fp16 LEANSONIC_PDL=1 LEANSONIC_SEQS=1 LEANSONIC_CUBIN_PATH="$PINNED/matched.cubin" "$PINNED/target/release/examples/inkling_sglang_replay" "$MODEL" inkling-readme1024.ids "$bench/continuation.ids"
```

[Native phase metrics and predictions](timing.json) · [configuration and all requests](functional.json) · [answers and token hashes](scores.json) · [SGLang peak sample](sglang-peak.json).
