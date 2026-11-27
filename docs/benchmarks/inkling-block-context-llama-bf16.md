# Inkling-2-Mini with lossless block-context weights vs llama.cpp BF16

On RTX PRO 5000 72GB, both engines predicted all 32 supplied continuation tokens in each of 100 requests. Each fresh process made five requests. The table gives the mean of the four warm requests, over five mirrored pairs of processes per engine (two processes × five cycles). Prefill used the same 1,024 input IDs and the decode phase executed 32 greedy forward calls with FP16 KV, context 8,192 and speculation off. Model load and reset are excluded from both phase timers. Peak VRAM is the largest per-PID NVML reading including load, sampled every 4 ms.

| Engine | Prefill ms | Decode32 ms | Process peak bytes | Text model incl. tokenizer, bytes |
|---|---:|---:|---:|---:|
| LeanSonic native + lossless NVFP4 codes | 90.07 | 189.66 | 18,234,736,640 | 15,853,391,742 |
| llama.cpp official BF16 GGUF | 598.57 | 565.91 | 55,578,722,304 | 52,226,509,568 |

LeanSonic ran commit `1e4825b` (production kernel, core, CLI and packer sources identical to `32710d3` in #109). The comparison uses the pinned Inkling-capable llama.cpp revision `29b52d3` and its fully GPU-offloaded BF16 model. Model representations differ. The [reference instructions](../inkling-reference.md) build both fixed-token benchmarks. Run them with the [README token IDs](inputs/inkling-readme1024.ids) and [continuation](inputs/inkling-continuation32.ids). Build the LeanSonic weights using [block-context packing](inkling-nvfp4-context.md). Use the quoted commits and the [per-process data](inkling-block-context-llama-bf16.json) to verify file hashes, requests, memory and timings.
