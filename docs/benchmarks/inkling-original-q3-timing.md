## Original Q3 runtime on RTX PRO 4500 Blackwell

One sequence, 8,192 context, FP16 KV, greedy, no speculation. Both engines processed the same 1,024 IDs and each consumed the same 32 forced next inputs even where a prediction diverged. All 25 native layers were offloaded. Five alternating ABBA/BAAB cycles, ten fresh processes per engine, five requests per process, first excluded. The phase figures are averages over 40 warm requests per engine. Peak is the largest sampled per-process GPU allocation during loading and inference, sampled at approximately 4 ms intervals using NVML.

| Engine / package | Prefill 1024 (ms / tok/s) | Decode 32 (ms / tok/s) | Peak GPU bytes | Package bytes | Predicted continuation IDs matching fixture |
|---|---:|---:|---:|---:|---:|
| LeanSonic #117 | 122.860 / 8,334.7 | 201.557 / 158.76 | 18,106,810,368 | 15,724,161,014 | 32/32 |
| Unsloth llama.cpp original Q3_K_XL | 472.619 / 2,166.6 | 408.651 / 78.31 | 16,999,514,112 | 13,631,778,016 | 31/32 |

Use the [fixed-token driver and replay commands](https://github.com/tjm8874/LeanSonic/blob/5e9b9f7/docs/benchmarks/inkling-original-iq4.md), substituting the original Q3_K_XL GGUF for the IQ4_XS file. Both use Unsloth's release at commit `883286076efe5b85fddb31d9cc7035a6b698ba9b`. [Individual phase timings, prediction IDs, memory and file hashes](inkling-original-q3-timing.json). The 32 predictions establish task mechanics and do not substitute for BF16 distribution scoring.
