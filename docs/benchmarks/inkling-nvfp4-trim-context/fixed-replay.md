# Fixed-input replay with lossless trim storage

The exact 13,445,948,726-byte LeanSonic package in the [storage receipt](storage.json) was compared with the original 13,631,778,016-byte Unsloth Inkling Q3_K_XL GGUF. These are different quantizations of Inkling-2-Mini. Both tokenizers are included in the reported sizes. All weights remained GPU-resident. Both runs consume the same 1,024 prompt IDs and 32 fixed continuation IDs, single sequence, context 8,192, FP16 KV, no speculation. For each board, five mirrored cycles produced ten fresh processes per engine and four warm requests per process. Peak memory is the largest sampled process allocation including model load.

| GPU | Engine | Prefill ms (tok/s) | Decode 32 ms (tok/s) | Peak VRAM B | Predicted fixture IDs |
|---|---|---:|---:|---:|---:|
| 5060Ti | LeanSonic trim | 257.647 (3,974.4) | 339.854 (94.16) | 16,015,949,824 | 32/32 |
| 5060Ti | Unsloth llama.cpp | 882.146 (1,160.8) | 573.947 (55.75) | 16,030,629,888 | 30/32 |

## Quality of the same LeanSonic artifact

RTX PRO 4500, integrated CLI, prefill width 512 and FP16 KV. Each WikiText-2 split contains 100 chunks, 25,500 scored positions, context length 512.

| Split | Mean KL | Worst KL (chunk/position) | BF16 top-1 matches | PPL |
|---|---:|---:|---:|---:|
| test | 0.0081645 | 1.2202775 (99/398) | 24,500 | 7.081285 |
| valid | 0.0075089 | 1.4429343 (12/317) | 24,566 | 6.529914 |

## Reproduce

Use the [existing fixed-ID benchmark and native driver](https://github.com/tjm8874/LeanSonic/tree/b95ce21/docs/benchmarks/inkling-device-codebooks/integration) and the [build and environment for the trim loader](runtime.md). The integrated source is 8c77955c0cf04ebb7c69599612c6caf5dddee71a (bc526e3 plus the storage change), llama.cpp is 883286076efe5b85fddb31d9cc7035a6b698ba9b. The cubin is identical to bc526e3. Keep the same prompt, continuation and native microbatch of 512. Use the integrated replay executable and matching package for the LeanSonic arm. [Build digests, predicted IDs, per-request times, paired intervals, memory peaks and quality scores](fixed-replay.json).
