# Load both scale-coded layouts on the same RTX PRO 4500

Each process used the same executable and cubin, FP16 KV, PDL, one sequence, no speculation, context 2,048 and prefill width 512. Five paired cycles alternated ABBA and BAAB, with ten fresh processes per layout. File cache was not flushed. The engine printed model-load wall time, and GNU time recorded peak process RSS. The warm phase values below use the last three 1,024-token requests in each process. All 160 requests across both prefix lengths reproduced the same prefix-specific 64 greedy IDs, next ID and final FP32 logit digest.

| Metric | Split streams | Direct order | Direct minus split (paired 95% interval) |
|---|---:|---:|---:|
| Model load (seconds) | 14.573 | 8.181 | -6.392 [-6.698, -6.092] |
| Peak host RSS (KiB) | 6,650,351.2 | 6,650,220.8 | -130.4 [-405.2, +144.4] |
| 1,024-token prefill (ms) | 173.269 | 173.481 | +0.212 [+0.132, +0.288] |
| 64-token decode (ms) | 421.705 | 421.787 | +0.083 [-0.009, +0.166] |

Direct order reduced load time by 43.9% and saved 1,784 distribution bytes relative to the split layout. Warm prefill increased by 0.212 ms (0.12%). The intervals for peak host RSS and warm decode include zero. These measurements compare only the two listed files in the same cycles.

Use the [direct and split pack recipes](README.md) and [parity driver](https://github.com/tjm8874/LeanSonic/tree/cc62564/docs/benchmarks/inkling-nvfp4-trim-context). Wrap each process with GNU time -f %M. [Per-process data, IDs, hashes and intervals](startup.json).
