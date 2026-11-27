# Cost of reading the smaller model

RTX PRO 4500, Ryzen 9 9900X, same integrated executable and cubin, FP16 KV, explicit prefill width 512. Five alternating ABBA/BAAB cycles started ten fresh processes for each distribution. Each process ran eight fixed requests. All continuations, next IDs and hashes of the final FP32 logit vector agreed. The OS file cache was retained between processes. GNU time recorded peak host RSS for the entire process.

| Observation | 13,608,770,734 B trim | 13,479,712,078 B scale split | Paired difference (95% interval) |
|---|---:|---:|---:|
| Loader seconds | 9.129 | 17.260 | +8.131 [+7.948, +8.314] |
| Mean of process peak RSS (KiB) | 6,650,272.400 | 6,650,290.800 | +18.400 [-161.200, +230.000] |
| Warm prefill 1,024 (ms) | 174.723 | 174.465 | -0.258 [-0.367, -0.149] |
| Warm decode 64 (ms) | 422.543 | 422.464 | -0.079 [-0.311, +0.213] |

The largest measured host RSS was 6,650,716 KiB before and 6,650,944 KiB after. The main cost was startup: the smaller package took an extra 8.131 s to load. Keep this format opt-in while testing a faster decoder. [Full observations and generated IDs](load.json) · [Reproduction settings](runtime.md).
