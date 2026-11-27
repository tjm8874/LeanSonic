# Fixed-input comparison with public Inkling Q3_K_XL

The tested LeanSonic package including tokenizer occupies 13,479,710,294 bytes, and the reference GGUF occupies 13,631,778,016 bytes. The SHA-256 of the complete file, native driver digests and input digests are included in the receipt. Build a2008463cf028ee56bdefe39e79f6c2f95d57e73 uses the bc526e3 cubin. The comparator is Unsloth llama.cpp at 883286076efe5b85fddb31d9cc7035a6b698ba9b. The LeanSonic artifact resides on a separate local data volume. Warm phase timings exclude file load.

Both engines consume exactly the same 1,024 README IDs and fixed 32-token continuation, one sequence at context 8,192, speculation off, FP16 KV, all weights resident. Each of five paired cycles launches two fresh processes per arm; the phase means use four warm requests per process. Peak bytes cover the entire process including model load.

| GPU | Engine | Prefill ms (tok/s) | Decode 32 ms (tok/s) | Peak VRAM B | Matched continuation IDs |
|---|---|---:|---:|---:|---:|
| 5060Ti | lean | 257.289 (3,980.0) | 339.787 (94.18) | 16,009,658,368 | 32/32 |
| 5060Ti | llama | 881.167 (1,162.1) | 573.510 (55.80) | 16,030,629,888 | 30/32 |

## BF16 corpus on the same file

PRO 4500, 100 chunks and 25,500 scored positions per split, prefill width 512 and FP16 KV.

| Split | Mean KL | Worst KL | Top-1 matches | PPL |
|---|---:|---:|---:|---:|
| test | 0.0077481 | 1.4620404 | 24,537 | 7.083731 |
| valid | 0.0069527 | 0.7446714 | 24,645 | 6.530797 |

Build the local model using the [storage recipe](README.md) and use the [fixed-input driver, IDs, environment and procedure](https://github.com/tjm8874/LeanSonic/tree/18ae3c2/docs/benchmarks/inkling-nvfp4-trim-context/fixed-replay.md). [Individual requests, phase times, peaks, scores and checksums](fixed-q3.json).
