# Fixed-input comparison with public Inkling Q3_K_XL

The tested LeanSonic package including tokenizer occupies 13,608,770,734 bytes, and the reference GGUF occupies 13,631,778,016 bytes. The SHA-256 of the complete file, native driver digests and input digests are included in the receipt. Build 8c77955c0cf04ebb7c69599612c6caf5dddee71a uses the bc526e3 cubin. The comparator is Unsloth llama.cpp at 883286076efe5b85fddb31d9cc7035a6b698ba9b. Both model files reside on the model volume.

Both engines consume exactly the same 1,024 README IDs and fixed 32-token continuation, one sequence at context 8,192, speculation off, FP16 KV, all weights resident. Each of five paired cycles launches two fresh processes per arm; the phase means use four warm requests per process. Peak bytes cover the entire process including model load.

| GPU | Engine | Prefill ms (tok/s) | Decode 32 ms (tok/s) | Peak VRAM B | Matched continuation IDs |
|---|---|---:|---:|---:|---:|
| 5070Ti | lean | 121.806 (8,406.8) | 192.427 (166.30) | 16,112,418,816 | 32/32 |
| 5070Ti | llama | 526.444 (1,945.1) | 402.392 (79.52) | 16,150,167,552 | 31/32 |
| 5060Ti | lean | 257.397 (3,978.3) | 339.792 (94.18) | 15,965,618,176 | 32/32 |
| 5060Ti | llama | 881.623 (1,161.5) | 573.753 (55.77) | 16,030,629,888 | 30/32 |

## BF16 corpus on the same file

PRO 4500, 100 chunks and 25,500 scored positions per split, prefill width 512 and FP16 KV.

| Split | Mean KL | Worst KL | Top-1 matches | PPL |
|---|---:|---:|---:|---:|
| test | 0.0077481 | 1.4620404 | 24,537 | 7.083731 |
| valid | 0.0069527 | 0.7446714 | 24,645 | 6.530797 |

Build the local model using the [storage recipe](README.md) and use the [fixed-input driver, IDs, environment and procedure](https://github.com/tjm8874/LeanSonic/tree/18ae3c2/docs/benchmarks/inkling-nvfp4-trim-context/fixed-replay.md). [Individual requests, phase times, peaks, scores and checksums](fixed-q3.json).
