# Capped Inkling-2-Mini against native Q3_K_XL

Both engines see the same 1,024 README prompt IDs and consume the same 32 continuation IDs at context 8,192, one sequence, FP16 KV and speculation off. All 25 llama.cpp layers are GPU-offloaded with a 512-token microbatch. LeanSonic uses the capped 15-symbol package and scratch-free U4 lookup on the fused-verifier build. These weights differ from both original Q3 and the earlier compact LeanSonic distribution. The LeanSonic package includes its tokenizer; the Q3 GGUF embeds its tokenizer. They occupy 13,624,461,934 and 13,631,778,016 bytes respectively.

| GPU | Engine | Prefill ms (tok/s) | Decode 32 ms (tok/s) | Peak bytes | Matched fixture IDs |
|---|---|---:|---:|---:|---:|
| 5060Ti | LeanSonic capped | 257.592 (3,975.3) | 339.832 (94.16) | 16,013,852,672 | 32/32 |
| 5060Ti | Unsloth llama.cpp Q3_K_XL | 882.035 (1,161.0) | 574.012 (55.75) | 16,030,629,888 | 30/32 |

5060Ti LeanSonic retained 24/24 expert layers. Speedup over native Q3, with paired-cycle 95% intervals: prefill 3.424× (3.421–3.427), decode 1.689× (1.688–1.691).

## BF16 test with the same weights and integrated executable

| Device path on PRO 4500 | Mean KL | Worst KL | BF16 top-1 | PPL |
|---|---:|---:|---:|---:|
| scratch-free | 0.0081645 | 1.2202775 | 24,500/25,500 | 7.081285 |

## Reproduce

Use the [capped model build and fingerprint](https://github.com/tjm8874/LeanSonic/tree/c9a6a63/docs/benchmarks/inkling-codebooks/nvfp4-budget/capped). Build the integrated CLI and cubin at bc526e3. Use the [same native driver, model and fixed inputs as the earlier two-board protocol](https://github.com/tjm8874/LeanSonic/tree/b95ce21/docs/benchmarks/inkling-device-codebooks/integration). Each board ran five mirrored ABBA/BAAB cycles, ten fresh processes per arm. The phase means use four warm requests per process after the first excluded request. Process VRAM is the largest sampled allocation during load and inference. For the corpus score use explicit prefill width 512 with the same 100-chunk BF16 oracle. [All requests, predicted IDs, peaks, input and build digests](receipt.json).
