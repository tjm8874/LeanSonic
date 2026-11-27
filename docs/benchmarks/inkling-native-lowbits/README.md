# Original lower-bit Inkling-2-Mini distributions

The Unsloth public release contains models smaller than original Q3_K_XL and the 13,626,745,974-byte compact LeanSonic distribution. Each GGUF here includes its tokenizer. We verify the exact byte count and SHA-256 from the pinned release before measuring.

| Native package | Distribution bytes | Smaller than compact LeanSonic by |
|---|---:|---:|
| Inkling-2-Mini-UD-IQ3_XXS.gguf | 11,663,309,504 | 1,963,436,470 |
| Inkling-2-Mini-UD-IQ1_S.gguf | 7,764,291,776 | 5,862,454,198 |

The three original GGUFs (Q3_K_XL, IQ3_XXS, IQ1_S) have identical names and logical dimensions for all 734 tensors. The 12 tokenizer fields and 35 architecture-specific fields match byte for byte. Use [the standalone metadata check](verify_layout.py) with the three GGUF paths in that order. The receipt includes the individual field hashes.

## Quality on RTX PRO 5000

| Fixture | Engine | Mean KL | Worst KL | BF16 top-1 | PPL |
|---|---|---:|---:|---:|---:|
| test | Original IQ3 | 0.027728 | 4.049091 | 23,655/25,500 | 7.188809 |
| valid | Original IQ3 | 0.025635 | 2.807837 | 23,785/25,500 | 6.613878 |

For the same corpus oracles, compact LeanSonic used RTX PRO 4500 and the original native packages used RTX PRO 5000. The bilingual runs used RTX PRO 5000 for both. The following are the [pinned compact results](https://github.com/tjm8874/LeanSonic/blob/10425ae/docs/benchmarks/inkling-codebooks/results.json).

| Fixture | LeanSonic compact mean KL | Worst KL | BF16 top-1 | PPL |
|---|---:|---:|---:|---:|
| test | 0.0137735 | 0.7519345 | 24,289/25,500 | 7.131377 |
| valid | 0.0131739 | 0.9835867 | 24,341/25,500 | 6.564200 |

## Fixed-token replay

Five mirrored ABBA/BAAB cycles, ten fresh processes per engine, five requests per process, first excluded. Context 8,192, 1,024 fixed prompt IDs and 32 forced decode steps, one sequence, FP16 KV, no speculation. NVML samples the process during load and inference at about 4 ms intervals. Each row is a warm mean over 40 requests.

| Pair | Runtime | Prefill ms (tok/s) | Decode 32 ms (tok/s) | Peak GPU bytes |
|---|---|---:|---:|---:|
| IQ3 | Compact LeanSonic | 89.770 (11,407.0) | 155.449 (205.86) | 18,251,513,856 |
| IQ3 | Original llama.cpp | 442.906 (2,312.0) | 380.059 (84.20) | 15,187,574,784 |
| IQ1 | Compact LeanSonic | 89.695 (11,416.5) | 155.442 (205.87) | 18,232,639,488 |
| IQ1 | Original llama.cpp | 446.080 (2,295.6) | 361.137 (88.61) | 11,301,552,128 |

The latency pairs ran on RTX PRO 5000 72 GB. Exact-token replay keeps decoder history fixed across different predictions. Compact LeanSonic matched all 32 forced inputs, and each native GGUF matched 30. The IQ3 pair uses 13,626,745,974 versus 11,663,309,504 distributed bytes, and the IQ1 pair uses 13,626,745,974 versus 7,764,291,776 bytes. The original runtimes used less peak GPU memory in both pairs. [Individual requests](requests.csv) · [Speed ratios, paired cycle intervals and matches](timing-pairs.json).

The native corpus scorer requested 99 GPU layers. Its default corpus log reports CUDA and batch dimensions but omits the actual layer count. The matched latency driver separately checks full offload for each native process.

## Reproduce

Download the original model files named in the table from the [pinned public release](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/ae9327c5fbc7333116ba32eee78bb5d2bbefae4f). Use the same Unsloth llama.cpp revision `883286076efe5b85fddb31d9cc7035a6b698ba9b` and the [native WikiText scoring command](../inkling-original-gguf-quality.md), substituting each GGUF and the 100-chunk oracle for that split. For bilingual scores, use the [613-position fixture and native reference driver](../inkling-chat-quality.md). For speed, use the [fixed-token replay instructions](../inkling-original-q3-timing.md) and compact weights from [PR #147](https://github.com/tjm8874/LeanSonic/pull/147). The measured LeanSonic binary is d08c1cf with its default byte-storage path. [File and input digests, scores and traces](receipt.json).
