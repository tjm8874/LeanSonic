# The 571-expert package on RTX 5070 Ti

The 11,662,590,054-byte package reproduces every aggregate BF16 statistic from RTX PRO 4500 on both splits. Here the two LeanSonic packages and native IQ3_XXS were scored on one physical RTX 5070 Ti. Each split uses the same 100 WikiText chunks, 512-token context, FP16 KV and 25,500 scored positions. The bounded parent uses 11,662,534,370 bytes; original IQ3_XXS includes its tokenizer in 11,663,309,504 bytes.

| Split | Model | Mean KL | Worst KL | Top-1 / 25,500 | PPL |
| --- | --- | ---: | ---: | ---: | ---: |
| test | 571 experts | 0.0169571 | 2.8585658 | 24,084 | 7.127375 |
| test | Bounded 217 | 0.0197912 | 2.8942013 | 23,950 | 7.137518 |
| test | llama.cpp IQ3_XXS | 0.0280570 | 5.5014780 | 23,666 | 7.193132 |
| valid | 571 experts | 0.0157347 | 2.5524491 | 24,200 | 6.566551 |
| valid | Bounded 217 | 0.0181395 | 2.3339590 | 24,105 | 6.576801 |
| valid | llama.cpp IQ3_XXS | 0.0256590 | 3.7240360 | 23,786 | 6.616409 |

Against native the 571-expert model improves all four metrics on both splits. Compared with the bounded 217-expert package, it costs 55,684 more bytes, improves mean KL, top-1 agreement and PPL on both splits, and improves test worst KL, while validation worst KL increases. [Package assembly](../README.md) · [Raw scores and hashes](scores.json) · [Same-GPU native scorer, inputs and commands](https://github.com/tjm8874/LeanSonic/tree/3f05d37/docs/benchmarks/inkling-embedding-lut/refit/native-corpus).

## Matched timing and peaks

Same 1,024 input tokens and 32 forced decode steps, context 8,192, one sequence, FP16 KV, no speculation. Five mirrored ABBA/BAAB cycles, ten processes per engine, one excluded warmup and four timed requests per process. All LeanSonic expert layers and all 25 native layers remained resident.

| Engine | Prefill tok/s | Decode tok/s | Peak process GPU bytes |
|---|---:|---:|---:|
| 571 experts | 8,424.59 | 170.96 | 16,083,058,688 |
| llama.cpp IQ3_XXS | 1,934.82 | 80.03 | 15,049,162,752 |

Use the [same fixed-token runner and native engine](https://github.com/tjm8874/LeanSonic/blob/f6dbff3/docs/benchmarks/inkling-embedding-lut/refit/native-corpus/timing.md) with the 571-expert weights and release loader 726b6f2. The timing archive includes every request, both input digests, sampled peaks and paired cycle intervals. [Raw timing](timing.json.gz).
