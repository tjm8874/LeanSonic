# BF16 corpus comparison at the native IQ3_XXS size

The two LeanSonic packages used the same RTX PRO 4500, four-bank-capable executable, bc526e3 cubin, FP16 KV, prefill width 512, one sequence and no drafter. Both score 25,500 positions in each completed 100-chunk WikiText-2 split. All four parent metrics also exactly reproduce the [two-bank model](../quality.md). The native GGUF scored the same reference files using Unsloth llama.cpp 8832860 on RTX PRO 5000. Its scorer reports the maximum without its position.

| Split | Package (bytes) | Mean KL | Worst KL (chunk/position) | BF16 top-1 | PPL |
|---|---|---:|---:|---:|---:|
| test | four (11,577,239,270) | 0.0230838 | 2.4809246 (23/330) | 23,829/25,500 | 7.158300 |
| test | restored (11,648,582,790) | 0.0225754 | 2.5457894 (23/330) | 23,842/25,500 | 7.157658 |
| test | public IQ3_XXS (11,663,309,504) | 0.027728 | 4.049091 | 23,655/25,500 | 7.188809 |
| valid | four (11,577,239,270) | 0.0216190 | 2.9282704 (11/374) | 23,939/25,500 | 6.609503 |
| valid | restored (11,648,582,790) | 0.0209783 | 3.0298716 (11/374) | 23,957/25,500 | 6.604097 |
| valid | public IQ3_XXS (11,663,309,504) | 0.025635 | 2.807837 | 23,785/25,500 | 6.613878 |

[Raw results, maximum-divergence input histories and executables](corpus.json) · [Native scorer and pinned GGUF](https://github.com/tjm8874/LeanSonic/tree/f14d5c6/docs/benchmarks/inkling-native-lowbits).

Restoring the two complete planes improved mean KL, top-1 agreement and PPL on both splits. It increased worst KL on both, with the same worst token in each parent/candidate pair. On validation the native IQ3_XXS has lower worst KL than either LeanSonic package. Keep whole-plane restoration experimental while testing expert-level selection.
