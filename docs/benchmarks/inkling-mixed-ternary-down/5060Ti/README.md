# BF16 corpus scores with 19/20 down projections in two-bit mode

The 61-expert model used RTX 5060 Ti, all expert layers resident, one sequence, FP16 KV, prefill width 512, TERNARY_UP=1 and TERNARY_DOWN=auto. The original IQ1_S GGUF used the [pinned native llama.cpp scorer](https://github.com/tjm8874/LeanSonic/tree/d65d1a6/docs/benchmarks/inkling-native-lowbits) on RTX PRO 5000. Both score identical reference token positions. Each split has 25,500 rows. The LeanSonic package is 300,130 bytes smaller.

| Split | Package | Mean KL | Worst KL | Top-1 matches | PPL |
|---|---|---:|---:|---:|---:|
| test | LeanSonic 61-expert | 0.2247122 | 6.6547808 | 20,453/25,500 | 8.482171 |
| test | Original IQ1_S | 0.1130210 | 5.8886900 | 21,921/25,500 | 7.707463 |
| valid | LeanSonic 61-expert | 0.2077958 | 6.5760688 | 20,786/25,500 | 7.841980 |
| valid | Original IQ1_S | 0.1076170 | 5.5685180 | 22,138/25,500 | 7.121110 |

On the test split, the original IQ1_S is better on all four measures. The exact equivalence between two storage modes of LeanSonic on the smaller prompt set does not imply equivalence to BF16 or the native model. [Raw scores, worst prefix, token target and checksums](scores.json).
