# Three-way quality on the same RTX PRO 4500 scorer

Each model used the same 100 WikiText-2 chunks in each split (25,500 positions), one sequence, FP16 KV, a 512-token prefill workspace and no drafter. Scores compare each output distribution against the same BF16 reference. Physical package bytes include the tokenizer.

| Split | Model | Bytes | Mean KL | Worst KL (chunk/position) | Top-1 agreement | PPL |
|---|---|---:|---:|---:|---:|---:|
| test | ternary | 7,718,950,382 | 0.2341477 | 7.5035677 (90/424) | 20,385/25,500 (79.94%) | 8.566626 |
| test | ranked | 7,747,846,286 | 0.2255251 | 6.6990397 (90/424) | 20,448/25,500 (80.19%) | 8.508404 |
| test | expanded | 7,763,991,646 | 0.2247130 | 6.2064049 (1/260) | 20,453/25,500 (80.21%) | 8.492011 |
| valid | ternary | 7,718,950,382 | 0.2161375 | 6.5853805 (10/471) | 20,643/25,500 (80.95%) | 7.904899 |
| valid | ranked | 7,747,846,286 | 0.2088534 | 6.5426205 (10/471) | 20,775/25,500 (81.47%) | 7.849810 |
| valid | expanded | 7,763,991,646 | 0.2080759 | 6.6786801 (10/471) | 20,753/25,500 (81.38%) | 7.845188 |

Ranked restores 42 experts, expanded restores 61. Both retain the same training-only ranking. [Raw metrics, binary and cubin hashes, worst-position inputs and targets](corpus.json).

Moving from 42 to 61 improves all four test measures. On validation it improves mean KL and PPL but loses 22 top-1 matches and increases worst KL from 6.5426205 to 6.6786801. Keep both operating points visible. A separate newer runtime on RTX 5060 Ti changes some absolute scores, so this table compares all three using the same original scorer and cubin. [Newer runtime and native IQ1_S measurements](https://github.com/tjm8874/LeanSonic/tree/342f521/docs/benchmarks/inkling-mixed-ternary-down/5060Ti).
