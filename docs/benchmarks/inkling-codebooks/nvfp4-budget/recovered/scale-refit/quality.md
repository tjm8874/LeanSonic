# BF16 quality after scale-only refit

RTX PRO 4500 Blackwell, FP16 KV, batch one, prefill chunk 512, 100 chunks and 25,500 positions per split. The same native CLI (a200846) scores both sets of weights. [Predeclared gate](https://github.com/tjm8874/LeanSonic/issues/35#issuecomment-7220649399).

| Split | Model | Mean KL | Worst KL (chunk/position) | Top-1 matches | PPL |
|---|---|---:|---:|---:|---:|
| test | control | 0.0077481 | 1.4620404 (36/319) | 24,537 | 7.083731 |
| test | candidate | 0.0076817 | 1.2650036 (90/268) | 24,522 | 7.081468 |
| valid | control | 0.0069527 | 0.7446714 (12/317) | 24,645 | 6.530797 |
| valid | candidate | 0.0068496 | 1.3540535 (12/317) | 24,629 | 6.530331 |

Keep the control as the default. The candidate saves 90,728 bytes and improves mean KL and PPL on both splits. It loses 15 test and 16 validation top-1 matches. Validation worst KL rises from 0.7446714 to 1.3540535 at the same chunk and position. All four criteria must pass independently on both splits.

[Native metrics](quality.json) · [Model generation and SHA-256](README.md).
