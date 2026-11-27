# Accuracy gained and lost by restoring additional expert weights

The two packages are 13,445,948,726 and 13,608,770,734 bytes respectively. Both use the same executable and cubin on RTX PRO 4500 with scratch-free U4, FP16 KV, prefill width 512, speculation off and a common BF16 oracle for each 25,500-position split. The extra 162,822,008 bytes restore 585,066,515 NVFP4 weights.

| Split | Package | Mean KL | Worst KL (chunk/position) | BF16 top-1 matches | PPL |
|---|---|---:|---:|---:|---:|
| test | trim | 0.0081645 | 1.2202775 (99/398) | 24,500 | 7.081285 |
| test | recovered | 0.0077481 | 1.4620404 (36/319) | 24,537 | 7.083731 |
| valid | trim | 0.0075089 | 1.4429343 (12/317) | 24,566 | 6.529914 |
| valid | recovered | 0.0069527 | 0.7446714 (12/317) | 24,645 | 6.530797 |

On test the higher-precision package improves mean KL and top-1 agreement, while its worst KL and PPL increase. The previously largest divergence also moves from 99/398 to 36/319. A reduction in quantized weight error alone does not guarantee a smaller worst-row KL.

## Fixed-row diagnosis

The fixed-row driver replays the same input IDs with both the canonical and serial-prefill BF16 oracles. The table below quotes canonical KL for the fixed positions. Absolute values depend on the replay path.

| Split | Row | Trim | Recovered | Official weights on LeanSonic |
|---|---|---:|---:|---:|
| test | 23/432 | 0.6864532 | 0.5264494 | 0.0131257 |
| test | 29/443 | 0.3033687 | 0.0049556 | 0.1754755 |
| test | 69/353 | 0.1906996 | 1.0265868 | 0.6809774 |
| test | 80/452 | 0.2477859 | 0.1390385 | 0.0312497 |
| test | 99/398 | 0.1074545 | 0.0074948 | 0.0253805 |
| valid | 7/467 | 0.0360758 | 0.0267374 | 0.0024186 |
| valid | 12/317 | 1.9394759 | 0.8844182 | 0.0255825 |
| valid | 46/379 | 0.8818305 | 0.0106762 | 0.3524361 |

[Package build](README.md) · [Replay source and GPU settings](https://github.com/tjm8874/LeanSonic/tree/cc62564/docs/benchmarks/inkling-nvfp4-trim-context) · [Native scores and fixed-row output](quality.json).
