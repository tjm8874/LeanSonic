# Accuracy gained and lost by restoring additional expert weights

The two packages are 13,445,948,726 and 13,608,770,734 bytes respectively. Both use the same executable and cubin on RTX PRO 4500 with scratch-free U4, FP16 KV, prefill width 512, speculation off and a common BF16 oracle for each 25,500-position split. The extra 162,822,008 bytes restore 585,066,515 NVFP4 weights.

| Split | Package | Mean KL | Worst KL (chunk/position) | BF16 top-1 matches | PPL |
|---|---|---:|---:|---:|---:|
| test | trim | 0.0081645 | 1.2202775 (99/398) | 24,500 | 7.081285 |
| test | recovered | 0.0077481 | 1.4620404 (36/319) | 24,537 | 7.083731 |
| valid | trim | 0.0075089 | 1.4429343 (12/317) | 24,566 | 6.529914 |
| valid | recovered | 0.0069527 | 0.7446714 (12/317) | 24,645 | 6.530797 |

On test the higher-precision package improves mean KL and top-1 agreement, while its worst KL and PPL increase. The previously largest divergence also moves from 99/398 to 36/319. A reduction in quantized weight error alone does not guarantee a smaller worst-row KL.

[Package build](README.md) · [Replay source and GPU settings](https://github.com/tjm8874/LeanSonic/tree/cc62564/docs/benchmarks/inkling-nvfp4-trim-context) · [Native scores and fixed-row output](quality.json).
