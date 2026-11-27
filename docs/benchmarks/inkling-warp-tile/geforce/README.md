# One-warp expert tile scan on the 16 GB integration

Both arms use the same 15-symbol package (13,624,461,934 bytes), scratch-free lookup, FP16 KV and the fused-verifier main integration. The control cubin comes from bc526e3 and the candidate adds the one-warp scan at 4fdf53e. The same executable loads each cubin. Each board contributes five mirrored cycles, ten fresh processes per arm and three warm passes per process for each length. Times below are milliseconds; intervals resample complete paired cycles.

| GPU | Prompt | Prefill control | Prefill candidate | Difference (95% interval) | Decode 64 difference (95% interval) |
|---|---:|---:|---:|---:|---:|
| 5070Ti | 32 | 35.216 | 35.184 | -0.032 [-0.035, -0.028] | +0.045 [-0.043, +0.147] |
| 5070Ti | 1024 | 121.904 | 121.394 | -0.510 [-0.514, -0.507] | -0.072 [-0.169, +0.025] |
| 5070Ti | 2048 | 255.840 | 256.121 | +0.282 [-1.066, +2.454] | -0.076 [-0.175, +0.014] |

At every measured length and on each board, the two arms agreed for all 64 greedy IDs, the next prediction, and the hash of the last 200,058 FP32 logits. The 5070 Ti prefill gain at 1,024 tokens did not generalize conclusively to 2,048. Use the [kernel test and benchmark instructions](../README.md) with the [capped model](https://github.com/tjm8874/LeanSonic/tree/c9a6a63/docs/benchmarks/inkling-codebooks/nvfp4-budget/capped) and these environment overrides: `LEANSONIC_MXFP8_DEVICE_U4=1 LEANSONIC_MXFP8_DIRECT_ALL=1 LEANSONIC_PDL_ALL=1`. [Per-request times, output digests and binary digests](receipt.json).
