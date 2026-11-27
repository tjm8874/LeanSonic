# Replace a row-prefix scan with warp reduction

Candidate 865a279 computes the total row count for preceding expert groups using a warp-wide integer sum after scanning tile counts. Control 4fdf53e uses two shuffle scans. The SASS contains REDUX, and independent reference checks in Compute Sanitizer passed both tile widths (128 and 160 tokens), 104 count vectors each, three CTA widths. The full-engine experiment uses the same cubin build flags and driver as the [one-warp control](https://github.com/tjm8874/LeanSonic/tree/5c4ed7e/docs/benchmarks/inkling-warp-tile/geforce).

| GPU | Prompt | Prefill control ms | Candidate ms | Paired difference [95% interval] | Decode 64 difference ms |
|---|---:|---:|---:|---:|---:|
| 5060Ti | 32 | 69.915 | 70.015 | +0.100 [+0.078, +0.126] | +0.014 |
| 5060Ti | 1024 | 255.771 | 256.055 | +0.284 [+0.264, +0.312] | +0.029 |
| 5060Ti | 2048 | 424.835 | 425.462 | +0.627 [+0.495, +0.853] | -0.020 |

Keep the original scan. On the 5060 Ti the reduced-shuffle implementation increased prefill time at every width. All output tokens and final logit hashes agreed across arms. Each interval resamples five matched ABBA/BAAB cycles, ten fresh processes per arm. [Per-request data and sanitizer output](receipt.json).
