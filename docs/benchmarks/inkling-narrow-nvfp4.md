# Narrow NVFP4 Tensor Core gate/up on short split-prefill suffixes

The trial at `72e2b18` uses a 128-weight-row by 32-token tile for the gate/up projection on 2–31-row suffixes. The down projection keeps its original tile. Both arms use one identical native SM120a cubin (SHA-256 `0561c38ebc7c247346836582a088541de34fa4f9b025525c77d512c34fcef8e6`). The control driver is `689363a`, which selects the wide gate/up. On PRO 4500 the short tile preserved all compared logits, but increased prefill latency at 66, 80 and 95 tokens. Keep this implementation experimental.

| GPU | Input tokens | Control prefill ms | Trial prefill ms | Trial minus control (95% interval) ms |
|---|---:|---:|---:|---:|
| PRO4500 | 66 | 62.976 | 63.149 | +0.173 [+0.148, +0.197] |
| PRO4500 | 80 | 76.807 | 77.346 | +0.539 [+0.515, +0.564] |
| PRO4500 | 95 | 87.164 | 88.149 | +0.985 [+0.946, +1.025] |
| PRO4500 | 1024 | 794.172 | 793.920 | -0.252 [-0.516, -0.032] |
| 5070Ti | 66 | 174.340 | 175.072 | +0.732 [+0.300, +1.281] |
| 5070Ti | 80 | 218.037 | 219.256 | +1.219 [+0.766, +1.673] |
| 5070Ti | 95 | 248.326 | 249.999 | +1.672 [+1.062, +2.282] |
| 5070Ti | 1024 | 2167.596 | 2170.475 | +2.879 [-0.848, +7.069] |

Five mirrored four-process cycles; each process performed one warm-up then three measured passes for every length. Fixed prefill width 64, context 2048, FP16 KV, PDL on, one sequence, identical #117 weights and README prefix. Decode replay used the same eight input tokens. The paired interval enumerates the 3,125 ordered bootstrap samples of the five per-cycle deltas. The separate 64-slot parity case compares the 64-row workspace against independent runs with a 256-row workspace and covers 25,607,424 FP32 values. See [the earlier kernel-level profile](inkling-tail-profile/README.md).

[All lengths and parity results](inkling-narrow-nvfp4.json) · [PRO4500 raw requests](inkling-narrow-nvfp4-PRO4500.csv) · [5070Ti raw requests](inkling-narrow-nvfp4-5070Ti.csv)
