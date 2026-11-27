# Bound the expert grid for Tensor Core prefill tails

For a short prefill tail, each input token routes to at most eight experts. The candidate at `1261bfd` bounds the launch using this count and the total number of experts. Control `c7b03ae` and candidate use the same wide Tensor Core kernel, the same arithmetic and the same cubin (SHA-256 `93463fd100ab8fcd2b397137c7861f4b65e8cef0c6dcc20fbfe4150e9b6548d2`). The 66-token request, which contains a two-token tail, improved by 0.070 ms on PRO 4500. This comparison does not establish a win over the main DP4A tail path. Keep the change experimental.

| GPU | Input tokens | Control prefill ms | Trial prefill ms | Trial minus control (95% interval) ms |
|---|---:|---:|---:|---:|
| PRO4500 | 66 | 62.955 | 62.885 | -0.070 [-0.094, -0.052] |
| PRO4500 | 80 | 76.776 | 76.762 | -0.015 [-0.033, -0.000] |
| PRO4500 | 95 | 87.140 | 87.121 | -0.019 [-0.039, -0.000] |
| PRO4500 | 1024 | 793.927 | 793.818 | -0.109 [-0.170, -0.048] |

Five mirrored four-process cycles; each process performed one warm-up then three measured passes for every length. Fixed prefill width 64, context 2048, FP16 KV, PDL on, one sequence, identical #117 weights and README prefix. Decode replay used the same eight input tokens. The paired interval enumerates the 3,125 ordered bootstrap samples of the five per-cycle deltas. The separate 64-slot parity case compares the 64-row workspace against independent runs with a 256-row workspace and covers 25,607,424 FP32 values. See [the earlier kernel-level profile](inkling-tail-profile/README.md).

[All lengths and parity results](inkling-tight-expert-grid.json) · [PRO4500 raw requests](inkling-tight-expert-grid-PRO4500.csv)
