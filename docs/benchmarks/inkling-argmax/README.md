# Greedy vocabulary reduction on Blackwell

On PRO 5000 the single-block vocabulary argmax took 25.258 µs per non-speculative step in the [earlier trace](../inkling-post89-decode.json). The candidate uses 98 independent blocks of 2,048 logits and one final block over the 98 partial results. Both select the lowest vocabulary index on a tie. The final kernel increments the sequence position once. The workspace adds 784 bytes to the existing device logits allocation. The original one-block function remains available for older cubins.

The control host at `5f1c7ff` and candidate at `6544318` both used the same native SM120a cubin (SHA-256 `623c6cf3b75b79e9378ef241069054d9e0e32cfa4612f31f5af631ae4776eb7f`). These are production greedy-engine calls, context 2,048, speculation off, FP16 KV, PDL enabled, one sequence. The canonical 15,548,157,542-byte model and README prefix are identical. Each process ran one warmup and three timed requests at each length. Five cycles alternate control/trial/trial/control and its reverse. The interval enumerates the 3,125 bootstrap samples over the five per-cycle timing differences.

| GPU | Prefix tokens | Control decode 64 (ms) | Candidate (ms) | Candidate minus control (ms, 95% interval) |
|---|---:|---:|---:|---:|
| 5060Ti | 32 | 1266.579 | 1265.349 | -1.230 [-1.319, -1.117] |
| 5060Ti | 1024 | 1294.905 | 1293.629 | -1.276 [-1.361, -1.186] |
| 5070Ti | 32 | 541.942 | 540.843 | -1.099 [-1.306, -0.921] |
| 5070Ti | 1024 | 553.563 | 552.475 | -1.088 [-1.217, -0.959] |
| PRO4500 | 32 | 390.066 | 388.792 | -1.273 [-1.381, -1.174] |
| PRO4500 | 1024 | 401.718 | 400.476 | -1.242 [-1.339, -1.138] |
| PRO5000 | 32 | 304.670 | 303.084 | -1.586 [-1.652, -1.519] |
| PRO5000 | 1024 | 311.660 | 310.025 | -1.635 [-1.696, -1.580] |

For every listed GPU, all emitted IDs and the terminal token matched across builds, repeats, and warmups for each prefix. The hash of the final full FP32 logit row also matched for each prefix. A separate GPU test compares CPU, one-block and tiled argmax over 80 synthetic vocabulary-sized rows, including ties spanning tiles, the last partial tile and all-zero rows. It checks the position after each invocation.

[Source drivers and reproduction](reproduce.md) · [Raw request times including prefill](runs.csv) · [Final-logit hashes, emitted token IDs and paired summaries](traces.json).

## Independent comparison after the decode update

The control at `f556d07` includes the newer decode kernels from #140. The matched candidate at `ac3a6e0` uses the same newly compiled module (SHA-256 `f4842b5358173ed3ae85cd4b4708d5afc9d61f95e1c26b0464ce41378da32f04`). Inputs, five mirrored cycles, per-card token and final-logit checks are as above. These are independent runs with new control and candidate binaries.

| GPU | Prefix tokens | Control decode 64 (ms) | Candidate (ms) | Paired difference (ms, 95% interval) |
|---|---:|---:|---:|---:|
| PRO4500 | 32 | 385.307 | 384.193 | -1.114 [-1.196, -1.020] |
| PRO4500 | 1024 | 428.250 | 427.325 | -0.925 [-1.040, -0.800] |
| PRO5000 | 32 | 299.180 | 298.004 | -1.175 [-1.301, -1.038] |
| PRO5000 | 1024 | 345.071 | 344.221 | -0.850 [-0.999, -0.688] |

[Request-level measurements](main-runs.csv) · [Tokens, hashes, prefill and decode summaries](main-traces.json). Reuse the [two-checkout procedure](reproduce.md) with the two source revisions above.
