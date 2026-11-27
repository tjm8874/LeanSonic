# Greedy vocabulary reduction on Blackwell

On PRO 5000 the single-block vocabulary argmax took 25.258 µs per non-speculative step in the [earlier trace](../inkling-post89-decode.json). The candidate uses 98 independent blocks of 2,048 logits and one final block over the 98 partial results. Both select the lowest vocabulary index on a tie. The final kernel increments the sequence position once. The workspace adds 784 bytes to the existing device logits allocation. The original one-block function remains available for older cubins.

The control host at `5f1c7ff` and candidate at `6544318` both used the same native SM120a cubin (SHA-256 `623c6cf3b75b79e9378ef241069054d9e0e32cfa4612f31f5af631ae4776eb7f`). These are production greedy-engine calls, context 2,048, speculation off, FP16 KV, PDL enabled, one sequence. The canonical 15,548,157,542-byte model and README prefix are identical. Each process ran one warmup and three timed requests at each length. Five cycles alternate control/trial/trial/control and its reverse. The interval enumerates the 3,125 bootstrap samples over the five per-cycle timing differences.

| GPU | Prefix tokens | Control decode 64 (ms) | Candidate (ms) | Candidate minus control (ms, 95% interval) |
|---|---:|---:|---:|---:|
| PRO4500 | 32 | 390.066 | 388.792 | -1.273 [-1.381, -1.174] |
| PRO4500 | 1024 | 401.718 | 400.476 | -1.242 [-1.339, -1.138] |

For every listed GPU, all emitted IDs and the terminal token matched across builds, repeats, and warmups for each prefix. The hash of the final full FP32 logit row also matched for each prefix. A separate GPU test compares CPU, one-block and tiled argmax over 80 synthetic vocabulary-sized rows, including ties spanning tiles, the last partial tile and all-zero rows. It checks the position after each invocation.

[Source drivers and reproduction](reproduce.md) · [Raw request times including prefill](runs.csv) · [Final-logit hashes, emitted token IDs and paired summaries](traces.json).
