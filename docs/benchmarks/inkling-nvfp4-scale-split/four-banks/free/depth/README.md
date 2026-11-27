# Draft depth on RTX 5060 Ti

Every mode used the same four-bank target, tokenizer, FP16 KV, context 2,048, one sequence, and 128 freely generated greedy tokens. The row-exponent DSpark sidecar was loaded for depth 2, 4 and 7. The minimum match for suffix copy was 32. Each depth ran once in each of four launch positions across four counterbalanced cycles. Each launch ran two passes on four inputs. Rates below divide 128 by the mean second-pass decode time. Prefill times include the first prediction.

| Input | Decode tok/s, K=0 / 2 / 4 / 7 | Prefill ms, K=0 / 2 / 4 / 7 |
|---|---|---|
| q00_code_merge | 98.89 / 132.08 / 144.21 / 166.02 | 64.02 / 64.36 / 64.33 / 64.35 |
| q11_json_schema | 98.85 / 130.96 / 156.34 / 169.55 | 76.58 / 77.02 / 77.01 / 77.01 |
| ja_reasoning | 98.49 / 97.74 / 94.88 / 94.44 | 89.02 / 89.44 / 89.47 / 89.48 |
| readme1024 | 82.27 / 97.57 / 101.79 / 103.06 | 259.40 / 266.76 / 266.71 / 266.78 |

| Depth | Distribution bytes incl. tokenizer | Largest observed process VRAM (bytes) |
|---:|---:|---:|
| 0 | 13,455,139,382 | 15,810,428,928 |
| 2 | 13,781,658,950 | 16,238,247,936 |
| 4 | 13,781,658,950 | 16,259,219,456 |
| 7 | 13,781,658,950 | 16,275,996,672 |

All 128 requests agreed with the existing scalar baseline for all 128 emitted IDs, the successor ID and the terminal FP32 row fingerprint. Code and JSON benefited most from K=7. On the Japanese reasoning fixture K=2 narrowed the loss relative to scalar, but still trailed it. A single Japanese fixture does not establish a language-wide policy.

Build [the measured revision](https://github.com/tjm8874/LeanSonic/commit/5ab84b70b821f59baeaa1d2b4f177e605d5f748b) with the [driver](lean.rs) as crates/leansonic-core/examples/depth_replay.rs. Use the [parent recipe and ID files](../), set LEANSONIC_SPEC_COPY=32 and pass depth 0, 2, 4 or 7 as the second argument. Use the plain target directory for 0 and the target with the compact drafter for positive depths. [Per-request phase times, tokens, counters, loader settings and file hashes](verified.json) · [NVML sampler](leg.py).
