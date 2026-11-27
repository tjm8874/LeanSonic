# Single-stream price policy results

Each board has its own scalar anchor. The control (stock K=7) and candidate (priced K=7) share the same release executable. After one warm pass per launch, each input contributed ten measured requests per arm over five alternating ABBA/BAAB cycles. Positive throughput changes favor the policy. The intervals resample whole cycles. The four extra inputs are en_count, en_extract, ja_json, ja_filter.

## RTX 5070 Ti

| Input | Stock / priced prefill ms | Stock / priced decode tok/s | Decode change | Speed ratio 95% interval | Priced skips per 128 tokens, range |
|---|---:|---:|---:|---:|---:|
| q00_code_merge | 34.18 / 34.19 | 290.03 / 289.83 | -0.07% | 0.998–1.000 | 0–0 |
| q11_json_schema | 38.09 / 38.10 | 296.53 / 296.37 | -0.05% | 0.999–1.000 | 0–0 |
| ja_reasoning | 43.95 / 43.96 | 162.64 / 171.32 | +5.34% | 1.053–1.054 | 59–59 |
| readme1024 | 127.55 / 127.53 | 168.26 / 172.23 | +2.36% | 1.023–1.024 | 24–24 |
| en_boolean | 40.74 / 40.75 | 225.31 / 225.27 | -0.02% | 0.999–1.000 | 0–0 |
| en_sort | 42.65 / 42.65 | 295.72 / 295.55 | -0.06% | 0.998–1.000 | 0–0 |
| ja_stock | 45.14 / 45.15 | 199.59 / 199.02 | -0.28% | 0.997–0.998 | 24–24 |
| ja_name | 44.60 / 44.59 | 188.94 / 183.71 | -2.77% | 0.972–0.973 | 46–46 |
| en_count | 40.13 / 40.13 | 231.57 / 235.21 | +1.57% | 1.015–1.016 | 2–2 |
| en_extract | 44.30 / 44.29 | 289.23 / 289.12 | -0.04% | 0.999–1.000 | 0–0 |
| ja_json | 46.64 / 46.65 | 226.71 / 226.57 | -0.06% | 0.999–1.000 | 0–0 |
| ja_filter | 46.40 / 46.41 | 241.99 / 241.91 | -0.03% | 0.999–1.000 | 0–0 |

Stock and priced peak per-process VRAM were 16,301,162,496 and 16,301,162,496 bytes. Both used the 13,781,658,950-byte four-bank target, tokenizer and compact drafter. All 480 speculative requests matched the corresponding scalar result on the 128 emitted IDs, successor and final FP32 fingerprint. Each readback also checked the actual FP32 argmax against the returned ID.

The predeclared 1% per-input bound failed for ja_name. Keep the policy experimental.
[All runs, input fingerprints, GPU memory samples and scalar anchors](5070Ti.json).
