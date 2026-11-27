# One early neural probe on the two GeForce boards

Each input contributes ten measured requests per arm on each board. Each run checked all 128 generated IDs, their successor and the terminal FP32 logit fingerprint against an independent scalar run on the same GPU. Across the 20 fresh launches per board this covered 480 requests per board including warm-up. The speculative package including the tokenizer and compact draft model occupies 13,781,658,950 bytes. Probe and resumption counts below are the observed range per measured 128-token request. Intervals resample the five paired cycles.

## RTX 5060 Ti

| Input | Stock tok/s | Early probe tok/s | Change | Ratio 95% interval | Probes | Resumed early |
|---|---:|---:|---:|---:|---:|---:|
| q00_code_merge | 165.99 | 165.97 | -0.01% | 1.000–1.000 | 0 | 0 |
| q11_json_schema | 169.52 | 169.50 | -0.01% | 1.000–1.000 | 0 | 0 |
| ja_reasoning | 94.43 | 96.19 | +1.86% | 1.018–1.019 | 2 | 1 |
| readme1024 | 103.04 | 105.39 | +2.28% | 1.023–1.023 | 1 | 0 |
| en_boolean | 129.46 | 129.45 | -0.01% | 1.000–1.000 | 0 | 0 |
| en_sort | 169.73 | 169.71 | -0.01% | 1.000–1.000 | 0 | 0 |
| ja_stock | 115.51 | 114.42 | -0.94% | 0.990–0.991 | 1 | 0 |
| ja_name | 110.55 | 107.69 | -2.59% | 0.974–0.974 | 2 | 2 |
| en_count | 132.65 | 134.75 | +1.59% | 1.016–1.016 | 0 | 0 |
| en_extract | 166.92 | 166.89 | -0.01% | 1.000–1.000 | 0 | 0 |
| ja_json | 130.98 | 130.96 | -0.01% | 1.000–1.000 | 0 | 0 |
| ja_filter | 139.07 | 139.05 | -0.02% | 1.000–1.000 | 0 | 0 |

The 1% per-input regression bound failed on ja_name. Keep the option experimental. [All requests, counters, individual timings and scalar output IDs](5060Ti.json).

## RTX 5070 Ti

| Input | Stock tok/s | Early probe tok/s | Change | Ratio 95% interval | Probes | Resumed early |
|---|---:|---:|---:|---:|---:|---:|
| q00_code_merge | 289.54 | 290.00 | +0.16% | 1.000–1.003 | 0 | 0 |
| q11_json_schema | 296.03 | 296.46 | +0.15% | 1.000–1.003 | 0 | 0 |
| ja_reasoning | 162.60 | 166.47 | +2.38% | 1.023–1.024 | 2–4 | 1–3 |
| readme1024 | 168.16 | 170.52 | +1.41% | 1.014–1.014 | 1 | 0 |
| en_boolean | 225.12 | 225.32 | +0.09% | 1.000–1.001 | 0 | 0 |
| en_sort | 295.20 | 295.66 | +0.16% | 1.000–1.003 | 0 | 0 |
| ja_stock | 199.42 | 198.64 | -0.39% | 0.995–0.997 | 1 | 0 |
| ja_name | 188.87 | 184.74 | -2.18% | 0.978–0.978 | 2 | 2 |
| en_count | 231.20 | 235.28 | +1.76% | 1.017–1.019 | 0 | 0 |
| en_extract | 288.91 | 289.22 | +0.11% | 1.000–1.002 | 0 | 0 |
| ja_json | 226.51 | 226.67 | +0.07% | 1.000–1.002 | 0 | 0 |
| ja_filter | 241.66 | 241.98 | +0.13% | 1.000–1.002 | 0 | 0 |

The 1% per-input regression bound failed on ja_name. Keep the option experimental. [All requests, counters, individual timings and scalar output IDs](5070Ti.json).

[Fixture, binary and cubin SHA-256 digests](digests) · [GPU UUIDs, driver and power caps](gpu.csv) · [Controls and reproduction](../).
