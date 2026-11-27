# Quality of the fifteen-symbol routing fit

WikiText-2 rows below use the same release scorer on RTX PRO 4500, FP16 KV and automatic prefill; each split contains 100 chunks and 25,500 scored positions. The bilingual comparison uses the pinned PRO 5000 teacher-forced fixture. The two prior models were scored on the same fixtures and configurations. The new package occupies 13,624,461,934 bytes.

| Split | Package | Mean KL | Worst KL and position | BF16 top-1 matches | PPL |
|---|---|---:|---:|---:|---:|
| test | previous compact | 0.0137735 | 0.7519345 (69/353) | 24,289/25,500 | 7.131377 |
| test | routing-weighted | 0.0081565 | 1.4537516 (36/319) | 24,514/25,500 | 7.084181 |
| test | fifteen-symbol capped | 0.0081320 | 0.6864531 (23/432) | 24,476/25,500 | 7.085819 |

[Build, distribution bytes and weight-file fingerprints](README.md) · [Native scorer output](quality.json).
