# Quality of the fifteen-symbol routing fit

WikiText-2 rows below use the same release scorer on RTX PRO 4500, FP16 KV and automatic prefill; each split contains 100 chunks and 25,500 scored positions. The bilingual comparison uses the pinned PRO 5000 teacher-forced fixture. The two prior models were scored on the same fixtures and configurations. The new package occupies 13,624,461,934 bytes.

| Split | Package | Mean KL | Worst KL and position | BF16 top-1 matches | PPL |
|---|---|---:|---:|---:|---:|
| test | previous compact | 0.0137735 | 0.7519345 (69/353) | 24,289/25,500 | 7.131377 |
| test | routing-weighted | 0.0081565 | 1.4537516 (36/319) | 24,514/25,500 | 7.084181 |
| test | fifteen-symbol capped | 0.0081320 | 0.6864531 (23/432) | 24,476/25,500 | 7.085819 |
| valid | previous compact | 0.0131739 | 0.9835867 (7/467) | 24,341/25,500 | 6.564200 |
| valid | routing-weighted | 0.0073719 | 1.4013990 (46/379) | 24,559/25,500 | 6.521989 |
| valid | fifteen-symbol capped | 0.0076511 | 1.9394759 (12/317) | 24,588/25,500 | 6.530274 |
| chat | previous compact | 0.0126191 | 0.4765965 (q06_translate_fr/13) | 585/613 | 3.732578 |
| chat | routing-weighted | 0.0108587 | 0.4207198 (q10_ml_gradient/21) | 589/613 | 3.810583 |
| chat | fifteen-symbol capped | 0.0083901 | 0.1302973 (q11_json_schema/25) | 595/613 | 3.790257 |

[Build, distribution bytes and weight-file fingerprints](README.md) · [Native scorer output](quality.json).
