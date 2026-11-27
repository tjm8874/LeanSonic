# BF16 reference scores for the routing-weighted package

The fitter saw only WikiText-2 train windows. Both WikiText splits use the 100-chunk BF16 oracle, 512-token contexts, FP16 KV and the same PRO 4500 binary and cubin. Bilingual scoring uses a separate PRO 5000 full-logit driver. Reported package sizes include the tokenizer.

| Fixture | Package | Bytes | Mean KL | Worst KL (position) | BF16 top-1 matches | PPL |
|---|---|---:|---:|---:|---:|---:|
| test | compact | 13,626,745,974 | 0.0137735 | 0.7519345 (69/353) | 24,289/25,500 | 7.131377 |
| test | recalibrated | 13,623,652,382 | 0.0100863 | 1.8988259 (99/398) | 24,362/25,500 | 7.091487 |
| test | routing-weighted | 13,612,680,454 | 0.0081565 | 1.4537516 (36/319) | 24,514/25,500 | 7.084181 |
| valid | compact | 13,626,745,974 | 0.0131739 | 0.9835867 (7/467) | 24,341/25,500 | 6.564200 |
| valid | recalibrated | 13,623,652,382 | 0.0088730 | 2.3469483 (46/379) | 24,476/25,500 | 6.522603 |
| valid | routing-weighted | 13,612,680,454 | 0.0073719 | 1.4013990 (46/379) | 24,559/25,500 | 6.521989 |
| chat | compact | 13,626,745,974 | 0.0126191 | 0.4765965 (q06_translate_fr/13) | 585/613 | 3.732578 |
| chat | routing-weighted | 13,612,680,454 | 0.0108587 | 0.4207198 (q10_ml_gradient/21) | 589/613 | 3.810583 |

On test, routing improves all four metrics over the previous recalibrated fit and saves 10,971,928 bytes. Compared with compact, the mean improves but worst KL rises from 0.7519345 to 1.4537516. The worst position moves to chunk 36, position 319. Keep the candidate experimental.

The [calibration and package rebuild](README.md) reproduce the weight file. Use the [parent evaluation commands](../../README.md) on this file, including the same test and validation oracle fingerprints. [Scorer output](quality.json).
