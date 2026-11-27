# BF16 quality at the IQ1_S size budget

LeanSonic uses its a200846 release host with bc526e3 cubin, RTX PRO 4500, spec 0, FP16 KV, one sequence, context 8,192 and prefill width 512. Each split uses the same 100 fixed WikiText chunks of 512 with 25,500 scored positions. Lower KL and PPL and higher BF16 top-1 agreement are better. Byte counts include the tokenizer.

| Split | Model | Bytes | Mean KL | Worst KL (chunk/position) | Top-1 / 25,500 | PPL |
|---|---|---:|---:|---:|---:|---:|
| test | 13.48 GB control | 13,479,710,294 | 0.0077481 | 1.4620404 (36/319) | 24,537 (96.22%) | 7.083731 |
| test | ternary | 7,718,950,382 | 0.2341477 | 7.5035677 (90/424) | 20,385 (79.94%) | 8.566626 |
| test | scale refit | 7,734,665,510 | 0.5942670 | 9.7883801 (1/260) | 17,668 (69.29%) | 12.167307 |
| test | rate-constrained | 7,679,612,078 | 0.7000745 | 10.0854794 (1/260) | 17,118 (67.13%) | 13.475101 |
| test | energy refit | 7,748,141,470 | 0.2770098 | 9.2712852 (90/424) | 19,995 (78.41%) | 8.947962 |
| valid | 13.48 GB control | 13,479,710,294 | 0.0069527 | 0.7446714 (12/317) | 24,645 (96.65%) | 6.530797 |
| valid | ternary | 7,718,950,382 | 0.2161375 | 6.5853805 (10/471) | 20,643 (80.95%) | 7.904899 |
| valid | scale refit | 7,734,665,510 | 0.5779062 | 12.2230370 (76/476) | 17,813 (69.85%) | 11.258599 |

ternary / test: fails the four-metric nonregression criterion against the 13.48 GB control.

ternary / valid: fails the four-metric nonregression criterion against the 13.48 GB control.

scale refit / test: fails the four-metric nonregression criterion against the 13.48 GB control.

scale refit / valid: fails the four-metric nonregression criterion against the 13.48 GB control.

rate-constrained / test: fails the four-metric nonregression criterion against the 13.48 GB control.

energy refit / test: fails the four-metric nonregression criterion against the 13.48 GB control.

Native IQ1_S corpus scores are pending. The size comparison alone does not establish equivalent quality.

Use the same corpus evaluation [command and settings](../iq3-palette/quality.md) and the [predeclared comparison criteria](https://github.com/tjm8874/LeanSonic/issues/35#issuecomment-7221275174). [Raw scores and oracle digests](quality.json).
