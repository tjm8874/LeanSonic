# Higher-budget routing fit

The 13,628,279,942-byte package uses the same training windows and routing factors as the [smaller fit](quality.md). The additional 15,599,488 bytes restore precision in 779 per-expert cutoffs. Both WikiText splits use the same BF16 reference and automatic prefill configuration.

| Split | Mean KL | Worst KL | BF16 top-1 matches | PPL |
|---|---:|---:|---:|---:|
| test | 0.0081364 | 1.3625982 | 24,514/25,500 | 7.085631 |
| valid | 0.0073278 | 1.6855848 | 24,576/25,500 | 6.513571 |

[Raw scores](higher-budget-quality.json) · [Rebuild the exact distribution](README.md#second-build-within-the-public-q3-size-budget).
