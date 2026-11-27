# Quality of 68 ranked expert restorations

All three LeanSonic packages used the same RTX PRO 4500 scorer, CUDA module, 100 chunks and 25,500 positions per split. The ranked package is 11,663,036,518 bytes. Compare with the native [IQ3_XXS scores and BF16 input fixture](../restored-four-banks/corpus.md).

| Split | Package | Mean KL | Worst KL (chunk/position) | Top-1 / 25,500 | PPL |
|---|---|---:|---:|---:|---:|
| test | four | 0.0230838 | 2.4809246 (23/330) | 23,829 | 7.158300 |
| test | whole_planes | 0.0225754 | 2.5457894 (23/330) | 23,842 | 7.157658 |
| test | ranked68 | 0.0216220 | 2.7777589 (23/330) | 23,871 | 7.139651 |
| valid | four | 0.0216190 | 2.9282704 (11/374) | 23,939 | 6.609503 |
| valid | whole_planes | 0.0209783 | 3.0298716 (11/374) | 23,957 | 6.604097 |
| valid | ranked68 | 0.0199156 | 2.3529866 (41/499) | 24,013 | 6.594276 |

The ranked model improves mean KL, top-1 agreement and PPL versus the two-plane candidate on both splits. Its worst KL rises on test (2.5457894 to 2.7777589) and falls on validation (3.0298716 to 2.3529866). Against the published native IQ3_XXS run on RTX PRO 5000, this PRO 4500 result is better on all four metrics on both splits, with a 272,986-byte smaller package. The native worst KL was 4.049091 on test and 2.807837 on validation. The native and LeanSonic runs used different GPUs. Validate the release-source and same-card scores along with the paired latency before adoption.

Use the package recipe in this directory and the scorer recipe in the linked corpus report. The scorer and CUDA module digests and raw score summaries are included in [corpus.json](corpus.json).
