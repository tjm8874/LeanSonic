# Output quality at the IQ3 byte budget

RTX PRO 4500, FP16 KV, batch one, spec 0, prefill width 512. Each WikiText split scores 25,500 positions across 100 chunks of 512. The 13.48 GB control and candidate use the same a200846 CLI and bc526e3 cubin. Public IQ3_XXS uses Unsloth llama.cpp 8832860 and the same corpus and BF16 oracle files. The native corpus summary does not expose the worst-row coordinate. The oracle is BF16 and uses the repository’s compact KLD reference format. [Native comparator](https://github.com/tjm8874/LeanSonic/tree/53a4369/docs/benchmarks/inkling-native-lowbits) · [Decision criteria](https://github.com/tjm8874/LeanSonic/issues/35#issuecomment-7221051793).

| Split | Model | Mean KL | Worst KL (chunk/position) | Top-1 matches / 25,500 | PPL |
|---|---|---:|---:|---:|---:|
| test | 13.48 GB control | 0.0077481 | 1.4620404 (36/319) | 24,537 (96.22%) | 7.083731 |
| test | fixed seven-level | 0.0239480 | 2.6846245 (23/330) | 23,734 (93.07%) | 7.171151 |
| test | candidate | 0.0230838 | 2.4809246 (23/330) | 23,829 (93.45%) | 7.158300 |
| test | public IQ3_XXS | 0.027728 | 4.049091 | 23,655 (92.77%) | 7.188809 |
| valid | 13.48 GB control | 0.0069527 | 0.7446714 (12/317) | 24,645 (96.65%) | 6.530797 |
| valid | fixed seven-level | 0.0224466 | 3.0877419 (11/374) | 23,916 (93.79%) | 6.611086 |
| valid | candidate | 0.0216190 | 2.9282704 (11/374) | 23,939 (93.88%) | 6.609503 |
| valid | public IQ3_XXS | 0.025635 | 2.807837 | 23,785 (93.28%) | 6.613878 |

Keep the higher-precision control as the default. The smaller distribution fails the predeclared nonregression check against it.

Against public IQ3_XXS on test, the candidate meets 4 of four directional criteria (lower mean/worst KL and PPL, higher BF16 top-1 count).

Against public IQ3_XXS on valid, the candidate meets 3 of four directional criteria (lower mean/worst KL and PPL, higher BF16 top-1 count).

Compared with the fixed seven-level package, the candidate saves 3,392,944 bytes. The four-metric check against that model by split is test pass, valid pass.

Rebuild the package with [the encoder recipe](README.md), then run the release CLI for each oracle file. Use the source revisions above and set the cubin path to the bc526e3 build.

```sh
export LEANSONIC_KV=fp16 LEANSONIC_PDL=1 LEANSONIC_PDL_ALL=1 LEANSONIC_SEQS=1
export LEANSONIC_PREFILL_CHUNK=512 LEANSONIC_MXFP8_DEVICE_U4=1 LEANSONIC_MXFP8_DIRECT_ALL=1 LEANSONIC_CTX=8192
leansonic eval-kld --weights "$NEW_MODEL_DIR" --ref "$BF16_KLD" --chunks 100
```

[Raw scores, model and reference hashes](quality.json).
