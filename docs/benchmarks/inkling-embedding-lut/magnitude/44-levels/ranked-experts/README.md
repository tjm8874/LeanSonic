# Reallocate the smaller 44-magnitude embedding to 354 more experts

At the same 11,663,309,504-byte IQ3_XXS limit, using the smaller input embedding allowed 571 higher-precision experts across 36 projections. The 217-expert and 571-expert packages use the same 259,021,360-byte input embedding. Package bytes including tokenizer are 11,419,458,046 and 11,662,590,054. The larger candidate is 719,450 bytes below native IQ3_XXS. The independent decoder checked all 9,342,812,160 reconstructed expert-weight bytes and 1,167,851,520 scale codes across the changed projections, then package assembly verified the 998 retained tensor payloads. The two full-network BF16 splits are reported below.

Use the training ranking, low and high source packages and mixing tool from the [68-expert experiment](https://github.com/tjm8874/LeanSonic/tree/aaceb42/docs/benchmarks/inkling-codebooks/nvfp4-budget/recovered/variable-palette/global-experts). For the intermediate model with its original input embedding, set its rank.json cap to the native limit plus the observed difference between 11,779,027,886 bytes (the 217-expert intermediate with original input embedding) and 11,419,458,046 (the same experts with the 44-magnitude input). Run that experiment's assemble.py and verify.py at the revised cap. Then run the local assemble.py with the intermediate model, the identical [44-magnitude embedding frame](../README.md) and an empty destination. [Selected expert IDs and measured trial bytes](expert.json) · [Independent decoded tensors](experts-verified.json) · [Final SHA-256 and package bytes](receipt.json).

## Whole-model BF16 quality on RTX PRO 4500

Both packages share the identical 44-magnitude embedding. Raising the ranked allocation from 217 to 571 experts costs 243,132,008 distribution bytes. Each split contains 25,500 positions. The scorer, CUDA module, GPU, FP16 KV, histories and context are matched.

| Split | Experts | Mean KL | Worst KL | Top-1 / 25,500 | PPL |
|---|---:|---:|---:|---:|---:|
| Test | 217 | 0.0196979 | 2.4135362 | 23,940 | 7.134607 |
| Test | 571 | 0.0169571 | 2.8585658 | 24,084 | 7.127375 |
| Validation | 217 | 0.0185062 | 3.0118662 | 24,094 | 6.581250 |
| Validation | 571 | 0.0157347 | 2.5524491 | 24,200 | 6.566551 |

The 571-expert model improves both mean KL and PPL on both splits, gains 144/106 top-1 matches on test/validation, and improves the validation worst case. Test worst KL increases, peaking at chunk 36 position 375. Against the 452-magnitude, 217-expert package at a similar byte budget, the 571-expert model has lower mean KL and PPL and more top-1 matches on both splits, while validation worst KL rises from 1.9770640 to 2.5524491. Continue to matched speed, VRAM and bilingual checks. [Raw results, model bytes and executable hashes](corpus.json) · [Fixed 217-expert controls](../README.md).
