# Spending the recovered bytes on 354 additional experts

Use the same training-only ranking, byte limit and 44-level input embedding as the 571-expert parent. Separating the four entropy tables during each candidate encode lets 925 ranked gate/up or down expert slices fit in 11,663,106,190 bytes including the tokenizer. Original IQ3_XXS uses 11,663,309,504 bytes, leaving 203,314 bytes. At the intermediate-package search stage the 926th slice exceeded the allocation budget. The offline training objective fell from 243,418.67 to 202,755.71. Held-out BF16 quality and throughput require their own measurements.

The independent verifier compared every retained and restored expert against its source in 38 affected projections, covering 9,852,420,096 weight bytes and 1,231,552,512 scale codes. To reproduce, use the [parent ranking, source archives and instructions](../../README.md), substitute [mix.rs](mix.rs) for the mixing example in the [original toolchain](https://github.com/tjm8874/LeanSonic/tree/aaceb42/docs/benchmarks/inkling-codebooks/nvfp4-budget/recovered/variable-palette/global-experts), and run the same assembly and verification commands with the unchanged rank.json. [Prefix trials and selected expert IDs](expert.json) · [Independent verification](experts-verified.json) · [Final package hashes](receipt.json).

## BF16 reference on RTX 5070 Ti

All three packages used the same 100-chunk inputs and reference; 25,500 positions were scored per split with FP16 KV and 512-token contexts. [Inputs, scorer, module hashes and full scores](5070-scores.json).

| Split | Model | Mean KL | Worst KL | Top-1 matches | PPL |
| --- | --- | ---: | ---: | ---: | ---: |
| test | 925 experts | 0.0146735 | 2.9098074 | 24,195 | 7.107475 |
| test | 571 experts | 0.0169571 | 2.8585658 | 24,084 | 7.127375 |
| test | llama.cpp IQ3_XXS | 0.0280570 | 5.5014780 | 23,666 | 7.193132 |
| valid | 925 experts | 0.0139915 | 1.9965633 | 24,269 | 6.557166 |
| valid | 571 experts | 0.0157347 | 2.5524491 | 24,200 | 6.566551 |
| valid | llama.cpp IQ3_XXS | 0.0256590 | 3.7240360 | 23,786 | 6.616409 |

The 925-expert package uses 175,638,400 additional bytes over the class-specific 571-expert package. It improves mean KL, top-1 agreement and PPL on both splits, and improves validation worst KL. Test worst KL rises to 2.9098074 at chunk 98, position 465. All four metrics improve against native IQ3_XXS on both splits.
