# Reinvest lossless packing savings in two complete expert upgrades

The scale refit on the [927-slice parent](../README.md) freed 1,721,464 bytes. Giving that much extra provisional capacity to the training-only paired projection search retained 931 slices. The two additional pairs are layer 11 expert 53 and layer 15 expert 76. Their gate/up and down projections are both restored. The fixed training objective changes from 202,963.93 to 202,573.19. The following group at 933 slices exceeded the provisional encoding limit. Final size is checked after fitting the candidate's own scale frames.

| Package | Physical bytes including tokenizer | Bytes below IQ3_XXS |
| --- | ---: | ---: |
| 927 after scale refit | 11,660,934,270 | 2,375,234 |
| 931 after scale refit | 11,662,708,470 | 601,034 |
| Native GGUF | 11,663,309,504 | 0 |

The two additional pairs cost 1,774,200 bytes against the refitted 927. The independent precision verifier checked 9,852,420,096 weight bytes and 1,231,552,512 scales across the 38 altered projections against the low- and high-precision sources. After fitting, all 1,274,019,840 scale codes matched across 40 planes; the final shard passed readback of 999 tensor payloads. A lower training objective alone does not establish higher held-out quality.

## Reproduction

Use the same paired training ranking and mixer as the 927 package. Copy the rank JSON and add 1,721,464 to its provisional `cap` field, retaining the expert order, group boundaries, parent bytes and all gains. Run the parent's paired assembler and independent verifier. Attach the same 44-magnitude embedding. Finally apply the [scale-bin fit and complete package assembler](https://github.com/tjm8874/LeanSonic/tree/077bf7e/docs/benchmarks/inkling-class-frames/scale-bins) to the resulting materialized directory. Compare both file digests and verify the 11,663,309,504-byte limit on the finished files.

[Selected expert IDs and trials](expert.json) · [Direct source precision checks](experts-verified.json) · [Embedding attachment](embedding.json) · [Each fitted scale plane](receipt.json) · [Final bytes and model hashes](package.json).

## Charge each entropy table to the package

The first refit kept all 144 entropy tables per projection. Each table takes 512 serialized bytes. We tested dropping neighboring context bins where their estimated reduction in coding efficiency costs fewer bytes than keeping the extra tables. Both axes then receive two more boundary-fitting sweeps. We encode complete frames for the pruned and fixed-grid candidates, decode and compare every scale, and retain the smallest physical frame including its header. All 40 expert scale planes improve; selected shapes range from 6 to 12 bins per axis.

This second lossless pass reduces the same 931-slice package from 11,662,708,470 to **11,661,987,870 bytes**, saving 720,600 bytes. The resulting model is 1,321,634 bytes below native IQ3_XXS. There are 1,274,019,840 decoded scale codes and 999 final tensor payloads in the checks. The model loader uses the existing mean-scale format. [Package hashes](grid-package.json) · [All candidate bins and real sizes](grid-planes.json).

From a fresh directory, run `python3 grid.py PARENT_931_MODEL ORIGINAL_REFIT_WORKDIR PROBE`, using the 931 model above, its original per-plane feature files and fit receipt, and the compiled scale_refit probe. Then run the unchanged scale-bin `assemble.py PARENT_931_MODEL . materialized` and compare both final file hashes.

## Package and attention rounding interaction on RTX PRO 5000

We scored a fixed seven-chunk diagnostic set under the packed GPU runtime a1592f3, toggling only the G1 hunk from cce6f21. Five test chunks contain the largest measured KL excesses above one nat in the 571 vs vLLM comparison. The sixth contains vLLM's test maximum; the seventh contains the current 571 validation maximum. All 1,785 positions use the same BF16 references, scorer and inputs. The candidate packages here include the first scale refit. They differ by layer 11 expert 53 and layer 15 expert 76.

| Position | 927 current KL | 931 current KL | 927 earlier G1 KL | 931 earlier G1 KL |
| --- | ---: | ---: | ---: | ---: |
| test 2, 478 | 1.525412 | 3.573213 | 3.640435 | 3.969963 |
| test 36, 375 | 2.259930 | 2.077238 | 2.746056 | 3.079025 |
| test 81, 291 | 1.535650 | 1.548244 | 1.347333 | 1.529870 |
| test 98, 465 | 2.514592 | 2.645702 | 0.241216 | 0.179064 |
| test 5, 342 | 0.997490 | 0.889339 | 1.070802 | 0.963962 |
| test 23, 330 | 1.383126 | 1.380741 | 1.823065 | 1.566801 |
| valid 85, 317 | 2.475259 | 2.417674 | 2.279424 | 2.118983 |

At test 2,478 the extra expert pairs increase KL under current G1, while at test 36,375 they reduce it. At test 98,465 the 927 earlier-G1 result is 0.241216, but the same package with current G1 produces 2.514592. These interactions require both package and cubin digests when reporting a fix. The targeted chunks were selected for known errors; full holdouts govern adoption. [All selected predictions, full-subset scores and exact executable and package hashes](seven-chunks.json).
