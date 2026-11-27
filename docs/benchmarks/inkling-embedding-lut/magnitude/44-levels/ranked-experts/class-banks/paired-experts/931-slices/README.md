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
