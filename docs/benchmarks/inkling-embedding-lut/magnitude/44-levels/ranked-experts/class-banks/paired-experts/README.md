# Rank gate/up and down precision together at the native IQ3_XXS byte ceiling

Each ranking entry combines the training gains and estimated costs of one expert's gate/up and down projections. A pair enters the model together. Layers 16 and 17 retain single gate/up entries because their down planes are already restored in the parent. The search measures actual archive sizes at complete group boundaries. It uses the same 44-magnitude embedding, parent weights, class-specific encoder and training observations as the independent-projection baseline.

| Selection | Retained ranked slices | Partially upgraded experts outside L16/L17 | Weight shard + tokenizer, bytes | Training objective (lower is better) |
| --- | ---: | ---: | ---: | ---: |
| Independent projection ranking | 925 | 20 | 11,663,106,190 | 202,755.71 |
| Atomic pairs | 927 | 0 | 11,662,655,734 | 202,963.93 |
| Native IQ3_XXS GGUF | | | 11,663,309,504 | |

Pairing completes 11 previously half-retained experts and drops the single retained projection for the other nine. It saves 450,456 distribution bytes against the independent ranking, and its training objective is 208.22 higher. The independent verifier decoded 9,852,420,096 weight bytes and 1,231,552,512 scale codes across 38 affected planes. The prefix containing 927 slices fits, while the next complete group (929 slices) exceeds the search limit. [Selected experts and prefix costs](expert.json) · [Membership changes](selection-difference.json) · [Decoded weights and scales](experts-verified.json) · [Final package hashes](receipt.json).

## Reproduction

Start with the [925-expert recipe](../925-experts/README.md), its original training rank, lower- and higher-precision source archives and 44-magnitude embedding. Use the physical class-bank mixer from that recipe. Run the following in a new working directory, giving paired_encoder.py the original global expert assembly script.

```sh
python3 pair_rank.py "$ORIGINAL_RANK" rank.json
python3 paired_encoder.py "$ORIGINAL_ASSEMBLER"
python3 assemble_pairs.py rank.json "$LOW_MODEL" "$HIGH_MODEL" "$MIXER" original-embedding >expert.json
```

As in the baseline recipe, independently verify the expert frame bytes against the two source precisions, then attach the identical embedding frame and compare the two final file hashes. The full BF16 test and validation runs and targeted comparison of test chunk 98 position 465, test chunk 36 position 375, and validation chunk 85 position 317 are tracked in issue #35.

## Three targeted BF16 chunks on RTX PRO 4500

We scored all 765 positions from test chunks 36 and 98 and validation chunk 85 in one file. All packages used the same scorer, cubin, FP16 KV, prefill chunk 512 and 24 resident layers. Each chunk starts from a fresh context. The row IDs below are zero-based in the original split.

| Split, chunk, position | BF16 top ID | 571 KL / top ID | 925 KL / top ID | 927 paired KL / top ID |
| --- | ---: | ---: | ---: | ---: |
| test 36, 375 | 150299 | 2.858566 / 220 | 2.801739 / 220 | 2.746056 / 220 |
| test 98, 465 | 152134 | 0.684851 / 152134 | 2.909807 / 157728 | 0.241216 / 152134 |
| validation 85, 317 | 197045 | 2.552449 / 196977 | 1.996563 / 196977 | 2.279424 / 196977 |

Across these three selected chunks, 925 to 927 lowers mean KL from 0.0286131 to 0.0253014 and PPL from 9.058370 to 9.038856. Top-1 agreement falls from 715 to 707 of 765. These chunks were selected for known outliers. Use the independent 100-chunk test and validation runs for model selection.

Extract the exact subset with `python3 three_chunks.py TEST.kld VALID.kld subset.kld`, then run each model with `eval-kld --ref subset.kld --weights MODEL --chunks 3` on the [same scorer configuration](https://github.com/tjm8874/LeanSonic/issues/35#issuecomment-7228918097). [Oracle and executable digests, full aggregates, selected rows and row-file hashes](three-chunks.json).
