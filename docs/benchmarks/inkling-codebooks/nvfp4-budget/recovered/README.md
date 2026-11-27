# Restore NVFP4 precision within the public Q3 byte budget

The [first lossless repack](https://github.com/tjm8874/LeanSonic/pull/167) saved 178,513,208 bytes without changing any decoded weights. This experiment spends part of that saving on precision using the same train-only routing factors, deterministic scale samples, and 15-symbol MXFP8 codebooks. The estimator measures entropy with the new prefix context and predicts the difference from an actual package of known size. The target was fixed at 13,611,778,016 B (20,000,000 B below Q3). The cutoff and final model were frozen before scoring this candidate.

| Package | Distribution bytes | NVFP4 weights rounded |
|---|---:|---:|
| 15-symbol capped with lossless repack | 13,445,948,726 | 2,723,001,508 |
| Recovered precision with lossless repack | 13,608,770,734 | 2,137,934,993 |
| Unsloth Q3_K_XL GGUF | 13,631,778,016 | different quantization |

The additional 162,822,008 bytes restore 585,066,515 weights to their original NVFP4 values. The final shard is 13,605,384,240 B with SHA-256 `310c60d26b0dfc46ae6c143d9981410ec8d21cdfac9565f4aa7f37c80b24fffd`. The tokenizer is 3,386,494 B. All 40 repacked NVFP4 planes decode exactly to the higher-precision intermediate weights. These counts measure weight changes and storage; the BF16 corpus comparison evaluates their effect on predictions.

## Reproduce

Prepare the official shard, intermediate codebook folder, training-only routing JSON and NVFP4 encoder using the [capped recipe](../capped). Build the trim encoder and obtain the repacking script from [the storage implementation](https://github.com/tjm8874/LeanSonic/tree/5767592/tools/inkling_storage_probe/trim_context). Use new output directories.

```sh
tool="$PWD/tools/inkling_storage_probe"
data="$PWD/docs/benchmarks/inkling-codebooks/nvfp4-budget"
export PYTHONPATH="$tool" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 "$tool/nvfp4_budget/select_cutoffs.py" "$SHARD" "$INTERMEDIATE/codebook" new.json 13611778016 "$ROUTING_JSON" --trim-baseline "$data/capped/cutoffs.json" --trim-bytes 13445948726
cmp new.json "$data/recovered/selection.json"
python3 "$tool/nvfp4_budget/assemble.py" "$SHARD" "$INTERMEDIATE/codebook" new.json "$NVFP4_ENCODER" "$NEW_OLD_FORMAT"
python3 "$REPACK_PY" "$NEW_OLD_FORMAT" "$SHARD" "$TRIM_ENCODER" "$FINAL"
```

The refactored estimator reproduced both the new selection and the previous selection byte-for-byte on the build host. [New fit](selection.json) · [Per-plane changed weights](intermediate.json) · [Final payload sizes and hashes](receipt.json).
