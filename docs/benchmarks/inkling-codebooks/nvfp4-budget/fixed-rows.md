# Layer restoration at fixed BF16 reference positions

The first precision-reallocation fit reached its maximum test divergence at chunk 99, position 398. Keep its MXFP8 values fixed and restore both NVFP4 expert planes in selected physical layers from the middle package. Each hybrid changes only those compressed tensors. The table uses the same fixed token prefixes, PRO 4500, FP16 KV, context 512, and PDL. Each row is one actual FP32 engine output compared with both reference schedules.

| Package / restored layers | Test 99/398 KL, canonical / sequential | Predicted ID | Valid 46/379 KL, canonical / sequential | Predicted ID |
|---|---:|---:|---:|---:|
| first fit | 1.8819804 / 1.7127683 | 220 | 1.6231833 / 1.9431331 | 17 |
| middle | 0.0255543 / 0.0063688 | 158000 | 0.3668402 / 0.5005905 | 17 |
| compact | 0.0304287 / 0.0117143 | 158000 | 0.5060771 / 0.6515247 | 17 |
| layers 0 through 5 | 0.1274341 / 0.0757932 | 158000 | 0.5606973 / 0.7454026 | 17 |

At test 99/398 both BF16 schedules predict 158000. At valid 46/379 both predict 170558. The corpus target is separately recorded in the receipt; agreement here refers to BF16 top-1. Two rows diagnose localized failures. Adoption requires independent full-corpus and bilingual checks.

## Reproduce

Use the [first-fit cutoff file and model recipe](README.md) and the [middle model](../README.md). The reproduction example below restores the first six physical layers. Each group uses an exclusive upper bound. Use a fresh output directory.

```sh
export PYTHONPATH="$PWD/tools/inkling_storage_probe"
python3 tools/inkling_storage_probe/nvfp4_budget/restore.py "$FIRST" "$MIDDLE" "$NEW_MODEL" 0 6
```

Use the [same-row BF16 scorer](https://github.com/tjm8874/LeanSonic/blob/29c7d6c/docs/benchmarks/inkling-bf16-serial.md) and the same CLI/cubin digests as the [parent study](../README.md). Set LEANSONIC_CTX=512, LEANSONIC_KV=fp16 and LEANSONIC_PDL=1. Score test positions 29/443, 69/353, 80/452 and 99/398, and validation positions 30/331 and 46/379. [All positions, model digests, full-logit hashes and BF16 predictions](fixed-rows.json).
