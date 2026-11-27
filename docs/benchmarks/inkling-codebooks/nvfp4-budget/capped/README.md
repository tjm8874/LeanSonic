# Fit Inkling precision to the four-bit device path

This Inkling-2-Mini distribution occupies **13,624,461,934 bytes** including its tokenizer, 7,316,082 bytes below the original Unsloth Q3_K_XL GGUF. Its final eight MXFP8 expert planes each use 15 signed byte values. The four-bit device loader supports up to 16. The earlier compact file uses 9 values on gate/up and 13 on down; the two earlier routing-weighted candidates inherited the middle model's 31–43 values and cannot enable this loader. Throughput for the compact file therefore cannot be transferred to those two candidates.

For each MXFP8 plane, dynamic programming assigns the 127 finite E4M3 magnitudes to at most eight contiguous bins. The existing block-scale-weighted squared-error objective selects a finite representative per bin, with zero fixed in the first bin. Every nonzero representative has both signs. This selects 15 symbols without changing the device storage width or its reconstruction arithmetic. The intermediate package occupies 14,122,391,478 bytes. We then used the [train-only routing factors](../routing/README.md) to spend the remaining bytes on the 40 NVFP4 expert planes. This changes 2,723,001,508 NVFP4 weights, compared with 4,178,544,265 in the previous higher-budget fit. This count describes changes to the NVFP4 weights only.

## Rebuild

Use the [pinned official model, canonical package and MXFP8 encoder](../../README.md) and the [NVFP4 encoder recipe](../README.md). The following shell variables denote absolute paths. Use new output directories.

```sh
tool="$PWD/tools/inkling_storage_probe"
data="$PWD/docs/benchmarks/inkling-codebooks/nvfp4-budget/capped"
export PYTHONPATH="$tool" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 "$tool/pack_mxfp8.py" "$OFFICIAL" "$CANONICAL" "$INTERMEDIATE" --encoder "$MXFP8_ENCODER" --codebooks "$data/maps.json"
python3 "$tool/nvfp4_budget/assemble.py" "$OFFICIAL/model-00001-of-00001.safetensors" "$INTERMEDIATE/codebook" "$data/cutoffs.json" "$NVFP4_ENCODER" "$NEW_MODEL"
(cd "$NEW_MODEL" && sha256sum model-00001-of-00001.safetensors tiktoken/tokenizer.model)
```

The final shard SHA-256 is `d4bdf49edd6ae8f891367a96eedc1dea2cef7eb341cd4d9c60137abfb6f83d4c`. The [build receipt](build.json) includes both file sizes, both digests, the intermediate shard and per-plane frame hashes. To refit, run `capped.py OFFICIAL_SHARD maps.json` using the same PYTHONPATH, then `select_cutoffs.py OFFICIAL_SHARD INTERMEDIATE/codebook cutoffs.json 13623991319 routing.json`. Generate routing.json from the training-only calibration in the linked recipe. The selection procedure uses the checkpoint and training windows; the source maps and cutoffs above are frozen before output scoring.
