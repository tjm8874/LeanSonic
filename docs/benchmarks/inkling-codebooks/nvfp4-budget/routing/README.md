# Allocate precision using observed expert routing

This Inkling-2-Mini package occupies **13,612,680,454 bytes**, including its tokenizer. At the same fitting target as the previous allocation (13,607,950,944 bytes), weighting experts by their observed routing reduces the actual distribution by 10,971,928 bytes. The fitter estimated 13,607,893,757 bytes. The package exceeds this estimate by 4,786,697 bytes and the fitting target by 4,729,510 bytes. The file size alone does not establish output quality.

The calibration build ran the **middle** model on 32 non-overlapping, 512-token windows from WikiText-2 raw **train**. Each prefill selected eight experts per position across 44 physical NVFP4 layer visits. The capture checked the expert IDs and gate weights and recorded 5,767,168 selections. For each physical layer, we divided each expert's sum of squared routing weights by the mean across its 128 experts, then averaged the result with one. The resulting multipliers range from 0.5000 to 8.3574. The fitter already counts the repeated visits to the recurrent layers. It uses each expert's multiplier for both weight planes. The repacker reports 4,217,469,960 modified weights across 40 planes.

## Rebuild from the published cutoffs

Use the official Inkling-2-Mini shard, middle package and native encoder from the [parent recipe](../README.md). From the repository root:

```sh
tool="$PWD/tools/inkling_storage_probe"
data="$PWD/docs/benchmarks/inkling-codebooks/nvfp4-budget/routing"
export PYTHONPATH="$tool" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 "$tool/nvfp4_budget/assemble.py" "$OFFICIAL_SHARD" "$MIDDLE" "$data/cutoffs.json" "$ENCODER" "$NEW_MODEL"
```

Choose a new output directory. Verify that the single shard has SHA-256 `b9c20b062e60b7e0efa4dab495bf8a95673f62ed290dbb94500a5238e50a44ef`. All frame sizes, digests and modified-weight counts are in the [receipt](receipt.json).

## Recompute the calibration

The instrumentation is for an isolated calibration checkout at `387d5a6dd36f299739f77205ea886a7fdcdfd64c`. Install Python NumPy and tokenizers in the calibration environment. Use the same SM120a cubin as the [parent quality study](../../README.md), SHA-256 `3714fd830d6cfedbf6954f21cad460489173be36566b0cb4b78173d42da87f0b`. Run on one RTX PRO 4500 Blackwell. Use WikiText-2 raw wiki.train.raw and the official tokenizer.json. Their SHA-256 digests and all 32 token offsets are in the receipt.

```sh
cal="$tool/nvfp4_budget/calibration"
git worktree add --detach "$CAL_SOURCE" 387d5a6dd36f299739f77205ea886a7fdcdfd64c
git -C "$CAL_SOURCE" apply "$cal/router.patch"
cp "$cal/calibrate.rs" "$CAL_SOURCE/crates/leansonic-core/examples/inkling_route_calibrate.rs"
(cd "$CAL_SOURCE" && cargo build --release -p leansonic-core --example inkling_route_calibrate)
mkdir "$CAL_OUTPUT"
cd "$CAL_OUTPUT"
python3 "$cal/windows.py" "$TRAIN_RAW" "$TOKENIZER_JSON" windows.ids > fixture.json
export LEANSONIC_PREFILL_CHUNK=512 LEANSONIC_KV=fp16 LEANSONIC_PDL=1 LEANSONIC_SEQS=1
export LEANSONIC_CUBIN_PATH="$QUALITY_CUBIN"
"$CAL_SOURCE/target/release/examples/inkling_route_calibrate" "$MIDDLE" windows.ids > predictions
python3 "$cal/reduce.py"
python3 "$tool/nvfp4_budget/select_cutoffs.py" "$OFFICIAL_SHARD" "$MIDDLE" selection.json 13607950944 routing.json
```

Compare the SHA-256 of windows.ids and routes.bin with fixture.window_sha256 and routing.source_sha256 in the receipt. An independent run of the published preprocessing and fitter reproduced windows.ids, routing.json and selection.json byte for byte; selection.json had SHA-256 `4767bbd768f3f83b9da50ab3f4526a41e15a733f899e4eddb862df8018a6ff00`. The evaluation uses separate WikiText-2 test, validation and bilingual fixtures.

[Held-out quality scores and adoption decision](quality.md).
