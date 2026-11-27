# Train-only gate/up channel importance for ternary fitting

For each physical NVFP4 layer, selected expert and input channel we accumulated the sum of squared router weight times squared normalized activation. The tap reads the FP32 gate/up input before activation quantization. It uses the same 32 WikiText-2 train windows, 512 tokens each, as the [routing calibration](../README.md). All 32 predicted tokens and the 5,767,168 routing selections and weight summaries reproduce the prior calibration hashes on RTX PRO 4500. The resulting 20 × 128 × 2,304 FP64 array is 47,185,920 bytes. Of 2,560 layer/expert pairs, 2,535 were selected. The median within-expert ratio of the 90th to 10th channel percentile is 2.36. [Array and fixture checks](verified.json).

For every gate/up tensor, an independent sample of 64 blocks per expert (163,840 blocks total, seed 8874) compares the original bounded ternary fitter to a channel-weighted fit. Both permit the original scale byte minus two through plus six, clipped to 64–126, and fit magnitudes to zero or signed six. The trial weight is one half uniform plus one half normalized train energy, with uniform weights for unvisited experts. Of the sampled scales, 816 changed (0.50%). The weighted squared error on this sample fell 0.0060% and ordinary squared error rose 0.0070%. The experiment produced neither a new checkpoint nor held-out logits. We retained the original fitter: the improvement in the fitted surrogate is small, and its predictive value for BF16 KL is unestablished. [Per-layer totals](pilot.json).

## Reproduce

Use the source, middle calibration model, tokenizer, cubin, environment and corpus in the [routing recipe](../README.md). Set RECIPE to this directory and CAL to the parent recipe's calibration tools. From a fresh scratch directory create the source checkout as a subdirectory called src, apply the original router capture, then add this instrumentation before building.

```sh
git clone "$LOCAL_REPO" src
git -C src checkout --detach 387d5a6dd36f299739f77205ea886a7fdcdfd64c
git -C src apply "$CAL/router.patch"
cp "$RECIPE/moments.rs" .
python3 "$RECIPE/instrument.py"
cp "$CAL/calibrate.rs" src/crates/leansonic-core/examples/inkling_route_calibrate.rs
(cd src && cargo build --release -p leansonic-core --example inkling_route_calibrate)
python3 "$CAL/windows.py" "$TRAIN_RAW" "$TOKENIZER_JSON" windows.ids >fixture.json
export LEANSONIC_PREFILL_CHUNK=512 LEANSONIC_KV=fp16 LEANSONIC_PDL=1 LEANSONIC_SEQS=1
export LEANSONIC_CUBIN_PATH="$QUALITY_CUBIN"
src/target/release/examples/inkling_route_calibrate "$MIDDLE" windows.ids >predictions
python3 "$CAL/reduce.py"
python3 "$RECIPE/check.py" >checked.json
cmp checked.json "$RECIPE/verified.json"
python3 "$RECIPE/pilot.py" moments.f64 "$OFFICIAL_SHARD"
cmp pilot.json "$RECIPE/pilot.json"
```
