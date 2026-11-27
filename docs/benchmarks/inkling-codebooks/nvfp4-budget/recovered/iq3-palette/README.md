# Seven-level NVFP4 expert weights within the public IQ3_XXS size

The final Inkling-2-Mini shard occupies 11,605,977,408 bytes. Its tokenizer adds 3,386,494 bytes for a **total of 11,609,363,902 bytes**. This saves 1,870,346,392 bytes against the 13,479,710,294-byte direct-format control and 53,945,602 bytes against the public IQ3_XXS GGUF. The public IQ1_S GGUF is 7,764,291,776 bytes. [Native distributions and hashes](https://github.com/tjm8874/LeanSonic/tree/53a4369/docs/benchmarks/inkling-native-lowbits).

For each of 40 NVFP4 expert weight planes, fit three choices using only the official weights, recovered cutoffs and training-derived routing factors. One choice retains the direct-format control. The other two reduce magnitudes to 0, 1, 2 or 4 in either previously selected blocks or the entire plane. Each reduced 16-weight block chooses a UE4M3 scale from two encoded steps below through six above the original, clamped to [64,126]. Minimize squared error in dequantized weights. Break ties by nearest original scale, then lower scale, and use the smaller magnitude on a per-weight tie. The existing decoder handles all three cases. The new model is lossy.

Sample one row in each successive group of 128 rows per expert using a deterministic seed. For each choice, measure both encoded frames and scale the byte difference to the whole plane. Weight error by squared expert amplitude and training routing factor, normalize by plane energy, and apply the visit multiplier for recurrent layers. Select minimum weighted error by dynamic programming with per-plane savings rounded down to whole MiB and a target 32 MiB below IQ3_XXS. The projection was 11,613,052,502 bytes. Full encoding reduced 998,989,178 blocks across 27 full planes and two partially reduced planes. The remaining 11 retained the control frames.

All 29 re-encoded weight and scale pairs round-tripped in full. The assembler read back and verified all 999 tensor payloads. A separate FP64 implementation decoded the final artifact, recomputed the optimal weight and scale for 32 sampled blocks per expert on each plane, and compared both signs and scale codes. All 163,840 blocks matched (38,759 original, 10,181 with the original rounding, 114,900 with seven levels). Separately regenerating the complete control frames for layer 0 down and layer 11 gate/up produced the original four frames byte for byte. [Plan](plan.json) · [Final sizes and hashes](receipt.json) · [FP64 audit](audit.json) · [Control frame hashes](control.json).

## Reproduce

Use the [official model and direct-format control recipe](https://github.com/tjm8874/LeanSonic/tree/897b812/docs/benchmarks/inkling-nvfp4-scale-split/direct). Set OFFICIAL_SHARD to the official safetensors file, DIRECT_CONTROL to the control model directory and NEW_MODEL_DIR to a fresh output directory. Set RECIPE to this folder in the evidence checkout. Run the following from a separate runtime checkout at a2008463cf028ee56bdefe39e79f6c2f95d57e73 (Python 3.11+, Rust release build). Use a fresh plan output path.

```sh
export SELECTION="$RECIPE/../selection.json"
ex=crates/leansonic-core/examples
mkdir -p "$ex/palette_budget"
cp "$RECIPE/palette.rs" "$ex/palette_budget/"
for tool in sample pack; do
  sed 's|include!("palette.rs")|include!("palette_budget/palette.rs")|' "$RECIPE/$tool.rs" >"$ex/palette_$tool.rs"
done
cp "$RECIPE/audit.rs" "$ex/audit_palette.rs"
cargo build --release -p leansonic-core --example palette_pack --example palette_sample --example audit_palette
python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["routing"],separators=(",",":")))' "$RECIPE/../../routing/receipt.json" > routing.json
python3 "$RECIPE/select.py" "$OFFICIAL_SHARD" "$SELECTION" routing.json "$PWD/target/release/examples/palette_sample" plan.json
cmp plan.json "$RECIPE/plan.json"
python3 "$RECIPE/assemble.py" "$OFFICIAL_SHARD" "$DIRECT_CONTROL" "$SELECTION" plan.json "$PWD/target/release/examples/palette_pack" "$NEW_MODEL_DIR" > receipt.json
cmp receipt.json "$RECIPE/receipt.json"
python3 "$RECIPE/audit.py" "$OFFICIAL_SHARD" "$NEW_MODEL_DIR" "$SELECTION" plan.json "$PWD/target/release/examples/audit_palette" > audit.json
cmp audit.json "$RECIPE/audit.json"
```

The [predeclared quality gate](https://github.com/tjm8874/LeanSonic/issues/35#issuecomment-7220905362) compares both 25,500-position splits with the higher-precision direct-format control. Byte counts and codec checks alone do not establish output quality or throughput.

[BF16 corpus scores and decision](quality.md).

[Fixed-input speed and peak VRAM versus public IQ3_XXS](fixed-iq3.md).
