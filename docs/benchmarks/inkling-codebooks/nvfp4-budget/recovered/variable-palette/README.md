# Fit the number of NVFP4 levels per plane

The final distribution is **11,605,970,958 bytes** including tokenizer. Relative to the [fixed seven-level fit](../iq3-palette), it saves another 3,392,944 bytes. It sits 57,338,546 bytes below the public IQ3_XXS GGUF. The optimizer used the same training factors, sampled rows, 32 MiB planning margin, per-plane choices and block-scale search window. The fitted training-weighted squared-error objective fell from 400,896.146 to 360,348.698 (10.11%). This statistic measures weights. Use the BF16 tests to judge output distributions.

A preliminary screen kept the number of levels constant and added four other magnitude sets. It chose the existing fixed palette on every changed plane and reproduced the previous score. We skipped rebuilding that identical model. Its [plan, alternative palettes and selector](fixed-cardinality-screen) preserve that result. For this screen use the outer folder's observer, sample program and shared palette include, substituting the screen's palette.rs and select.py.

The next fit offered 3, 4, 5 or 6 nonnegative magnitudes including zero, with both signs available for nonzero magnitudes. The chosen 40-plane plan uses 16 full planes with {0,1,2,4}, 16 with {0,1,2,3,4}, six with {0,1,2,3,4,6}, and two planes from the original control. All 38 newly encoded pairs passed full round-trip. The assembler verified all 999 final tensor payloads. An independent FP64 implementation checked decoded signs, weight magnitudes and encoded scale for 32 sampled blocks per expert on all 40 planes. All 163,840 matched. The complete shard SHA-256 is `dd0839f3d61525a730eaf921d2b6243f2bb98f74238d11466e84cb66692db785`. [Plan](plan.json) · [Sizes and hashes](receipt.json) · [Independent verification](audit.json).

## Reproduce

Use the same official shard, direct-format control, cutoffs, training-only routing.json and runtime checkout at a200846 as in the [fixed palette recipe](../iq3-palette). Set RECIPE to this folder and keep SELECTION pointing to recovered/selection.json. The prior plan used below is the published fixed palette plan. Regenerate it with the parent recipe for a completely independent run. Start with a fresh working/output directory. Compile and execute from the runtime checkout.

```sh
ex=crates/leansonic-core/examples
mkdir -p "$ex/variable_palette"
cp "$RECIPE/palette.rs" "$ex/variable_palette/"
for tool in sample pack; do
  sed 's|include!("palette.rs")|include!("variable_palette/palette.rs")|' "$RECIPE/$tool.rs" >"$ex/variable_palette_$tool.rs"
done
cp "$RECIPE/audit.rs" "$ex/audit_variable_palette.rs"
cargo build --release -p leansonic-core --example variable_palette_pack --example variable_palette_sample --example audit_variable_palette
bin="$PWD/target/release/examples"
python3 "$RECIPE/select.py" "$OFFICIAL_SHARD" "$SELECTION" routing.json "$bin/variable_palette_sample" "$RECIPE/../iq3-palette/plan.json"
cmp plan.json "$RECIPE/plan.json"
python3 "$RECIPE/assemble.py" "$OFFICIAL_SHARD" "$DIRECT_CONTROL" "$SELECTION" plan.json "$bin/variable_palette_pack" "$NEW_MODEL_DIR" > receipt.json
cmp receipt.json "$RECIPE/receipt.json"
python3 "$RECIPE/audit.py" "$OFFICIAL_SHARD" "$NEW_MODEL_DIR" "$SELECTION" plan.json "$bin/audit_variable_palette" > audit.json
cmp audit.json "$RECIPE/audit.json"
```

Compare both BF16 splits against the original 13.48 GB control, fixed seven-level model and public IQ3_XXS. [Predeclared decision criteria](https://github.com/tjm8874/LeanSonic/issues/35#issuecomment-7221051793).

[Fixed-input speed and peak VRAM versus public IQ3_XXS](fixed-iq3.md).

[BF16 corpus scores and decision](quality.md).
