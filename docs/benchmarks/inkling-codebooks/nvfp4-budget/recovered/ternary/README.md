# Ternary Inkling experts below the public IQ1_S file size

| Distribution | Shard or GGUF bytes | Separate tokenizer bytes | Total bytes |
|---|---:|---:|---:|
| LeanSonic ternary | 7,715,563,888 | 3,386,494 | **7,718,950,382** |
| Unsloth IQ1_S | 7,764,291,776 | bundled | **7,764,291,776** |

The measured difference is 45,341,394 bytes. The LeanSonic shard SHA-256 is `a3c815c1e7ce2c4e1e2bb32f694ecc16684f5143d3062f46e9cbeac967b40dc1`. The GGUF SHA-256 is `c09c028b050c108624682baeb28ac6e0cdb936af021ac637c8c3609f923ea4b8`. The file sizes use the actual model and tokenizer needed by each pinned runtime. File size does not predict device allocation. [Native file and release revision](https://github.com/tjm8874/LeanSonic/tree/53a4369/docs/benchmarks/inkling-native-lowbits).

Reduce all 40 NVFP4 expert weight planes to three signed levels (−6, 0, +6) times one UE4M3 scale per 16 weights and the original expert multiplier. This covers 1,274,019,840 blocks. The last four MXFP8 layers use the existing compact control representation. No new kernel or per-plane decoder metadata is needed. The optimizer evaluated five ternary alphabets and the previous variable-palette choices under a target 16 MiB below IQ1_S. Only the ±6 ternary solution was selected. The sampled projection of 7,724,879,446 bytes was 5,929,064 bytes higher than the resulting file.

All 40 encoded weight/scale pairs passed full decode comparison and the final assembler reread all 999 tensor payloads. A separate FP64 evaluator tested 32 blocks per expert across all 40 planes (163,840 blocks). It recomputed the optimal scale and every ternary weight, including signs, and matched the final artifact. The training-weighted distortion was 5,276,621 compared with 360,349 for the 11.606 GB variable-palette model. Only BF16 corpus and task measurements can establish the quality cost. [Target, other size projections](feasibility.json) · [Chosen plan](plan.json) · [Physical files and per-plane counts](receipt.json) · [Independent audit](audit.json) · [Comparison criteria](https://github.com/tjm8874/LeanSonic/issues/35#issuecomment-7221275174).

## Reproduce

Use the [variable-palette workflow](../variable-palette) to generate its six sample-*.json files. Set PALETTE_PREVIOUS to the directory containing those outputs. Use the same frozen fixed seven-level plan, direct control, official weights, cutoffs and training routing.json. Build in an a200846 runtime checkout. Select a fresh output directory for each model and a fresh work directory for the selector.

```sh
export RECIPE=/absolute/path/to/this/ternary/folder
export PYTHONPATH="$RECIPE/../variable-palette"
ex=crates/leansonic-core/examples
mkdir -p "$ex/ternary_palette"
cp "$RECIPE/palette.rs" "$ex/ternary_palette/"
for tool in sample pack; do
  sed 's|include!("palette.rs")|include!("ternary_palette/palette.rs")|' "$RECIPE/../variable-palette/$tool.rs" >"$ex/ternary_palette_$tool.rs"
done
cp "$RECIPE/../variable-palette/audit.rs" "$ex/audit_ternary_palette.rs"
cargo build --release -p leansonic-core --example ternary_palette_pack --example ternary_palette_sample --example audit_ternary_palette
bin="$PWD/target/release/examples"
mkdir ternary-build
cd ternary-build
python3 "$RECIPE/select.py" "$OFFICIAL_SHARD" "$SELECTION" "$ROUTING_JSON" "$bin/ternary_palette_sample" "$RECIPE/../iq3-palette/plan.json"
cmp plan.json "$RECIPE/plan.json"
python3 "$RECIPE/../variable-palette/assemble.py" "$OFFICIAL_SHARD" "$DIRECT_CONTROL" "$SELECTION" plan.json "$bin/ternary_palette_pack" "$NEW_MODEL_DIR" > receipt.json
cmp receipt.json "$RECIPE/receipt.json"
python3 "$RECIPE/../variable-palette/audit.py" "$OFFICIAL_SHARD" "$NEW_MODEL_DIR" "$SELECTION" plan.json "$bin/audit_ternary_palette" > audit.json
cmp audit.json "$RECIPE/audit.json"
```
