# Rate-constrained ternary expert weights

The final model shard plus tokenizer occupies **7,679,612,078 bytes**. This is 55,053,432 bytes below the fixed-support scale refit, 39,338,304 bytes below the initial ternary model, and 84,679,698 bytes below the public IQ1_S GGUF (7,764,291,776 bytes). These are actual file lengths. The package contains the target model and tokenizer. Model-quality decisions use the separate held-out BF16 comparison. [Predeclared criteria](https://github.com/tjm8874/LeanSonic/issues/35#issuecomment-7222000695).

For each block of 16 original E2M1 coefficients, evaluate candidate UE4M3 scales 64 through 126. For each coefficient choose zero or the signed magnitude 6 by minimizing squared reconstruction error plus a cost for selecting a nonzero coefficient. The cost is the plane penalty times the square of half the original block scale. Ties favor zero, then the scale nearest the original, then the lower code. The physical encoder and inference math are the same as in the [ternary parent](../).

On the fixed calibration sample, compare penalties 0, 4, 12, 24 and 48 with the [fixed-support scale refit](../scale-refit/). Measure actual encoded sample bytes, and weight reconstruction SSE by expert amplitude, training routing frequency and loop participation. Apply a 1 MiB-bucket knapsack to select one penalty per plane with a size target of 7,702,173,166 bytes. The best allocation selected 4 for 24 planes, 12 for 7 and 48 for 9. Projected size was 7,679,775,014 bytes. Calibration weighted error fell 2.76% against the scale refit. Held-out corpus logits were excluded from selection.

All 1,274,019,840 blocks of the 40 modified weight and scale planes passed encode/decode roundtrip. The assembler verified all 999 final tensors. A separate FP64 implementation enumerated all 63 possible scales on 163,840 deterministic samples across every plane and expert, and matched their final scales and every decoded coefficient. [File sizes, hashes and plane byte counts](receipt.json) · [Independent audit](audit.json) · [Selected penalties](plan.json).

## Reproduce

Use the a200846 runtime with source inputs from the [fixed ternary recipe](../). Set CONTROL_MODEL to the 13.48 GB direct scale-coded package and select a fresh output directory. The published sample files and plan are sufficient to reproduce the selection. To regenerate calibrated samples as well, run probe.py first in an empty working directory with the original routing weights and cutoffs.

```sh
ex=crates/leansonic-core/examples
mkdir -p "$ex/ternary_rd"
cp "$RECIPE/palette.rs" "$ex/ternary_rd/palette.rs"
for task in sample pack; do
  sed 's|include!("palette.rs")|include!("ternary_rd/palette.rs")|' "$RECIPE/$task.rs" >"$ex/ternary_rd_$task.rs"
done
cp "$RECIPE/audit.rs" "$ex/ternary_rd_verify.rs"
cargo build --release -p leansonic-core --example ternary_rd_sample --example ternary_rd_pack --example ternary_rd_verify
bin="$PWD/target/release/examples"
cd "$EMPTY_WORK_DIR"
export INKLING_OFFLINE_PALETTE=06
python3 "$RECIPE/probe.py" "$OFFICIAL_SHARD" "$SELECTION" "$ROUTING" "$bin/ternary_rd_sample"
python3 "$RECIPE/choose_costs.py"
cmp plan.json "$RECIPE/plan.json"
python3 "$RECIPE/assemble.py" "$OFFICIAL_SHARD" "$CONTROL_MODEL" "$SELECTION" plan.json "$bin/ternary_rd_pack" "$NEW_MODEL_DIR" > receipt.json
cmp receipt.json "$RECIPE/receipt.json"
python3 "$RECIPE/audit.py" "$OFFICIAL_SHARD" "$NEW_MODEL_DIR" "$SELECTION" plan.json "$bin/ternary_rd_verify" > audit.json
cmp audit.json "$RECIPE/audit.json"
```
