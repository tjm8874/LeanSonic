# Refit scale with fixed ternary positions and signs

Start from the [7,718,950,382-byte ternary distribution](../). Keep its 40 encoded weight frames unchanged. For each 16-weight block, minimize squared error against the original official E2M1 weights by changing its UE4M3 scale, within the range that preserves the decoder's original bank. All-zero blocks retain the original scale. Compute the exact least-squares optimum in integer units, bracket it in feasible code space, and evaluate its two neighbors and original code with a deterministic tie break.

All 1,274,019,840 blocks have a nonzero ternary weight and 1,073,305,366 chose another scale. The summed decoded weight squared error before the per-expert multiplier fell by 15.53%. All 40 weight frames decode to the same nibbles with both sets of scales. The final assembler reread and verified all 999 payloads. A separate FP64 implementation exhaustively checked every allowed scale for 32 blocks per expert and plane; all 163,840 sampled codes matched. [Plane statistics, package bytes and SHA-256](receipt.json) · [Independent audit](audit.json).

The new actual package occupies **7,734,665,510 bytes** with tokenizer. It costs 15,715,128 bytes over the original ternary distribution, and is 29,626,266 bytes smaller than the native public IQ1_S GGUF. [Quality decision criteria](https://github.com/tjm8874/LeanSonic/issues/35#issuecomment-7221275174).

Use a separate a200846 runtime checkout and the official shard, cutoffs and ternary model from the parent recipe. Set RECIPE to this directory. NEW_MODEL_DIR must be fresh.

```sh
ex=crates/leansonic-core/examples
cp "$RECIPE/refit.rs" "$ex/refit_ternary_scale.rs"
cp "$RECIPE/audit.rs" "$ex/audit_ternary_refit.rs"
cargo build --release -p leansonic-core --example refit_ternary_scale --example audit_ternary_refit
bin="$PWD/target/release/examples"
python3 "$RECIPE/assemble.py" "$OFFICIAL_SHARD" "$TERNARY_MODEL" "$SELECTION" "$bin/refit_ternary_scale" "$NEW_MODEL_DIR" > receipt.json
cmp receipt.json "$RECIPE/receipt.json"
python3 "$RECIPE/audit.py" "$OFFICIAL_SHARD" "$TERNARY_MODEL" "$NEW_MODEL_DIR" "$SELECTION" "$bin/audit_ternary_refit" > audit.json
cmp audit.json "$RECIPE/audit.json"
```
