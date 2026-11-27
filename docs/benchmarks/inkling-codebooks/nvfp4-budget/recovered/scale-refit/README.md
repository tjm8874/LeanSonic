# Adjust NVFP4 block scales with fixed rounded weights

Start from the [13,479,710,294-byte direct-coded distribution](https://github.com/tjm8874/LeanSonic/tree/897b812/docs/benchmarks/inkling-nvfp4-scale-split/direct) and its original training-derived cutoffs. Only the 40 mean-coded scale frames can change. For each previously rounded 16-weight block, minimize squared dequantized weight error to the original official NVFP4 values, holding the nibbles fixed. Use the nearest feasible positive finite UE4M3 scale to the least-squares minimum. The search preserves the code range and entropy bank of every block. Blocks outside the original selection remain unchanged.

The objective uses integer E4M3 units of 1/512 and doubled E2M1 magnitudes. Reported sums precede the expert-wide multiplier; that multiplier is constant within each optimized block. The encoder starts from the control, moves downward only while error strictly falls, and preserves the original value on ties. The fixed rounding maps give a nonpositive residual/quantized cross term, so a larger scale cannot improve the objective.

Of 570,353,287 eligible blocks, 15,293,276 accepted a new scale. Summed squared error over eligible blocks fell by 0.5397%. The final distribution occupies **13,479,619,566 B** including tokenizer, 90,728 B smaller than the control. The assembler verified every final payload against its intended source, covering all 999 tensors. All encoded scale frames round-tripped. An independent FP64 brute-force evaluator compared 32 blocks per expert in every plane (163,840 blocks total) against every allowed code with the original code as tie-break by distance.

## Reproduce

Use the same weights as the [direct-format control](https://github.com/tjm8874/LeanSonic/tree/897b812/docs/benchmarks/inkling-nvfp4-scale-split/direct), including its recovered selection.json. At integrated a200846 place refit.rs in crates/leansonic-core/examples/refit_block_scale.rs.

```sh
cargo build --release --offline -p leansonic-core --example refit_block_scale
python3 assemble.py "$OFFICIAL_SHARD" "$DIRECT_CONTROL" "$SELECTION" \
  "$PWD/target/release/examples/refit_block_scale" "$NEW_MODEL_DIR"
```

[Model and tokenizer hashes, byte counts, errors per plane](receipt.json) · [Independent per-plane audit](audit.json).
