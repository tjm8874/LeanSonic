# Equal-error block rounding for NVFP4 experts

For each selected 16-weight E2M1 block, choose where to round magnitudes 0.5 and 1.5 between their adjacent integer values. The four candidate maps have identical elementwise squared error. Minimize the absolute signed sum of the weight errors, then the absolute change in squared norm, then prefer the first map. This minimizes the block output error for constant unit input. Actual activation directions can differ. The same 2,137,934,993 weights change as in the recovered-precision control, with the same original scales and frozen train-derived cutoffs.

The summed absolute block error in doubled E2M1 units fell from 856,058,287 to 552,701,807 (35.44%). The control encoder reproduced the complete existing frames for layer 0 down and layer 11 gate/up byte for byte. Each of the 40 candidate planes passed full decode/compare.

The trim distribution totals 13,656,270,126 bytes including tokenizer. Repacking the same weights with the direct scale-coded loader produces **13,529,187,270 bytes**, 102,590,746 below the public Q3_K_XL GGUF and 49,476,976 above the previous direct distribution. The repacker checked all 999 final tensors.

The same a200846 executable and bc526e3 cubin loaded both candidate formats on PRO 4500. All eight 32/1,024-token prefix requests matched on the 64 generated IDs, next token and final FP32 logit hash. [Per-request evidence](direct-parity.json).

PRO 4500 scored the intermediate trim file against BF16 with FP16 KV and fixed prefill width 512. Both splits use the [recovered control](../quality.md).

| Split | Positions | Mean KL | Worst KL (chunk/position) | BF16 top-1 matches | PPL |
|---|---:|---:|---:|---:|---:|
| test | 25,500 | 0.0075593 | 0.5778290 (30/293) | 24,515 | 7.091517 |
| valid | 25,500 | 0.0069436 | 1.0516098 (76/501) | 24,630 | 6.529624 |

The [predeclared non-regression check](https://github.com/tjm8874/LeanSonic/issues/35#issuecomment-7220164203) failed. Against the original rounding, test worst KL fell from 1.4620404 to 0.5778290 but test PPL rose from 7.083731 to 7.091517 and BF16 top-1 fell by 22. Validation mean KL fell slightly, worst KL rose from 0.7446714 to 1.0516098 and top-1 fell by 15. Keep the previous distribution as the comparison control. The encoder remains an offline experiment.

Use the [recovered-precision assembly](../README.md) to prepare the source directory and selection.json. At integrated commit 8c77955, place encode.rs in crates/leansonic-core/examples/block_sum.rs and run:

```sh
cargo build --offline --release -p leansonic-core --example block_sum
PYTHONPATH=tools/inkling_storage_probe python3 "$EXPERIMENT/assemble.py" \
  "$OFFICIAL_SHARD" "$RECOVERED_DIR" "$SELECTION_JSON" \
  target/release/examples/block_sum "$OUTPUT_TRIM_DIR"
```

Then pass that trim directory, the original official shard and the same selection to the [direct-scale repacker](https://github.com/tjm8874/LeanSonic/tree/897b812/docs/benchmarks/inkling-nvfp4-scale-split/direct). [Per-plane sizes, SHA-256 and observations](observations.json).
