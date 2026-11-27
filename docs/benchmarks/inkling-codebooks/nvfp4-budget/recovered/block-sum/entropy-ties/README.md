# Entropy tie-break for equal signed block errors

Retain the same 2,137,934,993 rounded values and the same four allowed rounding maps per 16-weight block as the [previous signed-error experiment](../). Estimate coded nibble costs from the control rounding histograms on each plane. Among choices tied on absolute signed error, prefer the lower estimated encoded cost, then the smaller energy change. The fixed cutoffs and the primary objective remain the same. All 40 per-plane sums of absolute signed block error match the previous encoder exactly, with the total falling from 856,058,287 original units to 552,701,807 units.

Full trim encoding produces 13,632,944,494 bytes. Direct scale coding of the same weights produces **13,506,056,734 bytes** including tokenizer. That saves 23,130,536 bytes against the signed-error version in direct format but costs 26,346,440 bytes against its original 13,479,710,294-byte control. Direct encoding verified all 40 planes and all 999 final payloads. This remains an offline experiment after both BF16 splits.

Follow the parent recipe, replacing its encode.rs with [encoder.rs](encoder.rs) at runtime commit a200846. Run the same intermediate assembly and direct repacking commands and compare the resulting shard and tokenizer with the [published hashes](receipt.json). [Per-plane primary objective](proxy.json) · [Two complete pilot planes](probe.json).

## BF16 evaluation

On PRO 4500 with the a200846 CLI, bc526e3 cubin, FP16 KV and prefill width 512, both splits use 100 chunks (25,500 positions).

| Split | Mean KL | Worst KL (chunk/position) | BF16 top-1 matches | PPL |
|---|---:|---:|---:|---:|
| test | 0.0076375 | 1.6098602 (36/319) | 24,558 | 7.087611 |
| valid | 0.0068880 | 1.2255167 (91/507) | 24,635 | 6.534697 |

Keep the original direct-format control. This candidate is 26,346,440 bytes larger and raises both test and validation worst KL. Equal signed-error sums in the weights did not reproduce the earlier signed-error model’s BF16 scores. [Exact native metrics](quality.json) · [Original control](https://github.com/tjm8874/LeanSonic/blob/d1721ec/docs/benchmarks/inkling-nvfp4-scale-split/direct/fixed-q3.md).
