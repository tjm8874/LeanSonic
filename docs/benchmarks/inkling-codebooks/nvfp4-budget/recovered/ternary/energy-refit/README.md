# Refit ternary scales for block energy

The [coefficient least-squares refit](../scale-refit/) reduced weight error but worsened BF16 next-token prediction. On the fixed WikiText test, its mean KL rose from 0.234148 to 0.594267 and top-1 agreement fell from 20,385 to 17,668 of 25,500. This candidate keeps the same nonzero positions, signs and weight payloads as the initial 7,718,950,382-byte ternary model, but selects each scale to minimize the absolute difference in squared norms of the 16 reconstructed and original coefficients. Selection stays within the scale bank that decodes the original weight frame. Ties retain the closest old code.

All 40 scale planes were rebuilt. 903,725,109 blocks changed scale. Summing in the original per-expert integer units before applying expert amplitude, the reconstructed energy went from 117.05% of source energy to 101.62%. The sum of absolute per-block energy errors fell by 72.19%. This measures the objective used to fit these scales. The WikiText BF16 evaluation is a separate required check.

The actual package including tokenizer occupies **7,748,141,470 bytes**, 16,150,306 bytes below native IQ1_S. All 40 original weight frames decoded identically and all 999 final tensors were reread. A separate FP64 implementation reproduced all 163,840 sampled best scales by enumerating the feasible codes. [Package and energy totals](receipt.json) · [Independent verification](audit.json).

Reproduce using the same [build and audit commands](../scale-refit/), pointing RECIPE to this directory and using a fresh model output directory.
