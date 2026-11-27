# Direct decoding in original expert-weight order

The direct format selects a frequency table using the existing lossless scale and a per-expert cutoff. It decodes the 16 weights of each block at their final position. The split format requires two separate block streams and a placement pass. Tensor names identify the format; both previous formats remain readable.

Run the [parent repack command](../README.md) with an empty output directory and append `--direct`. The input is the 13,608,770,734-byte restored-precision model, its original official shard and its frozen training-derived scale selection. Total size including tokenizer is **13,479,710,294 bytes**, 129,060,440 below the input and 1,784 below the split format. The repacker checked all 40 decoded expert planes and all 999 final tensors.

The same integrated executable (a2008463cf028ee56bdefe39e79f6c2f95d57e73) loaded each format on RTX PRO 4500 with FP16 KV, PDL, scratch-free lookup, width 512, context 2,048 and speculation disabled. The eight matched 32/1,024-token prefix runs produced the same 64 greedy IDs, next ID and final 200,058-logit hash in both formats. The same executable scored the direct package against BF16 on WikiText-2. Each reported summary exactly matched the split-format run, including every stored statistic.

| Split | Positions | Mean KL | Worst KL | BF16 top-1 matches | PPL |
|---|---:|---:|---:|---:|---:|
| test | 25,500 | 0.0077481 | 1.4620404 | 24,537 | 7.083731 |
| valid | 25,500 | 0.0069527 | 0.7446714 | 24,645 | 6.530797 |

[Actual file sizes, SHA-256, tensor checks, traces and oracle results](verified.json). The [parity and BF16 procedure](https://github.com/tjm8874/LeanSonic/tree/cc62564/docs/benchmarks/inkling-nvfp4-trim-context) uses the same prompts and references.
