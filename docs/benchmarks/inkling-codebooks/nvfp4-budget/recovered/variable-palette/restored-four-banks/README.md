# Keeping more expert precision below the IQ3_XXS byte budget

The [variable-palette parent](../) occupies 11,605,970,958 bytes. On the original WikiText-2 comparison it outperformed native IQ3_XXS on all four test measures and three of four validation measures. Its validation worst KL was 2.9282704 against the native 2.807837. This experiment uses the training objective for selection. The evaluation splits were not used to choose weights.

The ordinary four-bank encoder reduced the parent's shipped files to **11,577,239,270 bytes**, saving 28,731,688 bytes. A separate 238b46c decoder verified 10,192,158,720 reconstructed expert weight bytes, the other 959 tensors and tokenizer. All 40 planes shrank. [Actual frame sizes](lossless.json) · [Independent decoder receipts](lossless-decode.json)

Starting from this package, the allocator enumerates nondominated whole-plane choices under the published IQ3_XXS budget of 11,663,309,504 bytes. The cost of an option is the measured increase in both weight and scale frames. Its benefit is the decrease in the parent's frozen training objective. Within these whole-plane options and a 4 KiB header allowance, the largest benefit came from restoring both the weights and scales of the **layer 11 and 13 down projections**. This reduced the training objective from 360,348.70 to 343,710.64. The corpus result needs a separate measurement.

The resulting model and tokenizer occupy **11,648,582,790 bytes**, 14,726,714 below native IQ3_XXS. All four restored payloads came directly from the 13,455,139,382-byte four-bank control. A separate pass reopened the published shard and verified these four payloads against the control, the other 995 against the lossless parent, and the tokenizer against both. [Readback results](verified.json). [Frozen plan hash, costs, benefits and output hashes](restored.json)

## Reproduce

1. Build the [variable-palette parent](../) and retain its sample files and fixed seven-level plan. Use its original selections file. Convert the parent using the [four-bank encoder](https://github.com/tjm8874/LeanSonic/tree/16ee825/docs/benchmarks/inkling-nvfp4-scale-split/four-banks) and check with the [independent direct/four-bank decoder](../../ternary/ranked-experts/four-banks).
2. Use that actual four-bank package as LOW and the median-split four-bank control from the encoder recipe as HIGH. RECIPE points to this directory. PLAN is the variable-palette plan, SAMPLES its sample directory and ORIGINAL_PLAN the fixed seven-level plan.
3. Create the new model, reopen all 999 payloads, then evaluate LOW and OUTPUT with the same runtime, context, KV, GPU and BF16 reference files.

```sh
python3 "$RECIPE/allocate.py" "$LOW" "$HIGH" "$PLAN" "$SAMPLES" "$ORIGINAL_PLAN" "$OUTPUT" > restored.json
python3 "$RECIPE/verify.py" "$LOW" "$HIGH" "$OUTPUT" restored.json
```
