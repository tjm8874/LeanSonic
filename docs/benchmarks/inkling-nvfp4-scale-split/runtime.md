# GPU checks for scale-conditioned experts

On RTX PRO 4500, the same integrated executable loaded the 13,608,770,734-byte source and the 13,479,712,078-byte package. One fresh process per format alternated 32- and 1,024-token fixed README prefixes for four passes using FP16 KV, prefill width 512, scratch-free lookup, context 2,048 and no speculation. All eight matched requests produced identical 64-token continuations, next IDs and final hashes over 200,058 finite FP32 logits.

At explicit width 512 the complete BF16 WikiText-2 test scorer summaries also matched, including 25,500 positions, mean KL 0.0077481, worst KL 1.4620404, 24,537 top-1 matches, and PPL 7.083731.

Apply the scale-split patches to the [integrated trim loader](https://github.com/tjm8874/LeanSonic/tree/8c77955); the resulting source is 9ecca0858abc40d072cd04c47e73b47eaa805db5. The parity driver, environment settings, fixed input and BF16 test command are [documented here](https://github.com/tjm8874/LeanSonic/tree/cc62564/docs/benchmarks/inkling-nvfp4-trim-context). [Observations](runtime.json).
