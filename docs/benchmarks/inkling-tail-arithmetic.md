# Short prefill suffixes with Tensor Core arithmetic

Main `18141f3` and experimental `4a0f5d3`, unchanged CUDA kernel. The experimental engine retains full 64-row chunks and submits a 2–31-row tail through the same Tensor Core paths as its long request. Inputs of one token, standalone short prompts, decode and verification retain their original dispatch. Both arms use the same cubin and fixed IDs.

The separate correctness fixture explicitly forces a 256-row single-sequence workspace and a 64-row, 64-sequence workspace, then reverses the sequence order on the second forced round. Both GPUs matched all 25,607,424 FP32 logits and every immediate post-prefill and decode top-1 at lengths 32–95. RTX 5070 Ti had 22/24 vs 16/24 fully resident expert layers.

| GPU | Prompt | Main prefill ms | Tail candidate ms | Change |
|---|---:|---:|---:|---:|
| PRO4500 | 66 | 57.497 | 63.008 | +9.6% |
| PRO4500 | 80 | 70.306 | 76.834 | +9.3% |
| PRO4500 | 95 | 82.530 | 87.197 | +5.7% |
| PRO4500 | 96 | 87.682 | 87.591 | -0.1% |
| PRO4500 | 1024 | 794.349 | 794.338 | -0.0% |
| 5070Ti | 66 | 158.288 | 174.983 | +10.5% |
| 5070Ti | 80 | 180.014 | 219.484 | +21.9% |
| 5070Ti | 95 | 197.868 | 250.103 | +26.4% |
| 5070Ti | 96 | 249.309 | 250.448 | +0.5% |
| 5070Ti | 1024 | 2169.441 | 2174.384 | +0.2% |

Latency used fixed width 64, one sequence, FP16 KV, context 2,048, three mirrored ABBA/BAAB cycles, 18 warm requests per engine/length. Moving just the arithmetic to the tail preserves logits, but still costs too much on the 16-GB card for an unconditional default. At 66 tokens this approach is faster than the rebalancing candidate while still slower than main. [Machine-readable results](inkling-tail-arithmetic.json).
