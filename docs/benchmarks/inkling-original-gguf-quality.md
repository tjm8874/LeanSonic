# Original Unsloth GGUF size and BF16 fidelity on Blackwell

We measured the two public GGUF packages and a diagnostic hybrid with the same Unsloth llama.cpp release on RTX PRO 5000 72 GB. Each test and validation split contributes 25,500 next-token positions (100 independent 512-token chunks, 16 sequences per batch, FP16 KV). All three packages saw the same official-model BF16 oracle on each split. The native scorer compares with the stored 16-bit oracle, whose floor and rounding are documented in [the reference recipe](../inkling-reference.md).

| Split | Model | Distribution bytes | Mean KL | Worst KL | BF16 top-1 matches / 25,500 | PPL |
|---|---|---:|---:|---:|---:|---:|
| test | original IQ4_XS | 15,733,755,232 | 0.011125 | 7.915592 | 24,365 | 7.100445 |
| valid | original IQ4_XS | 15,733,755,232 | 0.009848 | 1.450225 | 24,458 | 6.522386 |
| test | IQ4 with three IQ2_S gate/up layers | 15,240,072,544 | 0.012631 | 9.332849 | 24,304 | 7.116852 |
| valid | IQ4 with three IQ2_S gate/up layers | 15,240,072,544 | 0.011219 | 2.102454 | 24,392 | 6.522536 |
| test | original Q3_K_XL | 13,631,778,016 | 0.017992 | 6.368902 | 24,081 | 7.134621 |
| valid | original Q3_K_XL | 13,631,778,016 | 0.016714 | 4.277821 | 24,179 | 6.565880 |

The hybrid swaps just the gate/up weights and associated scales in layers 1–3 of the original IQ4 package for their counterparts in original Q3. This saved 493,682,688 bytes. Compared with IQ4, the hybrid increased mean and worst KL on both splits. Its test worst KL even exceeded that of the entire Q3 package. The earlier 255-position CPU pilot had suggested a top-1 gain for the hybrid; the full GPU splits both show a loss. We are keeping the three-layer change as an external diagnostic.

For a separate LeanSonic reference point, the canonical #125 package is 15,548,157,542 bytes. Its test mean/worst KL were 0.003158426 / 0.680977332, with 24,865 matches and PPL 7.07295554. Validation measured 0.002760741 / 0.352436097, 24,963 matches and PPL 6.50462099. Original Q3 is 1,916,379,526 bytes smaller. LeanSonic uses its streaming scorer and original Unsloth uses batched native evaluation. These figures compare the deployed configurations against the same oracle; the differences include execution and evaluation schedule as well as weights. See the [matched Q3 latency and peak VRAM measurements](inkling-original-q3-timing.md) for the separate speed comparison against #117.

## Reproduce

Download the original [IQ4_XS and Q3_K_XL GGUFs](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/ae9327c5fbc7333116ba32eee78bb5d2bbefae4f) and the [matching Unsloth CUDA release](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8), source commit `883286076efe5b85fddb31d9cc7035a6b698ba9b`. Build the experimental hybrid with the [tensor-by-tensor repacking procedure](https://github.com/tjm8874/LeanSonic/blob/c0a1134/docs/benchmarks/inkling-mixed-iq2-pilot.md). Generate the BF16 oracles with the [reference recipe](../inkling-reference.md), once per split. Model, oracle, and input SHA-256 digests and full summary statistics are in the [measurement data](inkling-original-gguf-quality.json).

```sh
export LD_LIBRARY_PATH="$NATIVE:${LD_LIBRARY_PATH:-}"
"$NATIVE/llama-perplexity" -m "$GGUF" -f "$SPLIT_TEXT" --kl-divergence \
  --kl-divergence-base "$SPLIT_ORACLE" -c 512 -b 8192 -ub 8192 -ctk f16 -ctv f16 \
  -ngl 99 -t 8 --no-warmup --chunks 100
```
