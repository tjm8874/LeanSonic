# Training-ranked down experts under the IQ1_S byte budget

Inkling-2-Mini's 61-expert parent keeps all gate/up planes at their original low-bit tier. This trial uses the same training-only rank over the other down projections as the [143-expert control and corpus test](https://github.com/tjm8874/LeanSonic/tree/c591f04/docs/benchmarks/inkling-embedding-lut/iq1-down). The input embedding uses the [44-magnitude frame](../README.md).

| Package | Weight shard and tokenizer bytes |
| --- | ---: |
| 143 retained experts with 44-magnitude embedding | 7,520,403,846 |
| 487 retained experts with 44-magnitude embedding | 7,764,242,158 |
| Native llama.cpp IQ1_S GGUF | 7,764,291,776 |

The additional 344 slices cost 243,838,312 bytes. Attaching the 44-level embedding to the 61+426 expert archive leaves 49,618 bytes under the native file. The independent decoder checked 3,057,647,616 weight bytes and 382,205,952 scale codes across 18 changed planes. The first 61 retained experts were present in the parent. The BF16 comparison below measures the 487-expert candidate.

## Reproduction

1. Use the 61-expert four-bank parent, higher-precision control and frozen training ranking from the 143-expert experiment linked above. Build [mix.rs](mix.rs) as the expert-mixing example in the [four-bank toolchain](https://github.com/tjm8874/LeanSonic/tree/aaceb42/docs/benchmarks/inkling-codebooks/nvfp4-budget/recovered/variable-palette/global-experts).
2. Materialize the 143-expert package with the 44-level embedding to measure the embedding savings. Apply the same physical savings (359,569,840 bytes) to the native 7,764,291,776-byte ceiling while selecting the prefix. The intermediate archive cap is 8,123,861,616 bytes.
3. Encode prefixes with the class-specific tables, using the [44-level 925-expert assembly procedure](../ranked-experts/class-banks/925-experts/README.md) and the restricted down-expert ranking. Prefix 426 fits and 427 exceeds the cap. Replace the input embedding in the selected archive with the same frame. Verify the final shard and tokenizer SHA-256 before timing or scoring.

[143-expert control hashes](control.json) · [Selected ID list, capped ranking hash and prefix trials](expert.json) · [Expert readback](verified.json) · [Final package hashes](receipt.json).

## 100-chunk BF16 corpus comparison

LeanSonic scored the 487-expert package on RTX 5070 Ti with 24/24 layers resident, FP16 KV and 512-token prefill. The [original native IQ1_S measurements](https://github.com/tjm8874/LeanSonic/tree/d65d1a6/docs/benchmarks/inkling-native-lowbits) used RTX PRO 5000. Both engines used the same 25,500 positions per split. The device differs in this quality comparison.

| Split | Engine | Mean KL | Worst KL | BF16 top-1 / 25,500 | PPL |
| --- | --- | ---: | ---: | ---: | ---: |
| test | LeanSonic 487 | 0.1646306 | 8.5200744 | 21,179 | 8.038991 |
| test | llama.cpp IQ1_S | 0.1130210 | 5.8886900 | 21,921 | 7.707463 |
| valid | LeanSonic 487 | 0.1531214 | 5.8109892 | 21,413 | 7.426430 |
| valid | llama.cpp IQ1_S | 0.1076170 | 5.5685180 | 22,138 | 7.121110 |

The 49,618-byte size saving comes with poorer values on all four quality measures on both splits. Keep this allocation experimental. To reproduce, use engine 726b6f2 with the original 487-expert weights, device U4 and direct prefill enabled and both ternary device paths disabled, then run `eval-kld --weights MODEL --ref ORACLE --chunks 100`. [Model, executable and oracle digests, and full score fields](5070-quality.json).
