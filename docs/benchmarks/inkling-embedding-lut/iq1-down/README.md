# Reinvest the bounded embedding savings at IQ1_S size

The original 61-expert package used 7,763,991,646 bytes and missed native IQ1_S on both BF16 corpus splits. This candidate spends the isolated input-embedding savings on 82 additional experts across 12 down projections. The 61 retained in layer 17 and every gate/up weight remain fixed. In total 143 experts were restored. The shard and tokenizer are **7,763,480,170 bytes**, 811,606 below the public IQ1_S GGUF. This is a capacity and readback result; paired full-network BF16 evaluation is pending.

The search space is restricted to the other 19 NVFP4 down projections. Training routing activity and per-expert reconstruction error determine the rank. The restricted training objective falls by 248,858.75 versus 123,425.48 for the best complete-plane selection that fits. An independent decode checked 2,038,431,744 weight bytes and 254,803,968 scale codes in modified planes. The bounded embedding uses the same bytes as the isolated 68-expert experiment. The rank intentionally keeps gate/up weights compatible with two-bit residency; the number of remaining eligible down projections depends on the new weight layout.

Use prepare.py with the 61-expert four-bank parent, higher-precision four-bank control, frozen ternary plan.json, sample-06.json, and the isolated embedding receipt to generate offers.json. Run the [global expert rank and fitter](https://github.com/tjm8874/LeanSonic/tree/22ef9e3/docs/benchmarks/inkling-codebooks/nvfp4-budget/recovered/variable-palette/global-experts) using the original ternary_palette_sample executable and those offers. Validate the affected experts against the direct-frame pure ternary parent and higher-precision control. Finally run the [embedding assembler](../) with bounded.lut. [Expert selections and measured trial sizes](expert.json) · [Per-plane readback](expert-check.json) · [Distribution hashes](receipt.json).

## Matched BF16 corpus

The two LeanSonic arms used the same scorer, CUDA module, RTX PRO 4500, FP16 KV and prefill chunk 512. Each WikiText split used 100 reference chunks, 512-token contexts and 25,500 evaluated positions. The native IQ1_S control used the same oracle on RTX 5070 Ti. [Full precision LeanSonic scores and hashes](matched-corpus.json) · [Native scorer and input setup](https://github.com/tjm8874/LeanSonic/blob/bf1fa82/docs/benchmarks/inkling-mixed-ternary-down/5070Ti/native-iq1/same-card-quality.md).

| Split | Model | Mean KL | Worst KL | Top-1 matches | PPL |
| --- | --- | ---: | ---: | ---: | ---: |
| test | 61 experts | 0.224712 | 6.654781 | 20,453 | 8.482171 |
| test | 143 experts | 0.199565 | 5.593652 | 20,780 | 8.291805 |
| test | llama.cpp IQ1_S | 0.113543 | 5.392048 | 21,915 | 7.716134 |
| valid | 61 experts | 0.207796 | 6.576069 | 20,786 | 7.841980 |
| valid | 143 experts | 0.187337 | 6.201294 | 21,013 | 7.690245 |
| valid | llama.cpp IQ1_S | 0.107278 | 5.734114 | 22,181 | 7.117940 |

The 143-expert variant improves mean KL, worst KL, top-1 agreement and PPL on both splits against the 61-expert parent. Native IQ1_S retains the lead on these four metrics on both splits. The next IQ1-sized experiment spends the 44-magnitude embedding and class-specific entropy coding savings on additional experts.
