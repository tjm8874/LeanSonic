# Fixed English and Japanese continuations on the same RTX PRO 5000

The 68-expert package uses 11,663,036,518 bytes including tokenizer. The original IQ3_XXS in llama.cpp 8832860 uses 11,663,309,504 bytes. Both received identical histories across 20 prompts and 613 response positions. Each row is scored against full-vocabulary BF16 logits. The English and Japanese reset controls matched the original LeanSonic logits byte for byte.

| Prompt language | Engine | Rows | Mean KL | Worst KL | BF16 top-1 matches | PPL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| all | LeanSonic | 613 | 0.024085 | 0.514826 | 578 | 3.700575 |
| all | llama.cpp IQ3_XXS | 613 | 0.026935 | 0.396535 | 576 | 3.634267 |
| en | LeanSonic | 379 | 0.026092 | 0.370994 | 354 | 4.697894 |
| en | llama.cpp IQ3_XXS | 379 | 0.029812 | 0.341439 | 358 | 4.644604 |
| ja | LeanSonic | 234 | 0.020833 | 0.514826 | 224 | 2.514302 |
| ja | llama.cpp IQ3_XXS | 234 | 0.022276 | 0.396535 | 218 | 2.442708 |

Mean KL falls and top-1 matches rise by two across the combined fixture. Worst KL increases, from 0.396535 to 0.514826, and target PPL increases. LeanSonic’s largest divergence is on ja_story at offset 18, where both it and BF16 predict token 19955. These are teacher-forced scores on supplied continuations.

Build the public [fixture, dump harness and scorer](https://github.com/tjm8874/LeanSonic/tree/8132335/tools/inkling-chat-quality) at 8132335 and use the [68-expert package](../README.md). Set LEANSONIC_SEQS=1, LEANSONIC_PDL=1, LEANSONIC_PDL_ALL=1, LEANSONIC_PREFILL_CHUNK=64, LEANSONIC_KV=fp16, LEANSONIC_MXFP8_DEVICE_U4=1, LEANSONIC_MXFP8_DIRECT_ALL=1 and disable resident ternary gate/up and down. Use the published GPU module digest below. Run the dump example on all order.txt entries and score against the canonical BF16 dumps. [All positions](score.json.gz) · [Hashes and reset controls](receipt.json) · [Original IQ3 fixture and commands](https://github.com/tjm8874/LeanSonic/tree/feb9b8b/docs/benchmarks/inkling-native-lowbits).
