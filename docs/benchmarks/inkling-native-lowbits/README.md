# Original lower-bit Inkling-2-Mini distributions

The Unsloth public release contains models smaller than original Q3_K_XL and the 13,626,745,974-byte compact LeanSonic distribution. Each GGUF here includes its tokenizer. We verify the exact byte count and SHA-256 from the pinned release before measuring.

| Native package | Distribution bytes | Smaller than compact LeanSonic by |
|---|---:|---:|
| Inkling-2-Mini-UD-IQ3_XXS.gguf | 11,663,309,504 | 1,963,436,470 |
| Inkling-2-Mini-UD-IQ1_S.gguf | 7,764,291,776 | 5,862,454,198 |

## Quality on RTX PRO 5000

| Fixture | Engine | Mean KL | Worst KL | BF16 top-1 | PPL |
|---|---|---:|---:|---:|---:|
| test | Original IQ3 | 0.027728 | 4.049091 | 23,655/25,500 | 7.188809 |
| valid | Original IQ3 | 0.025635 | 2.807837 | 23,785/25,500 | 6.613878 |

The native corpus scorer requested 99 GPU layers. Its default corpus log reports CUDA and batch dimensions but omits the actual layer count. The matched latency driver separately checks full offload for each native process.

## Reproduce

Download the original model files named in the table from the [pinned public release](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/ae9327c5fbc7333116ba32eee78bb5d2bbefae4f). Use the same Unsloth llama.cpp revision `883286076efe5b85fddb31d9cc7035a6b698ba9b` and the [native WikiText scoring command](../inkling-original-gguf-quality.md), substituting each GGUF and the 100-chunk oracle for that split. For bilingual scores, use the [613-position fixture and native reference driver](../inkling-chat-quality.md). For speed, use the [fixed-token replay instructions](../inkling-original-q3-timing.md) and compact weights from [PR #147](https://github.com/tjm8874/LeanSonic/pull/147). The measured LeanSonic binary is d08c1cf with its default byte-storage path. [File and input digests, scores and traces](receipt.json).
