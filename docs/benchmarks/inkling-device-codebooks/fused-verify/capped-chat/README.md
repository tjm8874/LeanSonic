# Bilingual Inkling checks on 16 GB Blackwell

Build bc526e3 with the matching cubin, scratch-free U4, FP16 KV, one sequence and speculation disabled. The 5070 Ti fixture uses teacher forcing with prefill width 64 and 613 scored positions. The 5060 Ti uses the public CLI and greedy chat endpoint with a 384-token cap.

| Package | Mean KL | Worst KL | BF16 top-1 | PPL | Free-generation checks |
|---|---:|---:|---:|---:|---:|
| compact | 0.0133126 | 0.3825850 | 584/613 | 3.737550 | 13/13 |
| capped | 0.0086031 | 0.1235134 | 590/613 | 3.790252 | 13/13 |

The capped package improves mean KL, the largest KL and top-1 agreement on these inputs. Its PPL rises from 3.737550 to 3.790252 (BF16 3.738427). The free-generation suite contains 12 distinct tasks and a repeated Unicode answer. Both repeated teacher-forced outputs were byte-compared in each arm.

Use the [teacher-forced fixtures and scorer](../../../../../tools/inkling-chat-quality) and [functional suite](../../../../../tools/inkling_chat_tasks) from the integrated source commit, together with the [model recipe](https://github.com/tjm8874/LeanSonic/tree/c9a6a63/docs/benchmarks/inkling-codebooks/nvfp4-budget/capped). [All scores, responses and binary fingerprints](receipt.json).
