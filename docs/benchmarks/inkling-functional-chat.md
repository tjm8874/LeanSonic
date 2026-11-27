# Free-generation accuracy on twelve English and Japanese chat tasks

Six Japanese and six English questions cover exact Unicode copying, counting, reading comprehension, arithmetic, code and typed JSON. The expected answers were fixed before running the models. Each model generated its own greedy continuation on the same RTX PRO 5000 72 GB with the same 12 prompt token sequences, FP16 KV and a 512-token context. We scored only the visible final answer. A correct answer also had to stop within the generation budget. The answer may include surrounding whitespace. JSON must match the expected values and types, without duplicate keys.

| Model | Distribution bytes | Correct / 12 at 128 tokens | Correct / 12 at 384 tokens |
|---|---:|---:|---:|
| llama.cpp BF16 reference | reference | 7 | 12 |
| LeanSonic canonical | 15,548,157,542 | 8 | 12 |
| LeanSonic magnitude cutoff | 15,396,361,558 | 8 | 12 |
| LeanSonic per-plane cutoff | 15,396,357,022 | 8 | 12 |
| Unsloth IQ4_XS | 15,733,755,232 | 8 | 12 |
| Unsloth Q3_K_XL | 13,631,778,016 | 7 | 11 |

At the 384-token cap the BF16 reference, all three LeanSonic packages and IQ4 answered all twelve correctly. Q3 exhausted its budget on the Python slicing task before producing a visible answer; the expected answer was `pio`. Q3 is 1,916,379,526 bytes smaller than canonical LeanSonic. At 128 tokens, four canonical responses reached the length cap. These twelve examples expose effects of reasoning length and API correctness; they do not establish general quality equivalence. The [full-vocabulary bilingual test](inkling-chat-quality.md) and [WikiText comparisons](inkling-original-gguf-quality.md) measure separate properties.

## Unicode request encoding and reset

The public HTTP server received the 12 prompts as raw UTF-8, then in reverse order using ASCII JSON escapes, followed by the first raw prompt again. Before the parser fix, a Japanese store-name prompt used 53 vs 50 input tokens and an emoji question used 36 vs 28, changing the visible answers. The old canonical API scored 8/12 with UTF-8 and 7/12 with escapes. After the fix, every input token count, visible answer, completion count and finish reason matched across encodings on all three packages. Each first-case replay matched after the intervening requests. At the longer budget we ran the 12 raw prompts and the reset control. The C++ driver also verified the full generated token sequence on reset for each GGUF.

The HTTP measurements used separate generation runs at each cap. For the deterministic GGUF drivers we scored both the first 128 decoding steps of the 384-step run and the whole run. All models used the original no-effort template and were free to emit internal reasoning tokens, which count against the budget. The independent LeanSonic tokenizer and official template matched token for token on all twelve prompts.

[Visible outputs, finish reasons, token counts, repeat checks and hashes](inkling-functional-chat.json) · [Fixtures and reproduction](../../tools/inkling_chat_tasks/README.md) · [HTTP parser fix](https://github.com/tjm8874/LeanSonic/pull/139)
