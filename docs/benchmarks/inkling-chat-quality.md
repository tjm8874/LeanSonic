# BF16 fidelity on 20 English and Japanese chat prompts

All engines received the identical token IDs, including the assistant text header and each successive reference answer token. Each engine saw 379 English-prompt positions and 234 Japanese-prompt positions on RTX PRO 5000 72 GB, at context 512 with FP16 KV, one sequence and speculation off. The full-vocabulary KL compares raw FP32 logits against the official weights expanded to BF16, normalized in FP64 over all 200,058 tokens. No probability floor is applied. All five configurations (including BF16) reproduced every FP32 logit after resetting and repeating the first English and last Japanese cases.

| Model | Bytes | Prompt group | Rows | Mean KL | Worst KL | BF16 top-1 matches | PPL | BF16 PPL |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| LeanSonic canonical (#125) | 15,548,157,542 | all | 613 | 0.0026487 | 0.0956792 | 609 | 3.73697 | 3.73843 |
| LeanSonic canonical (#125) | 15,548,157,542 | en | 379 | 0.0023562 | 0.0956792 | 377 | 4.75739 | 4.75389 |
| LeanSonic canonical (#125) | 15,548,157,542 | ja | 234 | 0.0031225 | 0.0918076 | 232 | 2.52755 | 2.53315 |
| LeanSonic magnitude-rounded (#130) | 15,396,361,558 | all | 613 | 0.0026543 | 0.0953866 | 608 | 3.74013 | 3.73843 |
| LeanSonic magnitude-rounded (#130) | 15,396,361,558 | en | 379 | 0.0023750 | 0.0953866 | 376 | 4.75775 | 4.75389 |
| LeanSonic magnitude-rounded (#130) | 15,396,361,558 | ja | 234 | 0.0031067 | 0.0921274 | 232 | 2.53285 | 2.53315 |
| Unsloth original IQ4_XS | 15,733,755,232 | all | 613 | 0.0067935 | 0.1343043 | 595 | 3.71672 | 3.73843 |
| Unsloth original IQ4_XS | 15,733,755,232 | en | 379 | 0.0063782 | 0.1341770 | 364 | 4.73658 | 4.75389 |
| Unsloth original IQ4_XS | 15,733,755,232 | ja | 234 | 0.0074662 | 0.1343043 | 231 | 2.50958 | 2.53315 |
| Unsloth original Q3_K_XL | 13,631,778,016 | all | 613 | 0.0135265 | 0.2938108 | 583 | 3.66549 | 3.73843 |
| Unsloth original Q3_K_XL | 13,631,778,016 | en | 379 | 0.0124319 | 0.1269577 | 357 | 4.64474 | 4.75389 |
| Unsloth original Q3_K_XL | 13,631,778,016 | ja | 234 | 0.0152992 | 0.2938108 | 226 | 2.49795 | 2.53315 |

The 151,795,984-byte reduction cost one top-1 match overall. The two LeanSonic packages had their largest KL on the JSON-schema prompt (offset 25). On the Japanese prompts the mean KL decreased slightly and the worst increased slightly, both with 232/234 top-1 matches. The English subgroup had the opposite direction for mean and worst KL. We retain the canonical package as the default. Q3 had the lowest PPL on these supplied answer fragments, and also the highest full-distribution mean KL and lowest top-1 agreement. The tests score the supplied continuation; they do not evaluate free-form answer quality.

| Input | Zero-based response offset | BF16 top ID | Canonical top ID | Magnitude-rounded top ID |
|---|---:|---:|---:|---:|
| q07_poem_autumn | 16 | 197086 | 197086 | 220 |

[Fixture texts, build and scoring commands](../../tools/inkling-chat-quality/README.md) · [Per-case summaries, executable/input/model hashes and reset evidence](inkling-chat-quality.json) · [Every scored position](inkling-chat-quality-rows.jsonl). The teacher-forcing method and decision rule were [recorded before scoring](https://github.com/tjm8874/LeanSonic/issues/78#issuecomment-7214057248). English prompts had previously been used for speed measurements. Neither their fixed answers nor the Japanese fixtures were used to select the magnitude thresholds.

The BF16 run used mainline llama.cpp `29b52d3`, fully GPU-offloaded (49/49 reported); original IQ4 and Q3 used the matching Unsloth release `8832860` (25/25 reported). Both LeanSonic runs used reader `a9872cc` and cubin `48b7ec2`, SHA-256 `3714fd830d6cfedbf6954f21cad460489173be36566b0cb4b78173d42da87f0b`, PDL enabled, 64-token prefill workspace and all 24 physical expert layers resident. Compare separate timings in the [Q3](inkling-original-q3-timing.md), [IQ4](inkling-original-iq4.md) and [Basalt](inkling-basalt-canonical.md) reports.
