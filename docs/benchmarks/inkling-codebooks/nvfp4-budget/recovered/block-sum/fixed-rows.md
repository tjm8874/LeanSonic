# Replay of the worst rows from both block-rounding methods

Use the a200846 direct-format loader on RTX PRO 4500, with scratch-free U4, PDL, FP16 KV, prefill width 512 and speculation disabled. The four positions are the union of the baseline and trial corpus maxima. Both packages replay exactly the same prefix and forced continuation at each position. Each saved engine logit vector is scored against the canonical and serial-prefill BF16 rows. Replay scheduling can give different KL from a full-corpus run.

| Split | Chunk / position | Canonical KL, control → trial | Serial KL, control → trial |
|---|---:|---:|---:|
| test | 30/293 | 0.0303929 → 0.6744160 | 0.0435596 → 0.6167282 |
| test | 36/319 | 1.0469734 → 0.0172295 | 1.0770084 → 0.0145403 |
| valid | 12/317 | 0.8844182 → 0.0177243 | 0.3593229 → 0.0364009 |
| valid | 76/501 | 0.0081079 → 0.2900243 | 0.4899620 → 0.0102675 |

At validation 76/501 the direction changes with BF16 reference schedule. On test 30/293 the trial is worse under both references, while 36/319 improves under both. These differences retain the decision to leave the trial experimental. [Logit hashes, targets and argmax IDs](fixed-rows.json).
