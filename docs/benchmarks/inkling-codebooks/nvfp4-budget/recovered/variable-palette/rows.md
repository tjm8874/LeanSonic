# Replay of the worst rows from both IQ3-sized candidates

Union of the corpus maxima of the 13.48 GB control, fixed seven-level and variable-palette models. Each selected input is replayed through the single-row diagnostic and compared to two precomputed BF16 schedules: canonical and streaming (serial). The replay uses the same decoded target weights as the corpus evaluation but a separate invocation, so its KL need not match the 512-token corpus pass. Rows were selected after scoring; they diagnose the observed maxima and do not form a new held-out set.

| Split | Chunk/position | Package | Canonical KL | Serial KL | Reference top-1 (canonical/serial) | Engine top-1 |
|---|---|---|---:|---:|---:|---:|
| test | 23/330 | control | 0.272614 | 0.288612 | 17/17 | 17 |
| test | 23/330 | fixed | 2.698529 | 2.608293 | 17/17 | 20 |
| test | 23/330 | adaptive | 2.625127 | 2.538103 | 17/17 | 20 |
| test | 36/319 | control | 1.046973 | 1.077008 | 150080/150080 | 149876 |
| test | 36/319 | fixed | 1.598335 | 1.666010 | 150080/150080 | 149876 |
| test | 36/319 | adaptive | 0.191451 | 0.194361 | 150080/150080 | 149876 |
| valid | 11/374 | control | 0.138622 | 0.140787 | 293/293 | 293 |
| valid | 11/374 | fixed | 2.795289 | 2.790012 | 293/293 | 1479 |
| valid | 11/374 | adaptive | 2.956506 | 2.954935 | 293/293 | 1479 |
| valid | 12/317 | control | 0.884418 | 0.359323 | 196977/197045 | 197045 |
| valid | 12/317 | fixed | 0.556545 | 0.172994 | 196977/197045 | 197045 |
| valid | 12/317 | adaptive | 0.858994 | 0.344280 | 196977/197045 | 197045 |

[Per-row logits hashes and target IDs](rows.json) · [Held-out corpus scores](quality.md) · [Pinned single-row replay procedure](https://github.com/tjm8874/LeanSonic/issues/35).
