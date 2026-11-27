# BF16 quality after restoring 42 layer 17 experts

Each split uses exactly 100 WikiText-2 chunks and 25,500 scored next tokens. The two packages use the same RTX PRO 4500, a200846 CLI, bc526e3 module, FP16 KV, prefill width 512 and reference. The ternary baseline and 13.48 GB control are documented in the [parent comparison](../quality.md).

| Split | Model | Bytes | Mean KL | Worst KL (chunk/position) | Top-1 matches | PPL |
|---|---|---:|---:|---:|---:|---:|
| test | ternary | 7,718,950,382 | 0.2341477 | 7.5035677 (90/424) | 20,385/25,500 | 8.566626 |
| test | ranked | 7,764,131,886 | 0.2255251 | 6.6990397 (90/424) | 20,448/25,500 | 8.508404 |

On test, restoring 42 experts changed mean KL by -0.0086226, worst KL by -0.8045280, top-1 matches by +63/25,500, and PPL by -0.058222.


[Full aggregate outputs, executable hashes, worst-position input sequences and target IDs](quality.json).
