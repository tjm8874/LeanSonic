# BF16 corpus check on the four-bank file

The same RTX PRO 4500 and executable score the preceding two-bank package and the new four-bank release. Entire scorer outputs must compare byte-for-byte on each completed split. Each split uses 100 fixed 512-token chunks and 25,500 scored next-token positions with FP16 KV, context 8,192, prefill width 512 and one sequence. In addition to the pair comparison, the four headline metrics and the worst coordinates were compared with the previous two-bank run on the same integrated kernel build. Older measurements with a different cubin are a separate comparison.

| Split | Mean KL | Worst KL (chunk/position) | BF16 top-1 matches | PPL |
|---|---:|---:|---:|---:|
| test | 0.0076817 | 1.5215775 (36/319) | 24,542/25,500 (96.24%) | 7.079444 |
| valid | 0.0068862 | 1.1576402 (11/374) | 24,655/25,500 (96.69%) | 6.532434 |

Run the [published BF16 oracle procedure](../../../inkling-reference.md) on each file, with `LEANSONIC_PREFILL_CHUNK=512` and `leansonic eval-kld --weights "$MODEL" --ref "$ORACLE" --chunks 100`. Use the [four-bank conversion recipe](README.md) and the cubin pinned in the [throughput comparison](fixed-q3.md). [Complete score objects](paired-quality.json).
