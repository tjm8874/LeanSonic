# Paired full-corpus Inkling-2-Mini quality comparison

The two engines scored the same 25,500 next-token distributions on one RTX PRO 5000 72 GB Blackwell using the original 100-chunk BF16 reference. The input to each prediction used only corpus tokens. KV was FP16 and speculation was disabled. The reference PPL was 7.06440819. The LeanSonic aggregate reproduced the release report byte for byte. Dates and hashes for the public reference are in [the reference instructions](../../inkling-reference.md).

| Engine | Mean KL | P99 KL | Worst KL (chunk / position) | Top-1 matches | PPL |
| --- | ---: | ---: | --- | ---: | ---: |
| LeanSonic b207e5f4, text shard | 0.003039 | 0.028631 | 0.663992 (69 / 353) | 24,880 | 7.071986 |
| Basalt 5fe1928d, native MXFP8 file | 0.002984 | 0.027942 | 0.330612 (2 / 476) | 24,948 | 7.068542 |
| LeanSonic experimental FP64 expert accumulation, RTX 5060 Ti | 0.003094 | 0.030793 | 0.382984 (44 / 435) | 24,894 | 7.067199 |

The release engine and Basalt exceeded KL 0.05 on 84 and 76 rows respectively, with only 23 in common. Both agreed with BF16's leading token on 24,600 rows. The per-position comparison checked that the oracle NLL of the actual next token matched exactly on all rows. See `paired.json` for the largest shared and engine-specific errors and `basalt-per-chunk.csv` for the full per-chunk breakdown.

**Replay.** Apply `basalt-corpus.patch` to the upstream Basalt checkout `5fe1928dcc82ad62286d2a622bcbb5d85ebfba2b` and rebuild the Inkling CLI. The only changed file is the CLI adapter. With `BASALT_REPLAY_STREAM=1`, it processes the first 256 IDs in four 64-token blocks, then decodes one supplied ID per call. For each chunk reset state, pass IDs 0 through 510 inclusive, and record rows 256 through 510. For example, this extracts the 511 comma-separated input IDs:

```python
import struct
from pathlib import Path

chunk = 69
with open("inkling2-bf16-base.kld", "rb") as f:
    assert struct.unpack("<8sIII", f.read(20)) == (b"_logits_", 512, 200058, 100)
    f.seek(20 + chunk * 512 * 4)
    ids = struct.unpack("<512I", f.read(512 * 4))
Path("prefix.ids").write_text(",".join(map(str, ids[:511])) + "\n")
```

```sh
BASALT_REPLAY_STREAM=1 basalt --model Inkling-2-Mini-NVFP4-mxfp8.basalt --tokens-file prefix.ids --gpu 0 --kv fp16 --max-context 8192 --spec 0 --temperature 0 --max-new 1 --diag-logits chunk69.f32
leansonic eval-kld --ref inkling2-bf16-base.kld --start-chunk 69 --chunks 1 --logits-f32 chunk69.f32
```

The second command uses [PR #72](https://github.com/tjm8874/LeanSonic/pull/72). For a single report across the full corpus, concatenate the 100 chunks' raw rows in increasing chunk order and use `--logits-f32 -`. Per-chunk PPLs and percentiles cannot be averaged to recover the corpus metrics. The recorded run scored every row with the existing evaluator's `score_row`. Two independently captured streaming prefixes, 69/353 and 29/443, matched all 200,058 logit words within the full run. The CLI's external file and stdin modes also reproduced the separately aggregated chunk-38 metrics.

**FP64 trial.** The attached LeanSonic patch changes only the eight-expert weighted sum in the existing NVFP4, MXFP8 and batched combine CUDA paths. Apply it to b207e5f4 and rebuild the SM120a cubin with `nvcc -O3 -arch=sm_120a --cubin kernels/inkling2_sm120.cu`. The input shard and its stored bytes stay the same for this trial. On 5060 Ti, the four selected chunks had exact stock controls. Despite reducing the full-corpus maximum, the candidate worsened mean and P99 KL and missed both the maximum-KL and top-1 gates. We kept the release arithmetic.
