# Inkling-2-Mini install size for text inference

Both deployments below include tokenization and use greedy text generation with speculation off. Counted sizes are the actual sum of model and tokenizer file lengths. Engine executables, the CUDA runtime and filesystem allocation units are outside both totals.

| Materialized deployment | Bytes |
| --- | ---: |
| LeanSonic text shard on b207e5f4 | 16,086,645,336 |
| LeanSonic tokenizer.model | 3,386,494 |
| LeanSonic text total | **16,090,031,830** |
| Basalt text-only container for 5fe1928d, including tokenizer and chat template | **18,327,104,737** |
| Published full Basalt container tested, including vision and DSpark | 19,606,374,154 |

LeanSonic's text model and tokenizer are 2,237,072,907 bytes (12.21%) smaller than this Basalt text deployment. With the original Basalt file the comparison also counts vision and the drafter. These numbers describe unarchived deployment files; transport encodings and the two engines' output distributions can differ.

The script uses `tools/model_file.py` from upstream Basalt commit `5fe1928dcc82ad62286d2a622bcbb5d85ebfba2b`. Run it against the MXFP8 Inkling container, writing to a fresh path:

```sh
PYTHONPATH="$BASALT_SRC/tools" python3 make-text-package.py source.basalt text-only.basalt
```

The script copies the 656 main dense tensor ranges and their original offsets as a contiguous prefix. All expert sections and tokenizer sections retain their original digests. The 54 DSpark tensors and the vision projector account for most of the size removed. The resulting SHA-256 was `0f589432ce7ec1d18bb7a77f105fb11ccb19a68dd41f6711145e84f7c7ba6b1c`. The LeanSonic shard SHA-256 was `94c0a91688cd06b449f10529277c250b640f5accc928a83126c5d0a343dceb99`.

On RTX PRO 4500, the upstream Basalt CLI loaded the new file and reproduced the full file's 200,058 FP32 logits byte-for-byte at both WikiText 69/353 and 29/443. LeanSonic reproduced its own control at both positions with only the model shard and `tiktoken/tokenizer.model` present. The reduced LeanSonic deployment also matched its control's full logit row and four generated token IDs on the Japanese chat input `日本の首都はどこですか。地名だけで答えてください。`. The full-column timing comparison is recorded in [the original three-arm benchmark](../inkling-entropy-basalt-current.json).
