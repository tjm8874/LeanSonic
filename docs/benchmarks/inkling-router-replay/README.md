# Inkling-2-Mini router replay

This experiment records the normalized router input and selected experts at the final prompt token. The two investigated positions were selected for their known divergence. Each of the 48 visits has its own row. Expert lists in the CSV give the set difference from the streaming BF16 reference. Relative L2 uses the BF16 input norm as the denominator.

Apply the adjacent patch with `git apply` to Basalt at `5fe1928dcc82ad62286d2a622bcbb5d85ebfba2b`, then rebuild the Inkling CLI. Run on one RTX PRO 4500 Blackwell. Use the same public token IDs and BF16 oracle as [the prefix comparison](../inkling-basalt-oracle-prefixes.json). Use a fresh trace file for every process. The trace appends one little-endian record per visit, consisting of four int32 fields (visit, physical layer, embedding width, expert count), 2,304 FP32 normalized activations and eight int32 expert IDs.

```sh
BASALT_TRACE_FINAL=prefix.route basalt --model Inkling-2-Mini-NVFP4-mxfp8.basalt --tokens-file prefix.ids --gpu 0 --kv fp16 --max-context 8192 --spec 0 --temperature 0 --max-new 1 --diag-logits prefix.logits
```

For the streaming layout, also export `BASALT_REPLAY_STREAM=1`. It processes four 64-token prompt blocks, then consumes each remaining corpus token individually. The trace is enabled only when the CLI starts the final token. For each streaming control, keep this schedule and omit the trace variable. All 200,058 FP32 logit words matched within each trace/control pair. Native controls at 0/256, 0/400, 29/443 and 69/353 matched unmodified upstream Basalt.

The archived LeanSonic routing probe and streaming BF16 captures are described in [the prior router experiment](https://github.com/tjm8874/LeanSonic/issues/35#issuecomment-7201813771). To reproduce the interventions, extract the normalized vector from visit 11 at 69/353 or visit 13 at 29/443. Pass that vector to the same single-visit LeanSonic probe in router-only mode. The probe restores the original activation before evaluating the experts. Both stock controls reproduced their archived logit bytes and current main's reported row KL. Scores use the full vocabulary against the original KLD reference. The streaming BF16 capture supplies the expert sets and activations for the layer comparisons.

| Chunk / position | LeanSonic stock KL | KL with native Basalt router input | KL with streaming Basalt router input | BF16 top token | LeanSonic top token, stock / native input / stream input |
| --- | ---: | ---: | ---: | ---: | --- |
| 69 / 353 | 0.663992 | 0.012323 | 0.005933 | 2952 | 1057 / 1057 / 2952 |
| 29 / 443 | 0.193547 | 0.145719 | 0.147087 | 149918 | 149918 / 149918 / 149918 |

This intervention changes both expert selection and the associated mixing weights. The two prefixes establish a local causal effect. A production change still needs full-corpus testing.
