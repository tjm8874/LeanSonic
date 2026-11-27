# Quality at fixed high-divergence positions

Each arm uses the canonical NVFP4 parent and identical FP16 KV and PDL settings. The `last` and `all` arms round the selected MXFP8 expert codes. For each requested position the scorer captured one full FP32 logit vector from the running engine and compared that same vector against both the canonical and sequential BF16 references. Locations were selected from the original quality investigations before evaluating either rounded package.

| Split | Chunk / position | Canonical BF16 KL parent / last / all | Sequential BF16 KL parent / last / all | Predicted IDs parent / last / all | BF16 IDs canonical / sequential |
|---|---|---:|---:|---|---|
| test | 29 / 443 | 0.175476 / 0.173497 / 0.175237 | 0.061269 / 0.060237 / 0.060978 | 149918 / 149918 / 149918 | 149918 / 149918 |
| test | 69 / 353 | 0.680977 / 0.691573 / 0.735488 | 0.517345 / 0.525985 / 0.558376 | 1057 / 1057 / 1057 | 2952 / 1057 |
| test | 80 / 452 | 0.031250 / 0.029361 / 0.022973 | 0.827418 / 0.816762 / 0.774029 | 197045 / 197045 / 197045 | 197045 / 196977 |
| valid | 30 / 331 | 0.272334 / 0.252094 / 0.117765 | 0.515020 / 0.490123 / 0.294589 | 20 / 20 / 20 | 18 / 18 |
| valid | 46 / 379 | 0.352436 / 0.379498 / 0.375053 | 0.483306 / 0.514457 / 0.508554 | 17 / 17 / 17 | 170558 / 170558 |

Use the [same-row BF16 reference scorer](https://github.com/tjm8874/LeanSonic/blob/29c7d6c/docs/benchmarks/inkling-bf16-serial.md) with CLI `64200ed`, PDL cubin `48b7ec2` and the [three storage packages](inkling-storage-candidates.md). The [captured FP32 vector digests and per-reference scores](inkling-rounded-hard-rows.json) identify the outputs used.
