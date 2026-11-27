# Integrated cross-format verification

At e030d0c, the same release CLI and cubin scored the original trim file and the new direct-format file on RTX PRO 4500, FP16 KV, prefill chunk 512. Both distributions use the same decoded weights. All native metrics matched exactly on both independent 25,500-position splits. Eight additional free-generation requests (32/1024 prefixes, 64 new tokens) matched in every generated token, next token and final FP32 logit hash.

| Split | Mean KL | Worst KL (chunk/position) | Top-1 matches | PPL |
|---|---:|---:|---:|---:|
| test | 0.0076817 | 1.5215775 (36/319) | 24,542 | 7.079444 |
| valid | 0.0068862 | 1.1576402 (11/374) | 24,655 | 6.532434 |

The older a200846-runtime report has different scores. The integrated runtime changed between studies. On each pinned runtime the formats match. [Earlier control and direct scores](../direct/README.md).

## Original Q3_K_XL on the same inputs

Use the [unchanged fixed IDs, native revision, ABBA ordering and sampling procedure](../direct/fixed-q3.md). Only the LeanSonic executable and cubin change in this study. Both engines ingest 1024 fixed tokens and the same 32 forced continuation tokens. Warm means exclude the first of five requests; peak VRAM spans the complete process.

| GPU | Engine | Prefill tok/s | Decode tok/s | Peak process VRAM B |
|---|---|---:|---:|---:|
| 5070Ti | lean | 8,347.0 | 170.94 | 16,093,544,448 |
| 5070Ti | llama | 1,944.9 | 79.46 | 16,150,167,552 |
| 5060Ti | lean | 3,955.6 | 94.49 | 15,969,812,480 |
| 5060Ti | llama | 1,164.0 | 55.81 | 16,030,629,888 |

[Native score objects, request traces, process peaks and binary hashes](verified.json).
