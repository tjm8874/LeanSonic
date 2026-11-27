# Compact Inkling on the global-PDL engine

This integration combines staged NVFP4 scales, compact MXFP8 indices, direct expert reads and automatic workspace placement with the global-PDL engine. The added CUDA entry points wait before reading predecessor outputs. All four storage/scheduling modes use the same engine and cubin. The byte-storage mode uses the same compact distribution with expanded device weights.

## Greedy generation at context 2,048

The input alternates 32 and 1,024 tokens for four passes, resetting the engine between requests. Each case checks all 64 free tokens, one further prediction and a hash of the last 200,058 FP32 logits against the direct path with global PDL.

| GPU | Mode | Resident expert layers | Matching traces |
|---|---|---:|---:|
| 5070Ti | byte | 21/24 | 8/8 |
| 5070Ti | expansion | 24/24 | 8/8 |
| 5070Ti | legacy | 24/24 | 8/8 |
| 5070Ti | direct | 24/24 | 8/8 |

## Reproduce

Use the [parent build commands, canonical input files and compact model fingerprint](../README.md) at a567839d. Set LEANSONIC_PDL=1, LEANSONIC_KV=fp16 and LEANSONIC_SEQS=1, then unset LEANSONIC_PREFILL_CHUNK for the GeForce tests. Use the four combinations below. The timing arm uses the direct configuration. The corpus comparison uses `leansonic eval-kld --weights "$MODEL" --ref "$TEST_KLD" --chunks 100` and explicit LEANSONIC_PREFILL_CHUNK=512.

| Mode | LEANSONIC_MXFP8_DEVICE_U4 | LEANSONIC_MXFP8_DIRECT_ALL | LEANSONIC_PDL_ALL |
|---|---:|---:|---:|
| byte | 0 | 0 | 1 |
| expansion | 1 | 0 | 1 |
| legacy | 1 | 1 | 0 |
| direct | 1 | 1 | 1 |

[Raw timings, predicted IDs, score summaries and build hashes](receipt.json).
