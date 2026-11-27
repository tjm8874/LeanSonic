# Automatic prefill workspace placement

The loader estimates the peak cost of growing a live workspace after expert placement. If that allocation would exceed the remaining budget, it widens before uploading experts, then checks their actual placement. Both arms use the same cubin, compact model, four-bit expert storage and automatic workspace selection.

## Free-generation regression at context 2,048

| GPU | Startup capacity, control → candidate | Resident layers | Warm and cold traces |
|---|---|---|---|
| 5070Ti | 64..2048 → 2048 | 24/24 → 24/24 | 8 matched |

Lengths 32 and 1,024 alternate on one engine for four passes. Each trace includes all 64 greedy tokens, one extra prediction and a hash of all 200,058 final FP32 logits. The driver checks finite logits and the argmax against the reported final token.

## Fixed-input latency at context 8,192

| GPU | Phase | Control ms | Candidate ms | Paired change ms (95% interval) |
|---|---|---:|---:|---:|

Waiting for five complete mirrored cycles per GPU.

## Reproduce

Check out each source commit in the [receipt](receipt.json) and use the compact package from [the device-storage baseline](../README.md). Build the parity and replay examples with its commands. Set LEANSONIC_MXFP8_DEVICE_U4=1, LEANSONIC_PDL=1, LEANSONIC_KV=fp16 and LEANSONIC_SEQS=1. Unset LEANSONIC_PREFILL_CHUNK. Use the same SM120a cubin for both arms. For 8K replay use the 1,024-token input and fixed 32-token continuation, five fresh-process mirrored cycles, alternating ABBA/BAAB. Each process issues five requests; exclude its first request from phase timing. Sample process VRAM throughout load and execution. The receipt records individual requests, predictions, peak bytes and cycle-level intervals for completed cycles.
