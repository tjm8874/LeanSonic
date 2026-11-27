# Compact expert storage on main with fused speculative verify

The control is main at 1bcf36a, including #156 and #164. The integrated build at bc526e3 additionally combines #153, #157, #159 and #160. Both engines use their matching cubin and the same 13,626,745,974-byte compact weight file, with four-bit device storage enabled. Mode 0 expands the expert weights and mode 1 reads their codes directly in the integrated build.

## Independent greedy generation at context 2,048

Each configuration alternates 32 and 1,024 input IDs for four passes. Each result includes 64 generated IDs, one further prediction and the hash of the last 200,058 finite FP32 logits. The comparison here includes all three configurations on a board.

| GPU | Configurations | Requests per configuration | Matched outputs |
|---|---:|---:|---|
| 5070Ti | 3 | 8 | All 64 IDs, next prediction and logit hash |

## Four-sequence verification

For each completed case on PRO 4500, four prompts (1,024, 1,021, 1,018 and 1,015 IDs) were run alone, interleaved and in batched rounds of four. The driver checks 32 free tokens for every sequence. Hashes across all listed cases also agree.

| Engine | Scale mode | Direct lookup | Speculative depth |
|---|---|---:|---:|
| Pending | — | — | — |

## Reproduce

Use the [parent build and compact model recipe](../README.md) independently at both commits. Pin LEANSONIC_KV=fp16, LEANSONIC_PDL=1, LEANSONIC_PDL_ALL=1 and LEANSONIC_MXFP8_DEVICE_U4=1. For the GeForce tests leave prefill width automatic, set LEANSONIC_SEQS=1, and run the parent parity example on the pinned README input. Test LEANSONIC_MXFP8_DIRECT_ALL=0 on main and both values on the integration.

For four-sequence verification, additionally set LEANSONIC_CTX=2048 and LEANSONIC_PREFILL_CHUNK=512. For packed scales use LEANSONIC_GPU_SCALES=packed, LEANSONIC_TC_MXFP8=1 and LEANSONIC_MOE_DOWN_XW=1; for raw use raw, 0 and 0 respectively. Run the CLI below at each speculative depth:

```sh
leansonic multiseq --weights "$MODEL" --tokens "@$INPUT" --seqs 4 --round 4 --gen 32 --spec "$SPEC"
```

[Per-request IDs, final hashes, placement and binaries](receipt.json).
