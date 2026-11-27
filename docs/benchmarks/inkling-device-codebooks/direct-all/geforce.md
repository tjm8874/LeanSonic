# GeForce regression and memory checks

Both arms use the same four-bit device weights, cubin, single-token gate/up lookup and grouped Tensor Core prefill lookup. Only LEANSONIC_MXFP8_DIRECT_ALL changes. Each generation test alternates 32 and 1,024 input tokens four times at context 2,048. The comparison checks 64 free tokens, the next prediction and a hash over all 200,058 final FP32 logits.

| GPU | Matched traces | Resident expert layers, control / direct |
|---|---:|---:|
| 5060Ti | 8/8 | 24 / 24 |

Use the [parent build and input commands](../README.md) at the source in the [receipt](geforce.json). Pin LEANSONIC_MXFP8_DEVICE_U4=1, LEANSONIC_MXFP8_LOOKUP_UP=1 and LEANSONIC_MXFP8_LOOKUP_PREFILL=1 in both arms, and set LEANSONIC_MXFP8_DIRECT_ALL to 0 or 1. Use automatic workspace selection for these GeForce trials. For the additional PRO 4500 corpus comparison, use the real release CLI, the 100-chunk BF16 reference and explicit 512-row prefill width; leave both individual lookup overrides unset and vary DIRECT_ALL alone.
