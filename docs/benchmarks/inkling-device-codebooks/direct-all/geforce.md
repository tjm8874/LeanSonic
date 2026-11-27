# GeForce regression and memory checks

Both arms use the same four-bit device weights, cubin, single-token gate/up lookup and grouped Tensor Core prefill lookup. Only LEANSONIC_MXFP8_DIRECT_ALL changes. Each generation test alternates 32 and 1,024 input tokens four times at context 2,048. The comparison checks 64 free tokens, the next prediction and a hash over all 200,058 final FP32 logits.

| GPU | Matched traces | Resident expert layers, control / direct |
|---|---:|---:|
| 5060Ti | 8/8 | 24 / 24 |
| 5070Ti | 8/8 | 24 / 24 |

## Fixed-token measurements at context 8,192

A 1,024-token prompt and 32 forced decode steps, FP16 KV, spec 0. Phase means use four warmed requests per process, ten processes per arm, five paired cycles. The receipt retains all requests and process peaks.

| GPU | Phase | Control ms | Scratch-free ms | Paired difference ms (95% interval) |
|---|---|---:|---:|---:|
| 5060Ti | prefill_ms | 368.475 | 258.015 | -110.460 [-110.509, -110.400] |
| 5060Ti | decode32_ms | 345.638 | 339.613 | -6.025 [-6.059, -5.984] |
| 5070Ti | prefill_ms | 470.825 | 122.521 | -348.304 [-353.097, -345.674] |
| 5070Ti | decode32_ms | 195.412 | 192.311 | -3.101 [-3.209, -3.020] |

| GPU | Control peak bytes | Scratch-free peak bytes | Saved bytes |
|---|---:|---:|---:|
| 5060Ti | 16,605,249,536 | 16,013,852,672 | 591,396,864 |
| 5070Ti | 16,594,763,776 | 16,112,418,816 | 482,344,960 |

## WikiText-2 test, RTX PRO 4500

| Storage | Mean KL | Worst KL | Matching BF16 top tokens | PPL |
|---|---:|---:|---:|---:|
| control | 0.01378376 | 0.70781849 | 24298/25500 | 7.1274818 |
| direct | 0.01378376 | 0.70781849 | 24298/25500 | 7.1274818 |

Use the [parent build and input commands](../README.md) at the source in the [receipt](geforce.json). Pin LEANSONIC_MXFP8_DEVICE_U4=1, LEANSONIC_MXFP8_LOOKUP_UP=1 and LEANSONIC_MXFP8_LOOKUP_PREFILL=1 in both arms, and set LEANSONIC_MXFP8_DIRECT_ALL to 0 or 1. Use automatic workspace selection for these GeForce trials. For the additional PRO 4500 corpus comparison, use the real release CLI, the 100-chunk BF16 reference and explicit 512-row prefill width; leave both individual lookup overrides unset and vary DIRECT_ALL alone.
