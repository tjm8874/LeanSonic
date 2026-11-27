# Arithmetic changes at a split prefill boundary

PRO 4500 and RTX 5070 Ti independently produced identical FP32 files for every row of the table. We disabled all Tensor Core prefill paths and enabled exactly one named option at a time. Parallel convolution was the non-Tensor-Core control. The 10 prompt lengths were 63, 64, 65, 66, 71, 76, 82, 89, 91 and 95. Each engine received the same IDs either as one prefill or in chunks of 64; after two identical forced decode steps per prompt we saved all 20 × 200,058 logits. The first three prompt lengths served as controls. KL uses the same BF16 CPU reference for both arms.

| Only enabled feature | Unequal FP32 values | Changed top-1 of 20 | Mean KL whole | Mean KL split |
|---|---:|---:|---:|---:|
| PAR_CONV | 0 | 0 | 0.0158240 | 0.0158240 |
| TC_ATTN | 2,800,760 | 4 | 0.0159287 | 0.0160239 |
| TC_BF16 | 2,800,768 | 1 | 0.0193837 | 0.0212233 |
| TC_MXFP8 | 2,800,494 | 0 | 0.0153838 | 0.0153507 |
| TC_NVFP4 | 2,800,761 | 1 | 0.0200674 | 0.0169703 |
| TC_Q8 | 2,800,765 | 0 | 0.0142391 | 0.0189685 |

Independent changes in attention, BF16, MXFP8, NVFP4 and Q8 rule out a single offending dispatch in this fixture. Parallel convolution alone reproduced the same output. The two prefill widths trade places on BF16 KL depending on the active feature. Each JSON entry includes corresponding GPU digests and per-position KL. See the [data](inkling-prefill-ablation.json).
