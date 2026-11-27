# Gate/up decode with two scale values per lane

The two GeForce [activity traces](https://github.com/tjm8874/LeanSonic/tree/b95ce21/docs/benchmarks/inkling-device-codebooks/integration/profile) identify the NVFP4 gate/up kernel as the largest summed kernel duration in 32 decode steps. This trial extracts just the two scale values needed by each lane. The model and loader are identical across arms.

CUDA 13.0.88 reports 61 registers for the prior kernel and 62 for this version. Both use 3,968 shared bytes, zero local bytes and a zero-byte stack. Compute Sanitizer found no errors while comparing all 4,096 possible pairs at each of eight bit alignments, in raw and packed storage. On each GPU that passed the free-generation check, eight cold/warm requests at 32 and 1,024 prompt tokens agreed on all 64 free tokens, the next prediction and the final FP32 logit hash.

| GPU | Phase | Integrated prior ms | Two-scale candidate ms | Paired change ms (95% interval) |
|---|---|---:|---:|---:|
| 5070Ti | prefill_ms | 121.9561 | 121.9583 | +0.0023 [-0.0047, +0.0103] |
| 5070Ti | decode32_ms | 192.3589 | 192.5125 | +0.1536 [+0.0509, +0.2381] |
| 5060Ti | prefill_ms | 257.4656 | 257.4800 | +0.0145 [-0.0217, +0.0400] |
| 5060Ti | decode32_ms | 339.6868 | 340.1322 | +0.4454 [+0.3764, +0.4986] |

The RTX 5070 Ti and RTX 5060 Ti comparisons both show a decode regression. Keep the existing implementation. This experiment isolates the source edit; the extra register alone does not establish the cause.

## Reproduce

Use [the integrated engine build and compact inputs](https://github.com/tjm8874/LeanSonic/tree/b95ce21/docs/benchmarks/inkling-device-codebooks/integration). Build the prior cubin at `a567839d` and the candidate at `b381e78` with the same compiler and SM120a options. Leave the replay driver on the prior commit and vary only LEANSONIC_CUBIN_PATH. Use the direct configuration with PDL, FP16 KV and one sequence. For both arms, run the same 1,024/32 fixed input at context 8,192 in five alternating ABBA/BAAB cycles, with one warm-up and four warm requests per process. Every predicted continuation ID matched the fixture. The interval resamples paired cycles.

Build the standalone check on the candidate source with `nvcc -Ikernels -arch=sm_120a -std=c++17 -O3 docs/benchmarks/inkling-device-codebooks/decode-scale-pairs/check.cu -o check` and run `compute-sanitizer --tool memcheck --error-exitcode 4 ./check`. [Individual requests, build digests and parity](receipt.json).
