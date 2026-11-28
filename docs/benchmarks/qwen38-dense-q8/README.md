# Native group-32 Q8 projection experiment

The runtime reads signed codes and per-group BF16 scales for attention and PLE input/output matrices. Reconstructed weights are rounded to BF16 before the FP32 dot product. The first experiment covers 242 matrices. Other model components retain their existing Q4 layout. This is a development branch.

The standalone driver launches the cubin compiled from the normal production CUDA translation unit. It checks the result against a host implementation using independent BF16 rounding and a double accumulator. The 28 cases combine four input widths and seven batch sizes. Nine output rows also exercise the partial final block. The RTX PRO 4500, RTX 5070 Ti and RTX 5060 Ti reported identical per-case errors. On the PRO 4500, compute-sanitizer memcheck returned zero errors with exactly sized allocations. Removing the final output float produced the expected invalid global write with exit 77. [Per-shape errors, cubin digest and memcheck results](kernel.json)

Run from the repository root with a reserved GPU. The red control intentionally faults and should exit 77 under memcheck.

```sh
nvcc -O3 -std=c++17 -arch=sm_120a -Ikernels --cubin kernels/qwen38_sm120.cu -o q8.cubin
g++ -O2 -std=c++17 -I/usr/local/cuda/include docs/benchmarks/qwen38-dense-q8/check_gemv.cpp -o check_gemv -L/usr/local/cuda/lib64/stubs -lcuda
./check_gemv q8.cubin green
compute-sanitizer --tool memcheck --padding 65536 --error-exitcode 77 ./check_gemv q8.cubin green
compute-sanitizer --tool memcheck --padding 65536 --error-exitcode 77 ./check_gemv q8.cubin red
```

## First 242 native projections

The compact Q4 target used the corrected norm loader from #276. The mixed arm changed only the 242 QSA, Gated DeltaNet and PLE matrix precisions. The unchanged Q4 model ran on the experimental binary and matched the previous engine on all 233 FP32 logit rows and eight continuations. Quality was measured against the upstream BF16 teacher on an RTX PRO 5000 72 GB. The Q8 variant here uses the original MLX residual norms.

| Positions | Mean KL, Q4 → 242 Q8 | Worst KL | Top-1 match | PPL |
|---|---:|---:|---:|---:|
| 49 | 0.10005 → 0.08010 | 0.58455 → 0.38864 | 41 → 41 | 33.2775 → 33.0302 |
| 233 | 0.15861 → 0.09956 | 1.92162 → 0.76126 | 184 → 195 | 20.6235 → 19.6779 |
| 494 | 0.11960 → 0.09061 | 2.32175 → 1.32242 | 414 → 421 | 12.3159 → 12.0495 |

For four fixed prompts and 128 generated tokens each, median warmed prefill fell from 405.57–453.51 to 361.84–391.35 tok/s and decode from 160.91–161.66 to 132.68–133.18 tok/s. Sampled peak VRAM increased from 75,486,986,240 to 76,055,314,432 bytes. With automatic placement all 24,576 experts were in VRAM in the Q4 arm and 24,377 in the Q8 arm. The difference combines matrix precision, kernel implementation and expert placement. Exact distribution bytes for the 726-matrix artifact and timings from this 242-matrix indexed overlay describe different configurations. [Pinned package format and size](https://github.com/tjm8874/LeanSonic/blob/e5e59dd/docs/benchmarks/qwen38-pack/dense-q8/README.md) · [All timing/quality records and input/output IDs](native-242.json.gz)

## Hyper-connection precision and expert placement

Adding all 290 hyper-connection projections to the first 242 gives 532 Q8 matrices. On the same 494 positions, mean/worst KL improved from 0.09061/1.32242 to 0.05170/0.53670, top-1 matches increased from 421 to 432 and PPL improved from 12.0495 to 11.6515. On the 233 positions, mean KL improved from 0.09956 to 0.06356 and PPL from 19.6779 to 18.6362, but worst KL rose from 0.76126 to 1.49901. Four warmed decode medians with automatic placement lay between 107.07 and 109.68 tok/s.

With the same solver, revision, cubin and an explicit cap of 46 resident layers, each arm kept the same 23,552 of 24,576 routed experts in VRAM (45 contiguous layers and the same three partially resident layers). The fixed placement also reduced Q4 throughput relative to its automatic placement. Each comparison used the same four prompts, one sequence per request, 128 outputs, one warmup and three measured cycles. Output IDs may vary with the weight precision.

| Fixed-placement model | Prefill tok/s | Decode tok/s | Sampled VRAM bytes |
|---|---:|---:|---:|
| Q4 | 308.72–385.56 | 120.70–132.78 | 72,496,447,488 |
| 242 Q8 matrices | 292.52–356.13 | 106.57–123.20 | 73,645,686,784 |
| 532 Q8 matrices | 282.54–340.96 | 92.22–104.41 | 74,094,477,312 |

The shared/expert activation precision and execution paths outside the chosen families are unchanged in these three arms. Use `overlay.py Q4 Q8 OUT attention` or append `mixer` for the second arm. Build q38_stream with the q38-zstd-weights feature at `be2c93d` and set LEANSONIC_MAX_RESIDENT_LAYERS to 46. Both source directories must reside on the same filesystem for the diagnostic overlay. The file count includes superseded tensors and is unsuitable for release-size accounting.

The mixer driver compared both nonlinear stages and injection outputs against scalar reference dots across both final and intermediate mixers at batch sizes 1, 2, 5 and 8. The PRO 4500, 5070 Ti and 5060 Ti produced identical per-case maximum errors. Both kernels passed compute-sanitizer memcheck on the PRO 4500. Independently truncating the final down or up output by one float triggered an invalid write in each red control.

A separate Q4 experiment changed only the 148 residual norm vectors to official values. It left mean KL slightly higher on all three fixture sets. At 494 positions the corrected Q4 baseline's worst KL rose from 2.32175 to 2.70353 and top-1 agreement fell from 414 to 408. The main loader correction #276 and the separate 148-tensor substitution address distinct changes. [Inputs, outputs, quality metrics, controlled placement timings and mixer numerical errors](native-532.json.gz)
