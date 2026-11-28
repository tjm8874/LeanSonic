# Interleaved token groups for packed-scale NVFP4 gate/up prefill

Each token-warp pair covers a 128-token expert tile in groups of eight. Contiguous halves assign groups 0–7 to warp 0 and 8–15 to warp 1, leaving warp 1 idle for at most 64 live tokens. The interleaved mapping assigns even groups to warp 0 and odd groups to warp 1. Both warps can then contribute from nine live tokens onward. Each output keeps the same MMA operands, K loop, scales and destination. Gate/up with packed scales is the only kernel that selects the new mapping.

On RTX PRO 4500 Blackwell (200 W, driver 595.99.02), the seven-bit ranked distribution at 13,437,429,480 bytes gave the following results. Both arms used the same 1,024 prompt IDs and 32 teacher-forced decode inputs, one sequence, FP16 KV, speculation off and 24/24 resident layers.

| Phase | Baseline tok/s | Interleaved tok/s | Speed ratio (95% paired interval) |
| --- | ---: | ---: | --- |
| Prefill | 8,670.87 | 8,682.44 | 1.001334 (1.001050–1.001619) |
| Decode | 123.785 | 123.762 | 0.999815 (0.999378–1.000220) |

Five mirrored four-process cycles alternated which arm led. Each fresh process ran eight requests per mode, the last seven contributing to the timing (70 warm forced requests per arm). Both arms generated the same IDs on both forced and free paths. The decode interval spans one. The observed prefill improvement is 0.133%. [Process samples and paired intervals](timing.json).

Four held inputs, each with 1,024 prefill tokens and 32 teacher-forced steps, produced byte-identical FP32 logits in both arms on PRO 4500 and RTX 5060 Ti. Each comparison covers 26,407,656 finite logits. Both GPUs produced SHA-256 `21fe3c0c7f49883193c7717e6c363800dc6becf871a205dbdb5ecea7feddb55e`. The 5060 Ti replay used the losslessly repacked 13,407,173,808-byte variant (#236). [Per-card binary, fixture and result digests](parity.json).

## Reproduce

Build the [seven-bit model and rank reader](https://github.com/tjm8874/LeanSonic/tree/4d3bd11/docs/benchmarks/inkling-lossless-rank/measured-widths) and use the [timing harness and 1,024-token prompt](https://github.com/tjm8874/LeanSonic/tree/4d40691/docs/benchmarks/inkling-class-frames/three-engines). Use the same SM120a CUDA toolchain, timing executable and package for both arms. The baseline kernel source SHA-256 is ced5e872b1bffc45c46c01713acf58803b74fd42042c8be1177cefce4724e325 at both aa4d429 and the timed rank source 9bc4d48. Compile the baseline and this branch separately:

```sh
nvcc -Ikernels -arch=sm_120a -std=c++17 -O3 --cubin kernels/inkling2_sm120.cu -o engine.cubin
python3 ci/audit_sass.py --cubin engine.cubin
```

The trial passed the SASS audit. Six of 243 functions had different instruction fingerprints after recompilation of the common template. The measured speedup applies to that complete binary. [Function-level SASS comparison](sass.json).

Select the GPU by UUID. Set LEANSONIC_CTX=2048, LEANSONIC_PREFILL_CHUNK=1024, LEANSONIC_KV=fp16, LEANSONIC_SEQS=1 and enable LEANSONIC_PDL, LEANSONIC_PDL_ALL, LEANSONIC_MXFP8_DEVICE_U4 and LEANSONIC_MXFP8_DIRECT_ALL. Set LEANSONIC_CUBIN_PATH to each arm and alternate four fresh `bench MODEL` processes per cycle (base/trial/trial/base, then reverse). Use the [four-fixture runner and IDs](https://github.com/tjm8874/LeanSonic/tree/78def8f/docs/benchmarks/inkling-class-frames/three-engines/prefill-tiles) for the parity check. Compare both output files with `cmp` and SHA-256.

## Ragged expert counts and CUDA sanitizers

An isolated driver harness loads the exact baseline and trial cubins and calls the production gate/up kernel directly. It tests 18 counts including zero, 7/8/9, 63/64/65 and 127/128/129. On PRO 4500 all 1,456,128 active FP32 outputs agreed bit for bit. All 41,472 guard words retained their sentinels. For counts 9 and 129, memcheck, synccheck, racecheck and initcheck each checked both arms, with 158,976 outputs and 4,608 guards per tool. All four reported zero errors and racecheck also reported zero warnings. The full-engine memcheck additionally reported six missing-symbol lookups at model initialization; that run cannot establish a clean full-engine result. The direct harness requires only the two actual kernel entry points. [Outputs, sanitizer reports and digests](sanitizers.json).

Compile [check.cu](check.cu) with the CUDA toolkit. Run both binaries on the same GPU:

```sh
nvcc -arch=sm_120a -std=c++17 -O2 check.cu -L/usr/local/cuda/lib64/stubs -lcuda -o check
./check base.cubin trial.cubin
for tool in memcheck synccheck racecheck initcheck; do
    compute-sanitizer --tool "$tool" --error-exitcode 17 ./check base.cubin trial.cubin quick || exit
done
```
