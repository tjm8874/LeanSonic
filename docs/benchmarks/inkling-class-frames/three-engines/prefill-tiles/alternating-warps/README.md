# Alternate eight-token groups between the token warps

The default full-width, packed-scale NVFP4 gate/up kernel consumed **35.21 ms out of 118.92 ms** in the parent prefill profile. The two token warps of each weight-row warp pair normally handle tokens 0–63 and 64–127. This experiment interleaves eight-token groups between them, keeping the same weight tile, accumulation order, launch geometry and memory capacity. Both token warps can work once an expert receives more than eight tokens. The original threshold was 64. The number of weight tiles is unchanged.

The experimental SM120a cubin changes one of 243 disassembled functions. From a fresh a1592f3 checkout run `python3 isolate.py` at the root and compile kernels/inkling2_sm120.cu using CUDA 13.0.88, `-Ikernels -arch=sm_120a -std=c++17 -O3 --cubin`. The script duplicates the template only inside the disposable controlled build so that all other dispatches keep their original instruction/control words. The repository SASS audit passes. Compare the resulting cubin with the [binary and SASS receipt](parity.json).

On RTX 5060 Ti the experiment reproduces all 26,407,656 FP32 logits across four 1,024+32 fixtures (132 distributions), byte for byte. To replay, point LEANSONIC_CUBIN_PATH at the trial cubin, use the [parent full-logit checker](../README.md) with the default 128-token shape, and compare against the original 128 output. Throughput is being measured independently on PRO 4500; the numerical result alone does not establish a speedup.

## Paired warm profile

A baseline and trial capture on the same RTX PRO 4500, driver 595.99.02 at 200 W, each recorded 1,324 launches for the same warm 1,024-token prefill. The 36 gate/up calls took 35.407 ms in the baseline and 35.113 ms in the trial, a 0.83% reduction. The GPU span fell from 120.652 to 119.881 ms (0.64%). This is one profile per arm, in baseline-first order. The multiple-cycle uninstrumented comparison determines the end-to-end decision.

Use the [parent prefill profiling runner](../README.md) with the baseline then trial cubin, leaving the model, input and executable fixed. [Baseline kernel totals](baseline.json) · [Trial kernel totals](trial.json) · [Baseline overlap](baseline-overlap.json) · [Trial overlap](trial-overlap.json).
