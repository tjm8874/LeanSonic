# Comparing G1 and fallback attention with a CPU FP64 reference

At revision 5470f09 we ran the shipped FP16-KV decode kernels separately on one RTX 5060 Ti. For each input the reference computes attention from the original FP16 K/V, FP32 query/relative features, BF16 relative projection, sink and sigmoid gate. It then derives the scale and nearest-even signed Q8 codes for each group of 64. The 66 fixtures combine three fixed seeds, local or global attention, and eleven positions from 0 to 1535, including window wrap. Local and global modes see the same history below the window limit. Inputs are synthetic.

| Across 66 fixtures | Fallback | G1 |
|---|---:|---:|
| Q8 codes different from FP64 / 202,752 | 0 | 2 |
| Largest absolute block-scale error vs FP64 | 9.790e-10 | 9.790e-10 |
| Largest dequantized RMSE vs unquantized FP64 | 8.945e-4 | 8.945e-4 |

The two code differences occur on the same seed 303, position 31, evaluated once in each attention mode. Between kernels, 1,105 of 3,168 scale values differ bitwise across 48 of 66 cases. This isolates small numerical differences inside the standalone operator. How these differences amplify into 49 top-token changes on the six selected real chunks remains under investigation. No production dispatch change follows from the synthetic fixtures alone.

## Reproduction

From a 5470f09 checkout with the CUDA 13.0 toolchain, run the following on an SM120 GPU. Copy attention.cu from this directory into the working directory. Compare the resulting CSV to the archived file on the same toolchain and card.

```sh
nvcc -Ikernels -arch=sm_120a -std=c++17 -O3 --cubin kernels/inkling2_sm120.cu -o matched.cubin
nvcc -arch=sm_120a -std=c++17 -O3 attention.cu -lcuda -o attention
./attention matched.cubin > replay.csv
cmp replay.csv rows.csv
```

[Full operator rows](rows.csv) · [GPU and driver](gpu.csv) · [Measured binary digests](digests) · [Six-chunk engine replay](../README.md).
