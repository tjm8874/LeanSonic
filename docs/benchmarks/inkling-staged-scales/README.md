# Stage finite weight scales in the packed NVFP4 Tensor Core path

The [1,024-token PRO 5000 profile](../inkling-sglang/README.md) attributed 46.679 ms of GPU kernel time to gate/up and down. This experiment moves the existing four-byte finite-scale normalization to the producer side of shared memory in the 128-token tile. Every consumer receives the same normalized bits. The arithmetic and raw, asynchronously copied scale path retain their existing operations. The package, GPU allocations, launch dimensions and engine executable are identical in each pair.

| GPU | Prefix | Phase | Control ms | Candidate ms | Paired difference ms (95% interval) |
|---|---:|---|---:|---:|---:|
| PRO5000 | 32 | prefill_ms | 31.188 | 31.195 | +0.006 [-0.011, +0.029] |
| PRO5000 | 32 | decode64_ms | 299.328 | 299.279 | -0.049 [-0.139, +0.064] |
| PRO5000 | 1024 | prefill_ms | 89.785 | 89.587 | -0.198 [-0.216, -0.179] |
| PRO5000 | 1024 | decode64_ms | 345.307 | 345.290 | -0.017 [-0.112, +0.144] |
| PRO4500 | 32 | prefill_ms | 36.710 | 36.716 | +0.006 [+0.001, +0.012] |
| PRO4500 | 32 | decode64_ms | 384.288 | 384.317 | +0.029 [-0.065, +0.124] |
| PRO4500 | 1024 | prefill_ms | 120.326 | 120.096 | -0.230 [-0.363, -0.119] |
| PRO4500 | 1024 | decode64_ms | 426.921 | 426.959 | +0.038 [-0.071, +0.148] |
| 5070Ti | 32 | prefill_ms | 75.194 | 75.082 | -0.113 [-0.318, +0.047] |
| 5070Ti | 32 | decode64_ms | 536.330 | 536.190 | -0.140 [-0.494, +0.070] |
| 5070Ti | 1024 | prefill_ms | 160.420 | 159.963 | -0.457 [-0.696, -0.176] |
| 5070Ti | 1024 | decode64_ms | 576.002 | 575.979 | -0.023 [-0.408, +0.311] |
| 5060Ti | 32 | prefill_ms | 186.636 | 186.541 | -0.095 [-0.117, -0.077] |
| 5060Ti | 32 | decode64_ms | 1260.187 | 1260.252 | +0.065 [-0.063, +0.185] |
| 5060Ti | 1024 | prefill_ms | 363.525 | 362.801 | -0.724 [-0.785, -0.671] |
| 5060Ti | 1024 | decode64_ms | 1303.304 | 1303.330 | +0.026 [-0.119, +0.139] |

The PRO 5000 1K prefill improved by 0.198 ms. The paired intervals for short prefill and both decode lengths on that card include zero. All 64 generated IDs, the following greedy token, and the FNV-1a digest of the final 200,058 FP32 logits matched between arms for every measured card and prefix. This does not establish general BF16 fidelity.

Five mirrored ABBA/BAAB cycles use ten fresh processes per cubin. Within each process we drop one warm-up and time three requests per prefix (32 and 1,024 tokens). FP16 KV, context 2,048, one sequence, PDL, canonical weights, automatic prefill width. Intervals enumerate all 3,125 resamples of the five paired cycle differences. Use the benchmark driver from [the argmax comparison](https://github.com/tjm8874/LeanSonic/blob/b953894/docs/benchmarks/inkling-argmax/bench.rs) on its control host engine. Compile the kernel source at `1499e6b` and `9e38a4e` using `nvcc -arch=sm_120a -std=c++17 -O3 --cubin kernels/inkling2_sm120.cu`. Set LEANSONIC_CUBIN_PATH to each cubin in turn. Pass the canonical model directory and the published README token IDs to the driver.

[Request-level durations](runs.csv) · [Paired intervals](summary.json) · [Generated IDs and final-logit hashes](traces.json).
