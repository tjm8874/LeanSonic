# Direct lookups for compact MXFP8 weights

Both arms use the same compact model and four-bit device indices introduced in [the residency change](../README.md). The prefill trial varies only LEANSONIC_MXFP8_LOOKUP_PREFILL. It expands the weights directly into the grouped Tensor Core shared-memory tile, keeping the original accumulation sequence. The single-token trial varies LEANSONIC_MXFP8_LOOKUP_UP, which reconstructs gate/up operands in registers before the original dot product. Each trial uses a matched executable and cubin, FP16 KV, one sequence, no speculation, a 1,024-token prompt and 32 forced decode steps at context 8,192. Measurements below are means of 40 warm requests per arm in five mirrored cycles.

| GPU | Changed path | Phase | Control ms | Direct ms | Difference (95% interval) |
|---|---|---|---:|---:|---:|
| RTX 5060 Ti | single-token gate/up | prefill 1,024 | 403.466 | 403.928 | +0.462 [-1.998, +2.657] |
| RTX 5060 Ti | single-token gate/up | decode 32 | 366.418 | 345.533 | -20.885 [-20.914, -20.852] |
| RTX 5070 Ti | single-token gate/up | prefill 1,024 | 540.643 | 540.869 | +0.226 [-2.049, +2.422] |
| RTX 5070 Ti | single-token gate/up | decode 32 | 205.255 | 195.500 | -9.755 [-9.854, -9.655] |
| RTX 5060 Ti | Tensor Core prefill | prefill 1,024 | 404.020 | 370.029 | -33.990 [-36.072, -31.537] |
| RTX 5060 Ti | Tensor Core prefill | decode 32 | 366.425 | 366.554 | +0.129 [+0.101, +0.153] |
| RTX 5070 Ti | Tensor Core prefill | prefill 1,024 | 541.580 | 468.347 | -73.232 [-74.859, -71.606] |
| RTX 5070 Ti | Tensor Core prefill | decode 32 | 205.170 | 205.298 | +0.128 [+0.053, +0.204] |

Across each reported card and trial all 32 predictions matched between paths for every replay. Separately, every paired free-generation request at context 2,048 matched on all 64 generated IDs, the following token and the hash of all 200,058 final finite FP32 logits. The 8K peak allocation and individual request times are provided in the raw data. These tests measure execution of the compact model; its accuracy relative to BF16 is reported in the parent experiment.

## Reproduce

Use the exact compact model, input hashes and two driver builds from the [residency experiment](../README.md). Rebuild at the source revision in the receipt and export LEANSONIC_MXFP8_DEVICE_U4=1 for every run. Alternate only the flag associated with the path under test between 0 and 1, leaving the other at 0. Retain the first request for correctness, exclude it from timing, and repeat in five ABBA/BAAB cycles. To benchmark the same code after both features were added, use the Tensor Core prefill source revision.

[Phase times and memory](requests.csv) · [Intervals, binary hashes, generated IDs and FP32 row digests](receipt.json)

On the measured SM120a cubin the packed-scale 128-token up/down kernels use 124/120 registers per thread with direct lookup and 118/114 with staged byte weights. All four declare zero stack and local memory. The 160-token direct up/down kernels use 211/207 registers and also declare zero stack and local memory.

The [standalone CUDA test](check.cu) enumerates all 65,536 four-nibble combinations and checks the production expansion kernel on 128 experts and eight selected experts, including untouched sentinel bytes. Compile with `nvcc -arch=sm_120a -std=c++17 -O3 -Ikernels check.cu -o check` and run `compute-sanitizer --tool memcheck --error-exitcode 42 ./check`. [Memcheck output](memcheck.txt).
