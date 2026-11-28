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
