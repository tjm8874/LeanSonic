# Release-length prefill replay

The same release CLI loads either the control or candidate cubin. Inputs are the deterministic native synthetic token sequences. Each paired cycle runs four fresh processes in mirrored order and alternates the first arm in consecutive cycles. Each process executes two requests. The table uses the second request in each process, ten per arm per GPU and length. Every generated token also matched in the first request. Full sequences are in the receipt.

| GPU | Input tokens | Prefill ms, control → candidate | Candidate tok/s | Paired difference ms (95% interval) | Decode ms, control → candidate |
|---|---:|---:|---:|---:|---:|
| PRO4500 | 2,048 | 201.968 → 200.863 | 10,196.0 | -1.105 [-1.250, -0.961] | 201.745 → 201.689 |
| PRO4500 | 16,384 | 1765.818 → 1757.376 | 9,323.0 | -8.442 [-8.884, -8.046] | 884.982 → 885.213 |
| PRO4500 | 131,072 | 21585.658 → 21525.268 | 6,089.2 | -60.390 [-79.813, -36.302] | 1483.899 → 1484.019 |
| PRO5000 | 2,048 | 136.883 → 136.058 | 15,052.4 | -0.825 [-0.869, -0.793] | 156.067 → 156.100 |
| PRO5000 | 16,384 | 1208.809 → 1202.113 | 13,629.3 | -6.696 [-7.175, -6.177] | 676.901 → 677.092 |

Rebuild both sources at the commits in the [receipt](release-lengths.json) using CUDA 13.0.88, `nvcc -arch=sm_120a -std=c++17 -O3 --cubin kernels/inkling2_sm120.cu`. Use the same candidate release CLI for both arms. Set LEANSONIC_CUBIN_PATH separately for each cubin. Use the canonical LeanSonic model directory for MODEL.

```sh
export LEANSONIC_SEQS=1 LEANSONIC_PDL=1 LEANSONIC_KV=fp16
LEANSONIC_CTX=20480 leansonic bench --weights "$MODEL" --tokens synth:2048 --spec 0 --gen 32 --repeat 2
LEANSONIC_CTX=20480 leansonic bench --weights "$MODEL" --tokens synth:16384 --spec 0 --gen 128 --repeat 2
LEANSONIC_CTX=131584 leansonic bench --weights "$MODEL" --tokens synth:131072 --spec 0 --gen 128 --repeat 2
```
