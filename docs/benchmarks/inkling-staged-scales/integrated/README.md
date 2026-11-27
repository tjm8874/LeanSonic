# Packed scale staging on the integrated compact runtime

Host e030d0c, the [direct-coded model](https://github.com/tjm8874/LeanSonic/tree/897b812/docs/benchmarks/inkling-nvfp4-scale-split/direct), baseline cubin from main 2ba9b00 versus 7b23c40. Only the cubin differs.

The executable creates a 4096-token engine and runs batch 1 with FP16 KV, spec 0 and PDL enabled. Each process repeats 32/1024/2048-token requests four times, with a 64-token greedy continuation. The 2048 prefix repeats the 1024 input. Each of five cycles runs two processes per arm in ABBA or BAAB order. Exclude the first sweep per process from timings and retain it for exact-token and final-logit hash verification. Bootstrap intervals resample the five paired cycles.

| GPU | Prefix | Prior prefill ms | Candidate prefill ms | Paired difference, ms (95%) | Prior / candidate decode64 ms |
|---|---:|---:|---:|---:|---:|
| PRO5000 | 32 | 30.546 | 30.547 | +0.001 (-0.015, +0.017) | 290.035 / 289.963 |
| PRO5000 | 1024 | 89.953 | 89.769 | -0.184 (-0.208, -0.160) | 451.355 / 451.278 |
| PRO5000 | 2048 | 143.114 | 142.379 | -0.735 (-0.758, -0.713) | 478.288 / 478.236 |

On PRO 5000 the 1024- and 2048-token paired prefill intervals are below zero. The 32-token interval includes zero. The original long-context results use a separate fixture and are recorded in [the release-length study](../release-lengths.md).

Each GPU JSON contains timings for every request, paired intervals for both phases, all token IDs and final hashes, plus the binary, cubin and input hashes. Copy bench.rs into the host checkout as crates/leansonic-core/examples/check_scale_pair.rs. Build the release example and set LEANSONIC_CUBIN_PATH to the selected cubin for each leg. The positional arguments are the materialized model directory and the comma-separated 1024 prompt IDs. Set LEANSONIC_PDL=1, LEANSONIC_PDL_ALL=1, LEANSONIC_SEQS=1, LEANSONIC_KV=fp16, LEANSONIC_MXFP8_DEVICE_U4=1 and LEANSONIC_MXFP8_DIRECT_ALL=1. Compile kernels/inkling2_sm120.cu with nvcc -arch=sm_120a -std=c++17 -O3 --cubin.
