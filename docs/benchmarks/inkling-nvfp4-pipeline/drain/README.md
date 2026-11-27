# Omitting redundant pipeline drain barriers

Change only the CUBIN (a1bf5e3 versus 1cdc654). Both use the same indexed-weight host executable and [13,445,948,726-byte capped compact model recipe](https://github.com/tjm8874/LeanSonic/tree/58c993f/docs/benchmarks/inkling-codebooks/nvfp4-budget/capped).

The executable creates a 4096-token engine and runs batch 1 with FP16 KV, spec 0 and PDL enabled. Each process repeats 32/1024/2048-token requests four times, with a 64-token greedy continuation. The 2048 prefix repeats the 1024 input. Each of five cycles runs two processes per arm in ABBA or BAAB order. Exclude the first sweep per process from timings and retain it for exact-token and final-logit hash verification. Bootstrap intervals resample the five paired cycles.

| GPU | Prefix | Prior prefill ms | Candidate prefill ms | Paired difference, ms (95%) | Prior / candidate decode64 ms |
|---|---:|---:|---:|---:|---:|
| 5070Ti | 32 | 35.246 | 35.265 | +0.019 (+0.017, +0.022) | 363.403 / 363.350 |
| 5070Ti | 1024 | 122.851 | 122.855 | +0.004 (-0.005, +0.012) | 497.397 / 497.447 |
| 5070Ti | 2048 | 258.742 | 258.300 | -0.442 (-3.052, +1.787) | 521.903 / 522.047 |

The RTX 5070 Ti interval includes zero at 1024 tokens. Do not enable this change as a general prefill optimization.

Each GPU JSON contains timings for every request, paired intervals for both phases, all token IDs and final hashes, plus the binary, cubin and input hashes. Copy bench.rs into the host checkout as crates/leansonic-core/examples/check_scale_pair.rs. Build the release example and set LEANSONIC_CUBIN_PATH to the selected cubin for each leg. The positional arguments are the materialized model directory and the comma-separated 1024 prompt IDs. Set LEANSONIC_PDL=1, LEANSONIC_PDL_ALL=1, LEANSONIC_SEQS=1, LEANSONIC_KV=fp16, LEANSONIC_MXFP8_DEVICE_U4=1 and LEANSONIC_MXFP8_DIRECT_ALL=1. Compile kernels/inkling2_sm120.cu with nvcc -arch=sm_120a -std=c++17 -O3 --cubin.
