# Scratch-free indexed lookup on upstream P0-A

Compare upstream e9daee8 (#157 and #171) with PR #160 at a1bf5e3. Each revision uses its own release executable and its matching sm_120a cubin. Both use the same 13,445,948,726-byte distribution (shard SHA-256 9cc8302c5b382b9d0b1a9dcc60a26aeb115ed38833f2d838f6c815c4660e939d). Eligible expert lookups are scratch-free on the candidate. Both passed the repository SASS audit.

| GPU | Input | Prefill ms (main → PR) | Paired change, 95% interval | Decode 64 ms (main → PR) | Paired change, 95% interval |
|---|---:|---:|---:|---:|---:|
| PRO4500 | 32 | 45.310 → 36.059 | -9.251 [-9.260, -9.241] | 400.105 → 373.876 | -26.229 [-26.390, -26.067] |
| PRO4500 | 1024 | 194.886 → 176.794 | -18.092 [-18.125, -18.067] | 542.346 → 516.154 | -26.192 [-26.324, -26.031] |
| PRO4500 | 2048 | 391.352 → 354.918 | -36.434 [-36.522, -36.357] | 566.721 → 540.677 | -26.044 [-26.167, -25.865] |
| 5060Ti | 32 | 87.737 → 70.328 | -17.410 [-17.431, -17.385] | 698.967 → 642.957 | -56.010 [-56.099, -55.937] |
| 5060Ti | 1024 | 400.783 → 366.532 | -34.251 [-34.317, -34.144] | 833.109 → 777.094 | -56.015 [-56.408, -55.623] |
| 5060Ti | 2048 | 816.064 → 747.432 | -68.632 [-68.769, -68.445] | 855.724 → 799.740 | -55.984 [-56.402, -55.566] |

Each GPU ran five paired ABBA/BAAB cycles (ten processes per arm). Four passes per process exercised all three input lengths. Phase times exclude the first pass. Every request reproduced the 64 greedy IDs, next ID and final digest of 200,058 finite FP32 logits for its prompt across both revisions. All 24 expert layers remained resident. The loader lines in each JSON are initialization snapshots.

Build the driver from this directory as a leansonic-core example on each revision and compile kernels/inkling2_sm120.cu with nvcc -arch=sm_120a -std=c++17 -O3 --cubin. Select the corresponding cubin with LEANSONIC_CUBIN_PATH. Use FP16 KV, PDL and PDL_ALL on, MXFP8_DEVICE_U4=1, MXFP8_DIRECT_ALL=1, prefill chunk 512, one sequence, context 4,096, spec 0. Read the [fixed 1,024 README IDs and model recipe](https://github.com/tjm8874/LeanSonic/tree/cc62564/docs/benchmarks/inkling-nvfp4-trim-context). The driver repeats the input for the 2,048-token case. Each GPU JSON contains the binaries, input SHA-256, loader output, raw timing, token IDs, hashes and cycle intervals.

The PRO 4500 four-sequence check matched solo vs interleaved execution and all four hashes across e9daee8/a1bf5e3 with spec 0 and 3. Run each matched CLI/cubin with `multiseq --weights "$MODEL" --tokens "@$IDS" --seqs 4 --round 4 --gen 32 --spec 0`, then repeat with spec 3, context 2,048 and the settings above. [Outputs and hashes](multiseq.json).
