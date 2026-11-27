# Single-token controller inner products

Current main already launches the 128-row BF16 router with 128 CTAs. Its controller down kernel uses 256 threads to compute one output row. The first part of the dot product does not require the RMS scale. The change computes each adjacent weight/input pair before the shared RMS barriers. It applies the scale to those pairs in the original sequence and keeps both warp reductions. The Lean proof checks the accumulator order for arbitrary pair and step functions. The sm_120a cubin uses 37 registers, 1,104 bytes of shared memory and zero stack; the compiled SASS audit passes.

Mean warm 64-step decode times below use the same main host binary with baseline and candidate cubins. Five balanced ABBA/BAAB cycles, ten fresh processes per arm, four requests per process. The statistic within each process is the median of its last three requests. Input is the first 32 README token IDs, greedy, FP16 KV, ctx 8192, no speculation. All 64 IDs matched and a forced 17-position dump had identical FP32 logits across the entire vocabulary.

| GPU | Main (ms) | Early products (ms) | Time saved | Paired bootstrap 95% interval |
|---|---:|---:|---:|---:|
| 5070Ti | 599.73 | 591.45 | 1.38% | 1.35% to 1.42% |
| 5060Ti | 1330.16 | 1323.40 | 0.51% | 0.50% to 0.51% |
| PRO4500 | 455.55 | 447.47 | 1.77% | 1.73% to 1.80% |

For reproduction build the baseline and candidate kernels separately using `nvcc -std=c++17 -O3 -arch=sm_120a --cubin kernels/inkling2_sm120.cu`. Build `cargo build --release -p leansonic-cli`, then run the same binary with each `LEANSONIC_CUBIN_PATH`, setting `LEANSONIC_KV=fp16` and `LEANSONIC_CTX=8192`. Use `bench --weights "$WEIGHTS" --tokens "@$IDS32" --gen 64 --spec 0 --repeat 4` and the input fingerprint in the [per-process evidence](inkling-controller-early.json). Run `lake build` in spec and the repository SASS audit on the candidate cubin. Two-row RMS sharing was mixed across actual shapes and explicit weight prefetch alone slowed down both GeForce cards in the initial microbenchmark; neither was adopted.
