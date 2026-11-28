# Combined replay of PRs 254 and 255

The integration commit 5513028 combines Claude's PR 254 (79c613f) with the exponent-palette runtime (48bbbb8). The only merge conflict involved adjacent module declarations; both were retained. The integration harness emits every target and draft token. This branch is for reproducing measurements.

Use the [same native Q4 checkpoint](../README.md), fixture IDs and cubins on an idle 72 GB card. In one scratch directory, create a checkout of this integration branch at `exp4-mtp-probe`, place the checkpoint at `direct-engram` and copy the fixed `fixtures.ids` into the scratch root. Build in two separate target directories. The Python harness uses Python 3 and the NVML bindings and expects `CUDA_VISIBLE_DEVICES` to be the reserved GPU UUID.

```sh
cd exp4-mtp-probe
CARGO_TARGET_DIR="$PWD/stock" cargo test --release -p leansonic-core --lib --no-run
CARGO_TARGET_DIR="$PWD/palette" cargo test --release -p leansonic-core --lib --features q38-exponent-palette --no-run
nvcc -arch=sm_120a -O3 -std=c++17 -Ikernels --cubin kernels/qwen38_sm120.cu -o ../wave1.cubin
nvcc -arch=sm_120a -O3 -std=c++17 -Ikernels -DLEANSONIC_Q38_EXPONENT_PALETTE --cubin kernels/qwen38_sm120.cu -o ../exp4-word.cubin
nvcc -arch=sm_120a -O3 -std=c++17 -Ikernels --cubin kernels/qwen38_mtp_sm120.cu -o ../exp4-mtp.cubin
cd ..
python3 exp4-mtp-probe/docs/benchmarks/qwen38-pack/exponent-palette/mtp_replay.py
```

The Japanese input contains 16 IDs and the target generates 160 greedy tokens with default MTP position alignment. Each launch uses Claude's `mtp_greedy_acceptance` harness and preserves its 512 MiB MoE-placement reserve. Runs alternate stock, palette, palette, stock. The harness samples process VRAM every 25 ms and verifies that all draft and target IDs match across processes. Target forward time and drafter step time are measured separately. End-to-end speculative decode additionally requires a batched verification and rollback path.

## PRO 5000 72 GB results

| Target expert layout | MTP MoE placement | Target forward (ms/position) | MTP step (ms/position) | Accepted | Sampled process peak (GiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| Stock Z24 | Pinned host | 6.29 | 1.79 | 36/160 | 70.32 |
| Four-bit exponent palettes | VRAM | 6.50 | 0.58 | 36/160 | 70.00 |

Both runs per arm agreed at the precision printed by the MTP harness. Enabling the target palette freed enough VRAM for the unmodified MTP MoE, whose step became 3.09× faster on this prompt. The target forward took 0.21 ms longer. The 22.5% acceptance rate includes 160 generated positions. The target and draft sequences matched across all four processes. [Full traces and measurements](mtp-results.json)
