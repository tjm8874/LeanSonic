# Replaying the three largest departures observed in two corpus runs

The same 7,763,991,646-byte 61-expert model and RTX 5060 Ti produced different full-corpus maxima with two LeanSonic builds. We extracted original WikiText test chunks 0, 1, 76, 77, 89 and 90 from the BF16 oracle, retaining both the target chunks and their predecessors. Each arm scores 1,530 positions. The archive includes a 64-bit fingerprint over every FP32 vocabulary row and its KL, NLL, target and top token. Matching fingerprints are evidence of agreement, subject to hash collisions.

| Original chunk / position | Historical CLI and historical module | Current source pair, any tested packing |
|---|---:|---:|
| 1 / 260 | 6.206404867035 | 6.171709663084 |
| 77 / 324 | 6.115262033232 | 6.654780790238 |
| 90 / 424 | 5.930834931346 | 6.071825525325 |

The historical repeat matched every fingerprint. Switching from the historical pair to the current build changed all 1,530 fingerprints and 49 top tokens. On the current build, raw, compact gate/up, and compact gate/up plus automatic down packing yielded identical row files. A diagnostic attempt to combine the old CLI with the current cubin failed on the first prefill at chunk 0 position 256 (CUDA error 700). The release comparison therefore uses matched sources. The replay does not isolate which earlier source change caused the scores to move.

## Reproduction

1. Use the [model and test reference from the full corpus run](../scores.json). Run `python3 fixture.py TEST_ORACLE subset.kld`; verify the subset SHA-256 against result.json.
2. At each CLI revision pinned in result.json, run `python3 instrument.py crates/leansonic-cli/src/eval_kld.rs` and `cargo build --release -p leansonic-cli`. Compile the indicated module revision using `nvcc -Ikernels -arch=sm_120a -std=c++17 -O3 --cubin kernels/inkling2_sm120.cu -o engine.cubin`. The historical baseline used a separately sourced module.
3. Set LEANSONIC_KV=fp16, LEANSONIC_PDL=1, LEANSONIC_PDL_ALL=1, LEANSONIC_SEQS=1, LEANSONIC_PREFILL_CHUNK=512, LEANSONIC_MXFP8_DEVICE_U4=1 and LEANSONIC_MXFP8_DIRECT_ALL=1. Set LEANSONIC_CUBIN_PATH to the intended absolute module path and DIAG_ROWS to a fresh output path. Set LEANSONIC_TERNARY_UP and LEANSONIC_TERNARY_DOWN to 0/0 for raw, 1/0 for up, or 1/auto for mixed.
4. Run `leansonic eval-kld --weights MODEL --ref subset.kld --chunks 6`. The CSV columns are subset chunk, position, target, engine top token, KL, NLL, BF16 top match and fingerprint. Original chunk mapping, SHA-256 of decompressed CSVs, binary identities, all scores and comparisons are in [result.json](result.json).
