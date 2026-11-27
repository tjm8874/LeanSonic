# NVFP4 gate/up decode on Blackwell

A warp can calculate the gate and up rows for a channel together. Each half warp computes the partial sums for two of the original lanes, adds them in XOR-16 order, then reduces across 8, 4, 2 and 1 lanes. Lane 0 reads the up result from lane 16. Four warps produce four channels per CTA. The kernel uses direct activation reads and no block-wide synchronization. Raw and packed expert scales share the implementation.

Weight placement changes the result. On an RTX 5070 Ti with 21 fully resident expert layers and three mixed layers, using half warps for every NVFP4 up projection **increased** 64-step decode time. Restricting the new kernel to device-resident up weights recovered the loss. The dispatcher checks the buffer backing each up matrix. Older cubins can still supply the original kernel.

| RTX 5070 Ti, same cubin | Mean warm decode, 64 tokens | Elapsed time change vs original |
|---|---:|---:|
| Original kernel everywhere | 665.09 ms | — |
| Half warp everywhere (experiment) | 711.04 ms | +6.91% |
| Half warp for VRAM weights | 662.87 ms | −0.33% |

On RTX 5060 Ti the same three arms took 1,331.32, 1,329.05 and 1,329.09 ms. Its 17 full-vocabulary output rows produced the same SHA-256 as the 5070 Ti. On 5070 Ti the official model with its speculative drafter also returned identical 200-token greedy sequences with widths 0 and 7 on Basalt's conversation input (output SHA-256 `0d4643170122373b2361b2cee3f9443527f4e5c305fcf2166c21ec3b6c4791c6`).

Each of three balanced cycles rotated the starting arm and then ran the arms in reverse order. Every visit started a fresh CLI, ran four repeats, and contributed the median of the last three. All arms produced the same 64 token IDs. The stock and placement-aware arms also produced byte-identical FP32 logits for all 200,058 vocabulary entries at 17 positions (13,603,944 bytes, SHA-256 `68538b83e50b2df69e032bc05740bd5f019a9dd279f7077905553fe792df3489`). The checkpoint and tokenizer were identical. Full timing samples, build hashes and the independent five-cycle all-layer screen are in [the measurement file](inkling-nvfp4-half-warp.json).

The device-memory microbenchmark sampled eight distinct experts out of 128 across 128 selections. For packed scales, time per launch fell from 28.15 to 26.53 µs on the RTX 5070 Ti and from 55.89 to 53.38 µs on the RTX 5060 Ti. Testing raw and packed scales covered 9,216 output floats per selection for each candidate. These timings describe VRAM weights. Use an end-to-end decode with actual placement to evaluate the dispatcher.

## Reproduce

Build a reference CLI from `144430561eda9ff87bf557dbf984a169dd353810` and a CLI from this change with `cargo build --release -p leansonic-cli`. Compile the updated CUDA kernel once. Point both binaries at that cubin so that the host dispatcher is the only variable. To reproduce the all-layer experiment, build a third CLI with the `TierBuf::Vram(_)` pattern replaced by `_` in the match on `(w13, s13.kernels.nvfp4_dp4a_up_half)`.

Set `CUDA_VISIBLE_DEVICES` to a single Blackwell GPU and `COMPACT` to the compact text package. Run from the source root. The evaluation used FP16 KV, context 8192, greedy generation, and speculation disabled. The following 16 fixed IDs exercise teacher-forced decode.

```bash
nvcc -std=c++17 -O3 -arch=sm_120a --cubin kernels/inkling2_sm120.cu -o half.cubin
export LEANSONIC_CUBIN_PATH="$PWD/half.cubin" LEANSONIC_KV=fp16
python3 - <<'PY'
from pathlib import Path
ids = Path('docs/benchmarks/inputs/inkling-readme1024.ids').read_text().strip().split(',')
Path('short.ids').write_text(','.join(ids[:32]) + '\n')
Path('force.ids').write_text('9,9,68014,58,0,58,39446,220,19,60,7,68713,25,145055,3247,13\n')
PY

for cli in "$STOCK_CLI" "$CANDIDATE_CLI"; do
    "$cli" bench --weights "$COMPACT" --tokens "@$PWD/short.ids" --spec 0 --gen 64 --repeat 4
done
for arm in stock candidate; do
    if [ "$arm" = stock ]; then cli=$STOCK_CLI; else cli=$CANDIDATE_CLI; fi
    LEANSONIC_EVAL_FORCE="$(cat force.ids)" LEANSONIC_EVAL_DUMP="$arm.f32" \
        "$cli" bench --weights "$COMPACT" --tokens "@$PWD/short.ids" --spec 0 --gen 0 --repeat 1
done
cmp stock.f32 candidate.f32 && sha256sum candidate.f32
```

## Matched end-to-end comparison

On the RTX PRO 5000 72GB, the 1,024-token README prompt took 97.55 / 95.60 ms (prefill / 16 decode forwards) on V13, 97.53 / 94.87 ms on this branch, and 102.17 / 92.00 ms on Basalt. All runs used FP16 KV, ctx 8192, greedy sampling, speculation off, and the same first 16 output IDs. Three rotated mirrored cycles per card are in the [per-process data](inkling-nvfp4-half-warp-matched.json). The PRO 4500 and 5070 Ti separately reproduced all 200 greedy IDs with speculative width 7 and width 0 on the regression prompt (SHA-256 of ID list `0d4643170122373b2361b2cee3f9443527f4e5c305fcf2166c21ec3b6c4791c6`).
