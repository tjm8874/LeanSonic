# One byte permutation per eight resident ternary weights

Pair selectors 0–7 select the E2M1 byte pairs 00, 07, 0F, 70, 77, 7F, F0 and F7. Selector 15 sign-extends byte 7 (F7) to FF. This fits all nine legal pairs. A packed byte still holds four weights. The loader validates the original nibble pairs; the new expansion symbol distinguishes this device layout from the previous one. The model on disk is unchanged. [NVIDIA PTX prmt semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-prmt)

On RTX 5060 Ti, compared with the two-permutation variant in #203, tile latency fell 1.35–1.74% across the four measured gate/up cases.

| Workload | Tile | Two permutations, ms | Pair selectors, ms | Change |
|---|---:|---:|---:|---:|
| Ragged (1,376 tokens) | 128 | 0.218597 | 0.214792 | −1.74% |
| Ragged | 160 | 0.218054 | 0.215115 | −1.35% |
| Dense (16,384 tokens) | 128 | 2.547106 | 2.504550 | −1.67% |
| Dense | 160 | 2.828786 | 2.786679 | −1.49% |

The isolated microbenchmark used the existing #203 fixture and changes only the staged Tensor Core expansion and corresponding input layout. Five alternating four-leg cycles, 24 kernel launches per tile per leg. Every compared FP32 output and guard matched the full-width kernel. [Raw observations and paired cycle intervals](result.json)

To reproduce the microbenchmark, use the #203 tile harness with its original ternary.bin for the two-permutation module. Generate a new ternary.bin with `python3 encode.py OLD_TERNARY ORIGINAL_NIBBLES NEW_TERNARY`. The converter checks all 6,561 legal selector words and verifies all 339,738,624 original weight bytes. Substitute the single prmt.b32 expansion from this change into the #203 module for the second arm. Keep both tiles.cu binaries and all other inputs identical.

The integrated engine at 9f42e06 also changes DP4A decode and scratch expansion. On RTX PRO 4500, the 61-expert model with compact gate/up and automatic down matched its 3364f6d control for eight full FP32 vocabulary rows and four 64-token greedy continuations, with prefix lengths 32, 1024, 3072, 4096. The new CLI rejected the previous CUDA module when compact loading was requested. Reuse the #205 fixture and prompt, the included smoke.rs as a leansonic-core example, FP16 KV, one sequence, PDL/PDL_ALL, device U4 and direct MXFP8, prefill width 4096. All binaries and outputs are pinned in [checks.json](checks.json). End-to-end comparison on RTX 5070 Ti is in progress.
