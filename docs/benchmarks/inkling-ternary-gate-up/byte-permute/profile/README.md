# Warm 2,048-token prefill on RTX PRO 4500

Nsight Systems captured the second prefill after resetting the engine. Each arm loaded the same 7,718,950,382-byte ternary package. The 1,024-token README fixture was repeated twice. We used one sequence, context 4,096, FP16 KV, PDL_ALL and direct device codebooks. Two captures per arm were interleaved in forward/reverse order. Down weights used two-bit storage in every arm. All six captures returned token 2 and issued 1,324 CUDA kernels each, including 44 grouped gate/up launches and 44 grouped down launches.

| Gate/up storage and module | GPU kernel span (ms) | Gate/up cumulative (ms) | Down cumulative (ms) |
|---|---:|---:|---:|
| Four-bit, original | 205.667 | 55.623 | 30.821 |
| Two-bit, original | 206.734 | 57.059 | 30.554 |
| Two-bit, byte permutes | 205.775 | 56.618 | 30.111 |

The original two-bit gate/up kernel took longer than four-bit at this length. The byte permutations recovered part of the gate/up cost and also reduced grouped down time. GPU spans exclude work before the first launch and after kernel completion. These six profiles locate the cost. Use the [unprofiled inference pairs](../inference/README.md) to compare user-visible latency.

To reproduce, copy profile.rs into leansonic-core/examples, link libcuda and build the release example. Use the module hashes and fixture in captures.json. Run nsys profile --sample=none --cpuctxsw=none --trace=cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop on the example with model directory and fixture path, export SQLite, then run summarize.py on each export.
