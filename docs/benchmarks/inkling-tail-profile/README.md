# GPU work in the last split prefill batch

RTX PRO 4500 Blackwell, Nsight Systems 2025.3.2, #117 scale-neighbor weights, a 64-token prefill workspace, FP16 KV, PDL on and one sequence. The measurements cover only the final batch after a warm request and a reset. The preceding 64 tokens, model load, host-side token selection and subsequent decoding are outside the capture. Each cell is one trace. Use the interleaved end-to-end benchmarks in #120 for latency comparisons.

| Input / suffix | Dispatch | Kernel span, ms | Launches | NVFP4 expert gate/up, ms | NVFP4 expert down, ms |
|---|---|---:|---:|---:|---:|
| 66 / 2 | original | 10.867 | 1070 | 3.140 | 1.575 |
| 66 / 2 | Tensor Core tail | 16.258 | 1132 | 4.713 | 1.690 |
| 95 / 31 | original | 34.661 | 1092 | 13.508 | 6.557 |
| 95 / 31 | Tensor Core tail | 40.436 | 1132 | 20.772 | 6.217 |

Both variants dispatched 44 gate/up and 44 down kernels. No memcpy activity was recorded inside these four capture ranges. The large gate/up increase at 31 rows accounts for more than the overall span increase because other kernels, including down, became faster. These are GPU activity durations, not memory-bandwidth or stall-counter measurements. The control uses grouped DP4A; the candidate preserves the Tensor Core arithmetic of wide prefill. [Kernel totals and copy counts](traces.json).

To reproduce, place the original checkout at `prior/` (18141f3) and the candidate at `candidate/` (4a0f5d3) in a new directory. Copy these scripts there. In each checkout install `profile.rs` as `crates/leansonic-core/examples/inkling_profile_tail.rs`. Run `python3 instrument.py` once and build both examples in release mode. Use the #117 model and the repository's `inkling-readme1024.ids` (SHA-256 e93129d009d923541cd0b1ee39a6f121c11e000638ceb868258665ec974ed843). The shared cubin SHA-256 was 93463fd100ab8fcd2b397137c7861f4b65e8cef0c6dcc20fbfe4150e9b6548d2.

```sh
export LEANSONIC_PDL=1 LEANSONIC_KV=fp16 LEANSONIC_PREFILL_CHUNK=64 LEANSONIC_SEQS=1
export LEANSONIC_CUBIN_PATH="$CUBIN"
for arm in prior candidate; do
  for n in 66 95; do
    nsys profile --trace=cuda --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
      --capture-range-end=stop --output="$arm-$n" \
      "$arm/target/release/examples/inkling_profile_tail" "$MODEL" "$IDS" "$n"
    nsys export --type sqlite --output "$arm-$n.sqlite" "$arm-$n.nsys-rep"
  done
done
python3 analyze.py
```
