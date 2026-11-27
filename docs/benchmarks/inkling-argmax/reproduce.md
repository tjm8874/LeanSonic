# Reproduce the greedy vocabulary comparison

Use two clean checkouts, `$BASE` at `5f1c7ff` and `$TRIAL` at `6544318`. Set `$HERE` to this benchmark directory, `$CUDA` to the CUDA installation, and `$MODEL` to the canonical package. Pin one otherwise idle GPU by UUID. Build the candidate cubin once and use it for both executables. The control host calls the original one-block kernel from the same module.

```bash
nvcc -arch=sm_120a -std=c++17 -O3 --cubin "$TRIAL/kernels/inkling2_sm120.cu" -o trial.cubin
c++ -std=c++17 -O2 "$HERE/edges.cpp" -I"$CUDA/include" -lcuda -o edges
./edges trial.cubin
for tree in "$BASE" "$TRIAL"; do
  cp "$HERE/bench.rs" "$tree/crates/leansonic-core/examples/inkling_argmax_bench.rs"
  (cd "$tree" && cargo build --release -p leansonic-core --example inkling_argmax_bench)
done
export LEANSONIC_CUBIN_PATH="$PWD/trial.cubin" LEANSONIC_PDL=1 LEANSONIC_KV=fp16 LEANSONIC_SEQS=1
"$BASE/target/release/examples/inkling_argmax_bench" "$MODEL" "$BASE/docs/benchmarks/inputs/inkling-readme1024.ids"
"$TRIAL/target/release/examples/inkling_argmax_bench" "$MODEL" "$BASE/docs/benchmarks/inputs/inkling-readme1024.ids"
```

Each line contains a pass number, input length, prefill and 64-token decode milliseconds, terminal prediction, a hash of the 200,058 final FP32 logits, and every output token. Repeat the executable in the documented five mirrored cycles, validating all output IDs and hashes for each prefix before comparing times.
