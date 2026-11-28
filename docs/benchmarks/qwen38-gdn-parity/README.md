# Gated DeltaNet recurrence on Blackwell

The scalar decoder, batched prefill and checkpointed verifier update the same FP32 state. The expression with two products permits different FMA contraction choices. The explicit update rounds the key times delta, then uses one fused multiply-add for the decayed state plus that rounded product.

On RTX 5070 Ti, 78 of 110 comparisons differed with both modules built from `f9fe64d`. All 110 matched when the candidate module used the explicit update and the reference remained the original scalar binary. For a single row, 110,498 of 786,432 state elements differed in the original batched path. The maximum difference was 1.19209e-7. The opposite contraction order increased that count to 221,395. This test covers deterministic finite synthetic inputs, widths 1 through 8, both output gates, outputs, final states and every intermediate rollback checkpoint.

## Reproduce with the CUDA driver API

Run from the repo root on a compute capability 12.0 GPU. Set `CUDA_HOME` to the toolkit root and select the device with `CUDA_VISIBLE_DEVICES`.

```sh
d=docs/benchmarks/qwen38-gdn-parity
git show f9fe64d:kernels/qwen38_sm120.cu > /tmp/q38-control.cu
nvcc -arch=sm_120a -O3 -std=c++17 -Ikernels --cubin /tmp/q38-control.cu -o /tmp/q38-control.cubin
nvcc -arch=sm_120a -O3 -std=c++17 -Ikernels --cubin kernels/qwen38_sm120.cu -o /tmp/q38-candidate.cubin
c++ -std=c++17 -O2 -I"$CUDA_HOME/include" "$d/gdn_parity.cpp" -L"$CUDA_HOME/lib64/stubs" -lcuda -o /tmp/q38-gdn-test
/tmp/q38-gdn-test /tmp/q38-control.cubin /tmp/q38-control.cubin > control.tsv
/tmp/q38-gdn-test /tmp/q38-control.cubin /tmp/q38-candidate.cubin --require-exact > candidate.tsv
```

Columns give batch width, gate, implementation, field, number of floats, bit mismatches and maximum absolute error. The exact gate returns nonzero for any mismatching field. Compare whole requests separately using the four checked-in prompts. Each request resets the target and drafter, generates exactly 128 tokens, and records all IDs, including tokens following EOS. Greedy and speculative paths share target placement.

```sh
mkdir -p crates/leansonic-core/examples
cp "$d/requests.rs" crates/leansonic-core/examples/gdn_requests.rs
cargo build --release -p leansonic-core --example gdn_requests
nvcc -arch=sm_120a -O3 -std=c++17 -Ikernels --cubin kernels/qwen38_mtp_sm120.cu -o /tmp/q38-mtp.cubin
export LEANSONIC_QWEN38_MTP_CUBIN_PATH=/tmp/q38-mtp.cubin
for arm in control candidate; do
  LEANSONIC_QWEN38_CUBIN_PATH="/tmp/q38-$arm.cubin" target/release/examples/gdn_requests "$MODEL" "$d/fixtures.ids" 2 > "requests-$arm.tsv"
done
```

The request columns give cycle, zero-based case, draft depth (0 is greedy), prompt seconds, total seconds, accepted drafts, total drafts and output IDs.
