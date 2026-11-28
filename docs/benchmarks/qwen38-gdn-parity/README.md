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

Compare both complete TSV files using the release checkpoint's EOS IDs. The report distinguishes full-stream differences, agreement through the first EOS (inclusive), unchanged greedy control and within-arm repeatability.

```sh
python3 "$d/compare_requests.py" requests-control.tsv requests-candidate.tsv --stop-id 248046 --stop-id 248044
```

## Fixed-length generation after the rounding change

Using NVCC 13.0.88 and driver 595.99.02, both the RTX 5070 Ti and 5060 Ti reproduced the synthetic result (78/110 failing field comparisons on control, 0/110 after the change). The exact-match test exited with status 1 for control and 0 for the candidate on the 5060 Ti. The [5070 Ti control](5070ti-control.tsv) and [candidate](5070ti-fixed.tsv) give counts per array.

On the PRO 5000 72 GB, 12/16 speculative requests matched greedy across all 128 token IDs on control; all 16/16 matched with this patch. On the 5070 Ti the counts were 6/8 and 8/8, respectively (one cycle). Every control stream already matched its greedy result through the first EOS or the 128-token limit. For the English reproducer, the original greedy and K=1/K=2 streams diverged at zero-based output index 80, following EOS at index 64. The candidate changed the **greedy** continuation at 80 to match the previous speculative result. The speculative streams retained all 128 original IDs. Rounding a batched prefill state can change later greedy outputs even though the scalar kernel passes the fixed-input reference test. All twelve request hashes for each arm agreed across both GPUs. The other three prompts preserved every ID under the change.

[PRO 5000 timings, output hashes and comparisons](pro5000-requests.json) · [RTX 5070 Ti measurements](5070ti-requests.json)

On 494 extended teacher positions the PRO 5000 returned identical FP32 vocabulary logits and 16/16 eight-token native continuations before and after the change. [Native exact gate](stock-fixed.json) · [Official BF16 comparison](bf16-fixed.json)

The extended sand-weighing prompt (second line of the extended fixture) still diverges at output index 70 with K=2 before EOS, while the patch fixes K=1 on that input. Compare a forced history with forced_replay.rs by building the file as a leansonic-core example. It accepts MODEL, a prompt-ID text file and a space-separated 128-ID greedy reference. Each row gives verification width, rows committed per sweep, and any (index, observed ID, reference ID) differences. Running this with fixed inputs separates differences due to verification width from a change of token history in free generation.

Fixed-history replay on the PRO 4500 32 GB matched the 128-ID greedy reference for widths 1 and 2, including verifying two rows and committing only the first. Widths 3 and 4 reproduced the index-70 mismatch both with full acceptance and with one or two rows committed. When every row was accepted the probe performed no rollback, so a rejected draft is unnecessary to reproduce this error. [Exact prompt](extended-sand/prompt.ids) · [Greedy token history](extended-sand/greedy.ids) · [Ten width/commit combinations](extended-sand/results.tsv)

## Hyper-connection reduction with three or more rows

In the sand-weighing trace, with the GDN fix applied, scalar and verification widths 1 and 2 matched on all 756 comparable layer buffers, both for the first generated token and after forcing 69 greedy tokens. At width 3 the first differing buffer was the layer-0 hyper-connection down projection (107/320 differences at the first token, 117/320 after 69). The original scalar and B=2 kernels accumulate alternating 32-group tiles into the two partial sums; the B=4 kernel used the first 160 groups for one sum and the last 160 for the other. Selecting the accumulator by tile parity restores the scalar order while retaining batched weight reuse.

On the RTX 5060 Ti, the synthetic test of eight widths (1 to 64) and both injection settings found 48,016 unequal output elements across 24/32 field comparisons on control. With the change, all 32 comparisons matched bitwise, including the final partial tile for widths 3 and 5. Run the test with Python, NumPy and the NVIDIA driver, against cubins built before and after the hyper-connection change.

```sh
python3 "$d/hc_parity.py" /tmp/q38-control.cubin /tmp/q38-control.cubin
python3 "$d/hc_parity.py" /tmp/q38-control.cubin /tmp/q38-candidate.cubin --require-exact
```
