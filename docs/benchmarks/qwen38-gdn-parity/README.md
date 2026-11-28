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

## Fixed-length generation with the GDN change alone

Using NVCC 13.0.88 and driver 595.99.02, both the RTX 5070 Ti and 5060 Ti reproduced the synthetic result (78/110 failing field comparisons on control, 0/110 after the change). The exact-match test exited with status 1 for control and 0 for the candidate on the 5060 Ti. The [5070 Ti control](5070ti-control.tsv) and [candidate](5070ti-fixed.tsv) give counts per array.

On the PRO 5000 72 GB, 12/16 speculative requests matched greedy across all 128 token IDs on control; all 16/16 matched with the GDN change alone. On the 5070 Ti the counts were 6/8 and 8/8, respectively (one cycle). Every control stream already matched its greedy result through the first EOS or the 128-token limit. For the English reproducer, the original greedy and K=1/K=2 streams diverged at zero-based output index 80, following EOS at index 64. The candidate changed the **greedy** continuation at 80 to match the previous speculative result. The speculative streams retained all 128 original IDs. Rounding a batched prefill state can change later greedy outputs even though the scalar kernel passes the fixed-input reference test. All twelve request hashes for each arm agreed across both GPUs. The other three prompts preserved every ID under the change.

[PRO 5000 timings, output hashes and comparisons](pro5000-requests.json) · [RTX 5070 Ti measurements](5070ti-requests.json)

On 494 extended teacher positions the PRO 5000 returned identical FP32 vocabulary logits and 16/16 eight-token native continuations before and after the change. [Native exact gate](stock-fixed.json) · [Official BF16 comparison](bf16-fixed.json)

The extended sand-weighing prompt (second line of the extended fixture) with the GDN-only cubin diverges at output index 70 with K=2 before EOS, while the patch fixes K=1 on that input. Compare a forced history with forced_replay.rs by building the file as a leansonic-core example. It accepts MODEL, a prompt-ID text file and a space-separated 128-ID greedy reference. Each row gives verification width, rows committed per sweep, and any (index, observed ID, reference ID) differences. Running this with fixed inputs separates differences due to verification width from a change of token history in free generation.

Fixed-history replay on the PRO 4500 32 GB matched the 128-ID greedy reference for widths 1 and 2, including verifying two rows and committing only the first. Widths 3 and 4 reproduced the index-70 mismatch both with full acceptance and with one or two rows committed. When every row was accepted the probe performed no rollback, so a rejected draft is unnecessary to reproduce this error. [Exact prompt](extended-sand/prompt.ids) · [Greedy token history](extended-sand/greedy.ids) · [Ten width/commit combinations](extended-sand/results.tsv)

## Hyper-connection reduction with three or more rows

In the sand-weighing trace, with the GDN fix applied, scalar and verification widths 1 and 2 matched on all 756 comparable layer buffers, both for the first generated token and after forcing 69 greedy tokens. At width 3 the first differing buffer was the layer-0 hyper-connection down projection (107/320 differences at the first token, 117/320 after 69). The original scalar and B=2 kernels accumulate alternating 32-group tiles into the two partial sums; the B=4 kernel used the first 160 groups for one sum and the last 160 for the other. Selecting the accumulator by tile parity restores the scalar order while retaining batched weight reuse.

On the RTX 5060 Ti, the synthetic test of eight widths (1 to 64) and both injection settings found 48,016 unequal output elements across 24/32 field comparisons on control. With the change, all 32 comparisons matched bitwise, including the final partial tile for widths 3 and 5. Run the test with Python, NumPy and the NVIDIA driver, against cubins built before and after the hyper-connection change.

```sh
python3 "$d/hc_parity.py" /tmp/q38-control.cubin /tmp/q38-control.cubin
python3 "$d/hc_parity.py" /tmp/q38-control.cubin /tmp/q38-candidate.cubin --require-exact
```

## Generation with both arithmetic fixes

The production engine on the RTX PRO 5000 72 GB generated 128 tokens for each of 20 prompts (four original and sixteen additional English, Japanese, Python and arithmetic prompts). With the CUDA changes at `62fcedc`, every K=1 and K=2 output matched its same-run target-only greedy sequence, for **40/40** complete comparisons covering 5,120 speculative tokens. This includes the pre-EOS mismatch in the sand question. The 494 forced-token full-vocabulary FP32 rows and all sixteen 8-token native canaries also matched the stock native baseline. Against the official BF16 reference, mean/worst KL were 0.269670/2.499328, 406/494 top-1 predictions agreed and PPL was 15.9460 (BF16 11.3821).

[Input IDs](hc-width/all.ids) · [All 60 requests with token IDs and timings](hc-width/generation.tsv) · [Stock native comparison](hc-width/native.json) · [Official BF16 comparison](hc-width/bf16.json) · [Pre-fix layer-level trace summary](hc-width/trace-before.json)

To reproduce the 20-prompt comparison, run the example above with `hc-width/all.ids` and one cycle. For the layer trace, apply the diagnostic instrumentation in a fresh checkout, copy the tracing example, and run on an unused output directory. Use the same prompt and forced history for each cubin. The unconditional readback synchronizations are for diagnosis and do not measure inference speed.

```sh
d=docs/benchmarks/qwen38-gdn-parity
git apply "$d/hc-width/instrument.patch"
mkdir -p crates/leansonic-core/examples
cp "$d/hc-width/forced-ops.rs" crates/leansonic-core/examples/forced_ops.rs
cargo build --release -p leansonic-core --example forced_ops
mkdir "$TRACE_OUT"
LEANSONIC_QWEN38_CUBIN_PATH="$TARGET_CUBIN" target/release/examples/forced_ops "$MODEL" "$d/extended-sand/prompt.ids" "$d/extended-sand/greedy.ids" "$TRACE_OUT"
python3 "$d/hc-width/read-forced-ops.py" "$TRACE_OUT"
```

On the RTX PRO 4500 32 GB, rerunning the sand input with the combined fix gave **0/756 different buffers** for each of widths 1, 2 and 3 at both offsets. At width 3 the GDN-only baseline had 706/756 and 705/756 differences. The comparison excludes unused GDN output buffers on the QSA layers. [Post-fix trace](hc-width/trace-after.json)

The RTX 5070 Ti independently generated 128 tokens on the four additional English, Japanese, Python and arithmetic inputs. All eight speculative streams (K=1 and K=2) matched greedy. All twelve sequences including greedy matched the corresponding PRO 5000 runs (zero-based cases 5, 8, 12, 16). [Inputs](hc-width/inputs-5070.ids) · [All requests and output IDs](hc-width/generation-5070.tsv)

For the sixteen additional prompts alone, stock and GDN-only each matched their own greedy path for 30/32 speculative requests; the combined change matched for 32/32. Relative to stock greedy, the combined change preserved 15/16 full greedy streams. The sand input changed at index 70. Relative to GDN-only greedy, 14/16 stayed the same; the Japanese input also changed at index 37. These differences occurred before any EOS. The consistency gain is agreement between decoding modes with the same kernel revision. The updated greedy path can differ from previous revisions because they used different accumulation orders during batched prefill. [Stock and GDN-only raw control requests, grouped by four prompts](hc-width/heldout-controls.json.gz)

The final cubin passed the repository SASS audit. On the PRO 4500, compute-sanitizer memcheck with 65,536-byte padding reported zero errors for all four changed kernel entry points using exact-sized buffers; each corresponding planted-fault case returned the expected exit code 77. Use the qwen38_gpu exact_size_qwen38_kernels and exact_size_qwen38_red_control tests from the same checkout, with LEANSONIC_EXACT_ONLY set to each listed kernel. [Exit codes and cubin digest](hc-width/memcheck.txt)
