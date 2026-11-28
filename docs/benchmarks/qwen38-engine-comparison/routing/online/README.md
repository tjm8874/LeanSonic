# Updating expert placement from the current prompt

This opt-in experiment changes the contents of a bounded number of existing GPU slots after observing the first batched prefill chunk (up to 64 prompt rows). It uses only the current prompt and a calibration profile. The worker keeps its current slot contents across request resets. The model weights and the number of GPU expert records stay fixed.

The four calibration decodes determine priority within the same 9,728 records used by the fixed [Wave 3 profile](../wave3/README.md). A cold expert needs at least two router selections in the observed prompt chunk. Only configured low-priority experts absent from that chunk are eligible for eviction. Serial single-token prefill leaves placements as they are. Each mutable slot backs up its three GPU projections in immutable pinned host memory. A layer temporarily publishes host pointers while its slots change, then publishes the final device table. The stable variant also leaves a wanted expert in its existing physical slot when its priority changes.

## First online experiment

RTX PRO 4500 Blackwell 32 GB, Qwen3.8-Flash-Next, directly loadable MLX Q4, context 256, identical prompt IDs and 32 greedy tokens. Six fresh launches alternated slot limits 0, 4, 16, 16, 4, 0. Each warmed up on all four prompts and then repeated that order three times. Each cell reports the median time from request to DONE in ms, including route observation and all PCIe transfers. The checksum covers 65 full-vocabulary FP32 distributions and the test also compares four 8-token continuations. No prompt cache was reused.

| Held-out input | Fixed | Four slots per layer | Sixteen slots per layer |
| --- | ---: | ---: | ---: |
| English, 19 tokens | 618.30 | 571.38 | 579.29 |
| Japanese, 25 tokens | 849.00 | 799.83 | 816.78 |
| Code, 6 tokens | 1143.59 | 1073.85 | 1121.73 |
| Arithmetic, 19 tokens | 885.74 | 869.31 | 889.26 |

Four slots improved all four total times but added 17–27 ms to median time to first token. The per-case median update copied 499–540 MB across PCIe and took 24–26 ms. Sixteen slots increased decode throughput further and made the arithmetic request slower overall than fixed placement. FP32 logits at all 65 positions and all four canaries matched the original Q4 engine. The tested layouts still have the [stock quantization's measured gap to BF16](../wave3/README.md#matched-bf16-quality-gates-on-the-held-out-prompts).

All three arms peaked at 32,084,328,448 bytes of sampled process VRAM. Four or sixteen slots per layer reserve an additional 560,332,800 or 2,241,331,200 bytes of pinned **host** storage. The direct checkpoint occupies 110,833,479,038 bytes. The optional ordered profile adds 36,500 bytes. [Every sample, token IDs, peak and hash](initial32.json)

## Reproduce

Use the mixed cubin build and GPU selection from the [fixed-placement experiment](../wave3/README.md#reproduction). Commit `5e36e63` produced the first online result; `65968d2` adds overlap-preserving slot assignment. The extended stream harness supports multiple requested output lengths in one resident process. Use the benchmark scripts in this directory's revision.

```sh
d=docs/benchmarks/qwen38-engine-comparison
online=$d/routing/online
mkdir -p crates/leansonic-core/examples
cp "$d/q38_stream.rs" crates/leansonic-core/examples/
cp docs/benchmarks/qwen38-pack/teacher.rs crates/leansonic-core/examples/q38_teacher.rs
cargo build --release -p leansonic-core --example q38_teacher --example q38_stream
export Q38_EXPERT_PROFILE=$PWD/$online/profile-ranked-19layers.txt
export LEANSONIC_QWEN38_CUBIN_PATH=$PWD/mixed.cubin Q38_ADAPT_STATS=1
export Q38_ADAPT_SLOTS=16
tail -n 4 "$d/routing/route-fixtures.ids" > heldout.ids
Q38_TEACHER_ADAPT_PREFILL=1 target/release/examples/q38_teacher "$MODEL" heldout.ids adaptive-teacher
python3 docs/benchmarks/qwen38-pack/score.py "$STOCK_TEACHER" adaptive-teacher --require-exact > parity.json
for run in 1-0 2-16 3-16 4-0; do
  export Q38_ADAPT_SLOTS=${run#*-}
  python3 "$d/compare_stream.py" --held-out-routing --generate 8 --generate 32 --generate 128 "$d/routing/route-fixtures.ids" "$run.log" "$run.json" target/release/examples/q38_stream "$MODEL" 256
done
python3 "$online/summarize.py" . > medians.json
```

For the first six launches, specify only `--generate 32` and use their order above. The ranked profile can be regenerated from the four calibration route traces with `rank_profile.py TRACE_DIR FIXED_PROFILE NEW_PATH`. A fixed stock teacher dump on the identical four held-out prompts supplies `$STOCK_TEACHER`. All benchmark outputs should use a fresh directory. The normal engine default is zero adaptive slots.
