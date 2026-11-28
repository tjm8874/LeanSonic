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

## Output length and stable slot assignment

After retaining the physical slot for overlapping experts, four new launches alternated fixed, sixteen slots, sixteen slots and fixed. Each resident process ran all four inputs at 8, 32 and 128 generated tokens. The 192 complete generations, including warmups, reproduced each input's identical greedy prefix across both layouts and all requested lengths. For each input, the 128-token continuation extends the matching shorter ones. The sixteen-slot teacher also matched stock Q4 at all 65 positions and for all four canaries.

Median time from request to DONE in ms, **fixed → sixteen slots** (six timed samples per cell):

| Input | 8 output tokens | 32 output tokens | 128 output tokens |
| --- | ---: | ---: | ---: |
| English | 232 → 267 | 619 → 570 | 3,625 → 3,474 |
| Japanese | 391 → 463 | 849 → 802 | 2,614 → 2,268 |
| Code | 350 → 415 | 1,144 → 1,095 | 4,582 → 4,289 |
| Arithmetic | 284 → 319 | 886 → 867 | 3,525 → 3,352 |

Eight-token requests became slower on every input. With 32 and 128 outputs, decode savings outweighed the first-token penalty on all four. Keep adaptation optional. The data do not establish a single break-even length across workloads. [Peak VRAM, TTFT, prefill/decode throughput, transfers and sample-level results](lengths.json)

## Matched public-engine requests

Basalt v0.5.7 used the same PRO 4500, 256-token context, prompt IDs and requested output limits. Its UD-Q4_K_XL quantization differs from LeanSonic MLX Q4. One resident launch per Basalt arm warmed all cases once and timed three cycles. Native results below use fixed slots for the 8-token request and sixteen adaptive slots for the 32- and 128-token requests. These are separate measured settings. The engine did not automatically choose based on the requested length.

Median request-to-DONE time in ms, **LeanSonic / Basalt adaptive / Basalt fixed**:

| Input | 8 output tokens | 32 output tokens | 128 output tokens |
| --- | ---: | ---: | ---: |
| English | 232 / 426 / 462 | 570 / 591 / 820 | 3,474 / — / — |
| Japanese | 391 / 644 / 750 | 802 / 649 / 1,201 | 2,268 / 1,823 / 2,662 |
| Code | 350 / 244 / 370 | 1,095 / 522 / 1,035 | 4,289 / 2,122 / 3,673 |
| Arithmetic | 284 / 273 / 522 | 867 / 567 / 1,006 | 3,352 / 2,065 / 3,216 |

For the English 128-token request, Basalt adaptive returned 69, 70 and 66 tokens across the three measured cycles. Fixed returned 69 each time. Report these as early completions; their DONE times cannot rank equal-length generation. Both Basalt arms reached every other requested limit. Native peak VRAM was 32,084,328,448 bytes. Basalt peaked at 32,847,691,776 bytes adaptive and 32,845,594,624 bytes fixed. The fixed-output timings use resident processes without prompt reuse. The [common BF16 subset](../wave3/README.md#matched-bf16-quality-gates-on-the-held-out-prompts) still favors Basalt on KL. [Public engine samples and output IDs](public-lengths.json)

Two additional native four-slot launches followed the sixteen-slot study using the stable slot assignment. Across English, Japanese, code and arithmetic, medians were respectively 248/407/360/298 ms at eight output tokens, 570/805/1,074/869 ms at 32 and 3,489/2,431/4,351/3,435 ms at 128. All 96 additional greedy runs matched the tested prefixes. [Native results and transfer samples](lengths.json)

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
