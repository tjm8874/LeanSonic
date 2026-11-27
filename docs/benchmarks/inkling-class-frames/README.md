# Two entropy frames for mixed expert classes

The 571-expert Inkling-2-Mini distribution gives two groups of experts different weight precision within the same projection. Put the two groups in separate four-bank streams when the serialized pair, including its mask and lengths, is smaller. Out of 36 eligible projections, 24 select the new frame. The remaining tensors retain their original representation. The shard and tokenizer shrink from 11,487,467,790 to 11,483,431,150 bytes. The encoder checked all 9,342,812,160 decoded weight bytes across the 36 alternatives against the parent. [Per-projection trials](fit.json) · [Final model hashes](receipt.json).

Decoder 38e8d57 scatters each decoded chunk directly into the final expert buffer. With the initially implemented decoder ca0fff1, an entire class was materialized before scattering it. We compared them in fresh four-core CPU processes on the gate/up and down projections with the largest physical savings in each category. Five mirrored cycles gave ten observations per decoder per projection. Every output SHA-256 matched the original single-frame control.

| Projection | Initial decode, ms | Chunk scatter, ms | Initial peak RSS, KiB | Chunk scatter peak RSS, KiB |
| --- | ---: | ---: | ---: | ---: |
| Gate/up | 508.853 | 444.477 | 855,284 | 661,176 |
| Down | 258.683 | 224.373 | 441,112 | 341,316 |

Values are medians. Decode time excludes reading input files, decoding the shared scales and writing the SHA stream. RSS includes those buffers. The result measures the host reader for two projections. Whole-model startup and GPU inference have a separate replay. [All process timings, peak RSS and hashes](host/verified.json).

## Reproduction

Use the [571-expert class-bank parent and the original selected expert IDs](https://github.com/tjm8874/LeanSonic/tree/4c28635/docs/benchmarks/inkling-embedding-lut/magnitude/44-levels/ranked-experts). Build split.rs as a leansonic-core release example and copy the executable to a fresh working directory as banks. Copy fit.py and assemble.py there. Run the following with the original ranked-experts/expert.json.

```sh
python3 fit.py "$PARENT_MODEL" "$EXPERT_JSON" frames
python3 assemble.py "$PARENT_MODEL" frames model
```

Verify the hashes above. Use engine 38e8d57 to load the new weights. The 14 focused core tests cover interleaved expert scatter, truncated frames, invalid lengths and empty classes.

For the CPU comparison, copy the host scripts to a fresh timing directory. The build helper extracts 38e8d57 into a temporary directory and builds both readers into one executable. The source repository must contain both measured commits.

```sh
python3 build.py "$ENGINE_REPO" ./bench
python3 prepare.py "$PARENT_MODEL" "$NEW_MODEL" "$FIT_JSON"
bash time.sh
```

## Full-model startup with the initial reader

On RTX PRO 4500, the initial class-frame reader ca0fff1 ran original/split/split/original against the same six BF16 diagnostic chunks. All four 1,530-row full-vocabulary fingerprint files and score aggregates matched. For the original package, the loader reported 9.39 and 11.36 seconds; for the split package it reported 17.22 and 12.62 seconds. All four runs placed 24/24 layers on the device. Saving 4,036,640 distribution bytes added 4.55 seconds of startup on average in this single mirrored cycle. The streaming reader was introduced to lower the cost; the CPU results above cover two projections, and its full-model startup is measured separately. [Four full-model runs and hashes](pro4500-loader.json).

## Whole-model streaming reader on RTX 5060 Ti

With both packed device projections enabled at integration commit a1592f3, ten fresh processes per package ran in five mirrored cycles. Each used an 8,192-token context, one sequence, FP16 KV, the same 1,024 prompt IDs and 32 teacher-forced steps. Throughput uses four warm requests per process. Both packages placed 24/24 layers on the GPU with 16 packed projections, and all predictions matched for all 100 requests.

| Package | Prefill tok/s | Decode tok/s | Mean loader seconds | Peak GPU bytes | Median peak host KiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 3,952.25 | 95.421 | 23.264 | 14,686,355,456 | 6,656,272 |
| Split class frames | 3,952.13 | 95.418 | 23.412 | 14,686,355,456 | 6,656,288 |

The package saves 4,036,640 bytes, with 0.148 seconds added mean startup on this card (paired-cycle bootstrap 95% interval +0.117 to +0.179 seconds). The measured steady-state difference was below 0.004% in each phase. The six direct-prefill chunks (1,530 positions) and the one generic-prefill chunk (255 positions) produced byte-identical score and full-vocabulary fingerprint rows across the packages. The CPU-only comparison above covers the initial and streaming readers; this GPU comparison uses the streaming reader for both packages.

Use the [benchmark inputs and harness](https://github.com/tjm8874/LeanSonic/tree/cbd160a/docs/benchmarks/inkling-mixed-even-down) and pass each model to the same executable built from a1592f3. [Per-process measurements, model manifests, executable and input digests](5060-streaming.json). The [scale-bin refit](scale-bins/README.md) is a subsequent lossless size experiment.
