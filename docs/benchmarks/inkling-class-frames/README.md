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
