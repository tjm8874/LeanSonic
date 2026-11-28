# Lossless, directly loadable expert parameters

The original Qwen3.8-Flash-Next MLX Q4 distribution occupies 113,233,127,335 bytes across 22 safetensors shards, an index and 11 auxiliary files. Encoding Engram parameters alone reduced that to 110,833,479,038 bytes. Adding routed expert scale/bias planes reduced the physical sum of file lengths to **105,964,828,539 bytes**. The directly loadable model saves 4,868,650,499 bytes against the Engram-only package and 7,268,298,796 bytes against the original. Both candidates share the source Q4 weight codes.

The converter restored each encoded plane and compared it to its original before writing it. A separate reread compared all 544 frames (256 Engram and 288 expert) and all 3,203 ordinary tensors against the original across all 22 shards. The index refers to the transformed tensor names and its SHA-256 is `53f4ac329bd8068aabb6097fe5b795286a97a904613f433acaf66808f549873d`. The eleven auxiliary files matched the original bytes. [Independent shard reread](shard-audit.json)

The reader borrows raw BF16 data. For dictionary planes it reconstructs only the requested range. A VRAM or pinned expert layer decodes its six parameter planes (314,572,800 temporary host bytes per layer) before calling the same Z24 packing and placement code. An mmap-staged expert decodes just its selected slice. The extra cost at startup is separate from steady-state GPU inference.

## PRO 5000 72 GB, Wave 3

The directly loadable packages were compared with one unchanged binary at commit `7783d63`. Both held all 48 expert layers in VRAM. On the original four prompts the candidate matched every FP32 logit at 49 positions over 248,320 vocabulary entries and matched all four eight-token greedy continuations. Against official BF16, mean/worst KL remained 0.281135/1.469437, top-1 agreement 39/49, and PPL 52.9910 versus 28.0382 for BF16.

The streaming run order was Engram-only, combined, combined, Engram-only. Each process warmed both prompts once and measured each three times, producing 32 greedy tokens at context 256. Every generated sequence matched across all launches. Sampled peak process VRAM was 75,486,986,240 bytes for both packages. The median launch-to-ready time increased from **7.88 s to 24.73 s** (two launches per package in the host's existing filesystem cache state).

| Input length | Engram-only prefill / decode tok/s | Combined prefill / decode tok/s |
| --- | ---: | ---: |
| 16 tokens | 398.11 / 163.07 | 398.41 / 163.09 |
| 128 tokens | 493.76 / 159.07 | 494.24 / 159.09 |

Prompt throughput includes time to the first output token; decode spans the remaining 31 intervals. Steady-state speed was essentially unchanged here. The candidate trades startup latency for a smaller direct distribution. [Prompts, generated IDs, every timing sample and quality scores](pro5000.json)

## Reproduction

From the repository root, point `ORIGINAL_MLX_Q4` at the original checkpoint and `NEW_DIRECT_DIR` at a fresh output directory. Build the frame codec, repack, then compare the saved bytes.

```sh
d=docs/benchmarks/qwen38-pack
rustc --edition=2021 -O "$d/codec.rs" -o q38-codec
python3 "$d/pack_engram.py" --experts "$ORIGINAL_MLX_Q4" "$NEW_DIRECT_DIR" ./q38-codec
python3 "$d/audit_direct.py" "$ORIGINAL_MLX_Q4" "$NEW_DIRECT_DIR" ./q38-codec
```

Follow the [teacher and stream setup](../README.md) for the 49-position quality comparison and the 16/128-input, 32-output timing fixture. Use the same GPU, context 256, cubin and binary for the directly loadable packages. To exercise the mmap path on a 72 GB PRO 5000, run the compact-format candidate with `LEANSONIC_MAX_RESIDENT_LAYERS=47 LEANSONIC_SPILLED_MODE=compact` and compare it to the same configuration with the Engram-only model.
