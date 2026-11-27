# Basalt greedy history versus the canonical LeanSonic package

Each GPU first ran Basalt greedily on the 1,024-token README fixture and saved its 33 emitted IDs. Basalt’s first emission comes from prefill. Each subsequent trial on that GPU had to reproduce those exact IDs. LeanSonic then consumed the first 32 saved IDs, including any steps where its own prediction differed. This gives both engines exactly the same 32 decode-forward inputs at the same positions. Context 8,192, one sequence, FP16 KV, no speculation, temperature zero. Each row is the mean of ten fresh-process first requests, in five mirrored ABBA/BAAB cycles. The separate original-IQ4 comparison measures warm requests.

| GPU | Engine | Prefill 1,024 ms / tok/s | Decode 32 ms / tok/s | Peak process GPU bytes | Predictions matching Basalt |
|---|---|---:|---:|---:|---:|
| PRO4500 | LeanSonic | 127.572 / 8,026.9 | 201.551 / 158.77 | 18,148,753,408 | 33/33 |
| PRO4500 | Basalt | 132.800 / 7,710.8 | 239.800 / 133.44 | 19,430,113,280 | 33/33 |
| 5070Ti | LeanSonic | 160.436 / 6,382.6 | 304.508 / 105.09 | 16,546,529,280 | 32/33 |
| 5070Ti | Basalt | 450.100 / 2,275.0 | 635.900 / 50.32 | 15,521,021,952 | 33/33 |
| 5060Ti | LeanSonic | 380.463 / 2,691.5 | 747.828 / 42.79 | 16,554,917,888 | 32/33 |
| 5060Ti | Basalt | 1283.000 / 798.1 | 781.700 / 40.94 | 15,527,313,408 | 33/33 |

LeanSonic uses the canonical 15,548,157,542-byte distribution and Basalt uses its 18,327,104,737-byte text container. The LeanSonic reader/driver used packer revision `a9872cc` with the SM120a PDL kernel from `48b7ec2`, SHA-256 `3714fd830d6cfedbf6954f21cad460489173be36566b0cb4b78173d42da87f0b`. The reported Basalt phase timers are integer milliseconds, and GPU memory is the maximum per-process NVML sample during loading and inference, sampled approximately every 4 ms. The per-GPU predictions check the replay history. Quality against BF16 is measured separately.

## Reproduce

Build the Basalt Inkling binary at [5fe1928d](https://github.com/jesdga95/basalt/commit/5fe1928d). The text-container SHA-256, binary hash, exact prompt digest and per-GPU continuation are in the [per-process results](inkling-basalt-canonical.json). Use the public [canonical packer and model recipe](inkling-zero-codes.md). To rebuild the measured LeanSonic driver, copy `crates/leansonic-core/examples/inkling_basalt_replay.rs` from this revision to the same path in the `a9872cc` checkout and build its release example. Build the measured cubin from `48b7ec2` using `nvcc -std=c++17 -O3 -arch=sm_120a --cubin kernels/inkling2_sm120.cu -o inkling.cubin`. Select the same GPU for both engines.

```sh
prompt=docs/benchmarks/inputs/inkling-readme1024.ids
"$BASALT" --model "$BASALT_MODEL" --tokens-file "$prompt" --gpu 0 --kv fp16 \
  --max-context 8192 --spec 0 --temperature 0 --max-new 33 > basalt.out
awk '$1 == "T" {print $2}' basalt.out | head -32 | paste -sd, - > continuation.ids
LEANSONIC_PDL=1 LEANSONIC_KV=fp16 LEANSONIC_CUBIN_PATH="$CUBIN" \
  "$LEAN_REPLAY" "$CANONICAL" "$prompt" continuation.ids
```
