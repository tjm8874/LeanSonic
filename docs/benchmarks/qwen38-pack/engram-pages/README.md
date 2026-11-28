# Lossless four-row Engram tiles

Each Engram head reads 80 bytes of Q4 weights and five scale/bias pairs. In the existing direct dictionary package, weights and the two sets of ten-bit indices live in separate planes. A 370-byte tile stores the same data for four consecutive rows together. The scale and bias dictionaries remain local to the shard. All 128 tile frames fit in the original safetensors shard set. The loader also accepts existing raw and dictionary layouts.

On Qwen3.8-Flash-Next, rearranging the 110,833,479,038-byte direct Engram package produced **110,833,399,834 bytes**. The compact tile needs no page padding. The four-row grouping avoids the per-page capacity overhead of grouping 44 rows into exactly 4 KiB.

## Row access on SSD

On an AMD Ryzen 9 9900X with Linux, ext4 and NVMe, all 2,500,012 rows from each of two model shards were decoded and compared with the original. Six deterministic sets of 4,096 random row reads then ran through the three layouts in all six orders. Before each cold arm we evicted only the private benchmark files and verified zero resident pages with mincore. The second pass reused the same rows. Both parameter dictionaries were kept hot for all arms.

| Shard | Layout | Cold ms | Cold major faults | Warm ms |
| --- | --- | ---: | ---: | ---: |
| 0 | Three planes | 1315.795 | 9008 | 1.350 |
| 0 | Four-row tile | 557.409 | 4173.5 | 0.935 |
| 0 | 4 KiB page | 518.279 | 3952 | 0.988 |
| 64 | Three planes | 742.664 | 9008 | 1.037 |
| 64 | Four-row tile | 348.457 | 4173.5 | 0.741 |
| 64 | 4 KiB page | 342.702 | 3952 | 0.765 |

The table reports medians. Each published TSV row gives seed, layout (0 = planes, 1 = 4 KiB page, 2 = four-row tile), cold flag, rows read, elapsed ms, resident pages before/after, major faults and output checksum. These measurements isolate host row lookup.

## Native inference on RTX PRO 5000 72 GB

All 494 full-vocabulary FP32 rows and 16 eight-token continuations matched the original Q4 engine on the frozen extended inputs. The 16 held-out streaming requests (four prompts across four cycles) also produced identical IDs across layouts. With 32 generated tokens per request, warmed decode was 163.34–164.14 tok/s for original and 163.27–164.11 for tiled, depending on the prompt. Sampled process peak VRAM was 75,486,986,240 bytes for both. The matched per-prompt results include prefill and startup. [Run measurements](measurements.json)

## Reproduce

Use the directly loadable Q38DICT1 Engram package and current matching SM120 target cubin. The converter creates a new directory on the same filesystem. It hard-links files which need no changes and rewrites the affected safetensors and index. The optional audit reads both layouts and compares each tile's weight codes, scale indices, bias indices and dictionary values, plus the other tensor payloads.

```sh
d=docs/benchmarks/qwen38-pack/engram-pages
python3 "$d/pack.py" "$DIRECT" "$TILED"
python3 "$d/audit.py" "$DIRECT" "$TILED"
python3 "$d/build.py" "$DIRECT" private-shard0 --shard 0
c++ -std=c++20 -O3 -Wall -Wextra "$d/compare.cpp" -o compare-engram
./compare-engram private-shard0 > shard-0.tsv
```

For full-engine tests, run the existing q38_teacher and q38_stream with each model. Use the same CUDA module, device, max context and inputs. The separate cold_requests.py harness takes a private original copy, the newly written tiled model, the frozen 16-prompt file, and an unused output directory. Its first request follows eviction of the private Engram tensor ranges after READY. It measures actual residency and repeats the request warm.

```sh
python3 "$d/cold_requests.py" copy "$DIRECT" "$PRIVATE"
python3 "$d/cold_requests.py" run "$STREAM_BIN" "$PRIVATE" "$TILED" "$EXTENDED_IDS" "$OUT"
```

## First access in the native engine

Each row gives the first request on a fresh engine with 128 generated tokens. The benchmark evicted private Engram pages after READY and immediately repeated the request warm. The 128 output IDs matched across all four runs for each prompt. This initial run has one cold sample per layout per input. [Actual residency, per-request faults and output hashes](first-cold-summary.json)

| Prompt | Input tokens | Cold TTFT, original → tile (ms) | Cold decode, original → tile (tok/s) |
|---|---:|---:|---:|
| English | 33 | 939.8 → 408.6 | 48.06 → 73.48 |
| Japanese | 30 | 845.4 → 399.2 | 49.55 → 76.79 |
| Python | 23 | 685.2 → 296.4 | 150.89 → 157.03 |
| Arithmetic | 43 | 977.5 → 475.7 | 60.50 → 90.02 |

## With the lossless expert-weight frames

An integration of this branch with #268 at `d3ec717` also accepts the 103,556,413,821-byte expert-frame model as the converter input. The tiled distribution occupied **103,556,334,617 bytes**. The disk audit checked all 320,001,536 Engram rows and 3,363 other tensor payloads. On the RTX PRO 4500 32 GB, every one of 494 full-vocabulary rows, all 16 eight-token continuations and all 16 held-out 32-output requests matched the untiled compressed model. [Per-case timings, peak VRAM and exact counts](measurements.json)

## Reversed cold order and matched public engine

Repeating the private-page test in reverse layout order gave cold TTFTs of 910.9 → 404.0 ms (English), 879.8 → 389.5 ms (Japanese), 690.5 → 343.5 ms (Python), and 1006.2 → 492.3 ms (arithmetic). All eight cold/warm sequences per prompt matched across the two experiments. [Reverse-order measurements and hashes](reverse-cold-summary.json)

The warmed public-engine comparison used the same four input ID sequences, 128 output tokens, max context 256, no speculative decoding, and no prompt-cache reuse. We discarded the first cycle and took the median of the next three for each prompt. The ranges below span those four medians. Prefill throughput divides the input length by client time to the first token. Decode throughput covers the remaining 127 tokens. The finish column measures the interval from the final token to DONE. [All timestamps, IDs and process VRAM samples summarized per engine](public128.json.gz)

| Engine | Prefill tok/s | Decode tok/s | TTFT ms | Finish ms | Total ms | Peak process bytes |
|---|---:|---:|---:|---:|---:|---:|
| LeanSonic combined | 405.69–453.83 | 160.94–161.70 | 56.69–94.75 | 0.00–0.00 | 842.09–883.93 | 75,486,986,240 |
| Basalt adaptive | 138.45–139.13 | 138.20–138.81 | 165.72–309.06 | 0.24–0.27 | 1083.52–1228.46 | 75,608,621,056 |
| Basalt fixed | 104.05–135.09 | 106.93–134.85 | 170.45–318.30 | 0.24–0.24 | 1114.97–1476.19 | 75,608,621,056 |

LeanSonic used the 103,556,334,617-byte MLX Q4 derivative at `d3ec717`; Basalt v0.5.7 used the 115,307,561,176-byte UD-Q4_K_XL file. Basalt used FP16 KV, RAM Engram, automatic expert cache and prefill; adaptive used `--temperature 0`, fixed used `--greedy --adapt-every 0`. Each LeanSonic and fixed Basalt prompt produced the same sequence across four cycles. Each adaptive prompt produced three distinct sequences. Timing used the RTX PRO 5000 72 GB and the protocol in `docs/benchmarks/qwen38-engine-comparison/compare_stream.py --held-out-routing --generation-length 128`. Use the four exact prompt arrays in cycle 0 of the archive as the four measured fixture rows. The combined model and engine command are specified above.

BF16 quality on 176 common positions favors Basalt. Mean/worst truncated KL, top-1 matches and PPL were 0.25952/2.49949, 156/176, 10.8242 for native MLX Q4; 0.01609/0.18175, 170/176, 7.7054 for Basalt adaptive; and 0.01690/0.19575, 172/176, 7.6513 for Basalt fixed. BF16 reference PPL was 7.5174. The two distributed quantizations differ. [Quality protocol and measurements](https://github.com/tjm8874/LeanSonic/blob/c34a9b2/docs/benchmarks/qwen38-pack/input-embedding/README.md#sixteen-additional-prompts-on-the-rtx-pro-5000)
