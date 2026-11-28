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
