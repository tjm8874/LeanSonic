# Smaller NVFP4 package and staged decode in the same binary

The candidate is `f51cf61` (#98, #100 and #102 on V13 `1444305`). Both optimizations are enabled in the same run. The candidate reads a 15,901,490,534-byte text distribution including tokenizer, with shard SHA-256 `c0b42b8bf338a0690000e57668f03e9f2d27450c8187c89555165a4bf22588f4`. V13 reads its 15,982,139,902-byte compact text distribution. Basalt `5fe1928d` reads its 18,327,104,737-byte text distribution.

Timing units are ms for prefill / 16 decode forwards. Peak process allocation was sampled using NVML every 4 ms and is reported in bytes. Each table row includes three complete cycles with rotated, mirrored process order. All three engines generated the same first 16 IDs on the listed inputs. The later #97 global attention split change is measured separately.

| GPU | Input tokens | V13 prefill / decode16 | Combined prefill / decode16 | Basalt prefill / decode16 | Peak bytes combined / Basalt |
|---|---:|---:|---:|---:|---:|
| 5070Ti | 32 | 105.57 / 165.94 | 105.47 / 149.55 | 264.50 / 311.17 | 16,548,626,432 / 15,518,924,800 |
| 5070Ti | 1024 | 160.69 / 187.84 | 161.30 / 166.30 | 451.33 / 326.83 | 16,546,529,280 / 15,521,021,952 |
| 5060Ti | 32 | 242.69 / 332.57 | 242.65 / 332.04 | 290.67 / 376.33 | 16,559,112,192 / 15,525,216,256 |
| 5060Ti | 1024 | 380.86 / 390.11 | 380.94 / 389.43 | 1283.33 / 391.67 | 16,554,917,888 / 15,527,313,408 |

Use the checked-in README input, greedy decoding, FP16 KV, context 8192 and speculation disabled. In each arm use a fresh process. LeanSonic `bench --gen 16 --repeat 1` executes 16 decode forwards. Basalt `--max-new 17` includes the first prefill prediction. The [per-process evidence](inkling-combined-v13.json) includes inputs, outputs, timing and allocation samples. See the [staged decoder](inkling-selected-host-weights.md) and [lossless packaging](inkling-nvfp4-code-storage.md) instructions for building the candidate.
