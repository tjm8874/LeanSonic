# Smaller NVFP4 package and staged decode in the same binary

The candidate is `262d973` (#98, #100 and #102 integrated with main `eba8d8d`, including #97). Both optimizations are enabled in the same run. The candidate reads a 15,901,490,534-byte text distribution including tokenizer, with shard SHA-256 `c0b42b8bf338a0690000e57668f03e9f2d27450c8187c89555165a4bf22588f4`. Main reads its 15,982,139,902-byte compact text distribution. Basalt `5fe1928d` reads its 18,327,104,737-byte text distribution.

Timing units are ms for prefill / 16 decode forwards. Peak process allocation was sampled using NVML every 4 ms and is reported in bytes. Each table row includes three complete cycles with rotated, mirrored process order. All three engines generated the same first 16 IDs on the listed inputs.

| GPU | Input tokens | Main #97 prefill / decode16 | Combined prefill / decode16 | Basalt prefill / decode16 | Peak bytes combined / Basalt |
|---|---:|---:|---:|---:|---:|
| 5060Ti | 32 | 242.37 / 332.91 | 242.55 / 332.24 | 291.50 / 377.33 | 16,559,112,192 / 15,525,216,256 |
| 5060Ti | 1024 | 380.76 / 390.32 | 380.75 / 389.59 | 1283.50 / 392.50 | 16,557,015,040 / 15,527,313,408 |

Use the checked-in README input, greedy decoding, FP16 KV, context 8192 and speculation disabled. In each arm use a fresh process. LeanSonic `bench --gen 16 --repeat 1` executes 16 decode forwards. Basalt `--max-new 17` includes the first prefill prediction. The [per-process evidence](inkling-combined-pr97.json) includes inputs, outputs, timing and allocation samples. See the [staged decoder](inkling-selected-host-weights.md) and [lossless packaging](inkling-nvfp4-code-storage.md) instructions for building the candidate.
