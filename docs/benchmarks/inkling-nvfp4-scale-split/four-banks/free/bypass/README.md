# Scalar decode while DSpark stays resident

RTX 5060 Ti, one sequence and 128 greedy tokens per request, four-bank target, context 2,048 and FP16 KV. Compare a target alone, normal spec7, and the same spec7 loader stepping the target once per token with DSpark context maintained. Both spec7 modes include the 326,519,568-byte row-exponent sidecar. The minimum suffix match was 32. Each of the six launch orders ran once. A process evaluated all eight inputs twice. The second pass supplies the timings. All 288 requests matched the scalar run per fixture on all emitted IDs, successor and full vocabulary FP32 fingerprint.

| Fixture | Input tokens | Prefill ms (plain / spec7 / bypass) | Decode tok/s (plain / spec7 / bypass) | Bypass / spec7 paired 95% interval |
|---|---:|---|---|---|
| q00_code_merge | 29 | 64.02 / 64.31 / 64.37 | 98.92 / 166.02 / 98.22 | [0.591, 0.592] |
| q11_json_schema | 32 | 76.65 / 77.03 / 77.06 | 98.86 / 169.54 / 98.15 | [0.579, 0.579] |
| ja_reasoning | 49 | 89.04 / 89.50 / 89.46 | 98.51 / 94.45 / 97.81 | [1.035, 1.036] |
| readme1024 | 1024 | 259.43 / 266.78 / 266.82 | 82.31 / 103.07 / 81.86 | [0.794, 0.795] |
| en_boolean | 38 | 82.11 / 82.61 / 82.63 | 98.75 / 129.49 / 98.03 | [0.757, 0.757] |
| en_sort | 46 | 86.26 / 86.70 / 86.73 | 98.58 / 169.76 / 97.87 | [0.576, 0.577] |
| ja_stock | 55 | 91.72 / 92.22 / 92.19 | 98.38 / 115.52 / 97.68 | [0.845, 0.846] |
| ja_name | 53 | 90.70 / 91.11 / 91.08 | 98.43 / 110.56 / 97.71 | [0.884, 0.884] |

| Mode | Distribution bytes | Largest observed process VRAM bytes |
|---|---:|---:|
| 0 | 13,455,139,382 | 15,802,040,320 |
| 7 | 13,781,658,950 | 16,259,219,456 |
| bypass | 13,781,658,950 | 16,269,705,216 |

Use the [same model, cubin and build recipe](../depth/) at revision 7e85cc3c4d0a1439aa3109b3fa2f7b6c09a354bc. Copy lean.rs into crates/leansonic-core/examples/depth_replay.rs, build it in release mode and run the binary from the directory containing these token IDs. Set LEANSONIC_SPEC_COPY=32 and the environment from the parent recipe. Pass the model path and then 0, 7, or bypass. For bypass, supply the same directory including mtp.safetensors as for 7. [Driver](lean.rs) · [Full request data, counters and NVML peaks](verified.json).
