# PDL and compact NVFP4 versus Basalt

All three binaries processed the same prefix of the [1,024-token prompt](inputs/inkling-readme1024.ids) with greedy sampling, FP16 KV, context 8192, and speculation disabled. Each prompt and arm used a fresh process. Five cycles rotated and mirrored the three arms. Sixteen decode forwards were timed after prefill. Each candidate ran the fast controller and PDL from #113 plus the #109 model reader, commit `48b7ec2`. The older LeanSonic control uses the `1991ac6` kernel and Basalt uses `5fe1928d`.

| GPU | Input tokens | Older LeanSonic prefill / decode16 (ms) | PDL + compact prefill / decode16 (ms) | Basalt prefill / decode16 (ms) | Peak process bytes PDL / Basalt |
|---|---:|---:|---:|---:|---:|
| 5070Ti | 32 | 105.49 / 149.64 | 105.91 / 135.47 | 267.20 / 314.70 | 16,548,626,432 / 15,518,924,800 |
| 5070Ti | 1024 | 161.22 / 166.03 | 160.73 / 152.12 | 450.10 / 324.30 | 16,546,529,280 / 15,521,021,952 |

All three engines predicted the same first 16 IDs in each reported row. NVML was sampled at a 4 ms target interval. Distribution sizes including tokenizer are 15,901,490,534 B (older LeanSonic), 15,853,391,742 B (PDL + compact) and 18,327,104,737 B (Basalt). This comparison measures the complete configurations. It does not isolate the effect of PDL from the new controller kernel.

Use the [existing commands and input fixture](inkling-selected-host-weights.md). Set `LEANSONIC_PDL=1` and select the cubin built from the candidate checkout for the compact arm. Run LeanSonic `bench --gen 16 --repeat 1 --spec 0` and Basalt `--max-new 17 --spec 0 --temperature 0` (one prediction comes from prefill). [Per-process times, sample peaks, IDs, hashes and paired 95% intervals](inkling-pdl-compact-basalt.json).
