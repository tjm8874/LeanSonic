# Matched speed and footprints on RTX 5070 Ti

The 217-expert bounded parent and original llama.cpp IQ3_XXS received the same 1,024 README tokens and the same 32 teacher-forced continuation tokens. Both ran on one 16 GB RTX 5070 Ti, driver 595.99.02, 300 W, FP16 KV, context 8,192, one sequence and no speculation. Every LeanSonic process reported 24/24 expert layers resident. Every llama.cpp process offloaded 25/25 layers. The 20 processes alternate ABBA/BAAB over five cycles, with one warmup and four measured requests per process (40 observations per engine). All repeated predictions within each engine match. Both predict 31/32 forced tokens; they miss different positions.

| Engine | Model + tokenizer bytes | Prefill tok/s | Decode tok/s | Sampled peak process GPU bytes |
| --- | ---: | ---: | ---: | ---: |
| LeanSonic d15b1d6, bounded 217 experts | 11,662,534,370 | 8,425.93 | 170.97 | 16,097,738,752 |
| llama.cpp 8832860, original IQ3_XXS | 11,663,309,504 | 1,933.34 | 79.92 | 15,049,162,752 |

LeanSonic uses 775,134 fewer distribution bytes. Prefill takes 121.530 vs 529.654 ms and 32 decode steps take 187.166 vs 400.381 ms. The speed ratios are 4.3582× prefill and 2.1392× decode (paired-cycle bootstrap 95% intervals 4.3562–4.3605 and 2.1375–2.1408). Its peak process GPU allocation is 1,048,576,000 bytes higher. On the same GPU the bounded parent improves mean KL, worst KL, top-1 agreement and PPL versus native on both 25,500-position WikiText splits. [Full quality tables, inputs and corpus commands](README.md).

## Reproduction

Use the original IQ3_XXS and bounded parent hashes in timing.json.gz. Build [the traced fixed-token Rust and native examples](https://github.com/tjm8874/LeanSonic/tree/c30829e/docs/benchmarks/inkling-mixed-ternary-down/5070Ti/native-iq1) at LeanSonic d15b1d6 and Unsloth llama.cpp 8832860 respectively. Use the compiled sm_120a module and executable hashes below. Keep full-width gate/up and down, PDL and PDL_ALL enabled, MXFP8_DEVICE_U4=1 and MXFP8_DIRECT_ALL=1. Use the IQ3 model file in iq and the bounded package directory in model-path. Run the included forced.py for cycles 1 through 5 on an idle card with the GPU UUID in CUDA_VISIBLE_DEVICES. The native binary uses batch and microbatch 1,024. NVML samples per-process allocation approximately every 4 ms; peaks include loading. [Requests, executable and input hashes, traces and peaks](timing.json.gz) · [Matched runner](forced.py).


## Prefill workspace and the VRAM gap

A one-process pilot per setting used the same card, executable, 217-expert package and 1,024 + 32 fixed IDs. Each process ran one warmup and four measured requests. All 32 predictions match across settings. Observed process peaks can vary across process launches; the five-cycle default result above uses the maximum across ten LeanSonic processes.

| Prefill cap | Prefill tok/s | Decode tok/s | Peak process GPU bytes |
| --- | ---: | ---: | ---: |
| default | 8,423.2 | 170.97 | 15,950,938,112 |
| 512 | 5,962.1 | 170.91 | 15,634,268,160 |
| 256 | 3,790.3 | 170.95 | 15,577,645,056 |
| 64 | 1,365.4 | 171.00 | 15,521,021,952 |

Capping the workspace at 512 saves 316,669,952 bytes versus the pilot default and costs about 29% of prefill throughput. Even at 64 the peak is above the native five-cycle maximum. Keep the default as the speed candidate. To reproduce, run the same LeanSonic fixed-token command once with each LEANSONIC_PREFILL_CHUNK setting, and unset it for default. [Pilot traces, timings and sampling counts](workspace-pilot.json).

## Matching the workspace to the 1,024-token prompt

The same single-process pilot with two larger explicit caps preserved all predicted IDs. Each row includes four measured requests after a warmup. [Measurements and token traces](workspace-extended.json).

| Prefill cap | Prefill tok/s | Decode tok/s | Peak process GPU bytes |
| --- | ---: | ---: | ---: |
| 1024 | 8,427.8 | 171.10 | 15,758,000,128 |
| 2048 | 8,427.1 | 171.11 | 16,171,139,072 |

The explicit 1,024 cap saved 192,937,984 bytes versus the earlier one-process default while preserving prefill throughput for this prompt length. A separate default run instrumented allocations at the CUDA driver interface. At the high-water mark, the loader’s decoded expert buffers accounted for 10,192,158,720 bytes, compared with 15,561,009,448 tracked bytes and a sampled NVML process peak of 15,911,092,224 bytes. The profile includes every tracked allocation site and phase totals. To reproduce, apply the instrumentation to revision `d15b1d69f38eac343fb8cda80695552cb1266694` with the fixed-decode example installed, rebuild that example, and run the same default profile. [Instrumentation](instrument-allocations.py) · [Allocation profile](allocations.json).
