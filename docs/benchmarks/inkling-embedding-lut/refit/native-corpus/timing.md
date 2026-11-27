# Matched speed and footprints on RTX 5070 Ti

The 217-expert bounded parent and original llama.cpp IQ3_XXS received the same 1,024 README tokens and the same 32 teacher-forced continuation tokens. Both ran on one 16 GB RTX 5070 Ti, driver 595.99.02, 300 W, FP16 KV, context 8,192, one sequence and no speculation. Every LeanSonic process reported 24/24 expert layers resident. Every llama.cpp process offloaded 25/25 layers. The 20 processes alternate ABBA/BAAB over five cycles, with one warmup and four measured requests per process (40 observations per engine). All repeated predictions within each engine match. Both predict 31/32 forced tokens; they miss different positions.

| Engine | Model + tokenizer bytes | Prefill tok/s | Decode tok/s | Sampled peak process GPU bytes |
| --- | ---: | ---: | ---: | ---: |
| LeanSonic d15b1d6, bounded 217 experts | 11,662,534,370 | 8,425.93 | 170.97 | 16,097,738,752 |
| llama.cpp 8832860, original IQ3_XXS | 11,663,309,504 | 1,933.34 | 79.92 | 15,049,162,752 |

LeanSonic uses 775,134 fewer distribution bytes. Prefill takes 121.530 vs 529.654 ms and 32 decode steps take 187.166 vs 400.381 ms. The speed ratios are 4.3582× prefill and 2.1392× decode (paired-cycle bootstrap 95% intervals 4.3562–4.3605 and 2.1375–2.1408). Its peak process GPU allocation is 1,048,576,000 bytes higher. On the same GPU the bounded parent improves mean KL, worst KL, top-1 agreement and PPL versus native on both 25,500-position WikiText splits. [Full quality tables, inputs and corpus commands](README.md).

## Reproduction

Use the original IQ3_XXS and bounded parent hashes in timing.json.gz. Build [the traced fixed-token Rust and native examples](https://github.com/tjm8874/LeanSonic/tree/c30829e/docs/benchmarks/inkling-mixed-ternary-down/5070Ti/native-iq1) at LeanSonic d15b1d6 and Unsloth llama.cpp 8832860 respectively. Use the compiled sm_120a module and executable hashes below. Keep full-width gate/up and down, PDL and PDL_ALL enabled, MXFP8_DEVICE_U4=1 and MXFP8_DIRECT_ALL=1. Use the IQ3 model file in iq and the bounded package directory in model-path. Run the included forced.py for cycles 1 through 5 on an idle card with the GPU UUID in CUDA_VISIBLE_DEVICES. The native binary uses batch and microbatch 1,024. NVML samples per-process allocation approximately every 4 ms; peaks include loading. [Requests, executable and input hashes, traces and peaks](timing.json.gz) · [Matched runner](forced.py).
