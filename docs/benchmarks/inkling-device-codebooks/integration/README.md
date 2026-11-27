# Compact Inkling on the global-PDL engine

This integration combines staged NVFP4 scales, compact MXFP8 indices, direct expert reads and automatic workspace placement with the global-PDL engine. The added CUDA entry points wait before reading predecessor outputs. All four storage/scheduling modes use the same engine and cubin. The byte-storage mode uses the same compact distribution with expanded device weights.

## Greedy generation at context 2,048

The input alternates 32 and 1,024 tokens for four passes, resetting the engine between requests. Each case checks all 64 free tokens, one further prediction and a hash of the last 200,058 FP32 logits against the direct path with global PDL.

| GPU | Mode | Resident expert layers | Matching traces |
|---|---|---:|---:|
| 5060Ti | byte | 21/24 | 8/8 |
| 5060Ti | expansion | 24/24 | 8/8 |
| 5060Ti | legacy | 24/24 | 8/8 |
| 5060Ti | direct | 24/24 | 8/8 |
| 5070Ti | byte | 21/24 | 8/8 |
| 5070Ti | expansion | 24/24 | 8/8 |
| 5070Ti | legacy | 24/24 | 8/8 |
| 5070Ti | direct | 24/24 | 8/8 |

## Matched speed and peak VRAM at context 8,192

Both engines consume the same 1,024 prompt IDs and 32 forced continuation IDs, with FP16 KV, one sequence and speculation off. The native Unsloth llama.cpp build at `8832860` runs the original Q3_K_XL GGUF with all 25 layers offloaded and a microbatch of 512. LeanSonic uses the same source model with a different quantization, 24/24 expert layers resident and automatic workspace placement.

Ten fresh processes per engine ran in five mirrored ABBA/BAAB cycles, with one warm-up and four measured requests per process. Latencies are means across the 40 warm requests. Throughput divides the fixed token count by mean latency. NVML sampled per-process memory during load and inference. The peak is the largest observation. The 95% intervals resample whole paired cycles.

| GPU | Engine | Prefill ms (tok/s) | Decode 32 ms (tok/s) | Peak GPU bytes | Package bytes | Fixed IDs matched |
|---|---|---:|---:|---:|---:|---:|
| RTX 5070Ti | LeanSonic compact | 121.960 (8,396.2) | 192.392 (166.33) | 16,118,710,272 | 13,626,745,974 | 32/32 |
| RTX 5070Ti | Unsloth llama.cpp Q3_K_XL | 526.094 (1,946.4) | 402.464 (79.51) | 16,150,167,552 | 13,631,778,016 | 31/32 |
| RTX 5060Ti | LeanSonic compact | 257.696 (3,973.7) | 339.703 (94.20) | 16,020,144,128 | 13,626,745,974 | 32/32 |
| RTX 5060Ti | Unsloth llama.cpp Q3_K_XL | 881.869 (1,161.2) | 573.839 (55.76) | 16,030,629,888 | 13,631,778,016 | 30/32 |

LeanSonic speedup over native Q3. RTX 5070Ti prefill 4.314× (95% 4.311–4.316), RTX 5070Ti decode 2.092× (95% 2.089–2.095), RTX 5060Ti prefill 3.422× (95% 3.421–3.423), RTX 5060Ti decode 1.689× (95% 1.688–1.690).

The compact distribution includes its tokenizer and is 5,032,042 bytes smaller than the GGUF, which embeds its tokenizer. The original IQ3_XXS and IQ1_S GGUFs are smaller at 11,663,309,504 and 7,764,291,776 bytes. See the [native Q3 driver, release and model fingerprint](../../inkling-original-q3-timing.md) and [smaller public formats](https://github.com/tjm8874/LeanSonic/pull/161).

The 32 predictions above measure agreement with the supplied continuation. On a separate 613-position bilingual fixture the compact weights reached worst BF16 KL 0.476597 and original Q3 reached 0.293811. This tail regression keeps the compact weights experimental. [Compact weight quality study](https://github.com/tjm8874/LeanSonic/pull/147).

## BF16 reference on PRO 4500

The same compact model, full-PDL build and fixed 512-row prefill width score WikiText-2 test.

| Mode | Mean KL | Worst KL | BF16 top-1 | PPL |
|---|---:|---:|---:|---:|
| expansion | 0.01378376 | 0.70781849 | 24,298/25,500 | 7.1274818 |
| direct | 0.01378376 | 0.70781849 | 24,298/25,500 | 7.1274818 |

All reported 100-chunk BF16 metrics, including the worst position, were identical in the expansion and scratch-free paths. The distribution still has the bilingual tail discrepancy above.

[Instrumented kernel timings on the same fixed inputs](profile/README.md).

## Reproduce

Use the [parent build commands, canonical input files and compact model fingerprint](../README.md) at a567839d. Set LEANSONIC_PDL=1, LEANSONIC_KV=fp16 and LEANSONIC_SEQS=1, then unset LEANSONIC_PREFILL_CHUNK for the GeForce tests. Use the four combinations below. The timing arm uses the direct configuration. The corpus comparison uses `leansonic eval-kld --weights "$MODEL" --ref "$TEST_KLD" --chunks 100` and explicit LEANSONIC_PREFILL_CHUNK=512.

| Mode | LEANSONIC_MXFP8_DEVICE_U4 | LEANSONIC_MXFP8_DIRECT_ALL | LEANSONIC_PDL_ALL |
|---|---:|---:|---:|
| byte | 0 | 0 | 1 |
| expansion | 1 | 0 | 1 |
| legacy | 1 | 1 | 0 |
| direct | 1 | 1 | 1 |

[Raw timings, predicted IDs, score summaries and build hashes](receipt.json).
