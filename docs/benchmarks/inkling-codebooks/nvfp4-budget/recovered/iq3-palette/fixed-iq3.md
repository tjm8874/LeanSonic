# Matched native IQ3_XXS comparison

LeanSonic package including tokenizer: 11,609,363,902 bytes. Unsloth native IQ3_XXS GGUF including tokenizer: 11,663,309,504 bytes. LeanSonic a200846 uses the bc526e3 cubin and directly expands the stored expert representation. Unsloth llama.cpp 883286076efe5b85fddb31d9cc7035a6b698ba9b consumes the original GGUF. One sequence, 8192-token context, chunk/microbatch 512, speculation off and FP16 KV. All 24 LeanSonic expert layers and all 25 reported native layers were GPU resident.

Both engines prefill the same 1,024 README tokens, then predict and consume exactly the same 32 forced continuation IDs. Each of five paired cycles alternates four fresh process launches (ABBA/BAAB). Each process serves five resets. Exclude its first request from timings. Peak VRAM is the largest NVML per-process observation across load and all requests, sampled roughly every 4 ms. The peak can miss short-lived allocations. Warm phases exclude loading. Matches count the 32 pre-feed argmax IDs that equal the supplied continuation.

| GPU | Engine | Prefill ms (tok/s) | Decode 32 ms (tok/s) | Peak process VRAM, bytes | Matches |
|---|---|---:|---:|---:|---:|
| 5070Ti | lean | 121.829 (8,405.2) | 192.381 (166.34) | 16,055,795,712 | 30/32 |
| 5070Ti | llama | 564.700 (1,813.4) | 399.716 (80.06) | 14,241,759,232 | 30/32 |
| 5060Ti | lean | 257.211 (3,981.2) | 339.796 (94.17) | 15,999,172,608 | 30/32 |
| 5060Ti | llama | 961.544 (1,065.0) | 566.875 (56.45) | 14,124,318,720 | 30/32 |

5070Ti: prefill 4.635× (4.633–4.637), decode 2.078× (2.077–2.078) (five-cycle paired 95% intervals). LeanSonic peak VRAM minus native is +1,814,036,480 bytes.
5060Ti: prefill 3.738× (3.735–3.741), decode 1.668× (1.668–1.669) (five-cycle paired 95% intervals). LeanSonic peak VRAM minus native is +1,874,853,888 bytes.

[Pinned replay harness, inputs and environment](https://github.com/tjm8874/LeanSonic/blob/18ae3c2/docs/benchmarks/inkling-nvfp4-trim-context/fixed-replay.md) · [LeanSonic encoder and model file](README.md) · [Native file and revision](https://github.com/tjm8874/LeanSonic/tree/53a4369/docs/benchmarks/inkling-native-lowbits) · [Per-request times and predictions, paired intervals and binary hashes](fixed-iq3.json).
