# Matched native IQ3_XXS comparison

LeanSonic package including tokenizer: 11,605,970,958 bytes. Unsloth native IQ3_XXS GGUF including tokenizer: 11,663,309,504 bytes. LeanSonic a200846 uses the bc526e3 cubin and directly expands the stored expert representation. Unsloth llama.cpp 883286076efe5b85fddb31d9cc7035a6b698ba9b consumes the original GGUF. One sequence, 8192-token context, chunk/microbatch 512, speculation off and FP16 KV. All 24 LeanSonic expert layers and all 25 reported native layers were GPU resident.

Both engines prefill the same 1,024 README tokens, then predict and consume exactly the same 32 forced continuation IDs. Each of five paired cycles alternates four fresh process launches (ABBA/BAAB). Each process serves five resets. Exclude its first request from timings. Peak VRAM is the largest NVML per-process observation across load and all requests, sampled roughly every 4 ms. The peak can miss short-lived allocations. Warm phases exclude loading. Matches count the 32 pre-feed argmax IDs that equal the supplied continuation.

| GPU | Engine | Prefill ms (tok/s) | Decode 32 ms (tok/s) | Peak process VRAM, bytes | Matches |
|---|---|---:|---:|---:|---:|
| 5070Ti | lean | 121.576 (8,422.7) | 192.341 (166.37) | 16,005,464,064 | 29/32 |
| 5070Ti | llama | 563.981 (1,815.7) | 399.562 (80.09) | 14,241,759,232 | 30/32 |
| 5060Ti | lean | 257.020 (3,984.1) | 339.838 (94.16) | 15,990,784,000 | 29/32 |
| 5060Ti | llama | 963.006 (1,063.3) | 567.412 (56.40) | 14,124,318,720 | 30/32 |

5070Ti: prefill 4.639× (4.637–4.641), decode 2.077× (2.076–2.079) (five-cycle paired 95% intervals). LeanSonic peak VRAM minus native is +1,763,704,832 bytes.
5060Ti: prefill 3.747× (3.745–3.748), decode 1.670× (1.669–1.670) (five-cycle paired 95% intervals). LeanSonic peak VRAM minus native is +1,866,465,280 bytes.

[Pinned replay harness, inputs and environment](https://github.com/tjm8874/LeanSonic/blob/18ae3c2/docs/benchmarks/inkling-nvfp4-trim-context/fixed-replay.md) · [LeanSonic encoder and model file](README.md) · [Native file and revision](https://github.com/tjm8874/LeanSonic/tree/53a4369/docs/benchmarks/inkling-native-lowbits) · [Per-request times and predictions, paired intervals and binary hashes](fixed-iq3.json).
