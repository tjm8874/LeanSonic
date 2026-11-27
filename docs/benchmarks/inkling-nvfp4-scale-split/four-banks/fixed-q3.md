# Four scale banks versus the public Inkling Q3_K_XL in llama.cpp

The LeanSonic package (target and tokenizer) is 13,455,139,382 bytes. The original Q3_K_XL GGUF with embedded tokenizer is 13,631,778,016 bytes, a difference of 176,638,634 bytes. Both come from Inkling-2-Mini and use different weight representations. Neither measurement uses a speculative drafter.

Both drivers take the same 1,024 README token IDs and 32 fixed continuation IDs, FP16 KV, one sequence and context 8,192. LeanSonic keeps all 24 expert layers on GPU with zero mixed or spilled layers. Unsloth llama.cpp at 883286076efe5b85fddb31d9cc7035a6b698ba9b offloads all 25 layers with microbatch 512. Each GPU runs five mirrored ABBA/BAAB cycles with two fresh launches per arm per cycle. Timings average the four warm requests after one warm-up in each process (40 measured requests per arm). NVML samples per-process memory every 4 ms during loading and inference.

| GPU | Engine | Prefill ms (tok/s) | Decode 32 ms (tok/s) | Peak VRAM bytes | Top-1 matches to forced IDs |
|---|---|---:|---:|---:|---:|
| 5070Ti | LeanSonic | 122.684 (8,346.6) | 187.176 (170.96) | 16,013,852,672 | 32/32 |
| 5070Ti | llama.cpp Q3_K_XL | 526.028 (1,946.7) | 402.088 (79.58) | 16,150,167,552 | 31/32 |
| 5060Ti | LeanSonic | 258.972 (3,954.1) | 338.632 (94.50) | 16,009,658,368 | 32/32 |
| 5060Ti | llama.cpp Q3_K_XL | 880.835 (1,162.5) | 573.763 (55.77) | 16,030,629,888 | 30/32 |

On 5070Ti, LeanSonic's speed ratios over native Q3_K_XL were 4.288× for prefill (paired cycle bootstrap 95% interval 4.284–4.292) and 2.148× for decode (2.146–2.151).

On 5060Ti, LeanSonic's speed ratios over native Q3_K_XL were 3.401× for prefill (paired cycle bootstrap 95% interval 3.399–3.403) and 1.694× for decode (1.694–1.695).

Predicted sequences repeat across every request and match the [prior two-bank replay](../integrated/) per GPU and arm. The 32 fixed positions exercise replay parity. The [paired BF16 corpus replay](paired-quality.md) measured both packages on test and validation. Their scorer outputs agreed byte-for-byte on each split (25,500 positions each).

Rebuild the four-bank loader at [0b6793e8](https://github.com/tjm8874/LeanSonic/commit/0b6793e8ec25bbd1142c19928b644db35eb413f5) with the [package recipe](README.md). Compile the [same fixed-input driver and inputs](https://github.com/tjm8874/LeanSonic/blob/18ae3c2/docs/benchmarks/inkling-nvfp4-trim-context/fixed-replay.md) against that loader, using its recorded cubin. For llama.cpp use the [pinned GGUF and native driver](../../inkling-original-q3-timing.md). Enable global PDL, direct expert reads and compact U4 device indices. [Per-request times and predictions, phase ratios and executable digests](fixed-q3.json).
