# Size and quality of a smaller public Inkling-2-Mini GGUF

The Unsloth Mainline_Support UD-IQ4_XS GGUF from revision `ae9327c5` is 15,733,756,512 bytes, 119,635,230 bytes smaller than the current LeanSonic text distribution with block-context NVFP4 storage. Its full GGUF hash matches the publisher's LFS digest. The official BF16 GGUF is 52,226,509,568 bytes. Each of these sizes includes the tokenizer.

| Model/runtime on RTX PRO 5000 72GB | Prefill 1024 ms | Decode32 ms | NVML process peak bytes | Greedy matches in the 32 forced positions |
|---|---:|---:|---:|---:|
| LeanSonic `1e4825b`, native compact + contextual codes | 90.18 | 189.71 | 18,243,125,248 | 32 |
| llama.cpp `29b52d3`, Unsloth UD-IQ4_XS mainline layout | 421.30 | 389.88 | 19,432,210,432 | 10 |

Both engines processed exactly the same prompt and all 32 continuation IDs. Differences in their greedy predictions were recorded while the supplied continuation was teacher-forced. As in the [BF16 run](inkling-block-context-llama-bf16.md), five mirrored paired cycles each started four fresh processes; each process made five requests. Timings average the four warm requests, including sampling, excluding load and reset. KV FP16, ctx 8192, greedy, no speculation. The [raw data](inkling-public-iq4xs.json) contain all per-process timings, sampled peaks and the IQ4_XS predicted IDs. The [GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/ae9327c5fbc7333116ba32eee78bb5d2bbefae4f/Mainline_Support/Inkling-2-Mini-UD-IQ4_XS.gguf_file) is explicitly the mainline-format file.

## Quality of this GGUF/runtime combination

The exact-vocabulary FP32 rows from llama.cpp were passed to the same `eval-kld --logits-f32` scorer used for LeanSonic, using 100 chunks and the same canonical official BF16 references per split. The scorer validated all token IDs before accepting each corpus run. Context 512, batch and microbatch 8192, FP16 KV. The native llama-perplexity test PPL printed to stderr independently rounded to 116.9172.

| WikiText split | Rows | Mean KL | Worst KL (chunk / position) | Top-1 matches | PPL | BF16 PPL |
|---|---:|---:|---:|---:|---:|---:|
| test | 25,500 | 2.851343 | 22.017612 (13/475) | 8,313 | 116.917195 | 7.064408 |
| valid | 25,500 | 2.939909 | 21.467917 (50/293) | 8,236 | 122.070790 | 6.498589 |

These results characterize the pinned runtime, file layout and batch schedule together. Establishing which contributes to the high PPL requires controls. The repository-root GGUF at the same model revision used a different layout and the pinned mainline runtime rejected its missing virtual-layer tensor. Use the publisher's matching runtime for that file. For reproduction, follow the [canonical oracle procedure](https://github.com/tjm8874/LeanSonic/blob/f1e45f1/docs/benchmarks/inkling-bf16-serial.md) with the mainline IQ4_XS file passed as the candidate `--model`. To time the fixed workload retain the forced IDs on mismatches and record predictions as in the [comparison harness patch](../../tools/inkling_fixed_prediction_log.patch).
