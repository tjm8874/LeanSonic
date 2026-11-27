# Runtime checks for lossless NVFP4 storage

The integrated parity driver loads both packages with the same cubin. The corpus comparison uses the corresponding builds before and after the storage change. On RTX PRO 4500 at explicit prefill width 512, FP16 KV, scratch-free U4 and speculation off, the complete BF16 WikiText-2 test summaries agree across both formats (100 chunks, 25,500 positions).

| Mean KL | Worst KL (chunk/position) | Top-1 matches | PPL |
|---:|---:|---:|---:|
| 0.0081645 | 1.2202775 (99/398) | 24,500/25,500 | 7.081285 |

5060Ti completed 2 fresh processes. Every 64-token greedy continuation, next ID and hash of the final 200,058 finite FP32 logits matched at both lengths in every pass. Mean loader time was 13.20 s before and 14.38 s after (1 process(es) per format).
5070Ti completed 2 fresh processes. Every 64-token greedy continuation, next ID and hash of the final 200,058 finite FP32 logits matched at both lengths in every pass. Mean loader time was 12.08 s before and 13.01 s after (1 process(es) per format).
PRO4500 completed 20 fresh processes. Every 64-token greedy continuation, next ID and hash of the final 200,058 finite FP32 logits matched at both lengths in every pass. Mean loader time was 7.08 s before and 7.62 s after (10 process(es) per format).

The parity driver uses the first 32 and all 1,024 tokens of the fixed README prompt for four passes in each process with context 2,048. Run the attached driver as a leansonic-core example with the package path and the [README token IDs](../inputs/inkling-readme1024.ids). For the BF16 corpus check run the integrated CLI with `eval-kld --weights "$MODEL" --ref "$BF16_TEST" --chunks 100`. Apply the storage code in this PR to bc526e3. Use the same cubin as bc526e3 and set `LEANSONIC_PDL=1 LEANSONIC_PDL_ALL=1 LEANSONIC_KV=fp16 LEANSONIC_SEQS=1 LEANSONIC_MXFP8_DEVICE_U4=1 LEANSONIC_MXFP8_DIRECT_ALL=1`. Set `LEANSONIC_PREFILL_CHUNK=512` for the corpus comparison. [Driver](parity.rs) · [Native summaries and per-request observations](runtime.json).
