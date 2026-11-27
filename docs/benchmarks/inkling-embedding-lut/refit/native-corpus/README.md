# 217-expert models and native IQ3_XXS on RTX 5070 Ti

Each split uses the same 100 WikiText chunks, 25,500 BF16-scored positions and FP16 KV on one RTX 5070 Ti. Parent (11,662,534,370 bytes) uses the bounded embedding; refit (11,663,198,358) retains the same experts with a more accurate embedding. Native llama.cpp 8832860 loads the original IQ3_XXS GGUF (11,663,309,504 bytes). All three run on the same physical card.

| Split | Arm | Mean KL | Worst KL | Top-1 / 25,500 | PPL |
|---|---|---:|---:|---:|---:|
| test | parent | 0.0197912 | 2.8942013 | 23,950 | 7.137518 |
| test | refit | 0.0197682 | 3.4155332 | 23,938 | 7.130266 |
| test | native | 0.0280570 | 5.5014780 | 23,666 | 7.193132 |
| valid | parent | 0.0181395 | 2.3339590 | 24,105 | 6.576801 |
| valid | refit | 0.0181472 | 2.4311891 | 24,082 | 6.580083 |
| valid | native | 0.0256590 | 3.7240360 | 23,786 | 6.616409 |

On test, the refit improves mean KL and PPL relative to its parent but raises the worst KL (same chunk 23, position 330) and loses 12 BF16 top-token matches. Both LeanSonic models beat the native IQ3_XXS on all four test metrics.
On validation the refit loses on all four metrics against the parent and uses 663,988 more bytes. The parent and refit both improve all four validation metrics against native IQ3_XXS. Across both splits the bounded parent is the stronger candidate for the matched timing comparison. Keep the refit as a control.

Build the release CLI and sm_120a cubin as in the parent PR. Use one sequence, PDL, PDL_ALL, prefill chunk 512, MXFP8_DEVICE_U4=1 and MXFP8_DIRECT_ALL=1. For each model use `leansonic eval-kld --weights MODEL --ref ORACLE --chunks 100`. For llama.cpp use `llama-perplexity -m GGUF -f CORPUS --kl-divergence --kl-divergence-base ORACLE -c 512 -b 512 -ub 512 -ctk f16 -ctv f16 -ngl 99 -t 8 --no-warmup --chunks 100`. The native scorer requested full GPU offload (-ngl 99); its log records the CUDA backend and effective batch but omits the layer count. [Raw scores and hashes](results.json)
