# Paired BF16 quality on RTX 5060 Ti

The two arms from the [five-cycle speed and memory comparison](README.md#paired-native-comparison-on-rtx-5060-ti) were scored on the same RTX 5060 Ti using the same BF16 WikiText reference files, FP16 KV and 512-token contexts. Each split uses 100 chunks and scores 25,500 next-token distributions. LeanSonic bd048b8 loads the 571-expert package with both mixed switches enabled. Unsloth llama.cpp build 12588 (883286076) loads the original IQ3_XXS GGUF.

| Split | Engine | Mean KL | Worst KL | BF16 top-1 / 25,500 | PPL |
| --- | --- | ---: | ---: | ---: | ---: |
| test | LeanSonic | 0.0172238 | 6.1395663 | 24,071 (94.396%) | 7.128123 |
| test | llama.cpp IQ3_XXS | 0.0281680 | 4.2972870 | 23,635 (92.686%) | 7.197528 |
| valid | LeanSonic | 0.0157144 | 2.6382777 | 24,192 (94.871%) | 6.565670 |
| valid | llama.cpp IQ3_XXS | 0.0253550 | 3.3852140 | 23,836 (93.475%) | 6.610591 |

On test, LeanSonic improves mean KL, top-1 agreement and PPL. Its worst row reaches KL 6.1395663 versus the native maximum of 4.297287. On validation, LeanSonic improves all four measures; its worst KL is 2.6382777 versus 3.385214. The test tail remains the reason to hold a general fidelity claim.

Reproduce with the switches in the main report and the same BF16 oracle digest for both engines. For LeanSonic use `eval-kld --weights MODEL --ref ORACLE --chunks 100` with prefill chunk 512. For llama.cpp use `llama-perplexity -m GGUF -f CORPUS --kl-divergence --kl-divergence-base ORACLE -c 512 -b 512 -ub 512 -ctk f16 -ctv f16 -ngl 99 -t 8 --no-warmup --chunks 100`. The native scorer reports CUDA and effective batch size and omits the layer count. The native top-1 count is reconstructed from its three-decimal percentage (0.001 percentage points identifies a unique count at 25,500 positions). [Full score fields, hashes and model manifests](5060-quality.json).
