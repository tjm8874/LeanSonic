# Lossless device packing of gate/up and down experts

Commit `9f65bb2` extends the gate/up reader to down projections in fused decode and grouped Tensor Core prefill. In this experiment eight gate/up and eight down projections used the mixed format. All 24 layers were device resident. The loader uses the same [11,487,467,790-byte 571-expert package](https://github.com/tjm8874/LeanSonic/tree/4c28635/docs/benchmarks/inkling-embedding-lut/magnitude/44-levels/ranked-experts). Each eligible expert has its weight codes packed at load time. Other experts retain their original code width. Both flags are opt-in.

| RTX 5070 Ti, one sequence | Prefill tok/s | Decode tok/s | Peak process GPU bytes |
| --- | ---: | ---: | ---: |
| Raw | 8,423.46 | 170.95 | 15,760,097,280 |
| Mixed gate/up and down | 8,272.23 | 172.59 | 14,784,921,600 |

The pilot used one process per format with one warmup and four measured requests. Each request prefills the same 1,024 token IDs and decodes 32 forced inputs. Both formats predicted the same 32 IDs on all five requests. Packing saved 975,175,680 peak GPU bytes. The small throughput differences need a repeated paired comparison. [Requests, input and executable hashes, memory samples, score aggregates and fingerprint hashes](5070-parity.json).

The raw and mixed paths produced identical full-vocabulary FP32 fingerprints at 1,530 scored positions on six difficult chunks with Tensor Core prefill and at 255 positions with the generic path. The latter uses mapped host scratch. This covers two execution paths on one GPU with the stated inputs.

## Reproduction

Use engine commit `9f65bb2`, the weights above and the [build, observer and fixed-input instructions from the gate/up experiment](https://github.com/tjm8874/LeanSonic/blob/7f8ebe1/docs/benchmarks/inkling-mixed-even/README.md). Build both host and cubin from the candidate commit. For the raw arm set both variables to 0. For the packed arm set both to 1.

```sh
export LEANSONIC_MIXED_EVEN_UP=1 LEANSONIC_MIXED_EVEN_DOWN=1
```

Keep both ternary flags disabled, FP16 KV, one sequence, context 8,192 and prefill workspace 1,024. For the diagnostic chunks use workspace 512, a separate row output per invocation, and compare the raw and mixed row files byte-for-byte within each prefill mode.

## Full test split after integration with main

[Integration commit bd048b8](https://github.com/tjm8874/LeanSonic/commit/bd048b8) combines the candidate with main cce6f21. On the WikiText test split the four runs (raw and packed on each of RTX PRO 4500 and RTX PRO 5000) produced the same ordered full-logit fingerprint file across all 25,500 positions. Mean KL was 0.0172238, worst KL 6.1395663, BF16 top-1 agreement 24,071/25,500 and PPL 7.128123. The worst row is chunk 2, position 478. The pre-integration 571-expert build measured a worst KL of 2.8585658. Use the integration commit when reproducing the new result. [Four score reports and whole-row digests](main-cross-gpu.json).

For this split, use the observer and flags above with the full WikiText test reference and `--chunks 100`. To isolate the worst row, use `--start-chunk 2 --chunks 1`.

The original mixed-reader commit 9f65bb2 also completed both 100-chunk BF16 splits on RTX 5070 Ti. All score fields, including the worst row and top-1 counts, equal the corresponding uncompressed 571-expert results on that card. [Full test and validation scores with reference hashes](5070-corpus.json).

Comparing the original reader on RTX 5070 Ti with the integration build on RTX PRO 4500, the predicted token changed at 455 of 25,500 test positions. Of these changes, 215 lost BF16 top-1 agreement and 202 gained it. The largest increase in KL came at chunk 2, position 478, from 0.0354035 (candidate and BF16 both predict 44259) to 6.1395663 (candidate predicts 39618). These builds differ in more than one source change. [Largest increases and decreases, split-level metrics and row hashes](main-row-changes.json).

## RTX 5060 Ti reproduction

With bd048b8 and the same 571-expert package, raw vs packed peak process VRAM was 15,661,531,136 vs 14,686,355,456 bytes, a reduction of 975,175,680 bytes. Warm prefill measured 3,982.35 vs 3,959.01 tok/s and decode measured 94.54 vs 95.50 tok/s in the single-process pilot. All 32 predictions matched on five requests. The 1,530 Tensor Core and 255 generic-path full-logit fingerprints matched within each mode. [RTX 5060 Ti process readings and score files](5060-parity.json).

## Paired native comparison on RTX 5060 Ti

Five mirrored cycles used the same 1,024 IDs, 32 teacher-forced decode steps, one sequence, 8,192 context and FP16 KV. Each engine contributed 40 warm requests across ten fresh processes. The GPU peak is the maximum observed NVML process reading including loading. [Inputs, package and executable hashes, request times, predictions and paired confidence intervals](5060ti-native.json.gz).

| Engine | Prefill tok/s | Decode tok/s | Package bytes | Peak process GPU bytes |
| --- | ---: | ---: | ---: | ---: |
| LeanSonic | 3,949.17 | 95.42 | 11,487,467,790 | 14,686,355,456 |
| llama.cpp IQ3_XXS | 1,090.47 | 56.40 | 11,663,309,504 | 14,935,916,544 |

The LeanSonic package counts the weights and tokenizer. The GGUF includes its tokenizer. Timings cover the individual engines at this input shape and do not include process startup.

This measurement used LeanSonic bd048b8 and the Unsloth llama.cpp build 12588 (883286076) with Inkling-2-Mini-UD-IQ3_XXS.gguf. LeanSonic used 175,841,714 fewer distribution bytes and 249,561,088 fewer peak process GPU bytes. On the 5060 Ti, the paired prefill speedup was 3.622× (95% paired-cycle bootstrap interval 3.621–3.622) and decode speedup was 1.692× (1.691–1.693). The full same-card BF16 corpus comparison is tracked in issue #35.
