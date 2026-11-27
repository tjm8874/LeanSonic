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
