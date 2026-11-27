# Attention rounding on the 571-slice compact model

On the packed RTX 5060 Ti build bd048b8, the normal G1 dispatch and existing per-head fallback produce identical full-vocabulary fingerprint and score files on all 255 positions of WikiText test chunk 2. Changing only the G1 compiled function to its arithmetic before cce6f21 changes position 478 as follows. The BF16 reference predicts ID 44259.

| Control | Predicted ID | KL at 478 | Mean KL over the 255 positions |
| --- | ---: | ---: | ---: |
| Current G1 | 39618 | 6.139566282 | 0.045409443 |
| Current per-head fallback | 39618 | 6.139566282 | 0.045409443 |
| Earlier G1 weighted-value order | 44259 | 0.035403482 | 0.020050878 |

The current control reproduces every row from the complete 25,500-position same-card run. Comparing disassembly instruction and control words for 243 symbols finds one changed function, `ink2_gqa_attn_g1_q8_decode`, still 760 instructions long. This isolates a binary change on one package and one input. The earlier G1 alone loses the established dispatch-parity property. The candidate gives the grouped and per-head FP16 paths an explicit shared recurrence: round the rescaled prior accumulator, then fuse the probability/value product into it.

Reproduce the three control arms by building bd048b8 and its sm_120a cubin (CUDA 13.0.88), then separately reverting only the CUDA hunk from cce6f21 and rebuilding the second cubin with identical flags. Load the [571 package and switches](https://github.com/tjm8874/LeanSonic/tree/9fd67b1/docs/benchmarks/inkling-mixed-even-down). Enable both packed projections, use FP16 KV, context 8,192 and prefill width 512. Run `eval-kld --weights MODEL --ref TEST.kld --start-chunk 2 --chunks 1`. The fallback uses the current cubin with LEANSONIC_G1_ATTN=0. The other two use LEANSONIC_G1_ATTN=1. See the [three row hashes and full scores](isolation-5060.json) and [per-function comparison](isolation-sass.json).

The [native three-engine comparison](https://github.com/tjm8874/LeanSonic/tree/eab21bd/docs/benchmarks/inkling-class-frames/three-engines/quality) observed vLLM KL 0.0257771 with the same BF16 top on this position. The [existing FP64 attention fixtures and full corpus results](../inkling-g1-parity/README.md) remain relevant to evaluating the shared recurrence.
