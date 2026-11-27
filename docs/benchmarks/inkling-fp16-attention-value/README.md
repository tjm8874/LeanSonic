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

## Shared-recurrence candidate on RTX 5060 Ti

Candidate ea3a7cf passes **704/704** operator cases comparing per-head batched vs staged FP16 attention and **704/704** comparing staged per-head vs grouped attention. These checks compare all Q8 codes and FP32 block-scale bits. The SM120a main and integration cubins pass the SASS audit. In the integration, the grouped function has exactly the earlier G1 instruction/control words; only the two per-head kernels differ from that earlier module.

Both new dispatch paths produce identical score and full-vocabulary fingerprint files across all 255 positions of test chunk 2; they also match the isolated earlier G1 run for every row. At position 478 both predict BF16 ID 44259 with KL 0.035403482. On the 66 independent FP64 fixtures the two paths agree on every code and scale. Each has two Q8 codes different from FP64 out of 202,752 (seed 303, position 31, global and local). The maximum scale error is 9.79e-10. This low-level rounding tradeoff is relevant to adoption.

For exact operator reproduction compile the repository's kernels/tests/inkling_decode_attention.cpp and run against the same candidate cubin first without a flag and then with `g1`. Run the [independent FP64 executable](https://github.com/tjm8874/LeanSonic/tree/d654dad/docs/benchmarks/inkling-mixed-ternary-down/5060Ti/matched-revisions/g1/operator) against that cubin. For the compact model use the 571-slice package, bd048b8 host, the integrated candidate cubin and the scoring flags from the isolation experiment above. [Candidate binaries, operator and corpus summaries](candidate-5060.json) · [All FP64 fixtures](operator-5060.csv).
