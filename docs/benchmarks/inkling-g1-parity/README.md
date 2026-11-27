# FP16 G1 rounding parity and BF16 fidelity

At fb541e3 on RTX 5060 Ti, compiled with CUDA 13.0.88, the 8-CTA G1 path reproduces the 24-CTA fallback on 704 bitwise code/scale cases (up to 131,072 context), 66 independent CPU FP64 fixtures and all 1,530 real WikiText row fingerprints from the six selected chunks. Main 2d7db2a fails the first fixture at base 6, batch 5, local capacity 1024. See [the parity command, source and fixtures](https://github.com/tjm8874/LeanSonic/pull/217) and the [matched hashes and scores](5060ti.json).

## Full WikiText effect of the two existing attention paths

On RTX PRO 5000 the 13,448,974,014-byte high-quality package and both executables from 6aa971d are identical between arms. Each split uses 25,500 BF16 positions, FP16 KV, one sequence, 512-token context and 24/24 resident expert layers. The second arm sets LEANSONIC_G1_ATTN=0. This is a full-corpus test of the existing fallback. Direct patched-kernel parity above covers the six selected chunks.

| Split | Path | Mean KL | Worst KL | Top-1 matches | PPL |
|---|---|---:|---:|---:|---:|
| Test | G1 | 0.0076817 | 1.5215775 | 24,542 | 7.079444 |
| Test | Fallback | 0.0077481 | 1.4620404 | 24,537 | 7.083731 |
| Validation | G1 | 0.0068862 | 1.1576402 | 24,655 | 6.532434 |
| Validation | Fallback | 0.0069527 | 0.7446714 | 24,645 | 6.530797 |

Worst KL falls on both splits. Mean KL rises and top-1 agreement falls on both, while PPL moves in opposite directions. The reason to consider the patch is a consistent result across the two dispatch paths. The full-corpus evidence gives no across-the-board BF16 advantage. Repeat the [original full-corpus measurement](https://github.com/tjm8874/LeanSonic/tree/588e3cd/docs/benchmarks/inkling-sglang/4096) with only the G1 switch changed. [Raw results and executable hashes](pro5000-corpus.json).

## End-to-end latency

On RTX 5070 Ti the 61-expert IQ1-size package (7,763,991,646 bytes) received 128 prompt tokens and the same 32 fixed continuation tokens. Both modules used the same compiled Rust binary, raw resident expert mode, context 8,192, one sequence, speculation off and FP16 KV. Across five ABBA/BAAB cycles, 40 warm requests per arm, parent/candidate measured 2,317.43/2,318.00 prefill tok/s and 175.495/175.511 decode tok/s. Decode speed ratio 1.00009, paired bootstrap 95% interval 0.99984–1.00034. All 32 predictions matched between the two arms on this input. Sampled peak process GPU usage was 15,833,497,600 vs 15,904,800,768 bytes. This timing uses a different model and GPU than the full-corpus check.

Build trace.rs as a leansonic-core release example on fb541e3. Point LEANSONIC_CUBIN_PATH to the parent or candidate sm_120a module. Set LEANSONIC_TERNARY_UP=0, LEANSONIC_TERNARY_DOWN=0, LEANSONIC_MXFP8_DEVICE_U4=1, LEANSONIC_MXFP8_DIRECT_ALL=1, LEANSONIC_PDL=1, LEANSONIC_PDL_ALL=1, LEANSONIC_SEQS=1 and LEANSONIC_KV=fp16. Run the example with the model directory, prompt128.ids and continuation32.ids. Discard request 0 per process. Archive all four subsequent requests and per-process peaks. [Prompt IDs](prompt128.ids) · [Continuation](continuation32.ids) · [Replay example](trace.rs) · [Timings and fingerprints](5070ti-timing.json.gz).
