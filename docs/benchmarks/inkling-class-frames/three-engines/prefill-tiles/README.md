# NVFP4 prefill tile comparison at 1,024 tokens

On RTX 5060 Ti, the 128- and 160-token dispatches produced identical full-vocabulary FP32 logits at all **132 positions** (4 inputs × 33 positions, 200,058 logits each). The 105,630,624 bytes from each run have SHA-256 a71f51347b117e25e19e3eeaa93af223e0f80807054ba878ede5cc16fd719162. Each row was checked for finite values and its argmax against the engine. [Inputs](fixtures.ids) · [Binary and output digests](parity.json).

Use the 571-slice class-frame model and replay environment described in the [parent benchmark](../README.md). Compile check.rs as a leansonic-core release example at a1592f3. The fixtures are the README prompt with native vLLM history, and three contiguous 1,056-token sequences starting at WikiText test chunks 0 and 35 and validation chunk 84. The first 1,024 tokens form the prefill. Feed the remaining 32 tokens as fixed continuations. From separate processes run:

```sh
env -u LEANSONIC_NVFP4_T160_MIN ./check MODEL fixtures.ids tile128.f32
LEANSONIC_NVFP4_T160_MIN=1024 ./check MODEL fixtures.ids tile160.f32
cmp tile128.f32 tile160.f32
```

The existing 128-token grouped NVFP4 kernel tests `wt * 64 < live` and `wt * 64 + nt * 8 < live` before its MMA work. An average of 64 routed tokens per expert does not imply 64 unused MMA rows in every 128-row tile. A 64-token tile can reduce shared-memory and idle-warp costs, but can also reread the same expert weights when an expert receives more than 64 tokens. Measuring the per-expert counts and phase timings distinguishes these effects.

## Matched PRO 4500 timing

Five alternating ABBA cycles, 20 fresh processes and 70 warm forced-history requests per arm gave **8,545.68 vs 8,213.95 prefill tok/s** for 128 vs 160, and **124.666 vs 124.688 decode tok/s**. The paired 160/128 prefill speed ratio is 0.9612 (cycle-bootstrap 95% interval 0.96105–0.96134); decode is indistinguishable (0.99981–1.00055). Keep 128 for this GPU, workload and package. [Every process and phase timing](timing.json).

The Nsight Systems trace uses the same model, GPU, prompt and 32 forced decode inputs. Run one warm request, reset, then capture the second prefill or decode only using the [profile example](profile.rs) compiled at a1592f3. Use the parent replay environment and set LEANSONIC_CTX=2048. Run the following for each phase. Both generations must yield the same 33 predictions as the timing receipt.
```sh
nsys profile --sample=none --cpuctxsw=none --trace=cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop -o prefill ./profile prefill MODEL
nsys export --type sqlite --output prefill.sqlite prefill.nsys-rep
python3 analyze.py prefill.sqlite
```

The prefill trace spans 118.915 ms across 1,324 kernels, with 0.0049 ms between kernel intervals. Largest contributors:

| Kernel | Launches | Total ms | Fraction of span |
|---|---:|---:|---:|
| `ink2_moe_nvfp4_tc_up_situ_packed_scales` | 36 | 35.215 | 29.6% |
| `ink2_moe_nvfp4_tc_down_packed_scales` | 36 | 14.332 | 12.1% |
| `ink2_q8_tc_gemm_b` | 68 | 10.057 | 8.5% |
| `ink2_moe_mixed_even_tc_up_situ_packed_scales` | 8 | 8.559 | 7.2% |
| `ink2_q8_tc_qkv3_gemm_b` | 28 | 7.264 | 6.1% |
| `ink2_flash_attn_tc_q8_b` | 48 | 5.664 | 4.8% |

[Full prefill kernel totals](prefill.json). These are instrumented GPU times. The process latencies above come from the unprofiled replay.

The decode trace spans 271.865 ms across 23,328 kernels, with 14.2650 ms between kernel intervals. Largest contributors:

| Kernel | Launches | Total ms | Fraction of span |
|---|---:|---:|---:|
| `ink2_q8_dp4a_gemv_c3072_r1` | 1536 | 94.995 | 34.9% |
| `ink2_gqa_attn_g1_q8_decode` | 1536 | 94.452 | 34.7% |
| `ink2_moe_nvfp4_dp4a_up_half_q8_packed_scales` | 1152 | 36.671 | 13.5% |
| `ink2_q8_dp4a_qkv3_gemv_r1` | 896 | 24.692 | 9.1% |
| `ink2_moe_nvfp4_dp4a_down_xw_conv_packed_scales` | 1152 | 21.006 | 7.7% |
| `ink2_q8_dp4a_gemv_c2304` | 32 | 19.078 | 7.0% |

[Full decode kernel totals](decode.json). These are instrumented GPU times. The process latencies above come from the unprofiled replay.

Some decode kernels run concurrently. The summed durations in the table can exceed the wall-clock span. In this trace two or more kernels were active for **132.670 ms**. The longest intervals with just one active kernel belong to `ink2_moe_nvfp4_dp4a_up_half_q8_packed_scales` (35.121 ms) and `ink2_q8_dp4a_gemv_c2304` (18.591 ms). Single-kernel occupancy is a scheduling observation; it does not give the latency saved by optimizing that kernel. [Decode overlap](decode-overlap.json) · [Prefill overlap](prefill-overlap.json) · [SQLite interval analysis](overlap.py).
