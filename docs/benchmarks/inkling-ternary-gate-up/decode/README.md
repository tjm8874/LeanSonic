# Two-bit gate/up in full inference

Both arms run 1976fb3 with two-bit down enabled. The control reads full-width gate/up matrices and the candidate stores resident gate/up weights in two bits. This first runtime expands a complete gate/up matrix before grouped prefill.

Inkling-2-Mini uses its 7,718,950,382-byte direct-coded ternary target and tokenizer, one sequence, context 4,096, FP16 KV, no drafter, global PDL and 24 resident expert layers. Every fresh process ran four passes for each of three prefix lengths, resetting between requests. Each arm contributed 30 measured requests per prefix after omitting the first pass. Five four-leg cycles alternate ABBA/BAAB. The 64 freely emitted IDs, successor and full terminal FP32 logit fingerprint matched for all 240 requests on each reported GPU. Intervals resample whole cycles.

## 5070Ti

| Prefix | Prior / candidate prefill ms | Prefill difference 95% interval, ms | Prior / candidate decode tok/s | Decode 64 difference 95% interval, ms |
|---:|---:|---:|---:|---:|
| 32 | 35.140 / 66.169 | +31.012 to +31.047 | 184.82 / 203.44 | -31.863 to -31.563 |
| 1024 | 124.793 / 155.857 | +31.032 to +31.091 | 133.27 / 142.73 | -32.039 to -31.638 |
| 2048 | 208.243 / 239.429 | +31.157 to +31.216 | 127.34 / 135.89 | -31.853 to -31.436 |

Peak observed process memory was 15,028,191,232 bytes for the control and 12,039,749,632 bytes for the candidate. [GPU UUID, samples, complete per-request measurements and token traces](5070Ti.json).

## Reproduce

Use the [same model, decode driver and prompt](../../inkling-ternary-down/engine/). Build that driver against 1976fb3d70f6c50694f10370f070b57846f34c09 and compile its sm_120a cubin. Set LEANSONIC_TERNARY_DOWN=1 throughout. Keep one engine.cubin from this revision. Toggle LEANSONIC_TERNARY_UP between 0 (prior) and 1 (candidate). The [process launcher](leg.py) runs a fresh executable named bench in each leg. Provide prompt.ids and a file named model-path containing the absolute model directory, export the full GPU UUID and the same environment switches as the linked driver, then run the mirrored four-leg cycles described above.
