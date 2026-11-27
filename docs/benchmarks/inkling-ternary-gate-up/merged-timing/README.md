# Full-width versus two-bit gate/up on the integrated runtime

Both arms run 349c7e8 after integrating main, with two-bit down matrices. The control keeps four-bit gate/up weights. The candidate keeps two-bit gate/up weights for decode and stages tiles directly for prefill. Both use the same cubin.

Inkling-2-Mini uses its 7,718,950,382-byte direct-coded ternary target and tokenizer, one sequence, context 4,096, FP16 KV, no drafter, global PDL and 24 resident expert layers. Every fresh process ran four passes for each of three prefix lengths, resetting between requests. Each arm contributed 30 measured requests per prefix after omitting the first pass. Five four-leg cycles alternate ABBA/BAAB. The 64 freely emitted IDs, successor and full terminal FP32 logit fingerprint matched for all 240 requests on each reported GPU. Intervals resample whole cycles.

## 5070Ti

| Prefix | Prior / candidate prefill ms | Prefill difference 95% interval, ms | Prior / candidate decode tok/s | Decode 64 difference 95% interval, ms |
|---:|---:|---:|---:|---:|
| 32 | 34.832 / 28.211 | -6.626 to -6.617 | 184.85 / 203.44 | -31.747 to -31.555 |
| 1024 | 124.388 / 119.993 | -4.415 to -4.375 | 133.27 / 142.73 | -31.897 to -31.764 |
| 2048 | 208.945 / 210.354 | +1.390 to +1.428 | 127.34 / 135.92 | -31.790 to -31.626 |

Peak observed process memory was 15,057,551,360 bytes for the control and 12,058,624,000 bytes for the candidate. [GPU UUID, samples, complete per-request measurements and token traces](5070Ti.json).

## Reproduce

Use the [same model, decode driver and prompt](../../inkling-ternary-down/engine/). Build that driver against 349c7e82621767fe587911951f62c5dea2ebb1ee and compile its sm_120a cubin. Set LEANSONIC_TERNARY_DOWN=1 throughout. Use the same 349c7e8 cubin for both arms. Toggle LEANSONIC_TERNARY_UP between 0 (prior) and 1 (candidate). The [process launcher](leg.py) runs a fresh executable named bench in each leg. Provide prompt.ids and a file named model-path containing the absolute model directory, export the full GPU UUID and the same environment switches as the linked driver, then run the mirrored four-leg cycles described above.
