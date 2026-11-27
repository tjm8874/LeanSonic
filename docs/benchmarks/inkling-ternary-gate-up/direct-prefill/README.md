# Direct gate/up staging during prefill

Both arms keep gate/up and down matrices in two bits on the same runtime. The control module expands each gate/up plane before grouped prefill. The new module stages its tiles directly.

Inkling-2-Mini uses its 7,718,950,382-byte direct-coded ternary target and tokenizer, one sequence, context 4,096, FP16 KV, no drafter, global PDL and 24 resident expert layers. Every fresh process ran four passes for each of three prefix lengths, resetting between requests. Each arm contributed 30 measured requests per prefix after omitting the first pass. Five four-leg cycles alternate ABBA/BAAB. The 64 freely emitted IDs, successor and full terminal FP32 logit fingerprint matched for all 240 requests on each reported GPU. Intervals resample whole cycles.

## 5060Ti

| Prefix | Prior / candidate prefill ms | Prefill difference 95% interval, ms | Prior / candidate decode tok/s | Decode 64 difference 95% interval, ms |
|---:|---:|---:|---:|---:|
| 32 | 130.348 / 49.491 | -80.873 to -80.834 | 116.53 / 116.53 | -0.097 to +0.047 |
| 1024 | 323.791 / 238.355 | -85.537 to -85.329 | 93.57 / 93.57 | -0.113 to +0.188 |
| 2048 | 496.930 / 428.638 | -68.401 to -68.119 | 90.65 / 90.65 | -0.181 to +0.183 |

Peak observed process memory was 11,953,766,400 bytes for the control and 11,794,382,848 bytes for the candidate. [GPU UUID, samples, complete per-request measurements and token traces](5060Ti.json).

## Reproduce

Use the [same model, decode driver and prompt](../../inkling-ternary-down/engine/). Build that driver against 30a41f786913dfa77155ac8f70c2140d75c8c736 and compile its sm_120a cubin. Set LEANSONIC_TERNARY_DOWN=1 throughout. Set LEANSONIC_TERNARY_UP=1 in both arms. Compile the control cubin from 1976fb3 and the candidate from 30a41f7 using the same nvcc command, and vary LEANSONIC_CUBIN_PATH. The [process launcher](leg.py) runs a fresh executable named bench in each leg. Provide prompt.ids and a file named model-path containing the absolute model directory, export the full GPU UUID and the same environment switches as the linked driver, then run the mirrored four-leg cycles described above.
