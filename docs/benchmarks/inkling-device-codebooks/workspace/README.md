# Prefill workspace placement on 16 GB Blackwell

All runs use the same d08c1cf binary and compact model with four-bit device storage, FP16 KV, one sequence, context 8,192, a 1,024-token prompt and 32 forced decode steps. The control uses automatic workspace selection. The candidate sets LEANSONIC_PREFILL_CHUNK=1024 before loading expert weights. Each arm contributes 40 warm requests from ten processes, with five mirrored cycles. Peak is sampled process GPU allocation during load and execution.

| GPU | Workspace | Prefill ms (tok/s) | Decode 32 ms (tok/s) | Peak GPU bytes |
|---|---|---:|---:|---:|
| RTX 5060 Ti | Auto | 403.539 (2,537.6) | 366.379 (87.34) | 16,588,472,320 |
| RTX 5060 Ti | 1,024 at load | 274.258 (3,733.7) | 366.434 (87.33) | 16,338,911,232 |
| RTX 5070 Ti | Auto | 539.970 (1,896.4) | 205.207 (155.94) | 16,596,860,928 |
| RTX 5070 Ti | 1,024 at load | 130.724 (7,833.3) | 205.260 (155.90) | 16,437,477,376 |

## Allocation trace

On 5060Ti the active widths before the five automatic requests were [64, 512, 512, 512, 512]. With an explicit 1,024-row workspace they were [1024, 1024, 1024, 1024, 1024].

The separate diagnostic build adds logging inside ensure_prefill_capacity and uses the same CUDA module and inputs. On RTX 5070 Ti the automatic workspace started at 64. Each attempt to allocate 1,024, 512 or 256 rows failed once the experts were on the GPU. The first request expanded to 128. Every subsequent request retried the failed allocations and processed the prompt in 128-row chunks. With the explicit width, all five requests started at 1,024 rows with 24/24 expert layers still resident. The complete per-request prediction streams matched in the paired trial.

To reproduce, use the [parent build and fixed-token inputs](../README.md) on an otherwise idle GPU. Export LEANSONIC_MXFP8_DEVICE_U4=1 and leave LEANSONIC_MXFP8_LOOKUP_PREFILL unset. For the control, unset LEANSONIC_PREFILL_CHUNK. For the other arm, export LEANSONIC_PREFILL_CHUNK=1024. Restart the executable for each leg. The paired results include process startup memory but exclude the first request from latency. The workspace experiment changes neither the distribution bytes nor the on-disk quantization.

[All timings](runs.csv) · [Paired intervals, output IDs and allocation failures](receipt.json)
