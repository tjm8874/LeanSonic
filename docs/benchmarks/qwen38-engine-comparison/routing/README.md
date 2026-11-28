# Expert placement from router traces

At the 32GB profile's current budget of 9,216 equal-size expert slots, the stock layout places all 512 experts in the first 18 layers in VRAM. We recorded all ten selected expert IDs in each of the 48 layers on a PRO 4500. The trace captured eight cases, 32 generated tokens per case. A second uninstrumented run with CUDA graphs produced exactly the same 256 output IDs.

We counted expert selections in the 31 decode steps per case. The first four prompts selected the placement and the remaining four supplied 59,520 held-out selections. Both alternatives have exactly the same byte budget as the stock layout (3,072,000 bytes per expert).

| Placement policy | Fraction of held-out selections hitting resident experts |
| --- | ---: |
| First 18 layers | 37.50% |
| Most frequent 9,216 (layer, expert) pairs from calibration | 73.63% |
| Most frequent 192 experts within each layer from calibration | 73.69% |

The policies were replayed on recorded IDs. Realizing those gains requires a mixed host/device kernel and a measured decode run. Embeddings, shared experts, router cost, and the placement indirection are outside this hit-rate measure. The four held-out cases give an initial sample across English, Japanese, code and arithmetic. [Inputs](route-fixtures.txt) and [per-case counts](route-analysis.json) are included.

## Reproduce

Use a disposable checkout of `9b3e4f2aabc5e33e73b34e0aea8e031b48aba0fe` with the comparison runner copied to `crates/leansonic-core/examples/q38_stream.rs`. Save the unmodified release binary and then run `trace_instrument.py` from the disposable checkout's root. Rebuild the release example. The diagnostic synchronizes at the router in each layer and disables CUDA graph capture. Keep both invocations on the same reserved GPU, restored checkpoint and cubin as the speed benchmark. The instrumented runner is for tracing only.

```sh
awk '{gsub(/ /,",");print "GEN 32 " $0} END {print "QUIT"}' route-fixtures.ids > route-input
"$NORMAL_RUNNER" "$RESTORED" 256 <route-input >route-golden.txt 2>route-golden.log
"$TRACE_RUNNER" "$RESTORED" 256 <route-input >route-output.txt 2>route-output.log
cmp route-golden.txt route-output.txt
python3 trace_eval.py
```

The source IDs have SHA-256 `cd0a1459401deb950951cbe891cb7b49d79d04ecf4ba6688283593b4a5d78875`. The complete graph-enabled response stream has SHA-256 `7100e88eb378ee516ead21223b05eaffc50400919c78166adbf3fbf9827a93c6`. Filtering the diagnostic stderr to lines beginning with `ROUTE ` produces SHA-256 `e81cf70e65e4d952e2af18aac4fea6a4c23457f8e7c4d433d7b6df3751bbf64c`.
