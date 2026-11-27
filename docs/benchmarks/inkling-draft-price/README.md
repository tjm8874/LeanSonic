# Single-stream speculation with measured round prices

The optional LEANSONIC_SPEC_PRICE=1 experiment compares the cost of eight neural attempts, including attempts that produce zero candidates, with the cost per emitted token of scalar decode with the drafter loaded. A scalar step already updates drafter context. The first naturally occurring scalar step supplies the initial price. User callbacks are outside the timers. If the pooled neural cost per committed token exceeds the smoothed scalar price by more than 2%, suspend neural proposals for up to 24 steps. History copy remains eligible during this new pause. The engine then samples neural rounds again. No language classification, host-specific timing constant or new GPU synchronization is involved.

This experiment selects between the existing K and scalar, for one greedy stream. It does not estimate the incremental price of intermediate draft depths or change multi-sequence dispatch. The stock and price modes use the same binary from 3c843d8, model, cubin and 12 ID fixtures, with 128 output tokens per request and one row readback outside timing. The driver checks the returned ID against the argmax of 200,058 finite FP32 values. The analysis compares all generated IDs, next ID and final row fingerprint against a separate scalar process on the same card. Timing uses the warmed pass. Each of five mirrored cycles comprises four fresh processes, stock/price/price/stock in even cycles and the reverse in odd cycles.

## Reproduction

Use the four-bank target from #181, the row-exponent DSpark sidecar from #185 and the [depth replay settings](https://github.com/tjm8874/LeanSonic/tree/1afcdc4/docs/benchmarks/inkling-nvfp4-scale-split/four-banks/free/depth). Copy lean.rs to crates/leansonic-core/examples/depth_replay.rs and build it in release mode. In a fresh working directory place bench (the release binary), the 12 ID files, leg.py, the cubin, and model directories named plain and spec. The spec directory includes mtp.safetensors. Expose just the selected GPU via its UUID for the NVML sampler.

```sh
export LEANSONIC_KV=fp16 LEANSONIC_PDL=1 LEANSONIC_PDL_ALL=1 LEANSONIC_SEQS=1
export LEANSONIC_MXFP8_DEVICE_U4=1 LEANSONIC_MXFP8_DIRECT_ALL=1 LEANSONIC_SPEC_COPY=32
export LEANSONIC_CUBIN_PATH="$PWD/engine.cubin"
mkdir card
python3 leg.py card/scalar scalar
for cycle in 0 1 2 3 4; do
  if ((cycle % 2)); then arms=(price stock stock price); else arms=(stock price price stock); fi
  for leg in 0 1 2 3; do
    arm=${arms[$leg]}
    python3 leg.py "card/$cycle-$leg-$arm" "$arm"
  done
done
python3 analyze.py card verified.json
```

The four additional fixtures were en_count, en_extract, ja_json and ja_filter. Each input must stay within a 1% decode regression versus stock and the Japanese reasoning gain must reproduce before considering a default change. [Predeclared scope and decision criteria](https://github.com/tjm8874/LeanSonic/issues/177#issuecomment-7223531794).

[Paired performance and correctness results](results/).
