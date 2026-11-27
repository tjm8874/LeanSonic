# Non-speculative Inkling decode after PR #89

Main at `b02cce0` reduced the 16-step decode time on PRO 5000 from 98.32 to 95.79 ms at 1,024 input tokens, relative to V12 at `7a02b538`. Basalt `5fe1928d` took 93.20 ms in the same comparison. The paired interval for main's time saved versus V12 was 2.52–2.62%. Basalt was still faster on this card (main's time saved −3.59% to −1.99%).

| GPU / input tokens | Main prefill (ms) | Basalt prefill (ms) | Main decode 16 (ms) | Basalt decode 16 (ms) | Main / Basalt peak process VRAM (GiB) |
|---|---:|---:|---:|---:|---:|
| PRO 5000 / 32 | 33.17 | 49.70 | 93.75 | 86.40 | 16.447 / 18.174 |
| PRO 5000 / 1,024 | 102.28 | 102.90 | 95.79 | 93.20 | 16.996 / 18.174 |
| RTX 5070 Ti / 32 | 107.47 | 263.40 | 166.27 | 312.50 | 15.412 / 14.453 |
| RTX 5070 Ti / 1,024 | 165.04 | 449.60 | 187.99 | 322.80 | 15.410 / 14.455 |

The three arms ran in five rotated, mirrored cycles per card. Each process ran one of the [same two prompts and CLI commands](../inkling-reference.md). Main used #85's compact model, V12 used #77's dictionary package and Basalt used its text-only container. Each arm performed 16 decode forwards. All 120 processes agreed on the 16 output IDs for their respective prompt. [Timings, peaks, outputs and binary hashes](inkling-post89-decode.json).

## Decode kernel profile

On PRO 5000 we separately profiled 64 decode steps after the 32-token prompt with the same main binary, cubin, compact text package and FP16 KV. The output matched all 64 IDs of the unprofiled control. The middle 32 steps took 6.297 profiled ms/step, including 5.894 ms of GPU kernel execution per step. The packed NVFP4 up projection used 978 µs/step (16.59% of kernel time). The generic Q8 GEMV used 784 µs/step (13.31%), followed by the expert-parallel packed NVFP4 down projection at 555 µs/step (9.41%). Use the unprofiled table for engine comparisons.

```bash
python3 -c 'from pathlib import Path; p=Path("docs/benchmarks/inputs/inkling-readme1024.ids"); print(",".join(p.read_text().strip().split(",")[:32]))' > short.ids
export LEANSONIC_KV=fp16 LEANSONIC_CTX=8192 LEANSONIC_CUBIN_PATH="$CUBIN"
nsys profile -t cuda --sample=none --cpuctxsw=none --cuda-graph-trace=node -o decode64 \
    "$CLI" bench --weights "$COMPACT" --tokens @short.ids --spec 0 --gen 64 --repeat 1
nsys export --type sqlite --output decode64.sqlite decode64.nsys-rep
```

Join `CUPTI_ACTIVITY_KIND_KERNEL.shortName` to `StringIds.id` in the SQLite export. Take the last 64 `ink2_argmax_and_inc_pos` end timestamps and sum per-kernel durations from the end of step 16 to the end of step 48, counting the 32 terminating argmax launches. The JSON contains the time and call count for each kernel inside that window.
