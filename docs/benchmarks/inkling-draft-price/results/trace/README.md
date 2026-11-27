# Token-aligned cooldown observations on RTX 5070 Ti

This separate instrumented run collected one warm-up and one scored replay for each of four prompts with stock and priced K=7. All 16 requests matched the corresponding scalar anchor on 128 returned IDs, successor and final FP32 fingerprint. Every round was checked for contiguous token coverage. A trace entry is timed before log serialization. One trace per arm localizes the time spent; the [five-cycle uninstrumented results](../) measure throughput.

Each interval starts and ends on common generated-token boundaries and includes at least one priced skip. Accepted and proposed counts include any copied proposals within that interval.

| Input | Generated positions [start,end) | Stock ms | Priced ms | Stock accepted/proposed | Priced accepted/proposed | Priced skips |
|---|---|---:|---:|---:|---:|---:|
| ja_name | 31–40 | 35.13 | 52.08 | 6/6 | 0/0 | 9 |
| ja_name | 49–64 | 94.32 | 85.93 | 7/18 | 1/1 | 13 |
| ja_name | 100–124 | 136.84 | 142.78 | 11/19 | 0/0 | 24 |
| ja_reasoning | 22–40 | 114.77 | 102.81 | 9/25 | 1/1 | 16 |
| ja_reasoning | 71–95 | 157.44 | 140.95 | 4/11 | 0/0 | 24 |
| ja_reasoning | 108–127 | 125.70 | 113.17 | 3/8 | 0/0 | 19 |
| ja_stock | 101–128 | 153.26 | 154.91 | 11/16 | 2/2 | 24 |
| readme1024 | 41–65 | 203.88 | 188.29 | 7/25 | 0/0 | 24 |

The name prompt lost 16.95 ms over positions 31–39: stock accepted all six proposals while priced generation skipped nine rounds. Later the same prompt benefited during positions 49–63. On the reasoning prompt all three listed intervals benefited from suppressing proposals. These reversals within a prompt motivate an early probe during a long pause.

To repeat, use the source and environment from the [main experiment](../../), including the [fixture manifest](../5070Ti.json). Place a clean checkout of source `3c843d8` in a subdirectory named `src`. From its parent run `python3 instrument.py`, copy `trace.rs` to `src/crates/leansonic-core/examples/price_trace.rs`, then build that example in release mode. Use the four whitespace-separated token files with the names shown in the table. From their directory, run `price_trace "$SPEC_MODEL" 7` with `LEANSONIC_SPEC_COPY=32` and `LEANSONIC_SPEC_PRICE=0` then `1`; capture stdout and stderr separately. [Round timings, requests and interval totals](verified.json).
