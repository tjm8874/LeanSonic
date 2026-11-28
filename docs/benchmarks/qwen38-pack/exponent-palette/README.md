# Lossless expert exponent palettes on Blackwell

Each routed expert projection has 51,200 scale/bias pairs. Stock Z24 retains their sign and fraction in a 16-bit plane and their two exponent codes in an 8-bit plane. On all 73,728 expert projections in this checkpoint, the exponent-code pair took at most 16 distinct values within each expert. The experiment encodes the upper plane as one 128-byte aligned palette followed by 25,600 bytes of nibble indices per expert. The rank in the palette is assigned in first-seen order. The existing 16-bit plane is unchanged.

All 3,774,873,600 pairs were verified against the source BF16 values on the host. The new upper planes were physically written and reread (1,896,873,984 bytes vs 3,774,873,600). A separate CUDA decoder checked every group on the RTX PRO 4500. The upper-plane allocation effect in that stand-alone test includes allocator rounding per projection. [Host readback and palette sizes](z24-four-bit.json) · [Device verification](z24-gpu.json)

In the full engine on an RTX PRO 5000 72 GB, the new layout lowered the sampled process peak from **70.256 to 68.475 GiB**, freeing **1.781 GiB** while retaining 48/48 expert layers in VRAM. Raw and palette variants loaded the same directly loadable Q4 checkpoint. Full-vocabulary FP32 logits were byte-identical at all 49 teacher-forced positions, and all four greedy continuations matched. An alternating stock/palette/palette/stock trial used resident greedy processes at context 256, with one warmup and three measurements per input per process. All generated IDs matched. [Individual samples and quality scores](native72.json)

| Input / output | Median TTFT stock / palette (ms) | Decode stock / palette (tok/s) |
| --- | ---: | ---: |
| 16 / 32 | 92.44 / 95.92 | 162.97 / 157.39 |
| 128 / 32 | 745.96 / 773.72 | 158.98 / 153.69 |

The build is opt-in. The measured 3.3–3.4% standalone decode cost matters when deciding whether to spend the freed VRAM on another component such as MTP. With the drafter from PR 254 loaded, its MoE then fit in VRAM. The measured draft step fell from 1.79 to 0.58 ms, while the target forward increased from 6.29 to 6.50 ms. Batched verification and rollback are outside that fixture. [Combined run and token-by-token checks](https://github.com/tjm8874/LeanSonic/blob/550665c/docs/benchmarks/qwen38-pack/exponent-palette/mtp.md). The distribution checkpoint is shared between both runs.

For the full-engine run, copy the [teacher example](../teacher.rs) and [stream example](../../qwen38-engine-comparison/q38_stream.rs) into a scratch checkout's `crates/leansonic-core/examples`. Build both the Rust feature and its matching cubin. Use the quality and stream commands from the [parent benchmark](../README.md) with the same model and fixture IDs.

```sh
cargo build --release -p leansonic-core --features q38-exponent-palette --example q38_teacher --example q38_stream
nvcc -arch=sm_120a -O3 -std=c++17 -Ikernels -DLEANSONIC_Q38_EXPONENT_PALETTE --cubin kernels/qwen38_sm120.cu -o /tmp/q38-exp4.cubin
export LEANSONIC_QWEN38_CUBIN_PATH=/tmp/q38-exp4.cubin
```

For the independent upper-plane audit, run from a scratch directory containing the ordinary Q4 checkpoint at `restored`. Set `d` to this benchmark directory's absolute path.

```sh
python3 "$d/z24_spec.py" > pairs.spec
c++ -O3 -std=c++17 "$d/z24_palette.cpp" -o palette-audit
./palette-audit pairs.spec new-packed-planes
nvcc -O3 -arch=sm_120 -std=c++17 "$d/z24_gpu.cu" -o verify-on-gpu
./verify-on-gpu new-packed-planes
```

## 32 GB cross-check

On RTX PRO 4500 32 GB, automatic placement used the saving to fit **20/48** expert layers in VRAM, up from 19/48. Sampled process peak consequently rose from 29.793 to 30.443 GiB. For the 16-token input, TTFT was 723.92 → 698.66 ms and decode was 21.84 → 22.64 tok/s. For the 128-token input, TTFT was 5798.46 → 5591.14 ms and decode was 21.78 → 22.57 tok/s. These are one process per arm, one warmup and three repeats per case. The 49 full-vocabulary distributions, four eight-token continuations and repeated 32-token streams matched. On this card both changed MoE kernels also passed the exact-size Compute Sanitizer memcheck with 65,536 bytes of padding and zero reported errors. The compiled kernels used zero local bytes and zero stack bytes. [32 GB timing samples and quality](native32.json)

### Isolating the extra resident layer

On the PRO 4500, a follow-up pinned `LEANSONIC_MAX_RESIDENT_LAYERS` to 19 or 20. All six launches used the same directly loadable checkpoint, context 256, two prompts and 32 greedy output tokens. The order was stock 19, palette 19, palette 20, palette 20, palette 19, stock 19. The engine confirmed every requested placement. All 48 streams, including warmups, produced the same IDs for their respective prompts. Each timing below is a median over six measured requests. Peak VRAM was sampled for the process.

| Wave 1 layout | Peak GiB | TTFT, 16 / 128 tokens (ms) | Decode, 16 / 128 input tokens (tok/s) |
| --- | ---: | ---: | ---: |
| Stock, 19 layers | 29.793 | 724.73 / 5832.80 | 21.84 / 21.24 |
| Palette, 19 layers | 29.088 | 720.81 / 5768.68 | 21.85 / 21.92 |
| Palette, 20 layers | 30.443 | 701.62 / 5633.50 | 22.64 / 22.29 |

With the palette binary held fixed, moving one additional expert layer to VRAM increased median decode by 3.6% on the short prompt and 1.7% on the long prompt. At 19 layers the short-prompt decode rate differed by 0.01 tok/s between builds. The long-prompt trials showed greater launch-to-launch variation. The stock and palette executables also printed different host Engram size estimates (29.8 and 27.6 GiB) for the same checkpoint path. The table reports process VRAM from NVML. [Timings, loader logs, token IDs and binary hashes](matched32.json)

## Wave 3 integration

The optional palette build at `000c80e` uses the packed exponent reader in both grouped batched MoE kernels. The GPU test allocates each upper plane at its actual packed size. On the PRO 5000 both grouped kernels passed Compute Sanitizer memcheck with 65,536 bytes of padding. Per-kernel names and test output are in the [Wave 3 results](wave3.json).

The stock and palette builds produced byte-identical logits at all 49 scored positions on both the PRO 5000 and PRO 4500. All four eight-token continuations and both 32-token streams matched on each GPU. On the PRO 5000 we also submitted four fixed sequences containing 53 tokens to `forward_verify_tokens`, using chunk widths from 1 through 8 after a fresh reset each time. All 424 returned top-token IDs in each build matched its serial run and the other build. The verifier returns argmax IDs at this boundary. To repeat, copy [verify_widths.rs](verify_widths.rs) into the core crate's examples, build it in the same configurations as the teacher example, and run it with the model path and `fixtures.ids`.

| GPU and layout | Expert layers in VRAM | Sampled peak (GiB) | Prefill to first token, 16 / 128 (tok/s) | Decode, 16 / 128 (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| PRO 5000 stock | 48 | 70.303 | 398.17 / 494.25 | 163.18 / 159.14 |
| PRO 5000 palette | 48 | 68.521 | 395.77 / 488.94 | 157.40 / 153.69 |
| PRO 4500 stock | 19 | 29.840 | 57.53 / 66.39 | 21.85 / 21.78 |
| PRO 4500 palette | 20 | 30.490 | 57.17 / 66.05 | 22.66 / 22.60 |

Each cell uses one warm resident process per build and GPU, one warmup and three measured requests at context 256 with 32 greedy output tokens. On this checkpoint the 72 GB card can trade 3.4% of long-prompt decode throughput for 1.781 GiB of space. On the 32 GB card automatic placement spends the saving on one more resident expert layer and increases long-prompt decode by 3.8%. The default remains the stock layout.

### Cost of target verification on PRO 5000

The same 16-token Japanese prompt and 32 continuation inputs exercised the target only, in four stock/palette/palette/stock process launches. Each entry is the median elapsed host call after a warmup cycle, pooled across three measured cycles in each process. Every output ID was checked against serial greedy decoding at the corresponding position.

| Method or verify width | Stock ms/call | Palette ms/call |
| --- | ---: | ---: |
| Serial greedy, one token | 6.129 | 6.347 |
| Verify 1 | 10.126 | 9.979 |
| Verify 2 | 12.655 | 12.465 |
| Verify 4 | 17.804 | 17.584 |
| Verify 8 | 26.183 | 26.034 |

At width two, two serial palette calls cost about 12.694 ms at the measured median. A verify call cost 12.465 ms, leaving only 0.229 ms for other work even if both target predictions can be used. The stock verify call exceeded two stock serial calls. Actual speculative throughput additionally depends on the drafter, acceptance and state management. Reproduce with the same environment as the teacher example, copying [q38_verify_cost.rs](q38_verify_cost.rs) into the core examples directory and passing the checkpoint and fixture IDs. [All per-call timings and fixed IDs](verify-cost.json)

## Full greedy and speculative requests after main 49dade3

Using the same PRO 5000 72 GB and direct Q4 model, EXP4 freed room for the MTP MoE in VRAM. With stock Z24 the MTP MoE was pinned. Each resident process ran all four held-out prompts through warmup and three timed cycles, requesting 128 tokens with stop IDs disabled and context 256. Launch order was stock K0, EXP4 K1, stock K1, EXP4 K0, EXP4 K1, stock K0. Each generation starts with fresh recurrent and KV state. The timer covers the native library call and excludes model loading. Decode throughput counts the 127 inter-token intervals using total request time minus the library's measured prompt phase.

| Input | Stock greedy | EXP4 greedy | Stock K1 | EXP4 K1 | K1 accepted drafts |
| --- | ---: | ---: | ---: | ---: | ---: |
| English | 161.18 | 155.70 | 109.15 | 144.58 | 24.3% |
| Japanese | 160.93 | 155.46 | 141.99 | 183.45 | 56.1% |
| Code | 161.63 | 156.11 | 157.49 | 200.82 | 72.0% |
| Arithmetic | 161.16 | 155.68 | 152.80 | 195.85 | 66.2% |

EXP4 K1 beat stock greedy on the three inputs whose full outputs agreed. Median total request times (stock greedy → EXP4 K1) were 850 → 769 ms in Japanese, 803 → 652 ms on code, and 835 → 708 ms in arithmetic. The English sequence reached its first configured EOS at zero-based index 64 in both modes. With stopping disabled, K1 returned token 248045 at index 80, versus 271 for greedy. Ten of 128 English output positions differed. Within either decoding policy the palette and raw storage produced identical IDs. This English long-run timing includes unlike continuations.

To reproduce on the tested branch, copy [spec_requests.rs](spec_requests.rs) into the core crate's examples. Build stock and with `--features q38-exponent-palette`, supplying the matching target cubin and an MTP cubin compiled from `kernels/qwen38_mtp_sm120.cu` through `LEANSONIC_QWEN38_MTP_CUBIN_PATH`. Run each binary with `MODEL HELDOUT_IDS K` in the order above. The four held-out prompts are the last four lines of [route-fixtures.ids](../../qwen38-engine-comparison/routing/route-fixtures.ids). [Every request and acceptance count](spec-timings.json) · [All input and output IDs](spec-regression.json).
