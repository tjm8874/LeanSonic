# Seven-bit four-bank package versus native vLLM on RTX PRO 5000

LeanSonic 9bc4d48 uses the 13,437,429,480-byte seven-bit package. Native vLLM 8c17e469 uses the 19,555,785,477-byte official non-MTP package at b85c3cfa. Both disable speculation, use one sequence, 2,048-token context, FP16 KV, 1,024 input tokens and the same 32-token decode history. Each engine on PRO 5000 follows the PRO 5000 native history. Across GPUs the prompt and first three generated IDs agree, then vLLM chooses token 154978 on PRO 5000 and 220 on PRO 4500. Within-card timings remain paired; the fourth and subsequent decode positions do not share a fixed trajectory across GPUs.

4 of five mirrored cycles are complete, with 56 warm requests per engine and two fresh processes per arm per cycle. The NVML peak includes model loading and the vLLM child worker.

| Engine | Prefill tok/s | Decode tok/s | Peak GPU bytes | Distribution bytes |
| --- | ---: | ---: | ---: | ---: |
| LeanSonic seven-bit | 11,434.61 | 141.77 | 15,674,114,048 | 13,437,429,480 |
| vLLM native | 12,298.60 | 138.37 | 22,861,053,952 | 19,555,785,477 |

Use the [same two-engine protocol, runner and timer definitions](../../vllm/README.md#reproduce) with the seven-bit [model hashes](../package.json) and an RTX PRO 5000. Generate the anchor locally with `python3 leg.py anchor vllm` before the five cycles. [GPU, package and executable hashes, predicted IDs, warm/cold samples and all peaks](measurement.json.gz).

## BF16 quality on PRO 5000

The normal seven-bit CLI and native vLLM scored the same 25,500 target positions per split on this GPU. Both use the same BF16 WikiText oracle and Rust scorer. The native worker exposes the full FP32 vocabulary before forcing each ground-truth token. The rank row files also match the earlier PRO 4500 five-bit files byte for byte.

| Split | Engine | Mean KL | Worst KL | BF16 top-1 agreement | PPL |
| --- | --- | ---: | ---: | ---: | ---: |
| test | LeanSonic seven-bit | 0.0077481 | 1.4620404 | 96.224% | 7.083731 |
| test | Native vLLM | 0.0109014 | 7.3723311 | 95.553% | 7.098681 |
| valid | LeanSonic seven-bit | 0.0069527 | 0.7446714 | 96.647% | 6.530797 |
| valid | Native vLLM | 0.0093524 | 2.5101105 | 95.910% | 6.518925 |

On test, native vLLM chose different top-1 tokens on 1,354 of 25,500 identical prefixes between the two measured GPU runs. The PRO 5000 native worst position was chunk 2, position 478. At that same position KL was 7.372331 for native PRO 5000, 0.025777 for native PRO 4500 and 0.032423 for the ranked model. These observations do not identify the cause of the native difference. [Paired chunk intervals and outliers](paired-test.json).

On valid, native vLLM chose different top-1 tokens on 1,180 of 25,500 identical prefixes between the two measured GPU runs. The PRO 5000 native worst position was chunk 64, position 494. At that same position KL was 2.510111 for native PRO 5000, 2.894327 for native PRO 4500 and 0.231688 for the ranked model. These observations do not identify the cause of the native difference. [Paired chunk intervals and outliers](paired-valid.json).

[Score JSON, cross-card counts and row hashes](quality.json). Use the [pinned full-vocabulary capture and corpus reproduction](https://github.com/tjm8874/LeanSonic/tree/4d40691/docs/benchmarks/inkling-class-frames/three-engines/quality) on the selected GPU.

## Repeated native capture

Two new native vLLM processes per GPU each reran the first three WikiText test chunks, with the same capture and scorer as the 100-chunk measurement. On each GPU, all 765 rows from each new process matched the original rows on that GPU byte for byte, including every full-vocabulary fingerprint. This reproduces the PRO 5000 KL of 7.372331 at chunk 2 position 478 in both repeats. The PRO 4500 original and both repeats had KL 0.025777 at the same position. This is a repeatable difference between these two configurations. Its cause is open. To repeat, run the full-vocabulary capture in the preceding section twice in fresh processes, with the first three chunks of the pinned test oracle and the same device selected for all three captures. [Per-GPU row hashes, mismatches and maxima](repeated-native.json).

## Kernel timeline on RTX PRO 5000

A single Nsight Systems capture used the same seven-bit package, 1,024 prompt IDs and 32 forced decode IDs as the paired speed comparison. The 1,324 prefill launches occupied 89.924 ms of an 89.940 ms span. The NVFP4 gate/up kernel was the only active kernel for 33.012 ms (36.70% of the span). The corresponding down kernel ran alone for 13.277 ms (14.76%). This makes the gate/up token-warp mapping from #238 a candidate for a same-card replay.

The 23,328 decode launches spanned 235.277 ms. Two kernels overlapped for 137.466 ms. The 3072-column Q8 GEMV accumulated 100.536 ms, but ran alone for just 2.200 ms. The NVFP4 gate/up and down kernels were the largest sole-active intervals, 30.102 and 16.466 ms. Adding kernel durations would double-count overlapping work. [Timeline totals, overlap depths and named kernel pairs](phase-profile.json).

Reproduce with the [pinned CUDA profiler harness and analysis](https://github.com/tjm8874/LeanSonic/tree/4d40691/docs/benchmarks/inkling-class-frames/three-engines), the seven-bit model, the local PRO 5000 native anchor and two separate Nsight Systems captures. Enable CUDA graph node tracing and capture the cudaProfilerApi interval around each isolated phase.

## Requesting native top log probabilities

Two fresh vLLM processes per GPU generated 33 tokens on the original 1,024-token prompt. Each process alternated four uncached requests with logprobs unset, 8, unset, 8. Within each process the 33 IDs were identical in all four requests. Both PRO 4500 processes reproduced the timing anchor. Both PRO 5000 processes generated token 220 at the fourth position, where its earlier timing anchor has 154978. In these follow-ups, the first difference across GPU runs was token 23. At position four, the top-two IDs were 220 and 154978 on both cards, with different scores.

The original timings still used the asserted local history on every leg. These additional runs show that the fourth-token difference cannot be attributed to card model alone. Requesting the eight log probabilities did not change trajectories within these processes. [Per-request IDs and top scores](native-trajectories.json). To rerun, select each device, use the pinned vLLM environment and publisher model from the timing protocol, and run [native_top.py](native_top.py) twice in fresh directories with the prompt and matching historical anchor saved as prompt.ids and expected.ids.
