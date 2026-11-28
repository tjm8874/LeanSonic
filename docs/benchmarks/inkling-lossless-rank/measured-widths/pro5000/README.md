# Seven-bit four-bank package versus native vLLM on RTX PRO 5000

LeanSonic 9bc4d48 uses the 13,437,429,480-byte seven-bit package. Native vLLM 8c17e469 uses the 19,555,785,477-byte official non-MTP package at b85c3cfa. Both disable speculation, use one sequence, 2,048-token context, FP16 KV, 1,024 input tokens and the same 32-token decode history. Each engine on PRO 5000 follows the PRO 5000 native history. Across GPUs the prompt and first three generated IDs agree, then vLLM chooses token 154978 on PRO 5000 and 220 on PRO 4500. Within-card timings remain paired; the fourth and subsequent decode positions do not share a fixed trajectory across GPUs.

3 of five mirrored cycles are complete, with 42 warm requests per engine and two fresh processes per arm per cycle. The NVML peak includes model loading and the vLLM child worker.

| Engine | Prefill tok/s | Decode tok/s | Peak GPU bytes | Distribution bytes |
| --- | ---: | ---: | ---: | ---: |
| LeanSonic seven-bit | 11,436.94 | 141.80 | 15,674,114,048 | 13,437,429,480 |
| vLLM native | 12,294.23 | 138.38 | 22,861,053,952 | 19,555,785,477 |

Use the [same two-engine protocol, runner and timer definitions](../../vllm/README.md#reproduce) with the seven-bit [model hashes](../package.json) and an RTX PRO 5000. Generate the anchor locally with `python3 leg.py anchor vllm` before the five cycles. [GPU, package and executable hashes, predicted IDs, warm/cold samples and all peaks](measurement.json.gz).


## BF16 quality on PRO 5000

The normal seven-bit CLI and native vLLM scored the same 25,500 target positions per split on this GPU. Both use the same BF16 WikiText oracle and Rust scorer. The native worker exposes the full FP32 vocabulary before forcing each ground-truth token. The rank row files also match the earlier PRO 4500 five-bit files byte for byte.

| Split | Engine | Mean KL | Worst KL | BF16 top-1 agreement | PPL |
| --- | --- | ---: | ---: | ---: | ---: |
| test | LeanSonic seven-bit | 0.0077481 | 1.4620404 | 96.224% | 7.083731 |
| test | Native vLLM | 0.0109014 | 7.3723311 | 95.553% | 7.098681 |

On test, native vLLM chose different top-1 tokens on 1,354 of 25,500 identical prefixes between the two measured GPU runs. The PRO 5000 native worst position was chunk 2, position 478. At that same position KL was 7.372331 for native PRO 5000, 0.025777 for native PRO 4500 and 0.032423 for the ranked model. These observations do not identify the cause of the native difference. [Paired chunk intervals and outliers](paired-test.json).

[Score JSON, cross-card counts and row hashes](quality.json). Use the [pinned full-vocabulary capture and corpus reproduction](https://github.com/tjm8874/LeanSonic/tree/4d40691/docs/benchmarks/inkling-class-frames/three-engines/quality) on the selected GPU.
