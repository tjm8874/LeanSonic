# Seven-bit four-bank package versus native vLLM on RTX PRO 5000

LeanSonic 9bc4d48 uses the 13,437,429,480-byte seven-bit package. Native vLLM 8c17e469 uses the 19,555,785,477-byte official non-MTP package at b85c3cfa. Both disable speculation, use one sequence, 2,048-token context, FP16 KV, 1,024 input tokens and the same 32-token decode history. Each engine on PRO 5000 follows the PRO 5000 native history. Across GPUs the prompt and first three generated IDs agree, then vLLM chooses token 154978 on PRO 5000 and 220 on PRO 4500. Within-card timings remain paired; the fourth and subsequent decode positions do not share a fixed trajectory across GPUs.

2 of five mirrored cycles are complete, with 28 warm requests per engine and two fresh processes per arm per cycle. The NVML peak includes model loading and the vLLM child worker.

| Engine | Prefill tok/s | Decode tok/s | Peak GPU bytes | Distribution bytes |
| --- | ---: | ---: | ---: | ---: |
| LeanSonic seven-bit | 11,441.73 | 141.82 | 15,674,114,048 | 13,437,429,480 |
| vLLM native | 12,313.79 | 138.39 | 22,861,053,952 | 19,555,785,477 |

Use the [same two-engine protocol, runner and timer definitions](../../vllm/README.md#reproduce) with the seven-bit [model hashes](../package.json) and an RTX PRO 5000. Generate the anchor locally with `python3 leg.py anchor vllm` before the five cycles. [GPU, package and executable hashes, predicted IDs, warm/cold samples and all peaks](measurement.json.gz).
