# Where four near-equal-size Inkling models spend their bytes

Groups include scales and codebooks whose tensor names belong to the corresponding projection. Headers, tokenizer metadata, tokenizers, tensor alignment padding and remaining tensors are included in Other. All rows reconcile to each measured distribution size. The inventories use ggml_row_size from the native library to determine the actual payload for each original GGUF tensor.

| Package | Input embedding + output head | Expert gate/up | Expert down | Other | Total bytes |
|---|---:|---:|---:|---:|---:|
| native_iq1 | 492,106,752 | 4,140,574,720 | 2,388,797,440 | 742,812,864 | 7,764,291,776 |
| lean_iq1 | 1,064,789,824 | 3,981,464,880 | 2,041,317,864 | 676,419,078 | 7,763,991,646 |
| native_iq3 | 578,949,120 | 5,711,865,856 | 4,586,481,664 | 786,012,864 | 11,663,309,504 |
| lean_iq3 | 1,064,789,824 | 6,531,353,328 | 3,390,474,256 | 676,419,110 | 11,663,036,518 |

Lean IQ1 restores 61 experts; Lean IQ3 restores 68. Both retain the same lossless input embedding (618,590,352 bytes) and the same Q8-64 output head. Together these cost 572,683,072 more bytes than the input/output pair in native IQ1_S. Within that size class native IQ1_S spends more bytes on both expert projections. The native IQ3_XXS also assigns appreciably more space to down projections. These differences motivate isolating input embedding precision while keeping the tokenizer, output head, expert weights and BF16 scoring fixture fixed. A byte allocation alone cannot establish which tensors caused any observed quality difference.

For GGUF pass the original file and --ggml pointing to libggml-base.so.1 from Unsloth llama.cpp 8832860 to inventory.py. For LeanSonic pass its model directory. Each compressed inventory records every name, storage format, shape, byte offset and true payload size. Native files and immutable digests are recorded in the [native results](../README.md). The [61-expert](https://github.com/tjm8874/LeanSonic/tree/22ef9e3/docs/benchmarks/inkling-codebooks/nvfp4-budget/recovered/ternary/ranked-experts/expanded) and [68-expert](https://github.com/tjm8874/LeanSonic/tree/22ef9e3/docs/benchmarks/inkling-codebooks/nvfp4-budget/recovered/variable-palette/global-experts) packages include their tokenizer.
