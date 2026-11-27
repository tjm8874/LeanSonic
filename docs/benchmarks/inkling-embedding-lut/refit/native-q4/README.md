# Native IQ3_XXS input embedding control

The public IQ3_XXS GGUF stores its input embedding in Q4_K (260,527,104 bytes). Its 463,159,296 GGML-dequantized values were rounded to nearest BF16, ties to even, and compared with the official BF16 embedding. Relative RMS error is 0.07174278, maximum absolute error 0.42431641. At the same positions the higher-precision LeanSonic refit yields 0.00451878 and 0.0078125, using a larger frame. These are tensor errors; the whole-model corpus comparison is a separate measurement.

Run probe.py with the official safetensors shard, pinned original IQ3_XXS GGUF, refit BF16 decode, llama.cpp libggml-base.so and the [native tensor inventory](https://github.com/tjm8874/LeanSonic/tree/d65d1a6/docs/benchmarks/inkling-native-lowbits). It calls dequantize_row_q4_K on the entire original GGUF embedding; the source payload and BF16 result hashes are included in [result.json](result.json).
