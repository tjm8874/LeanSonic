# Four-bank Inkling with a 452-magnitude input embedding

The four-bank model with all 998 non-embedding tensors retained now occupies **13,331,932,142 bytes** including the tokenizer. The parent is **13,448,974,014 bytes**, a difference of **117,041,872 bytes (0.87%)**. The parent uses a 618,590,352-byte prefix embedding. The candidate replaces it with the 501,549,352-byte magnitude frame fitted to the original publisher BF16 weights. The safetensors header accounts for the remaining byte difference. All other tensor payload hashes, types and shapes, metadata and tokenizer bytes were checked during assembly. [Model, executable and input hashes](package.json).

Build the parent with the [second four-bank cost fit](https://github.com/tjm8874/LeanSonic/tree/588e3cd/docs/benchmarks/inkling-nvfp4-scale-split/four-banks/quartiles/cost-fit). Check the original shard digest against package.json. Follow the [452-magnitude fit and encode instructions](../README.md) to create embedding.frame. Run the following with Python 3.11 or newer. The output directory must be new.

```sh
python3 assemble.py FOUR_BANK_PARENT FIT_DIR/embedding.frame NEW_MODEL > measured.json
```

The source embedding has 463,159,296 BF16 elements. The fitted decoder reconstructs 222,538,439 exactly, with relative RMS error 0.00273538 and maximum absolute error 0.0078125. The new package is an accuracy experiment. Use the two complete WikiText BF16 references, 100 chunks per split and 25,500 scored positions each. For timing, use the [three-engine replay](https://github.com/tjm8874/LeanSonic/tree/eab21bd/docs/benchmarks/inkling-class-frames/three-engines) on RTX PRO 4500 with its 1,024 prompt IDs and 32 vLLM continuation IDs. Rotate parent, magnitude and official vLLM, then reverse the order in each of five cycles. For the two four-bank models check FP16 KV, spec off, 48 packed scale planes and all 24 expert layers resident. Corpus quality, timings and peak VRAM determine whether the byte saving is worthwhile.
