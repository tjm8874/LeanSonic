# Reduce the shipped input embedding while retaining exact high-error values

Near the public IQ3_XXS size, the LeanSonic input embedding and output head occupy 1,064,789,824 bytes. The original GGUF pair occupies 578,949,120 bytes. This experiment changes only the input embedding and leaves the head, tokenizer and all expert weights untouched. [Complete per-tensor allocation](https://github.com/tjm8874/LeanSonic/tree/feb9b8b/docs/benchmarks/inkling-native-lowbits/allocation).

Using only the official input weight values, we fitted a symmetric BF16 lookup table. Each of the 463,159,296 scalar inputs uses one byte to select a BF16 value. An escape index uses its original two bytes when the magnitude error would exceed the fitted threshold. The table and index stream precede the little-endian BF16 literals, in traversal order. Existing packages retain their original loaders. After load, the engine uses the same pinned BF16 embedding representation and inference kernels.

| Input embedding | Physical bytes | Relative RMS weight error | Maximum absolute weight error |
|---|---:|---:|---:|
| Parent lossless prefix | 618,590,352 | 0 | 0 |
| Unrestricted scalar lookup | 463,159,808 | 0.0065020 | 7.25 |
| Lookup with exact outlier literals | 502,097,692 | 0.0053960 | 0.00830078125 |

The 480 MiB embedding budget was set before corpus scoring. We store 19,468,942 literal values and preserve 107,927,968 values exactly, including the literals. The candidate distribution measures **11,546,543,002 bytes**, 116,493,516 bytes below its 68-expert parent. Readback verifies all 998 other tensor payloads and the tokenizer. The full Rust decode was compared to every official BF16 input value. These are weight errors. Paired WikiText-2 test/validation against the BF16 logits is required before an inference-quality claim.

## Reproduce

Use source 8132335 and the [68-expert four-bank parent](https://github.com/tjm8874/LeanSonic/tree/22ef9e3/docs/benchmarks/inkling-codebooks/nvfp4-budget/recovered/variable-palette/global-experts). Use the official Inkling-2-Mini unmodified model shard as OFFICIAL, the parent directory as PARENT, and an unused path as OUTPUT. Python 3 and NumPy are required for fitting.

```sh
python3 encode.py "$OFFICIAL" plain.lut > encoding-run.json
python3 bounded.py "$OFFICIAL" encoding-run.json bounded.lut > bounded-run.json
python3 assemble.py "$PARENT" bounded.lut "$OUTPUT" > receipt-run.json
```

For independent readback, copy verify.rs to crates/leansonic-core/examples/verify_embedding_lut.rs. Run the following from the repository root with paths to the generated artifacts.

```sh
cargo run --release -p leansonic-core --example verify_embedding_lut -- bounded.lut decoded.bf16
python3 docs/benchmarks/inkling-embedding-lut/verify.py "$OFFICIAL" decoded.bf16 bounded-run.json
```

[Distribution digests](receipt.json) · [Encoder error and budget](bounded.json) · [Independent decoded BF16 digest](verification.json).
