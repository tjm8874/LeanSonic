# Allocate BF16 embedding precision by distortion per bit

The 217-expert model with a sign/magnitude embedding weighs 11,661,986,038 bytes, including the tokenizer. That is 1,212,320 bytes less than the prior fixed-capacity refit and 1,323,466 bytes below the public IQ3_XXS GGUF. The 998 other tensor payloads and the tokenizer are byte-identical to the refit. The package remains experimental pending end-to-end corpus quality and timing.

| Metric, full official input embedding | Prior LUT refit | Magnitude candidate |
|---|---:|---:|
| Embedding frame bytes | 502,761,672 | 501,549,352 |
| Relative RMS weight error | 0.00451878 | 0.00273538 |
| Worst absolute weight error | 0.0078125 | 0.0078125 |
| Exact BF16 values out of 463,159,296 | 173,733,613 | 222,538,439 |

The fitter uses only the original BF16 tensor histogram. It greedily joins adjacent absolute values by the increase in squared error per empirical entropy bit saved. Each representative is rounded to BF16. The selected fit uses 452 magnitudes and keeps all sign bits, including negative zero. Its primary stream directly encodes 255 frequent magnitudes; 6,405,035 occurrences use the secondary stream. Both streams use the existing tiled four-lane rANS codec. The Rust release decoder wrote all 926,318,592 bytes with the same SHA-256 as the independent Python reconstruction. One CPU decode took 497 ms.

## Reproduce

1. Build the model and source in the [previous 217-expert refit](../refit). Use the original official safetensors shard from the same model revision. At source 726b6f2, copy pack.rs and decode.rs into crates/leansonic-core/examples as mag_pack.rs and mag_decode.rs, then run `cargo build --release -p leansonic-core --example mag_pack --example mag_decode`.
2. In a fresh working directory, run `python3 fit.py OFFICIAL_SHARD` (NumPy required). Run `python3 encode.py OFFICIAL_SHARD map-502000000.u16 /absolute/path/to/mag_pack FIT_DIR`. The packer verifies each encoded byte stream against its raw source. Candidate selection uses measured file bytes.
3. Run `mag_decode FIT_DIR/embedding.frame decoded.bf16`, then `python3 check.py OFFICIAL_SHARD decoded.bf16`. Run `python3 assemble.py REFIT_MODEL FIT_DIR/embedding.frame NEW_MODEL`; it checks every retained tensor and file after writing. Compare the model hashes with [receipt.json](receipt.json) and decoded hash with [check.json](check.json).
4. Build the release CLI and sm_120a CUDA module from the same sources. Score each pinned WikiText oracle using `leansonic eval-kld --weights NEW_MODEL --ref ORACLE --chunks 100`, FP16 KV, PDL and PDL_ALL, one sequence, prefill chunk 512, MXFP8_DEVICE_U4=1, MXFP8_DIRECT_ALL=1 and ternary up/down disabled. The input tensor metrics above are distinct from those full-network scores.
