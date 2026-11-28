# Group-32 Q8 dense matrices with Q4 experts and Engram

The packer starts from the compact 103,556,334,617-byte expert-frame and four-row tiled Engram checkpoint. It replaces the 726 target decoder matrices with signed INT8 codes (weights are stored row-major) and one BF16 scale for every 32 columns. Each scale is the group's maximum absolute official BF16 weight divided by 127, rounded to BF16. Codes are rounded to nearest even against that stored scale and clipped to [-127, 127]. There is no bias plane. Routed experts, the large Engram table and MTP stay in their original format.

The resulting self-contained directory totals **105,692,126,511 bytes** including its index and auxiliary files. It is 2,135,791,894 bytes larger than the compact MLX Q4 source and 9,615,434,665 bytes smaller than the 115,307,561,176-byte Basalt v0.5.7 UD-Q4_K_XL file. A separate NumPy implementation read every code and scale back from the safetensors files and recomputed them from the original BF16 bits. It verified all 4,882,227,200 codes, 152,569,600 scales and 1,313 other tensor payloads or hard-linked originals. [Converter result](build.json) · [Complete audit](audit.json)

Use a new destination on the same filesystem as the compact source. The new directory hard-links untouched files and rewrites the seventeen affected shards. Files with the same name retain their own safetensors headers. The numeric dtype I8 and scale shape identify the dense encoding. The package contains model parameters; upstream Python inference and native CUDA dispatch are separate validations. The native Q4 release predates this dense encoding.

```sh
d=docs/benchmarks/qwen38-pack/dense-q8
python3 "$d/pack.py" "$COMPACT_Q4" "$OFFICIAL_BF16" "$Q8"
python3 "$d/audit.py" "$COMPACT_Q4" "$OFFICIAL_BF16" "$Q8"
python3 "$d/packed_teacher.py" "$Q8" "$FIXTURES" "$NEW_OUTPUT"
```

The teacher requires the same Transformers Qwen4Exp implementation, CUDA-enabled PyTorch and reference dependencies as the repository BF16 teacher, plus zstandard. It reads its configuration and all stored parameters from the new directory, unpacks selected expert frames and Engram rows, and expands dense Q8 to BF16. An [independent precision study](https://github.com/tjm8874/LeanSonic/tree/experiment/q38-embedding-quality/docs/benchmarks/qwen38-pack/input-embedding) uses the original MLX and official checkpoints for comparison.
