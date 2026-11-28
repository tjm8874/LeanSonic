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

## Measured quality of the stored model

The upstream eager teacher on an RTX PRO 5000 72 GB loaded the completed directory with the reader above. Each comparison uses identical teacher-forced inputs and the original BF16 reference. The new and extended fixtures contain eight and sixteen prompts respectively.

| All-vocabulary fixture | Mean KL | Worst KL | Top-1 match | Candidate PPL | BF16 PPL |
|---|---:|---:|---:|---:|---:|
| Original, 49 positions | 0.13954 | 2.71342 | 43/49 | 26.7287 | 28.0382 |
| New, 233 positions | 0.02806 | 0.30415 | 215/233 | 18.5824 | 18.5885 |
| Extended, 494 positions | 0.02393 | 0.33591 | 460/494 | 11.4001 | 11.3821 |

On the common 104-position Basalt protocol, the stored Q8 model reached mean/worst truncated KL 0.03379/0.30399, top-1 92/104 and PPL 9.2392. The published Basalt fixed-placement measurement was 0.03287/0.65464, 98/104 and 9.1700. The original 49-position fixture had a much higher worst KL than the dense-Q8 simulation using the official residual norms. The experimental package retains the MLX residual norms. Attribution is being tested with a separate official-norm overlay. These measurements use the Python teacher for the packed candidate. [Full fixture and common-protocol results, comparison against the mixed-source simulation and hashes](packed-quality.json)

## Restoring the 148 official residual norms

Keeping every Q8 code, scale, routed expert, Engram row and input unchanged, replacing only the 148 residual decoder norms with their official BF16 values exactly reproduced the original mixed-source Q8 experiment. All 49 and 233 rows of full-vocabulary FP32 logits and every corresponding eight-token continuation were byte-identical. In the original fixture, relative to BF16, mean/worst KL fell from 0.13954/2.71342 to 0.02296/0.18979 and top-1 agreement rose from 43 to 46 of 49. PPL increased from 26.7287 to 28.7123 (BF16 28.0382). On the 233-position fixture, mean/worst KL changed from 0.02806/0.30415 to 0.02989/0.42574, while top-1 increased from 215 to 216. This control used an auxiliary shard. [Paired scores, byte-equality checks and hashes](official-norms-control.json)

A standalone distribution can store the same values in the existing BF16 slots. The converter reads the affected official tensors, updates their original offsets in seventeen shards, and independently compares the complete destination files with the sources. It verifies the original header and every byte outside the replacements and compares the replacement bytes directly to the official weights. Exactly 2,066,432 bytes of tensor slots were rewritten and the directory is still **105,692,126,511 bytes**. [Per-file SHA-256, total size and tensor counts](official-norms-build.json)

```sh
python3 "$d/restore_norms.py" "$Q8" "$OFFICIAL_BF16" "$Q8_OFFICIAL_NORMS"
python3 "$d/packed_teacher.py" "$Q8_OFFICIAL_NORMS" "$FIXTURES" "$OUTPUT"
```
