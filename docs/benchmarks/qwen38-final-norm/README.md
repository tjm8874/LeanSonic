# Folded gains in the Qwen3.8 final mixer and drafter

The official RMSNorm multiplies the normalized input by 1 + weight. In the published MLX export, some tensors already store that sum in BF16. The flashnext-norm-zero conversion record lists 147 decoder norms. It omits the target's final hyper-connection mixer and all nine MTP norms. Across the original MLX export and the converted checkpoint, all 54,528 elements of those ten tensors equal BF16(1 + official weight); zero equal the official raw weight. For the final mixer, the raw MLX mean is 3.74981 and the official mean is 2.74972. A decoder-wide zero-centered flag added one a second time to these gains.

In the loader at `cee3c46`, a known folded tensor is recognized individually by its name and whole BF16 payload SHA-256. Other payloads follow the checkpoint's existing norm convention. This includes models with one or more norms repaired, the official-weight overlay and original non-zero-centered exports. Hashing takes place on model load. Neither the CUDA kernels nor the model files change. [Ten tensor identities and element counts](folded.json)

## Whole-model measurements

The RTX PRO 5000 72 GB used fixed input IDs and the official BF16 next-token teacher over all 248,320 vocabulary entries. The candidate used the same MLX Q4 tensors and model size. Only the final target norm affects these teacher results. The MTP correction is measured separately with speculative generation. Values show the old native model → corrected final norm.

| Fixture | Mean KL | Worst KL | Top-1 matches | PPL |
|---|---:|---:|---:|---:|
| Original, 49 positions | 0.28114 → 0.10005 | 1.46944 → 0.58455 | 39 → 41 | 52.9910 → 33.2775 |
| Held out, 65 | 0.36938 → 0.16435 | 1.40790 → 1.11268 | 44 → 44 | 25.2686 → 20.3930 |
| Extended, 494 | 0.26967 → 0.11960 | 2.49933 → 2.32175 | 406 → 414 | 15.9460 → 12.3159 |
| New, 233 | 0.30923 → 0.15861 | 2.46654 → 1.92162 | 184 → 184 | 29.1715 → 20.6235 |

On the 494 positions, official BF16 PPL was 11.3821; the corrected model matched two of sixteen eight-token BF16 continuations. On the separate 233 positions it matched two of eight, and BF16 PPL was 18.5885. [All metrics, stock and BF16-replacement controls](quality.json)

## Reproduction

Run audit_folded.py with the official BF16 directory and the raw or restored MLX export. The audit compares every element and reports each tensor digest. audit_dense_export.py also inspects all unquantized decoder tensors. In the restored export it found 196 equal tensors and 148 differing tensors, of which 147 appear in the norm conversion record. The published MLX Q4 model and reproducible BF16 teacher are described in the [reference setup](../qwen38-pack/reference/README.md).

To independently exercise the previous loader, build an overlay from the original data and use that same binary and cubin for both arms. All 10,240 values of the final folded target gain round-trip exactly through BF16 subtraction of one followed by FP32 addition of one. An additional control directly installs the official raw BF16 vector. The overlay retains the superseded tensor for this diagnostic; the loader patch operates on the unchanged model.

```sh
d=docs/benchmarks/qwen38-final-norm
python3 "$d/audit_folded.py" "$OFFICIAL" "$DIRECT"
python3 "$d/final_norm.py" "$DIRECT" "$OFFICIAL" "$REPAIRED" --mode reconstructed
python3 "$d/final_norm.py" "$DIRECT" "$OFFICIAL" "$ORACLE" --mode official
```

Build q38_teacher from docs/benchmarks/qwen38-pack/teacher.rs as a leansonic-core example. Run it on the old loader + repaired model, updated loader + unchanged model, updated loader + repaired model, and official-weight control. Each time, use the same ID file and fresh output directory. Compare FP32 vocabulary rows and eight-token continuations by ID. The [494- and 233-position fixture files](https://github.com/tjm8874/LeanSonic/tree/9b1654b/docs/benchmarks/qwen38-pack/input-embedding) are frozen at the same revision used for the earlier quality study.
