# Widening Q8 scales in the Inkling attention projections

Each original BF16 attention matrix is quantized in blocks of 64, using the parent’s max-absolute scale as control. The trial considers the next 1, 2, 4 or 8 representable positive BF16 maxima. The first candidate selects the smallest squared weight error in each block, keeping the parent on ties. The second minimizes estimated code length under the original tensor’s smoothed byte histogram among candidates with no greater squared error than the parent; equal estimated costs prefer lower error. No text or logits enter selection. Both packages run with the existing Q8 CUDA kernels and unchanged expert weights.

| Package | Total bytes incl. tokenizer | Delta bytes | Attention weight squared error (sum) |
|---|---:|---:|---:|
| Canonical | 15,548,157,542 | 0 | 10.51251569 |
| refine | 15,549,155,830 | +998,288 | 9.61390097 |
| rate | 15,548,089,286 | -68,256 | 10.47542904 |

Both builders regenerated all 352 canonical control frames byte for byte. The alternate frames were decoded and compared with the intended Q8 codes and FP32 scale words, and every tensor payload was re-read after assembling the shard. The sum of squared weight differences does not measure language-model accuracy.

## BF16 fidelity

| Set | Candidate | Mean KL | Worst KL | BF16 top-1 / scored positions | PPL |
|---|---|---:|---:|---:|---:|
| test | canonical | 0.00315843 | 0.68097733 | 24865/25500 | 7.072956 |
| test | refine | 0.00315592 | 1.21701377 | 24901/25500 | 7.073757 |
| test | rate | 0.00313373 | 0.70959240 | 24915/25500 | 7.071119 |
| valid | canonical | 0.00276074 | 0.35243610 | 24963/25500 | 6.504621 |
| valid | refine | 0.00272742 | 0.51979952 | 24976/25500 | 6.507939 |
| chat | canonical | 0.00264874 | 0.09567916 | 609/613 | 3.736971 |
| chat | refine | 0.00244614 | 0.08607209 | 606/613 | 3.721961 |
| chat | rate | 0.00259262 | 0.08234757 | 606/613 | 3.725609 |

WikiText uses the canonical 512-context BF16 oracles and 100 chunks per split. The refine package was assigned PRO 4500 and the rate package PRO 5000. Chat uses the independent [613-position bilingual full-vocabulary suite](inkling-chat-quality.md) on PRO 5000.

The smaller rate candidate gained 50 BF16 top-1 matches on the test split and lowered mean KL from 0.00315843 to 0.00313373. Worst KL rose from 0.680977 to 0.709592, at chunk 69, token 353. In bilingual chat it lost three top-1 matches while both mean and worst KL fell. Keep the canonical default pending wider quality evaluation.

## Reproduce

Start from the official BF16 attention tensors and the complete [canonical expert-code package](inkling-storage-candidates.md). Use an empty output path. The example runs on CPU and the repacker verifies the expected canonical shard hash. The test workload and quality commands are available in the [corpus benchmark](inkling-storage-candidates.md).

```bash
cargo build --release -p leansonic-core --example inkling_refine_q8
for mode in refine rate; do
  python3 tools/inkling_storage_probe/pack_attention_q8.py \
    target/release/examples/inkling_refine_q8 "$OFFICIAL" "$PARENT" "$OUTPUT/$mode" \
    --mode "$mode" >"$OUTPUT/$mode.json"
done
```

[Complete model sizes, shard SHA-256, per-matrix errors, scale choices and available quality results](inkling-attention-q8-refinement.json).
