# Per-plane MXFP8 precision at a fixed storage budget

We enumerated 14 possible magnitude cutoffs (85, 87, …, 111) for each of the eight expert planes in layers 20–23. The search uses only official weight counts and block scales. It minimizes the mean of the eight per-plane relative squared weight errors among candidates whose predicted saving matches or exceeds the previous magnitude-rounded package by up to 100,000 bytes. A meet-in-the-middle search checks every feasible pair of four-plane assignments using an interval-minimum table. No prompts, corpus scores or model outputs enter the choice.

The actual distribution is **15,396,357,022 bytes**, 4,536 fewer than the [previous package](inkling-storage-candidates.md), and 151,800,520 fewer than the canonical package. Mean per-plane relative squared weight error fell from 0.00004988953 to 0.00003786044 (24.1%). This error is measured on weights; model fidelity is evaluated separately. The table below lists the selected E4M3 magnitude codes.

| Physical layer | Gate/up cutoff | Down cutoff |
|---|---:|---:|
| 20 | 95 | 91 |
| 21 | 101 | 95 |
| 22 | 101 | 95 |
| 23 | 101 | 95 |

The package was generated with the public #130 repacker. The reader and compressed format are unchanged. The [artifact receipt](inkling-plane-rounding.json) contains the shard SHA-256, exact sizes and selected-plane checksums. To reproduce, use a fresh output directory and run these commands in the LeanSonic checkout. The first step reads the original weights. The second uses only those saved rate/distortion curves.

```sh
python3 tools/inkling_storage_probe/choose_cutoffs.py "$OFFICIAL/model-00001-of-00001.safetensors" curves.json
python3 tools/inkling_storage_probe/choose_plane_cutoffs.py curves.json plane.json
cargo build --release -p leansonic-core --example inkling_mxfp8_round
python3 tools/inkling_storage_probe/pack_mxfp8.py "$OFFICIAL" "$CANONICAL" "$OUTPUT" --cutoffs plane.json
```

## WikiText held-out corpus

Each split uses the same 25,500 predicted positions, BF16 reference, and 100 chunks. Lower KL and PPL are better. The top-1 column counts predictions matching BF16. See the [frozen corpus and reproduction commands](inkling-storage-candidates.md).

| Split | Package | Mean KL | Worst KL | BF16 top-1 / 25,500 | PPL |
|---|---|---:|---:|---:|---:|
| test | Canonical | 0.00315843 | 0.68097733 | 24865 | 7.07296 |
| test | Earlier budget | 0.00317531 | 0.67173614 | 24868 | 7.07393 |
| test | Per-plane budget | 0.00316872 | 0.66990856 | 24859 | 7.07363 |

The per-plane candidate improves the test worst KL relative to the canonical package, while its test mean KL and PPL increase and six fewer predictions match BF16. Keep the canonical package as the default.
