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
| valid | Canonical | 0.00276074 | 0.35243610 | 24963 | 6.50462 |
| valid | Earlier budget | 0.00276864 | 0.35100907 | 24944 | 6.50413 |
| valid | Per-plane budget | 0.00276719 | 0.35047861 | 24948 | 6.50418 |

The per-plane candidate improves the test worst KL relative to the canonical package, while its test mean KL and PPL increase and six fewer predictions match BF16. Keep the canonical package as the default.

## Fixed chat response prefixes

The comparison uses the [613-position bilingual BF16 fixture](https://github.com/tjm8874/LeanSonic/pull/134). Every candidate saw the same inputs and forced targets on PRO 5000.

| Package | Mean KL | Worst KL | BF16 top-1 / 613 | PPL |
|---|---:|---:|---:|---:|
| Canonical | 0.00264874 | 0.09567916 | 609 | 3.73697 |
| Earlier budget | 0.00265428 | 0.09538664 | 608 | 3.74013 |
| Per-plane budget | 0.00268161 | 0.09608839 | 606 | 3.73782 |

The largest per-plane chat divergence was in the English JSON schema case (offset 25, KL 0.09608839); its top prediction still matched BF16. Across all 613 positions, this package lost three top-1 matches relative to canonical. Test and validation worst KL improved, but that advantage did not transfer to chat.
