# Two high-divergence positions under two BF16 reference computations

Five physical Inkling packages saw exactly the same token histories. At both positions, all four ternary candidates changed the next-token prediction relative to both independent BF16 computations. The 13.48 GB control retained the BF16 top token on both. Changing the BF16 computation alone does not explain the discrepancy on these two rows. These inputs were selected from previous test failures and do not estimate corpus-wide error.

| Model | Bytes incl. tokenizer | Test chunk 1 / position 260: KL (canonical / serial) | Top ID | Test chunk 90 / position 424: KL (canonical / serial) | Top ID |
|---|---:|---:|---:|---:|---:|
| Control | 13,479,710,294 | 0.00124 / 0.00093 | 57105 | 0.03677 / 0.01116 | 153890 |
| Ternary | 7,718,950,382 | 6.48808 / 6.48519 | 38520 | 8.01276 / 8.58312 | 149881 |
| Scale refit | 7,734,665,510 | 9.80864 / 9.80517 | 38520 | 7.53882 / 8.06728 | 149922 |
| Rate-constrained | 7,679,612,078 | 10.06235 / 10.05875 | 38520 | 7.51600 / 8.05539 | 149922 |
| Energy refit | 7,748,141,470 | 9.04088 / 9.03753 | 38520 | 9.21439 / 9.82989 | 149922 |

Reference top IDs are 57105 and 153890 respectively. Per-row capture uses the first 257 prompt IDs followed by the forced continuation through the scored position. The canonical and serial comparisons use exactly the same captured FP32 logit vector per model and position. For each row the diagnostic applies the native reference cutoff of −16. The capture uses the bench entry point with a forced continuation. The [whole-corpus eval-kld figures](../quality.md) come from its own evaluation entry point.

## Reproduce

Use the original 100-chunk WikiText-2 test reference (SHA-256 895fac2136c06dce1854568c83eb76a0225a05fab3c37897a12fec1d56949b97) and the independently generated serial reference (cdb0a2ede0a1313c14f9253170d374d027e09e1e481b55b6339cf1e5b085d826). Pin RTX PRO 4500, driver 595.99.02, full GPU residency with one sequence, context capacity 8,192, spec0 and prefill workspace 512. The release CLI from a200846 and the sm_120a cubin from bc526e3 have SHA-256 9c8e6b6360b91bc833bc3e1d1cdf31efd1a278ebc26fe9db437c047bfa10ae0c and 65fb004d7d44c3d5450b7146dc9e3cb302d728ac35f1bcda2cf162a51c6dfb6d. Set MODEL to each matching physical package in turn.

```sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-2de24713-6ee5-0469-ca31-719f8e9417c9
export LEANSONIC_CTX=8192 LEANSONIC_KV=fp16 LEANSONIC_SEQS=1 LEANSONIC_PREFILL_CHUNK=512
export LEANSONIC_PDL=1 LEANSONIC_PDL_ALL=1 LEANSONIC_MXFP8_DEVICE_U4=1 LEANSONIC_MXFP8_DIRECT_ALL=1 LEANSONIC_CUBIN_PATH="$CUBIN"
python3 analyze_rows.py --cli "$CLI" --weights "$MODEL" --canonical "$TEST_KLD" --serial "$SERIAL_TEST_KLD" --row 1/260 --row 90/424
```

[Physical package hashes, both reference scores and FP32 row fingerprints](results.json).
