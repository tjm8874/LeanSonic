# Measured expert routing on four 1,024-token prompts

RTX PRO 5000 recorded all 128 expert counts at each of 48 prefill visits. All 26,407,656 FP32 logits match the uninstrumented RTX 5060 Ti replay on the same inputs. Each prompt contributes 44 NVFP4 visits (including the mixed-width path) and four other visits. The table aggregates the NVFP4 visits. The count snapshot synchronizes the stream, so these captures measure routing only.

| Prompt | Active expert visits with ≤64 rows | Inactive lanes in issued eight-row MMA groups | Nonempty 64-row tiles | Nonempty 128-row tiles | Nonempty 160-row tiles |
| --- | ---: | ---: | ---: | ---: | ---: |
| readme | 67.38% | 5.18% | 8,895 | 6,474 | 6,095 |
| test-0 | 63.55% | 4.83% | 8,654 | 6,109 | 5,691 |
| test-35 | 66.05% | 4.82% | 8,682 | 6,213 | 5,755 |
| valid-84 | 65.04% | 4.84% | 8,738 | 6,249 | 5,816 |

The nominal average is 64 routed tokens per expert. The largest observed expert received 1,024 tokens. On the README input the 64-row shape requires 37.4% more live weight tiles than the 128-row shape. The 160-row shape eliminates 5.9% of weight tiles, but its larger tile was slower in the [end-to-end experiment](../README.md). Holes in the last eight-row MMA group account for about 5% of issued row slots. These counts constrain the benefit available from eliminating padding alone.

To reproduce, use the pinned a1592f3 source and the 571-slice package from the [matched replay](../../README.md). From a working directory with that checkout under src/, run `python3 instrument.py`. Build and run the [four-input full-logit checker](../README.md) in that checkout at the baseline 128-token shape with FP16 KV, a 2,048-token context and prefill chunk 1,024. Save stderr and extract its ROUTES lines. Run `python3 analyze.py routes.txt` to regenerate the histograms and tile counts. The collector asserts 8,192 assignments per visit. The [raw counts](routes.txt.gz) and [artifact hashes](receipt.json) are included.
