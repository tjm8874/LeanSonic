# Bilingual distribution fidelity of the original SGLang checkpoint

All engines saw identical token histories at 613 English and Japanese answer positions. SGLang used the pinned original checkpoint in the [matched timing comparison](../README.md) on RTX PRO 5000, with one offline request at a time. The reference is the same [published BF16 oracle for the bilingual suite](../../inkling-chat-quality.md).

| Engine and distribution | Mean KL | Worst KL | BF16 top-1 matches | PPL |
|---|---:|---:|---:|---:|
| LeanSonic canonical | 0.002649 | 0.095679 | 609/613 | 3.736971 |
| Original SGLang | 0.011697 | 0.286366 | 589/613 | 3.840579 |
| Original llama.cpp Q3_K_XL | 0.013527 | 0.293811 | 583/613 | 3.66549 |

The largest SGLang divergence was at offset 0 of the Japanese summary. Both engines picked ID 16621. The supplied target was 12263, with negative log likelihood 3.437218 under SGLang and 1.621185 under BF16. Original Q3 has the lowest PPL on this supplied continuation and the lowest top-1 agreement in the table. The free-generation tests in the timing comparison score a different property.

The offline SGLang processor records each pre-sampling logit vector, then selects the known next fixture token. The driver verifies the complete forced output stream and compares two repeated cases byte for byte. SGLang emits 201,024 logits, and the BF16 reference emits 200,058. The scorer normalizes the full SGLang row, gives the reference zero probability on the extra IDs, and checks their mass and top-1 status. The maximum measured extra probability was zero. All 613 rows, input digests and language results are in [quality.json](quality.json). The instrumentation disables scheduler overlap and serves only to capture logits. Timing uses the ordinary uninstrumented configuration.

## Reproduce

Prepare the fixture directories and native BF16 rows using the [public bilingual fixture procedure](../../../../tools/inkling-chat-quality/README.md). Use the exact SGLang revision and model from the [distribution manifest](../distribution.json). Set REPO, OFFICIAL, INKLING_FIXTURE, and BF16_ROWS to absolute paths. Run in a fresh directory on a dedicated GPU with offline SGLang and the fixture data. The driver uses only the bundled capture processor.

```sh
quality="$REPO/docs/benchmarks/inkling-sglang/quality"
export INKLING_CAPTURE="$PWD/sglang.rows" HF_HUB_OFFLINE=1
export PYTHONPATH="$quality:${PYTHONPATH:-}"
cp "$quality/../functional.json" .
python3 "$quality/teacher_force.py"
OPENBLAS_NUM_THREADS=1 python3 "$quality/score_full.py" \
  "$REPO/tools/inkling-chat-quality/fixtures.json" "$INKLING_FIXTURE" \
  "$BF16_ROWS" "$INKLING_CAPTURE" quality.json
```
