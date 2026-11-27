# Preselected validation suffix for the routing fits

After building both routing-weighted models, we fixed a new text sample to evaluate generalization before examining its scores. The usual 100-chunk validation prefix was already used to compare candidates. The new fixture starts at the first complete article heading after the halfway point, byte 576,881 of the original 1,146,846-byte WikiText-2 raw validation file. The suffix SHA-256 is `90ee94eb53b5833ceb64e56d6412a7d72c40c6bdcc0d0508f6314445201031c2`.

Use the unmodified BF16 reference generator and converted reference GGUF from the [canonical recipe](../../../../inkling-reference.md). Retain its 16-sequence reference schedule. Use a new output directory. From the directory containing the scripts below:

```sh
python3 suffix.py "$VALID_RAW" "$OUT/suffix.raw" > "$OUT/fixture.json"
"$LLAMA_PERPLEXITY" -m "$BF16_GGUF" -f "$OUT/suffix.raw" -c 512 -b 8192 -ub 8192 -ngl 99 -t 8 --no-warmup --chunks 100 --save-all-logits "$OUT/suffix.kld"
python3 verify.py "$OUT/suffix.kld" "$TEST_KLD" "$VALID_KLD" "$TRAIN_WINDOWS"
sha256sum "$OUT/suffix.kld"
```

The verifier compares every new 255-token scored input span against all offsets in the earlier 100 test windows, 100 validation windows and 32 [calibration windows](../routing/README.md). It also checks for repeated complete scored spans within the new fixture, the expected headers, row count and file size. This check addresses duplicate input spans at the stated length. Short phrases can recur.

Run the same PRO 4500 scorer, cubin, FP16 KV and automatic prefill policy on the original compact model and both routing-weighted builds. Pass `eval-kld --weights "$MODEL" --ref "$OUT/suffix.kld" --chunks 100`. The three package sizes are 13,626,745,974, 13,612,680,454 and 13,628,279,942 bytes respectively. The two fitted artifacts were already frozen before allocating this evaluation. [Input fingerprint](fixture.json).
