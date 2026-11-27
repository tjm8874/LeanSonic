# Retain the committed verification row after speculative generation

On RTX 5070 Ti, the old scalar-buffer readback after 128 greedy tokens could have a different argmax from the returned next token. A separate diagnostic preserved the scalar-buffer fingerprint and read the selected row directly from the batched verification buffer. Each of the 200,058 FP32 logits must be finite and the argmax must equal the engine’s returned ID. This audit is outside the timing runs.

| Input | Four-bank target: old readback argmax → returned ID | Scalar vs. actual spec7 final row |
|---|---:|---:|
| q00_code_merge | 3828 → 16 | 200,058/200,058 equal |
| q11_json_schema | 1 → 25 | 200,058/200,058 equal |
| ja_reasoning | 196994 → 196994 | 200,058/200,058 equal |
| readme1024 | 63 → 69057 | 200,058/200,058 equal |

On the fixed seven-level target we independently tested the official, raw Q8, compact Q8, low/high BF16 Markov and row-exponent Markov drafters. All five matched per fixture on all 128 emitted IDs, returned token, proposal and acceptance counts, rounds and SHA-256 of the actual returned FP32 row. For JSON all five had old scalar-buffer argmax 1 while the actual committed row selected ID 198. On the four-bank target, the scalar and speculative modes agreed on all output IDs and on every final FP32 logit in all four cases (KL between those two rows 0). This KL measures target-mode parity and uses no BF16 oracle.

## Reproduce

For the independent control, check out the [benchmark integration](https://github.com/tjm8874/LeanSonic/commit/1675f4bcb466907848bf3f625831955a3ca2dc39) into a directory named src. Run `python3 instrument.py` from its parent, copy audit.rs into src/crates/leansonic-core/examples/audit_commit_row.rs, and build the release example. The instrumentation records the scalar fingerprint before the committed row is copied. Use the four adjacent ID fixtures, context 2,048 and FP16 KV. To test the fixed API build audit.rs directly against the fix, omitting the instrumentation.

```sh
export LEANSONIC_KV=fp16 LEANSONIC_PDL=1 LEANSONIC_PDL_ALL=1 LEANSONIC_SEQS=1
export LEANSONIC_MXFP8_DEVICE_U4=1 LEANSONIC_MXFP8_DIRECT_ALL=1
export LEANSONIC_CUBIN_PATH=/path/to/engine.cubin
"$BINARY" "$FOUR_BANK_TARGET" 0 scalar-rows
"$BINARY" "$FOUR_BANK_TARGET_WITH_MTP" 7 spec-rows
for file in scalar-rows/*.f32; do cmp "$file" "spec-rows/${file##*/}"; done
```

[Input, binary and cubin fingerprints, complete emitted sequences, selected verification row, prior scalar readback and actual row digests](audit.json).
