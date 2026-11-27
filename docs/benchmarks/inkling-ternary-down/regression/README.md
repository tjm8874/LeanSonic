# Two-bit expert execution regression cases

Both production Tensor Core down kernels in ad3b390 were exercised directly on the original layer-0 ternary plane, using its actual amplitudes and six-bit scales. The test assigns 1,376 activation rows to eleven nonempty experts, reverses the output row mapping and launches two extra idle tiles. Expert counts include 63/64/65, 127/128/129 and 159/160/161. For each of the 128- and 160-token tile shapes all 3,170,304 FP16 outputs must agree with the nibble kernel. The test also checks finiteness, nontrivial nonzero output and 4,608 guard words.

Compute Sanitizer memcheck completed with zero errors on PRO5000, 5070Ti.

The engine test uses four sequence slots and the [fixed prompt](../engine/prompt.ids). It varies the four prefixes (31, 64, 129, 257 tokens), interleaves the slots, verifies 3–6 tokens per sequence, commits a different number of tokens in each slot, and compares two subsequent decode steps and the final FP32 logit hash with serial replay. Repeat after reset. Check the entire trace across four-bit vs two-bit resident down weights.

Verified on PRO4500, 5060Ti.

## Reproduce

Generate nibbles.bin, ternary.bin, amax.bin and scales.bin using the [layer-0 recipe](../). Build engine.cubin from ad3b390 using the command in the [engine recipe](../engine/). For the four-slot test use identical environment settings to that recipe plus the overrides below.

```sh
nvcc -std=c++17 -O3 "$RECIPE/tiles.cu" -lcuda -o tiles
compute-sanitizer --tool memcheck --error-exitcode 97 ./tiles
cp "$RECIPE/seqs.rs" crates/leansonic-core/examples/ternary_seqs.rs
cargo build --release -p leansonic-core --example ternary_seqs
export LEANSONIC_SEQS=4 LEANSONIC_ATTN_SPLIT=8
for arm in 0 1; do
  LEANSONIC_TERNARY_DOWN=$arm target/release/examples/ternary_seqs "$MODEL_DIR" "$PROMPT_IDS" >"seqs.$arm"
done
cmp seqs.0 seqs.1
```

[Output traces, driver/cubin SHA-256 and sanitizer reports](results.json).
