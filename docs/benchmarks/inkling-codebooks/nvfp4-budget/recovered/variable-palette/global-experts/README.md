# Restoring 68 experts within the public IQ3_XXS byte budget

The shard and tokenizer together occupy 11,663,036,518 bytes. The IQ3_XXS GGUF occupies 11,663,309,504 bytes. Our tokenizer is counted once. The runtime and KV cache are excluded from both model sizes.

Starting with the 11,577,239,270-byte four-bank variable-palette parent, we ranked 4,864 expert choices in 38 planes using training data. For each expert we used its training routing activity, normalized squared weight error, layer weight and the estimated cost of restoring it. Re-encoding complete planes gave the actual byte cost of each trial prefix. The resulting prefix restores 68 experts across 26 planes. The training objective falls from 360,348.698 to 334,273.447. The two-whole-plane candidate scored 343,710.640 on that same objective. Held-out KL, top-1 and PPL require separate inference runs.

To reproduce, use the four-bank variable-palette and higher-quality parent packages from the [whole-plane experiment](../restored-four-banks/README.md). Use its original direct-frame parent packages for the independent verification. On the same 238b46c source, install mix.rs as crates/leansonic-core/examples/mix_iq3_planes.rs and build with cargo build --release -p leansonic-core --example mix_iq3_planes. Build the independent verifier following the [expanded expert experiment](../../ternary/ranked-experts/expanded/README.md).

```sh
gzip -dc rank.json.gz > rank.json
python3 assemble.py rank.json "$LOW_FOUR" "$HIGH_FOUR" "$MIXER" "$OUTPUT" > reproduced.json
python3 verify.py "$OUTPUT" "$LOW_FOUR" "$LOW_DIRECT" "$HIGH_DIRECT" rank.json reproduced.json "$VERIFIER"
```

To regenerate the ranking, run rank.py with the official safetensors shard, training cutoffs (selection.json), training routing factors (routing.json), parent variable-palette plan.json, whole-plane receipt.json and variable_palette_sample executable, in that order. Its stdout must match the uncompressed ranking digest in receipt.json. The receipt includes selected expert IDs, measured prefix trials and artifact digests. verified.json records an independent decode of 6,794,772,480 weight bytes and 849,346,560 scale codes across the affected planes. The other 947 tensor payloads matched the parent.
