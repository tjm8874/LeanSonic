# Lossless class-specific entropy tables for the 571-expert package

The existing four-bank NVFP4 decoder accepts different boundaries for each expert. The retained and restored experts have different weight distributions. We compared the original frame with five ways of assigning the same four entropy tables to those two classes. For every one of the 36 mixed projections, three scale ranges for retained experts and one range for restored experts gave the smallest frame tested. Expert membership comes from the parent model’s frozen training ranking.

| Materialized model, including tokenizer | Bytes |
| --- | ---: |
| Original llama.cpp IQ3_XXS | 11,663,309,504 |
| Parent LeanSonic, 571 experts | 11,662,590,054 |
| Class-specific tables, same 571 experts | 11,487,467,790 |

The new package saves 175,122,264 bytes against its parent and 175,841,714 against original IQ3_XXS. The encoder used the existing decoder to check all 9,342,812,160 reconstructed weight bytes in changed tensors. Assembly copied the remaining 963 tensor payloads, including all scale frames, and checked all 999 payload hashes after reading back the written file. The tokenizer SHA-256 also matches the parent. [Per-projection trial sizes](fit.json) · [Final file lengths and hashes](receipt.json).

Use engine revision 726b6f2142a4095524e52f1877cffcf7bbd0ff19, the [571-expert parent model and expert.json](../README.md), and Python 3.11 or later. Run the following from the engine root. Set FIXTURE and PARENT to absolute paths and give the working directory enough free space for the selected frames and the final model.

```sh
cp "$FIXTURE/banks.rs" crates/leansonic-core/examples/refit_expert_tables.rs
cargo build --release -p leansonic-core --example refit_expert_tables
work=$(mktemp -d)
cp target/release/examples/refit_expert_tables "$work/banks"
cd "$work"
python3 "$FIXTURE/fit.py" "$PARENT" "$FIXTURE/../expert.json" frames
python3 "$FIXTURE/assemble.py" "$PARENT" frames materialized > receipt.json
cmp receipt.json "$FIXTURE/receipt.json"
```

The resulting weights load through the same four-bank reader. The [matched 5070 Ti timings](../5070Ti/README.md) apply to the parent package. Repacked-model GPU scoring and timing are tracked separately.
