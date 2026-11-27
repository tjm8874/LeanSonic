# Restoring selected experts under the IQ1_S byte budget

The layer 17 expert down projection occupied the highest place among the 40 NVFP4 planes when ranked by training-weighted error reduction divided by the measured byte cost of restoring the whole plane. Within that plane, 42 of 128 experts fit when restored in training-error order. This adds 45,181,504 bytes to the [ternary distribution](../). The shard and tokenizer total 7,764,131,886 bytes, 159,890 below the published Inkling IQ1_S GGUF. The next prefix of 43 experts exceeded the budget. This search considered a single plane and ranked prefixes. It does not establish an optimal allocation.

Selection uses fixed training-route weights and stratified expert rows. Restored experts use both the weight and scale from the 13,479,710,294-byte control, and the other experts retain their ternary values. All 997 other tensor payloads were copied from the ternary parent and verified on readback. An independent pass decoded the selected plane directly from the three shipped shards and compared all 169,869,312 weight bytes and 21,233,664 scale codes against the prescribed parents. All 128 expert pairs were distinguishable. [File hashes, full ranking and measured trials](receipt.json) · [Independent check](verify.out)

## Reproduce

Generate the ternary package, sample-06.json, cutoffs, training routing.json and sample executable by following the [parent recipe](../). Use the corresponding direct-frame control package described there. Set RECIPE to this directory's absolute path, TERNARY and CONTROL to the model directories, and OFFICIAL to the official safetensors shard. Build the two examples from the 238b46c runtime.

```sh
cp "$RECIPE/fit.rs" crates/leansonic-core/examples/fit_ranked_experts.rs
cp "$RECIPE/verify.rs" crates/leansonic-core/examples/verify_experts.rs
cargo build --release -p leansonic-core --example fit_ranked_experts --example verify_experts
bin="$PWD/target/release/examples"
mkdir ranked-build
cd ranked-build
python3 "$RECIPE/assemble.py" "$TERNARY" "$CONTROL" "$OFFICIAL" "$SAMPLE_EXE" "$SELECTION" "$SAMPLE_06" "$ROUTING_JSON" "$bin/fit_ranked_experts" "$PWD/model" > build.json
shard=model-00001-of-00001.safetensors
python3 "$RECIPE/verify.py" build.json "$OFFICIAL" "$TERNARY/$shard" "$CONTROL/$shard" "$PWD/model/$shard" "$bin/verify_experts"
```

Use the standard NVFP4 inference path for the mixed down plane. For quality, run the same a200846 scorer and bc526e3 cubin on the same two BF16 references and RTX PRO 4500 as the [ternary baseline](../quality.md). Each split has 25,500 scored positions. The file-size comparison alone cannot establish equivalent quality.
