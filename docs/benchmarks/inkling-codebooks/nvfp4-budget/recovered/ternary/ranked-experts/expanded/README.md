# Restoring 61 experts inside the IQ1_S budget

The same model can keep 19 additional experts at control precision after lossless re-encoding. The direct 42-expert package occupied 7,764,131,886 bytes. After four-bank encoding it occupied 7,747,846,286 bytes. Spending the recovered space on the next experts in the frozen training rank produced a 61-expert package of **7,763,991,646 bytes**, 300,130 bytes below IQ1_S and 140,240 bytes smaller than the direct 42-expert package. All sizes include the tokenizer.

Only the layer 17 down weight frame and its companion scales changed from the [four-bank parent](../four-banks/). An independent decoder compared all 128 expert blocks to their source models. The first 61 in training rank reproduced the control, the other 67 reproduced the ternary parent. The assembler reread the other 997 tensor payloads. [Actual trial sizes, expert IDs and final shard hash](receipt.json) · [Independent readback](verify.out)

## Reproduce

Start from the [ranked-experts build](../), including its four original direct weight/scale frames and rank.bin, and the [physical four-bank model](../four-banks/). Work in a new directory. Set RECIPE to this directory and RANK_WORK to the original direct build directory. Set DECODER to a checkout of 238b46c, FOUR_PARENT to the 42-expert four-bank model, and OFFICIAL to the original safetensors shard.

```sh
for name in fit.rs verify.rs verify.py; do
  patch -s -o "$PWD/$name" "$RECIPE/../$name" < "$RECIPE/$name.patch"
done
cp fit.rs "$DECODER/crates/leansonic-core/examples/fit_four_ranked.rs"
cp verify.rs "$DECODER/crates/leansonic-core/examples/verify_four_ranked.rs"
(cd "$DECODER" && cargo build --release -p leansonic-core --example fit_four_ranked --example verify_four_ranked)
bin="$DECODER/target/release/examples"
for name in rank.bin ternary.weights ternary.scales control.weights control.scales; do
  ln -s "$RANK_WORK/$name" "$name"
done
python3 "$RECIPE/assemble.py" "$FOUR_PARENT" "$RECIPE/../receipt.json" "$bin/fit_four_ranked" "$PWD/model" >receipt.json
shard=model-00001-of-00001.safetensors
python3 verify.py receipt.json "$OFFICIAL" "$TERNARY/$shard" "$CONTROL/$shard" "$PWD/model/$shard" "$bin/verify_four_ranked"
```

The selection used training data only and retained the same rank throughout. Additional restored weights need a corpus replay to determine their effect on BF16 agreement. The three-arm evaluation uses the original ternary model, the 42-expert four-bank package and this 61-expert package on the same executable, cubin, PRO 4500 and test/validation references.
