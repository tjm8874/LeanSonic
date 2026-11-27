# Condition the BF16 prefix rank on the low byte

The low byte of a BF16 word includes the low exponent bit. Separate seven-entry rank orders for even and odd exponents reduce the official input tensor from 618,590,352 to 616,383,952 bytes (−2,206,400). The existing sign and low-byte streams are retained. The new high-bit rank encoding requires a distinct format and reader. Its SHA-256 on the official 926,318,592-byte raw tensor is `d3d0ece2a3c3dd9c5860980aaf1a33de7747b07f97db5f5bb5f50bc6800cf958`.

On a Ryzen 9 9900X, sequential tile expansion averaged 55.181 ms for the current stream versus 59.666 ms for the conditional stream across five mirrored cycles after warm-up. A separate three-arm run rotated all six orders twice: current stream 55.043 ms, current stream with paired tiles 51.927 ms, conditional stream with paired tiles 54.611 ms. All decoded bytes were checked after each leg. The fixture also exercises all 65,536 BF16 patterns and partial tiles. The smaller tensor pays additional CPU decode cost compared with pairing the existing stream. We retained this as a codec experiment. The independent compatible loader change and GPU parity checks are in [PR #191](https://github.com/tjm8874/LeanSonic/pull/191).

Use a scratch source checkout at `00052c8384ac47dea1bc4b70f06824382cfe7425`. Set RECIPE to this directory, RAW to the official tensor payload, and PREFIX to the existing prefix payload; the [extractor and reference hashes](https://github.com/tjm8874/LeanSonic/tree/15670a3/docs/benchmarks/inkling-paired-embedding) are public. Build the serial example first to produce conditional-prefix.bin. Then build the three-arm example. Both use the production prefix decoder as control.

```sh
base=crates/leansonic-core/src/inkling_embedding.rs
for kind in serial paired; do
  ex=crates/leansonic-core/examples/conditional_$kind
  mkdir -p "$ex"
  cp "$RECIPE/$kind/full.rs" "$ex/main.rs"
  for patchfile in "$RECIPE/$kind/"*.rs.patch; do
    target=$(basename "$patchfile" .patch)
    patch -s -o "$ex/$target" "$base" <"$patchfile"
  done
  cargo build --release -p leansonic-core --example "conditional_$kind"
done
target/release/examples/conditional_serial "$RAW" "$PREFIX" >serial.csv
target/release/examples/conditional_paired "$RAW" "$PREFIX" conditional-prefix.bin >paired.csv
```
