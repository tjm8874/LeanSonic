# Lossless resident representation of 571-expert model weights

The materialized 571-expert model decodes to 10,192,158,720 bytes across 40 NVFP4 expert weight tensors. In 1,997 of the 5,120 expert slices, every nibble has its low bit clear. The packer stores each group of 32 such weights in 12 bytes and copies the remaining slices verbatim. It reconstructs and compares the complete decoded stream for each tensor. Including 512 descriptor bytes per tensor, the representation occupies 9,198,178,304 bytes and saves 993,980,416 bytes. These counts describe only the weight buffers. Application peak VRAM, kernel throughput and package size require separate measurements.

To reproduce, use the engine source at 726b6f2142a4095524e52f1877cffcf7bbd0ff19 and the [571-expert package](../README.md). Run the following from the engine root, supplying the paths to this fixture and to the materialized model. The Python harness writes its report and first-layer binary fixtures in a fresh temporary directory.

```sh
cp "$FIXTURE/pack.rs" crates/leansonic-core/examples/pack_mixed_even.rs
cargo build --release -p leansonic-core --example pack_mixed_even
work=$(mktemp -d)
cp target/release/examples/pack_mixed_even "$work/pack"
(cd "$work" && python3 "$FIXTURE/pack.py" "$MODEL")
cmp "$work/report.json" "$FIXTURE/report.json"
```

The [per-tensor receipt](report.json) records the raw and encoded byte counts and the number of eligible slices.
