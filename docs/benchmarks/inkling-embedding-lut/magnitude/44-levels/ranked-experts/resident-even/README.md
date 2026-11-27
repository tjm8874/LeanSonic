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

## CUDA reader

The layer-0 gate/up projection has 121 compact experts out of 128. Its weight buffer shrinks from 339,738,624 to 259,449,344 bytes. The harness runs the fused gate/up plus Q8 activation kernel for 16 selections of eight experts, covering each expert once. It compares every Q8 value and FP32 scale byte-for-byte and checks 512 gate/up dot products using a separate double-precision calculation. Each timing is the median of 18 CUDA-event observations per representation across nine mirrored four-leg cycles. Each observation covers the same 16 kernel launches.

| GPU | Raw ms | Mixed ms | Raw / mixed |
| --- | ---: | ---: | ---: |
| RTX 5060 Ti | 0.922160 | 0.786192 | 1.173× |
| RTX PRO 4500 | 0.497504 | 0.431632 | 1.153× |

To reproduce, first run the packer as above. In the directory containing its four binary fixtures, build and run the harness using the same engine checkout and an sm_120a GPU.

```sh
python3 "$FIXTURE/generate.py" "$ENGINE/kernels/inkling2_sm120.cu"
nvcc -I"$ENGINE/kernels" -arch=sm_120a -O3 -std=c++17 --cubin candidate.cu -o candidate.cubin
nvcc -std=c++17 -O3 -arch=sm_120a "$FIXTURE/up.cu" -lcuda -o up
./up
```

The log for each tested GPU includes all 36 timings plus the executable, module and input digests.
