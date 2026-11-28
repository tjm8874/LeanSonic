# Lossless expert exponent palettes on Blackwell

Each routed expert projection has 51,200 scale/bias pairs. Stock Z24 retains their sign and fraction in a 16-bit plane and their two exponent codes in an 8-bit plane. On all 73,728 expert projections in this checkpoint, the exponent-code pair took at most 16 distinct values within each expert. The experiment encodes the upper plane as one 128-byte aligned palette followed by 25,600 bytes of nibble indices per expert. The rank in the palette is assigned in first-seen order. The existing 16-bit plane is unchanged.

All 3,774,873,600 pairs were verified against the source BF16 values on the host. The new upper planes were physically written and reread (1,896,873,984 bytes vs 3,774,873,600). A separate CUDA decoder checked every group on the RTX PRO 4500. The upper-plane allocation effect in that stand-alone test includes allocator rounding per projection. [Host readback and palette sizes](z24-four-bit.json) · [Device verification](z24-gpu.json)

In the full engine on an RTX PRO 5000 72 GB, the new layout lowered the sampled process peak from **70.256 to 68.475 GiB**, freeing **1.781 GiB** while retaining 48/48 expert layers in VRAM. Raw and palette variants loaded the same directly loadable Q4 checkpoint. Full-vocabulary FP32 logits were byte-identical at all 49 teacher-forced positions, and all four greedy continuations matched. An alternating stock/palette/palette/stock trial used resident greedy processes at context 256, with one warmup and three measurements per input per process. All generated IDs matched. [Individual samples and quality scores](native72.json)

| Input / output | Median TTFT stock / palette (ms) | Decode stock / palette (tok/s) |
| --- | ---: | ---: |
| 16 / 32 | 92.44 / 95.92 | 162.97 / 157.39 |
| 128 / 32 | 745.96 / 773.72 | 158.98 / 153.69 |

The build is opt-in. The measured 3.3–3.4% standalone decode cost matters when deciding whether to spend the freed VRAM on another component such as MTP. A combined MTP speed or fit still requires testing. The distribution checkpoint is shared between both runs.

For the full-engine run, copy the [teacher example](../teacher.rs) and [stream example](../../qwen38-engine-comparison/q38_stream.rs) into a scratch checkout's `crates/leansonic-core/examples`. Build both the Rust feature and its matching cubin. Use the quality and stream commands from the [parent benchmark](../README.md) with the same model and fixture IDs.

```sh
cargo build --release -p leansonic-core --features q38-exponent-palette --example q38_teacher --example q38_stream
nvcc -arch=sm_120a -O3 -std=c++17 -Ikernels -DLEANSONIC_Q38_EXPONENT_PALETTE --cubin kernels/qwen38_sm120.cu -o /tmp/q38-exp4.cubin
export LEANSONIC_QWEN38_CUBIN_PATH=/tmp/q38-exp4.cubin
```

For the independent upper-plane audit, run from a scratch directory containing the ordinary Q4 checkpoint at `restored`. Set `d` to this benchmark directory's absolute path.

```sh
python3 "$d/z24_spec.py" > pairs.spec
c++ -O3 -std=c++17 "$d/z24_palette.cpp" -o palette-audit
./palette-audit pairs.spec new-packed-planes
nvcc -O3 -arch=sm_120 -std=c++17 "$d/z24_gpu.cu" -o verify-on-gpu
./verify-on-gpu new-packed-planes
```
