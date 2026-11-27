# Two-bit ternary weights in an Inkling down projection

The tested layer-0 down projection is decoded from the [7,718,950,382-byte ternary model](https://github.com/tjm8874/LeanSonic/tree/0db9fe7/docs/benchmarks/inkling-codebooks/nvfp4-budget/recovered/ternary). The native nibble representation needs 169,869,312 bytes of weight data; its fixed two-bit representation needs 84,934,656 bytes. The same 21,233,664 scale codes apply to both. The sign and zero pattern of all 339,738,624 weights round-tripped before timing.

The basic GEMV probe uses eight synthetic Q8 activation vectors with fixed seed and visits all 128 actual experts in groups of eight. Before timing, all 294,912 GPU outputs match bit-for-bit, and 1,024 sampled outputs also pass an independent FP64 bound. The second probe calls the actual expert-parallel production down kernel from c48e244 and its two-bit specialization at 4b8e9d6. Both receive identical six-bit packed scales, official per-expert amplitudes and seeded activations, gates and routing indices; they run through the same eight-expert reduction. All 36,864 FP32 output values match and 256 independent FP64 checks pass before timing. Neither test includes model loading, tokenization or end-to-end generation. Each leg times one sweep of 16 routing groups using CUDA events. Intervals resample nine paired four-leg cycles.

| GPU | Kernel | Nibbles (ms/sweep) | Two bits (ms/sweep) | Ratio (paired 95% interval) |
|---|---|---:|---:|---:|
| 5070Ti | GEMV | 0.268530 | 0.224224 | 1.198× (1.195–1.201) |
| 5070Ti | MoE down | 0.290633 | 0.184820 | 1.573× (1.568–1.577) |

The actual MoE specialization loads half-width words. PRMT expands each four two-bit codes directly into the signed byte values used by DP4A. The scale, activation, warp reduction and ordered expert gate math are shared with the nibble path. End-to-end prefill/decode, memory, and output checks use a separate full-engine experiment.

## Reproduce

At 4b8e9d6 compile the kernel and down probe on CUDA Blackwell. Use the a200846 workspace for the extract example (its direct scale-coded decoder is required). Set TERNARY_SHARD and OFFICIAL_SHARD to the final and original safetensors shards. Start from an empty data directory and pass absolute paths to the executable and both shards.

```sh
cp "$RECIPE/extract.rs" "$RUNTIME/crates/leansonic-core/examples/extract_ternary_probe.rs"
cargo build --release --manifest-path "$RUNTIME/Cargo.toml" -p leansonic-core --example extract_ternary_probe
python3 "$RECIPE/extract_plane.py" "$TERNARY_SHARD" "$OFFICIAL_SHARD" "$RUNTIME/target/release/examples/extract_ternary_probe"
sha256sum -c "$RECIPE/planes.sha"
nvcc -arch=sm_120a -std=c++17 -O3 --cubin "$KERNEL_SOURCE/kernels/inkling2_sm120.cu" -o candidate.cubin
nvcc -std=c++17 -O3 "$RECIPE/down.cu" -lcuda -o down
./down
nvcc -arch=sm_120a -std=c++17 -O3 "$RECIPE/gemv.cu" -o gemv
./gemv
```

[GEMV cycles](gemv.json) · [Production down cycles](down.json) · [Weight populations](plane.json).
