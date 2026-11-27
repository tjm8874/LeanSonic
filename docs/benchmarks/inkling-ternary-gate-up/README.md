# Resident two-bit ternary gate/up weights

Inkling-2-Mini's 20 NVFP4 gate/up planes contain twice as many weights as its down planes. On the 7,718,950,382-byte direct-coded ternary model the GPU can keep each resident gate/up plane at two bits per weight and consume it directly for one-token fused SiTU/Q8 decode. Enable with `LEANSONIC_TERNARY_UP=1`. The direct-coded loader validates the weight alphabet before packing. Host-backed gate/up matrices retain their original width for PCIe staging. Grouped and verify paths can expand one resident matrix in a reused buffer on the compute stream.

## Integrated parity and memory

On RTX 5070 Ti the sampled per-process peak fell from **13,922,992,128** to **10,865,344,512 bytes**, a reduction of **3,057,647,616 bytes**. Both arms enabled two-bit down. Each had one fresh process, used identical weights, retained all layers on the GPU, and ran a 32-token and a 1,024-token prefix followed by 64 free greedy steps. The four complete FP32 logit vectors (3,200,928 bytes), generated IDs and their successors were byte-identical. Observed peaks include model loading. This smoke test measures peak memory and exact outputs. [Arm peaks, GPU identity and output hashes](integrated/5070Ti/).

Rebuild 1976fb3 with the [integrated parity driver and README fixture](../inkling-ternary-down/integrated/). Set all runtime switches listed there, pin LEANSONIC_TERNARY_DOWN=1 for both arms, and vary LEANSONIC_TERNARY_UP from 0 to 1. Run each arm in a fresh process. To sample memory, use the [process-tree NVML sampler](../inkling-sglang/peak.py) with the process PID in a file and the GPU UUID in CUDA_VISIBLE_DEVICES.

## Layer-0 fused kernel

RTX 5070 Ti executed a batch of 16 independent launches covering all 128 experts. On average over nine mirrored four-leg cycles the nibble kernel took 0.486830 ms and the two-bit prototype took 0.313168 ms. The measurements cover only this kernel. They use actual layer-0 gate/up weights, scales and expert amplitudes, fixed seeded Q8 activations, and a fixed permutation of the experts. All 147,456 output Q8 bytes and all 9,216 float scale values matched bit-for-bit. An independent check recomputed 512 gate/up dots in FP64 and checked the resulting gated values against their Q8 reconstructions. The source nibble plane is 339,738,624 bytes and its two-bit version occupies 169,869,312 bytes.

Reproduce the prototype using an absolute SOURCE checkout of 8fbb862, absolute paths to the compressed and official shards, and RECIPE pointing to the kernel directory here. Start in an empty directory.

```sh
cp "$SOURCE/docs/benchmarks/inkling-ternary-down/extract.rs" "$SOURCE/crates/leansonic-core/examples/ternary_up_extract.rs"
cargo build --release --manifest-path "$SOURCE/Cargo.toml" -p leansonic-core --example ternary_up_extract
python3 "$RECIPE/extract_plane.py" "$TERNARY_SHARD" "$OFFICIAL_SHARD" "$SOURCE/target/release/examples/ternary_up_extract"
python3 "$RECIPE/generate.py" "$SOURCE/kernels/inkling2_sm120.cu"
nvcc -I"$SOURCE/kernels" -arch=sm_120a -std=c++17 -O3 --cubin trial.cu -o candidate.cubin
nvcc -std=c++17 -O3 "$RECIPE/up.cu" -lcuda -o up
./up >runs.csv
```

[Paired CUDA event measurements](kernel/5070Ti-runs.csv) · [Actual plane and binary digests](kernel/5070Ti-digests) · [GPU configuration](kernel/5070Ti-gpu.csv)
