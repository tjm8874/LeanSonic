# Locate a routed expert tile with one warp

Each warp lane reads four adjacent expert counts. An exclusive prefix over the 32 lanes locates the one lane that owns the requested 128- or 160-row tile. That lane selects the expert and publishes its offset, first row and count in one shared int4. The CTA crosses one barrier. The prior implementation had four warps populate two 129-element tables and used two barriers.

The device test covers 104 input count vectors per tile width, surplus grid tiles and block sizes 256, 384 and 512. Every thread agrees with the CPU reference. Compute Sanitizer memcheck reported zero errors. The full engine uses the canonical-zero Inkling-2-Mini package, global PDL, FP16 KV, one sequence and context 4,096. Each request generates 64 free tokens. For all completed comparisons the full token sequence, the next prediction and the hash of the final 200,058 finite FP32 logits matched across all processes and passes at each input length.

| GPU | Input tokens | Prior prefill ms | Candidate prefill ms | Change ms (95% paired interval) | Change in 64-token decode ms (95% interval) |
|---|---:|---:|---:|---:|---:|
| PRO4500 | 32 | 36.501 | 36.482 | -0.018 [-0.028, -0.009] | +0.038 [-0.174, +0.300] |
| PRO4500 | 1024 | 122.331 | 121.834 | -0.497 [-0.637, -0.342] | +0.110 [-0.041, +0.261] |
| PRO4500 | 2048 | 208.784 | 207.733 | -1.051 [-1.208, -0.867] | +0.079 [-0.082, +0.256] |

## Reproduce

Build the baseline cubin at `1a701b0` and candidate at `01c4a0c` with CUDA 13.0.88. Use the same Rust executable for both.

```sh
nvcc -Ikernels -arch=sm_120a -std=c++17 -O3 --cubin kernels/inkling2_sm120.cu -o "$OUT/candidate.cubin"
nvcc -Ikernels -arch=sm_120a -std=c++17 -O3 "$EVIDENCE/check.cu" -o "$OUT/check"
compute-sanitizer --tool memcheck --error-exitcode 42 "$OUT/check"
cp "$EVIDENCE/bench.rs" crates/leansonic-core/examples/routed_tile_check.rs
cargo build --release -p leansonic-core --example routed_tile_check
export LEANSONIC_KV=fp16 LEANSONIC_PDL=1 LEANSONIC_SEQS=1
LEANSONIC_CUBIN_PATH="$OUT/candidate.cubin" target/release/examples/routed_tile_check "$MODEL" docs/benchmarks/inputs/inkling-readme1024.ids
```

Start ten fresh processes per arm in five mirrored ABBA/BAAB cycles. Each process runs four passes of 32, 1,024 and 2,048 tokens (the 2,048 fixture repeats the README input twice). Discard pass zero for timings. Each paired interval resamples entire cycles. [Individual requests, generated IDs, build hashes and Compute Sanitizer output](receipt.json).
