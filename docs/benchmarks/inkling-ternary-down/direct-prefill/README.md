# Direct two-bit staging in grouped prefill

Both arms load the same two-bit down weights and run the same Rust binary. The prior cubin (f429b35) expands a full plane before grouped prefill. The candidate (ad3b390) stages the tile directly from two-bit weights in the packed-scale Tensor Core kernel.

The model and tokenizer total 7,718,950,382 bytes in both arms. One sequence, FP16 KV, 4096 context, no drafter; all 24 expert layers were resident. Each process runs four passes, resetting between each prefix. The prompt contains 1,024 IDs, repeated for the 2,048 case. Generate 64 greedy tokens. Check all generated IDs, tail ID, and the full final FP32 logit-vector hash across every leg, pass and arm per prefix. Each input made identical predictions in both arms. This verifies equivalence of these execution paths on this fixture. The separate BF16 corpus evaluation measures the effect of the ternary model itself.

Five four-leg cycles (ABBA alternating with BAAB), a fresh process per leg. Phase means exclude the first pass after loading. Intervals for candidate minus prior are the 2.5th/97.5th percentiles of the five-cycle paired bootstrap. Peak memory is per-process NVML sampled about every 4 ms during load and execution.

| GPU | Prompt tokens | Prefill, prior → candidate (ms) | Generation of 64 tokens, prior → candidate (ms) | Decode, prior → candidate (tok/s) |
|---|---:|---:|---:|---:|
| 5070Ti | 32 | 49.864 → 35.124 | 346.234 → 346.005 | 184.85 → 184.97 |
| 5070Ti | 1024 | 137.203 → 124.729 | 480.103 → 479.829 | 133.30 → 133.38 |
| 5070Ti | 2048 | 219.829 → 207.979 | 502.464 → 502.198 | 127.37 → 127.44 |

| GPU | Prior peak (bytes) | Candidate peak (bytes) | Difference (bytes) |
|---|---:|---:|---:|
| 5070Ti | 15,084,814,336 | 15,103,688,704 | +18,874,368 |

## Reproduce

Check out ad3b390daa1a101683bbb21fcb6e549d0d77a771 and build the [same model and tokenizer](https://github.com/tjm8874/LeanSonic/tree/0db9fe7/docs/benchmarks/inkling-codebooks/nvfp4-budget/recovered/ternary). Use a dedicated Blackwell GPU. From the repository root, with RECIPE pointing to this directory:

```sh
cp "$RECIPE/bench.rs" crates/leansonic-core/examples/resident_ternary_abba.rs
cargo build --release -p leansonic-core --example resident_ternary_abba
nvcc -Ikernels -arch=sm_120a -std=c++17 -O3 --cubin kernels/inkling2_sm120.cu -o engine.cubin
export LEANSONIC_KV=fp16 LEANSONIC_PDL=1 LEANSONIC_PDL_ALL=1 LEANSONIC_SEQS=1
export LEANSONIC_MXFP8_DEVICE_U4=1 LEANSONIC_MXFP8_DIRECT_ALL=1 LEANSONIC_CUBIN_PATH="$PWD/engine.cubin"
# Build prior.cubin separately using the same nvcc command at f429b35.
for module in prior.cubin engine.cubin; do
  LEANSONIC_TERNARY_DOWN=1 LEANSONIC_CUBIN_PATH="$PWD/$module" target/release/examples/resident_ternary_abba "$MODEL_DIR" "$RECIPE/prompt.ids"
done
```

[Full results, per-leg peak, loader, raw token predictions, hashes and paired intervals](results.json) · [Measurement harness](bench.rs) · [Leg sampler](leg.py).
