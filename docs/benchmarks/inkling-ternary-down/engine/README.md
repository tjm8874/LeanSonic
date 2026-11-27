# End-to-end resident ternary down weights

Both arms run f429b35 using the same cubin. Prior loads four-bit E2M1 down weights. Candidate validates the same weights and stores them in two bits per element. The single-token path consumes these packed codes directly. Model precision and the package are identical.

The model and tokenizer total 7,718,950,382 bytes in both arms. One sequence, FP16 KV, 4096 context, no drafter; all 24 expert layers were resident. Each process runs four passes, resetting between each prefix. The prompt contains 1,024 IDs, repeated for the 2,048 case. Generate 64 greedy tokens. Check all generated IDs, tail ID, and the full final FP32 logit-vector hash across every leg, pass and arm per prefix. Each input made identical predictions in both arms. This verifies equivalence of these execution paths on this fixture. The separate BF16 corpus evaluation measures the effect of the ternary model itself.

Five four-leg cycles (ABBA alternating with BAAB), a fresh process per leg. Phase means exclude the first pass after loading. Intervals for candidate minus prior are the 2.5th/97.5th percentiles of the five-cycle paired bootstrap. Peak memory is per-process NVML sampled about every 4 ms during load and execution.

| GPU | Prompt tokens | Prefill, prior → candidate (ms) | Generation of 64 tokens, prior → candidate (ms) | Decode, prior → candidate (tok/s) |
|---|---:|---:|---:|---:|
| PRO5000 | 32 | 30.609 → 39.493 | 289.919 → 277.857 | 220.75 → 230.33 |
| PRO5000 | 1024 | 89.637 → 98.963 | 451.293 → 439.133 | 141.81 → 145.74 |
| PRO5000 | 2048 | 142.189 → 151.702 | 478.311 → 466.645 | 133.80 → 137.15 |
| 5070Ti | 32 | 35.063 → 49.867 | 363.211 → 346.074 | 176.21 → 184.93 |
| 5070Ti | 1024 | 122.196 → 137.215 | 497.122 → 479.996 | 128.74 → 133.33 |
| 5070Ti | 2048 | 257.166 → 219.893 | 521.657 → 502.346 | 122.69 → 127.40 |

| GPU | Prior peak (bytes) | Candidate peak (bytes) | Difference (bytes) |
|---|---:|---:|---:|
| PRO5000 | 16,741,564,416 | 15,233,712,128 | -1,507,852,288 |
| 5070Ti | 16,586,375,168 | 15,065,939,968 | -1,520,435,200 |

## Reproduce

Check out f429b35e60968d418f928b5e9da75e1cecde4943 and build the [same model and tokenizer](https://github.com/tjm8874/LeanSonic/tree/0db9fe7/docs/benchmarks/inkling-codebooks/nvfp4-budget/recovered/ternary). Use a dedicated Blackwell GPU. From the repository root, with RECIPE pointing to this directory:

```sh
cp "$RECIPE/bench.rs" crates/leansonic-core/examples/resident_ternary_abba.rs
cargo build --release -p leansonic-core --example resident_ternary_abba
nvcc -Ikernels -arch=sm_120a -std=c++17 -O3 --cubin kernels/inkling2_sm120.cu -o engine.cubin
export LEANSONIC_KV=fp16 LEANSONIC_PDL=1 LEANSONIC_PDL_ALL=1 LEANSONIC_SEQS=1
export LEANSONIC_MXFP8_DEVICE_U4=1 LEANSONIC_MXFP8_DIRECT_ALL=1 LEANSONIC_CUBIN_PATH="$PWD/engine.cubin"
for representation in 0 1; do
  LEANSONIC_TERNARY_DOWN=$representation target/release/examples/resident_ternary_abba "$MODEL_DIR" "$RECIPE/prompt.ids"
done
```

[Full results, per-leg peak, loader, raw token predictions, hashes and paired intervals](results.json) · [Measurement harness](bench.rs) · [Leg sampler](leg.py).
