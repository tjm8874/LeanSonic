# Packed Inkling GPU kernel time on RTX PRO 5000

Profiled integrated commit 8c77955 with the 13,445,948,726-byte distribution and bc526e3 cubin, FP16 KV, scratch-free U4, one sequence, context 2,048 and speculation off. The README fixture contains 1,024 token IDs; SHA-256 e93129d009d923541cd0b1ee39a6f121c11e000638ceb868258665ec974ed843. The drivers synchronize before starting and before stopping each capture. Model load and a full warmup are outside the range. Percentages below use the sum of GPU kernel durations within each range; they do not measure end-to-end latency.

## 1,024-token prefill

30 kernel names, 90.925 ms summed kernel time.

| Kernel | Share | Instances | Total ms |
|---|---:|---:|---:|
| `ink2_moe_nvfp4_tc_up_situ_packed_scales` | 36.9% | 44 | 33.526 |
| `ink2_moe_nvfp4_tc_down_packed_scales` | 14.4% | 44 | 13.103 |
| `ink2_q8_tc_gemm_b` | 9.3% | 68 | 8.490 |
| `ink2_flash_attn_tc_q8_b` | 6.1% | 48 | 5.511 |
| `ink2_q8_tc_qkv3_gemm_b` | 5.9% | 28 | 5.379 |
| `ink2_bf16_tc_gemm_b` | 4.8% | 96 | 4.342 |

[All kernel rows](pre.csv).

## 32-token greedy decode

25 kernel names, 209.807 ms summed kernel time.

| Kernel | Share | Instances | Total ms |
|---|---:|---:|---:|
| `ink2_moe_nvfp4_dp4a_up_half_q8_packed_scales` | 15.9% | 1408 | 33.311 |
| `ink2_gqa_rel_sink_attn_q8_decode` | 15.8% | 1536 | 33.225 |
| `ink2_moe_nvfp4_dp4a_down_xw_packed_scales` | 8.5% | 1408 | 17.882 |
| `ink2_q8_dp4a_qkv3_gemv_r1` | 8.3% | 896 | 17.462 |
| `ink2_q8_dp4a_gemv_c2304` | 6.1% | 32 | 12.782 |
| `ink2_q8_dp4a_gemv_c2304_r1` | 5.8% | 640 | 12.090 |

[All kernel rows](dec.csv).

## Reproduce

Copy the two Rust files into `crates/leansonic-core/examples` at 8c77955 and build with the CUDA driver library available at link time. Use the [fixture and small-package recipe](https://github.com/tjm8874/LeanSonic/tree/cc62564/docs/benchmarks/inkling-nvfp4-trim-context). Export `LEANSONIC_KV=fp16`, `LEANSONIC_PDL=1`, `LEANSONIC_PDL_ALL=1`, `LEANSONIC_MXFP8_DEVICE_U4=1`, `LEANSONIC_MXFP8_DIRECT_ALL=1`, `LEANSONIC_SEQS=1` and the cubin path. Leave the prefill chunk at its dynamic default. Use Nsight Systems 2025.3.2.

```sh
cargo build --release -p leansonic-core --example pre --example dec
for phase in pre dec; do
  nsys profile --sample=none --cpuctxsw=none --trace=cuda --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop -o "$phase" \
    "target/release/examples/$phase" "$MODEL" "$PROMPT_IDS"
  nsys stats --report cuda_gpu_kern_sum --format csv --output - "$phase.nsys-rep"
done
```

Prefill reproduced top ID 145026. Decode reproduced all 32 warmup IDs and next ID 2512. The first targeted change at 1cdc654 retains MMA arithmetic while removing the metadata barrier and redundant trailing reuse fences in the 128-token NVFP4 expert tile. Compare against a1bf5e3 with matched host code. Adoption requires repeated timing plus equal greedy IDs and final logit hashes.
