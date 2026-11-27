# GPU time by kernel on the compact engine

Hardware activity records cover one cold request and two warm requests on the integrated build. Each engine consumed the same 1,024 prompt and 32 continuation IDs as the [unprofiled benchmark](../README.md). Each warm request reproduced all 32 predictions. Each trace passed the dropped-record and timestamp checks. The table shows the three largest kernels by the mean of their cumulative GPU durations across the two warm requests. Overlapping kernels can contribute to multiple totals.

| GPU | Phase | Kernel | Calls per request | GPU ms per request |
|---|---|---|---:|---:|
| 5070Ti | prefill | `ink2_moe_nvfp4_tc_up_situ_packed_scales` | 44 | 43.774 |
| 5070Ti | prefill | `ink2_moe_nvfp4_tc_down_packed_scales` | 44 | 17.579 |
| 5070Ti | prefill | `ink2_q8_tc_gemm_b` | 68 | 11.078 |
| 5070Ti | decode32 | `ink2_moe_nvfp4_dp4a_up_half_q8_packed_scales` | 1,408 | 44.994 |
| 5070Ti | decode32 | `ink2_q8_dp4a_qkv3_gemv_r1` | 896 | 24.143 |
| 5070Ti | decode32 | `ink2_moe_nvfp4_dp4a_down_xw_packed_scales` | 1,408 | 22.975 |
| 5060Ti | prefill | `ink2_moe_nvfp4_tc_up_situ_packed_scales` | 44 | 104.120 |
| 5060Ti | prefill | `ink2_moe_nvfp4_tc_down_packed_scales` | 44 | 36.626 |
| 5060Ti | prefill | `ink2_q8_tc_gemm_b` | 68 | 19.337 |
| 5060Ti | decode32 | `ink2_moe_nvfp4_dp4a_up_half_q8_packed_scales` | 1,408 | 83.467 |
| 5060Ti | decode32 | `ink2_q8_dp4a_qkv3_gemv_r1` | 896 | 45.614 |
| 5060Ti | decode32 | `ink2_moe_nvfp4_dp4a_down_xw_packed_scales` | 1,408 | 42.668 |

Use the uninstrumented runs for throughput. The [receipt](receipt.json) includes each capture’s kernel counts, register counts, shared-memory use, cumulative durations, wall and union times, input predictions and trace fingerprint. The activity stream identifies where time was spent. It does not distinguish memory stalls from instruction or synchronization costs.

## Reproduce

Use CUDA 13 with CUPTI and the build and input files of the [integrated speed comparison](../README.md). The following commands assume the Linux x86-64 toolkit layout; point CUDA_ROOT at the toolkit containing cupti.h and libcupti.so.13. Save the instrument sources outside the engine checkout before switching that checkout to `a567839d`. Use a new output directory because the tracer opens kernels.tsv exclusively.

```sh
cp "$CAPTURE_SRC/profile.rs" crates/leansonic-core/examples/inkling_phase_capture.rs
mkdir "$RUN_DIR"
cc -O2 -Wall -Wextra -Werror -fPIC -shared "$CAPTURE_SRC/activity.c" -I"$CUDA_ROOT/include" -L"$CUDA_ROOT/lib" -l:libcupti.so.13 -o "$RUN_DIR/libactivity.so"
export LIBRARY_PATH="$RUN_DIR:$CUDA_ROOT/lib/stubs:$CUDA_ROOT/lib"
export LD_LIBRARY_PATH="$RUN_DIR:$CUDA_ROOT/lib:${LD_LIBRARY_PATH:-}"
cargo build --release -p leansonic-core --example inkling_phase_capture
exe="$PWD/target/release/examples/inkling_phase_capture"
cd "$RUN_DIR"
export LEANSONIC_MXFP8_DEVICE_U4=1 LEANSONIC_MXFP8_DIRECT_ALL=1 LEANSONIC_PDL_ALL=1 LEANSONIC_PDL=1
export LEANSONIC_KV=fp16 LEANSONIC_SEQS=1 LEANSONIC_CUBIN_PATH="$MATCHED_CUBIN"
unset LEANSONIC_PREFILL_CHUNK
"$exe" "$MODEL" "$PROMPT_IDS" "$CONTINUATION_IDS" >ranges 2>capture.log
python3 "$CAPTURE_SRC/summarize.py" >summary.json
```
