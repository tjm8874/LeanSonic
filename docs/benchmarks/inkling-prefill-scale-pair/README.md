# Adjacent packed scale loads in 128-token NVFP4 prefill

Two 32-bit aligned reads cover eight UE4M3 six-bit scale codes. This candidate expands an adjacent pair during FP4 Tensor Core staging. On sm_120a, cuobjdump reports fewer static LDG instructions in both packed kernels (up 11 to 7, down 12 to 8). Registers remain at 126 and 127 respectively and local memory and stack remain zero. Instruction counts are static; the measured end-to-end prefill time determines adoption. [Opcode and resource counts](sass.json).

The same integrated e030d0c Rust executable loads prior and candidate cubins. Use the [13,479,710,294-byte direct-coded artifact](https://github.com/tjm8874/LeanSonic/tree/897b812/docs/benchmarks/inkling-nvfp4-scale-split/direct). FP16 KV, context 8192, batch 1, spec 0, PDL enabled. Five paired four-process cycles alternate ABBA and BAAB. Each process repeats the 32/1024/2048 prefix sweep four times, with the first sweep excluded from timings and included in output checks. The 2048 fixture repeats the 1024-token prompt. Per length, all final logits hashes and all 64 generated token IDs must agree. Intervals resample the five paired cycles.

| GPU | Input | Prior prefill ms | Candidate prefill ms | Difference ms (paired 95% interval) |
|---|---:|---:|---:|---:|
| 5070Ti | 32 | 35.015 | 35.078 | +0.063 (+0.057 to +0.068) |
| 5070Ti | 1024 | 122.679 | 122.959 | +0.280 (+0.267 to +0.291) |
| 5070Ti | 2048 | 257.468 | 258.103 | +0.635 (+0.573 to +0.731) |

Keep the existing kernel on RTX 5070 Ti. All three measured prefill lengths regress despite the lower static load count. Raw per-process output, timing, binaries and input hashes are in each GPU JSON. The standalone edge check covers both bit alignments, exact allocation tails and eight array lengths.

## Reproduce

Build kernels/inkling2_sm120.cu from 2ba9b00 and 0ac02bb with nvcc -arch=sm_120a -std=c++17 -O3 --cubin. Put bench.rs in crates/leansonic-core/examples/check_scale_pair.rs at e030d0c. Build the release example and run it once per leg, with LEANSONIC_CUBIN_PATH pointing to the selected cubin. Supply the input IDs used by the public fixed-Q3 driver. Use the environment below; set CUDA_VISIBLE_DEVICES to the selected board.

```sh
export LEANSONIC_PDL=1 LEANSONIC_PDL_ALL=1 LEANSONIC_KV=fp16 LEANSONIC_SEQS=1 LEANSONIC_CTX=8192
export LEANSONIC_MXFP8_DEVICE_U4=1 LEANSONIC_MXFP8_DIRECT_ALL=1
cargo build --release -p leansonic-core --example check_scale_pair
target/release/examples/check_scale_pair "$MODEL" "$PROMPT_IDS"
nvcc -arch=sm_120a -std=c++17 -O3 edge.cu -o edge
compute-sanitizer --tool memcheck --error-exitcode 9 ./edge
```
