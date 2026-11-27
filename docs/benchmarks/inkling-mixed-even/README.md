# Lossless mixed-width gate/up weights on SM120

In the 11,487,467,790-byte [571-expert package](https://github.com/tjm8874/LeanSonic/tree/6c1fe17/docs/benchmarks/inkling-embedding-lut/magnitude/44-levels/ranked-experts/class-banks), eight gate/up tensors have experts whose NVFP4 codes all have their low bit clear. Setting `LEANSONIC_MIXED_EVEN_UP=1` stores those expert slices in three bits per weight and the others in their original four bits. The decode and grouped Tensor Core kernels recover the original nibbles before arithmetic. On other kernel paths, the engine expands the plane into a shared 339,738,624-byte mapped, pinned host buffer before launching the original reader. The option currently requires one sequence and the companion CUDA module.

| GPU | Storage | Prefill tok/s | Decode tok/s | Peak process GPU bytes |
| --- | --- | ---: | ---: | ---: |
| RTX 5060 Ti | Raw | 3,983.21 | 94.53 | 15,661,531,136 |
| RTX 5060 Ti | Mixed | 4,004.12 | 95.22 | 15,007,219,712 |
| RTX 5070 Ti | Raw | 8,416.03 | 170.92 | 15,760,097,280 |
| RTX 5070 Ti | Mixed | 8,366.77 | 171.95 | 15,105,785,856 |

Each row is one process, with one excluded warmup followed by four requests using 1,024 fixed input IDs and 32 forced decode steps. Both arms on each GPU predicted the same 32 IDs on all five requests. Both GPUs saved 654,311,424 peak process GPU bytes in this pilot. The 5060 Ti measurements use the first implementation, the 5070 Ti uses 5b91ec4 with the revised host packer. The pilot does not establish a throughput ranking. The receipts include the inputs’ digests, all predicted IDs, host peak RSS and the sampled GPU peaks. [RTX 5060 Ti](5060ti.json) · [RTX 5070 Ti](5070ti.json).

On both the 5060 Ti pilot and the committed source on 5070 Ti, the two formats produced identical per-row full-vocabulary FP32 logit fingerprints, top tokens, NLL and KL on 1,530 positions with Tensor Core prefill enabled. With it disabled the fallback matched for all 255 positions of the first chunk. The six diagnostic chunks are intentionally difficult and have a separate [selection record and observer methodology](https://github.com/tjm8874/LeanSonic/tree/c2b7174/docs/benchmarks/inkling-g1-parity).

## Materialization cost on RTX PRO 4500

With device scratch for generic readers, an earlier single-process experiment reproduced the 32 reference predictions on RTX PRO 4500. Expanding the entire layer before grouped Tensor Core prefill lowered throughput from 8,285.7 to 7,911.8 tok/s. Reading the packed tiles directly measured 8,299.2 vs 8,360.4 tok/s for that session’s raw control, and decode measured 165.49 vs 164.44 tok/s. The device-scratch variant saved 314,572,800 peak GPU bytes. These observations led to the direct Tensor Core path and the mapped host buffer for generic readers in the current candidate. [Expand then prefill](4500-expand.json) · [Direct prefill](4500-direct.json).

## Reproduction

Use source 5b91ec4, the file hashes in the package link above and the 1,024 + 32 fixed token ID files used by the [matched native comparison](https://github.com/tjm8874/LeanSonic/blob/f6dbff3/docs/benchmarks/inkling-embedding-lut/refit/native-corpus/timing.md). Build the module and fixed-history driver from the engine root.

```sh
nvcc -Ikernels -arch=sm_120a -O3 -std=c++17 --cubin kernels/inkling2_sm120.cu -o candidate.cubin
cp docs/benchmarks/inkling-mixed-even/fixed_decode.rs crates/leansonic-core/examples/inkling_fixed_decode.rs
cargo build --release -p leansonic-core --example inkling_fixed_decode
export LEANSONIC_CUBIN_PATH="$PWD/candidate.cubin"
export LEANSONIC_PDL=1 LEANSONIC_PDL_ALL=1 LEANSONIC_KV=fp16 LEANSONIC_SEQS=1
export LEANSONIC_PREFILL_CHUNK=1024 LEANSONIC_MXFP8_DEVICE_U4=1 LEANSONIC_MXFP8_DIRECT_ALL=1
export LEANSONIC_TERNARY_UP=0 LEANSONIC_TERNARY_DOWN=0
for arm in 0 1; do
  LEANSONIC_MIXED_EVEN_UP="$arm" target/release/examples/inkling_fixed_decode "$MODEL" "$PROMPT_IDS" "$CONTINUATION_IDS"
done
```

For full-vocabulary observations, apply `observe_rows.py` to `crates/leansonic-cli/src/eval_kld.rs`, rebuild the release CLI, set the prefill chunk to 512, set `DIAG_ROWS` to a fresh filename per arm, and use `eval-kld --weights "$MODEL" --ref "$SUBSET" --chunks 6`. Repeat with `LEANSONIC_TC_PREFILL=0` and one chunk to exercise the generic reader.
