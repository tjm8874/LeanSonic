# Ternary gate/up parity after integrating main

Both arms used the same 349c7e8 executable and sm_120a cubin. The down matrices were resident in two bits in both arms. The control kept gate/up in four-bit nibbles and the candidate stored them at two bits with tile-local expansion. Each arm used a fixed 4,096-token workspace and an 8,192-token context. All 24 expert layers were resident.

The test repeated the same 1,024-ID fixture four times, reset the engine for each prefix of 32, 1,024, 3,072 or 4,096 IDs, and generated 64 tokens greedily with spec disabled. It checked every emitted ID, the successor and eight full finite FP32 logit vectors (6,401,856 bytes) for bit equality on each reported GPU.

| GPU | Peak process bytes, full / two-bit gate/up | SHA-256 of the eight logit vectors |
|---|---:|---|
| [5070Ti](5070Ti.json) | 15,497,953,280 / 12,440,305,664 | `1c6ace0d283340ee3ce893a8bdb6a6abca3914d9cc88f599d196082bcf2d85ab` |

## Reproduce

Build this directory's smoke.rs as a release leansonic-core example at 349c7e8. Compile kernels/inkling2_sm120.cu with nvcc -Ikernels -arch=sm_120a -std=c++17 -O3 --cubin. Use the [7,718,950,382-byte ternary model and fixed 1,024 prompt IDs](../../inkling-ternary-down/engine/). Export LEANSONIC_CTX=8192, LEANSONIC_KV=fp16, LEANSONIC_SEQS=1, LEANSONIC_PREFILL_CHUNK=4096, LEANSONIC_PDL=1, LEANSONIC_PDL_ALL=1, LEANSONIC_MXFP8_DEVICE_U4=1, LEANSONIC_MXFP8_DIRECT_ALL=1, LEANSONIC_TERNARY_DOWN=1 and LEANSONIC_CUBIN_PATH to the new module. Run each arm in a fresh process on the same pinned GPU.

```sh
LEANSONIC_TERNARY_UP=0 ./bench "$MODEL" prompt.ids 0
LEANSONIC_TERNARY_UP=1 ./bench "$MODEL" prompt.ids 1
cmp 0.ids 1.ids
cmp 0.f32 1.f32
```
