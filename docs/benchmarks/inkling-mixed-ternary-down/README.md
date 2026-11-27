# Selective two-bit down projections on a mixed Inkling model

With LEANSONIC_TERNARY_DOWN=auto the loader checks each direct or four-bank NVFP4 down projection independently. Every reachable weight nibble must encode 0, 7 or 15 for two-bit GPU residency. Other projections keep their existing representation and dispatch. Strict mode (1) requires all 20 planes to qualify. The option is disabled by default.

On RTX PRO 4500 (driver 595.99.02, 200 W), the 7,763,991,646-byte model with [61 experts restored in layer 17](https://github.com/tjm8874/LeanSonic/tree/22ef9e3/docs/benchmarks/inkling-codebooks/nvfp4-budget/recovered/ternary/ranked-experts/expanded) selected 19/20 down projections. Both runs kept all 24 expert layers on GPU. The observed per-process peak fell from 13,981,712,384 to 12,557,746,176 bytes, a difference of 1,423,966,208 bytes. Each arm was loaded once. This run measures representation parity and sampled peak memory. It does not estimate warm latency or corpus fidelity to BF16.

The input is the [1,024-token README fixture](../inputs/inkling-readme1024.ids), repeated up to 4,096 tokens. Each arm evaluates prefixes of length 32, 1,024, 3,072 and 4,096 with 64 greedy steps each. The predicted IDs, tails and all 1,600,464 finite FP32 values from the eight endpoint logit vectors match byte for byte. The strict-mode negative control failed on the restored weights as expected.

To reproduce, copy smoke.rs to crates/leansonic-core/examples/mixed_down_smoke.rs, build it with cargo build --release -p leansonic-core --example mixed_down_smoke, and build kernels/inkling2_sm120.cu for sm_120a using nvcc -Ikernels -arch=sm_120a -std=c++17 -O3 --cubin. Set LEANSONIC_CUBIN_PATH to that cubin. The measured source was 3364f6d. The runtime sources and kernels match 0b6a7c9 on the #196 base. Use the model shard with SHA-256 affb919ab701a4edff6aaaa18c8726989e1b6040104b4ec303c805f0097dcc84.

```sh
export LEANSONIC_CTX=8192 LEANSONIC_KV=fp16 LEANSONIC_PDL=1 LEANSONIC_PDL_ALL=1
export LEANSONIC_SEQS=1 LEANSONIC_PREFILL_CHUNK=4096 LEANSONIC_MXFP8_DEVICE_U4=1
export LEANSONIC_MXFP8_DIRECT_ALL=1 LEANSONIC_TERNARY_UP=1
for mode in 0 auto; do
  LEANSONIC_TERNARY_DOWN="$mode" target/release/examples/mixed_down_smoke "$MODEL" docs/benchmarks/inputs/inkling-readme1024.ids "$mode"
done
cmp 0.ids auto.ids
cmp 0.f32 auto.f32
```

The PRO4500 directory includes the binary digests, endpoint and ID digests, full generated ID histories, and per-arm peak samples.
