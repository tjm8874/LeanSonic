# Integrated two-bit down parity

RTX 5070 Ti, driver 595.99.02, executed 8fbb862 with its own release CLI example and sm_120a cubin. Each arm used the 7,718,950,382-byte direct-coded ternary model and full GPU residency. Prefixes were the first 32 and all 1,024 IDs from the [README fixture](../../inputs/inkling-readme1024.ids). For each prefix the probe compared the full FP32 logits after prefill and after 64 freely generated tokens. Every value was finite, the returned greedy ID pointed to the maximum logit, and the four vectors (3,200,928 bytes in total) and generated IDs were byte-identical between arms.

Set ENGINE to a checkout of 8fbb862, MODEL to the ternary package, FIXTURE to the absolute path of the 1,024 IDs, and CUDA_VISIBLE_DEVICES to the target GPU UUID. Use a fresh output directory.

```sh
cp smoke.rs "$ENGINE/crates/leansonic-core/examples/ternary_smoke.rs"
(cd "$ENGINE" && cargo build --release -p leansonic-core --example ternary_smoke && nvcc -Ikernels -arch=sm_120a -std=c++17 -O3 --cubin kernels/inkling2_sm120.cu -o engine.cubin)
export LEANSONIC_CUBIN_PATH="$ENGINE/engine.cubin" LEANSONIC_CTX=2048 LEANSONIC_KV=fp16 LEANSONIC_PDL=1 LEANSONIC_PDL_ALL=1 LEANSONIC_SEQS=1 LEANSONIC_PREFILL_CHUNK=512 LEANSONIC_MXFP8_DEVICE_U4=1 LEANSONIC_MXFP8_DIRECT_ALL=1
for arm in 0 1; do LEANSONIC_TERNARY_DOWN=$arm "$ENGINE/target/release/examples/ternary_smoke" "$MODEL" "$FIXTURE" "$arm"; done
cmp 0.f32 1.f32 && cmp 0.ids 1.ids && sha256sum -c output.sha
```

[Executable, cubin and input digests](inputs.sha) · [Output digests](output.sha) · [Both generated continuations](0.ids)
