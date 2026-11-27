# DFlash-only compact target replay on RTX PRO 4500

The compact Inkling target with 68 restored experts occupies 11,663,036,518 bytes including tokenizer. The DFlash2 checkpoint adds 989,975,616 bytes. The optional DSpark checkpoint adds 635,801,416 bytes. The complete DFlash-only model is 12,653,012,134 bytes. With both drafters it is 13,288,813,550 bytes.

We compared a3548ca (#212) with 5fe287c (#216), loading the same target with DSpark present and absent. Each process generated 64 tokens after a 64-token or 4,096-token prompt, reset, and repeated once. At the end a separate reset/prefill and one scalar greedy step provided four target tap vectors (36,864 bytes total). The observer synchronizes before reading them. Context was 8,192, one sequence, FP16 KV, spec 7, the prefill workspace explicitly set to 4,096 and context copy disabled.

At a3548ca the target without DSpark fails on both prompts with `missing drafter tap storage`. With DSpark present both pass. At 5fe287c both configurations exactly match the non-speculative target on all 64 emitted IDs plus the following token and both resets. The four FP32 taps match byte for byte with and without DSpark at both prompt lengths, including the previously unguarded final tap. Without DSpark, the sampled process peak is 17,951,621,120 bytes, compared with 18,494,783,488 with both. Both are fully GPU resident. The non-speculative runner uses a host-only tap-preservation diagnostic change on the #212 source; no target or drafter weights differ between controls.

On the second run of each process, #216 without DSpark took 248.807 ms / 209.634 ms for the 64-token decode (64 / 4,096 input). With both drafters it took 249.611 / 210.160 ms. These single-process smoke timings do not establish a throughput ranking. At the 4,096-token prefix both runs accepted 53 of 58 drafted tokens.

## Reproduce

Use the [68-expert model and exact package digests](https://github.com/tjm8874/LeanSonic/tree/aaceb42/docs/benchmarks/inkling-codebooks/nvfp4-budget/recovered/variable-palette/global-experts). Make a second directory with the same weights and tokenizer plus the official mtp.safetensors; the first directory must contain just the target weights and tokenizer. Verify both drafter hashes in results.json. Build the target and DFlash modules from each checkout using CUDA 13.0.88. For the diagnostic executable, copy probe.rs into crates/leansonic-core/examples/dflash_only_probe.rs, run instrument.py on crates/leansonic-core/src/inkling2_gpu.rs and build with cargo build --release -p leansonic-core --example dflash_only_probe.

Set LEANSONIC_DFLASH2 to the DFlash2 checkpoint, LEANSONIC_DRAFTER=dflash2, LEANSONIC_DF2_CUBIN_PATH and LEANSONIC_CUBIN_PATH to the matching sm_120a modules. Set LEANSONIC_PREFILL_CHUNK=4096, LEANSONIC_KV=fp16, LEANSONIC_PDL=1, LEANSONIC_PDL_ALL=1, LEANSONIC_SEQS=1, LEANSONIC_MXFP8_DEVICE_U4=1, LEANSONIC_MXFP8_DIRECT_ALL=1 and LEANSONIC_SPEC_COPY=0. Run the following in a writable working directory. Repeat with size 4096, the second target directory and spec 0 controls. Compare complete .ids and .taps files for the corresponding arms. The full results contain per-tap digests, output digests, warm and cold timings and sampling counts.

```sh
nvcc -Ikernels -arch=sm_120a -O3 -std=c++17 --cubin kernels/inkling2_sm120.cu -o engine.cubin
nvcc -Ikernels -arch=sm_120a -O3 -std=c++17 --cubin kernels/inkling2_dflash2_sm120.cu -o df2.cubin
./target/release/examples/dflash_only_probe "$MODEL" prompt.ids 64 7 author-only-64
```
