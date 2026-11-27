# Free greedy generation on RTX 5060 Ti with a four-bank target

Compare the same Inkling-2-Mini input token IDs on the same GPU at context 2,048, FP16 KV and one request at a time. Both implementations emit 128 greedy tokens without stopping at EOS and advance through the 128th token to obtain its successor. Timings separate prompt evaluation (including the first greedy prediction) and decode. Each of the six permutations of the three engines/configurations runs exactly once. Each fresh process executes two passes over all four inputs, with reset before each request. Timings are means of the second passes. Peak process VRAM includes loading and is sampled via NVML every 4 ms.

| Configuration | Distribution bytes incl. tokenizer | Peak observed VRAM bytes |
|---|---:|---:|
| LeanSonic, spec0 | 13,455,139,382 | 15,816,720,384 |
| LeanSonic, spec7 row-exponent DSpark | 13,781,658,950 | 16,141,778,944 |
| Unsloth llama.cpp original Q3_K_XL | 13,631,778,016 | 14,938,013,696 |

Values below follow the order **LeanSonic spec0 / spec7 / llama.cpp**. Same source architecture, different target quantization in the public GGUF.

| Input | Input length | Prefill ms | Decode tok/s | DSpark accepted/proposed | Plain vs. spec emitted IDs | Native vs. plain shared prefix |
|---|---:|---:|---:|---:|---:|---:|
| q00_code_merge | 29 | 63.967 / 64.324 / 76.243 | 98.96 / 166.13 / 57.54 | 103/136 | 128/128 | 122/128 |
| q11_json_schema | 32 | 76.556 / 76.989 / 78.137 | 98.93 / 169.66 / 57.63 | 104/135 | 128/128 | 126/128 |
| ja_reasoning | 49 | 88.962 / 89.465 / 95.701 | 98.57 / 94.48 / 58.31 | 39/88 | 128/128 | 10/128 |
| readme1024 | 1024 | 259.264 / 266.661 / 887.276 | 82.35 / 102.81 / 56.05 | 71/118 | 128/128 | 17/128 |

Decode speed ratios from the six paired cycles, with whole-cycle bootstrap 95% intervals. Ratios below one favor the comparator.

| Input | spec7 / spec0 | spec7 / llama.cpp |
|---|---:|---:|
| q00_code_merge | 1.679× [1.678, 1.680] | 2.887× [2.886, 2.888] |
| q11_json_schema | 1.715× [1.714, 1.716] | 2.944× [2.941, 2.948] |
| ja_reasoning | 0.959× [0.958, 0.959] | 1.620× [1.619, 1.623] |
| readme1024 | 1.248× [1.248, 1.249] | 1.834× [1.833, 1.836] |

The GGUF embeds its tokenizer. Adding the 326,519,568-byte drafter makes the spec7 distribution 149,880,934 bytes larger than that GGUF. The spec0 distribution saves 176,638,634 bytes. Since each engine follows its own greedy history, different emitted IDs can change later decode work. The [fixed-continuation comparison](../fixed-q3.md) holds that history constant. The timed readback hash is from the scalar logits buffer. A dedicated diagnostic runs against the selected verification row when the last round is speculative.

## Reproduction

Use the [integrated benchmark checkout](https://github.com/tjm8874/LeanSonic/commit/1675f4bcb466907848bf3f625831955a3ca2dc39), [target packer](../), and [sidecar recipe](https://github.com/tjm8874/LeanSonic/tree/16bf706/docs/benchmarks/inkling-dspark-package/markov/row-exponents). The benchmark predates the suffix-copy change. Build the native driver against Unsloth llama.cpp at 883286076efe5b85fddb31d9cc7035a6b698ba9b and the original Q3_K_XL at model revision ae9327c5fbc7333116ba32eee78bb5d2bbefae4f. Native microbatch is 512 and all 25 layers offload. Both LeanSonic modes have all 24 expert layers resident.

```sh
cp "$RECIPE/lean.rs" crates/leansonic-core/examples/free_decode.rs
cargo build --release -p leansonic-core --example free_decode
cp target/release/examples/free_decode "$RECIPE/lean"
nvcc -Ikernels -arch=sm_120a -std=c++17 -O3 --cubin kernels/inkling2_sm120.cu -o "$RECIPE/engine.cubin"
cd "$RECIPE"
c++ -std=c++17 -O2 native.cpp -I"$LLAMA_HEADERS/include" -I"$LLAMA_HEADERS/ggml/include" -Llib -l:libllama.so.1 -l:libggml.so.1 -l:libggml-base.so.1 -Wl,-rpath-link,lib -o native
export LEANSONIC_KV=fp16 LEANSONIC_PDL=1 LEANSONIC_PDL_ALL=1 LEANSONIC_SEQS=1
export LEANSONIC_MXFP8_DEVICE_U4=1 LEANSONIC_MXFP8_DIRECT_ALL=1 LEANSONIC_CUBIN_PATH="$PWD/engine.cubin"
./lean "$TARGET_DIR" 0
./lean "$TARGET_WITH_MTP_DIR" 7
LD_LIBRARY_PATH="$PWD/lib:${LD_LIBRARY_PATH:-}" ./native "$Q3_GGUF"
```

[Individual timings, IDs, sampler peaks and build hashes](verified.json).

[Fixed depth 0, 2, 4 and 7 with 32-token context matching](depth/).
