# End-to-end DSpark on a fixed seven-level Inkling target

Fixed target including tokenizer: 11,609,363,902 bytes. Target plus official BF16 drafter: 12,245,165,318 bytes. Target plus raw Q8 drafter: 11,996,333,974 bytes. Target plus entropy-coded Q8 drafter: 11,971,005,038 bytes. Only the sidecar changes. The loader and cubin are identical across all arms. One sequence, FP16 KV, spec7, 1,024-token context. All three sidecars must activate DSpark. The hash and generated token sequence, actual proposed/accepted counts and number of verification rounds must agree in every run per GPU and fixture.

Run each of the six permutations of official/raw/compact once, a fresh process per leg. Each process runs all three fixtures twice, resetting before each request. Phase measurements use the second pass. Sample per-process NVML about every 4 ms across load and generation. A separate untimed replay records the proposed IDs and accepted prefix length in every speculative round.

| GPU | Prompt | Input tokens | Drafts accepted/proposed | Prefill ms (official / raw / compact) | Decode tok/s (official / raw / compact) |
|---|---|---:|---:|---:|---:|
| 5070Ti | q00_code_merge | 29 | 97/142 | 34.162 / 34.128 / 34.171 | 257.53 / 257.45 / 257.48 |
| 5070Ti | q11_json_schema | 32 | 99/159 | 38.083 / 38.071 / 38.074 | 246.85 / 246.82 / 246.77 |
| 5070Ti | ja_reasoning | 49 | 59/113 | 43.785 / 43.778 / 43.778 | 184.57 / 184.54 / 184.54 |
| PRO4500 | q00_code_merge | 29 | 97/142 | 34.357 / 34.368 / 34.346 | 270.67 / 270.73 / 270.69 |
| PRO4500 | q11_json_schema | 32 | 99/159 | 39.166 / 39.194 / 39.161 | 260.68 / 260.58 / 260.71 |
| PRO4500 | ja_reasoning | 49 | 59/113 | 45.043 / 45.052 / 45.044 | 188.88 / 188.82 / 188.97 |

5070Ti: maximum observed process VRAM (official / raw / compact) was 15,546,187,776 / 15,546,187,776 / 15,546,187,776 bytes. All 18 processes and both passes produced identical outputs and acceptance totals per fixture.

5070Ti: per-round proposal IDs and accepted-prefix counts also matched at every round in all three arms, including the second pass after reset.

[Raw requests, token IDs, hashes, confidence intervals and loader logs for 5070Ti](5070Ti-verified.json).


PRO4500: maximum observed process VRAM (official / raw / compact) was 15,684,599,808 / 15,684,599,808 / 15,684,599,808 bytes. All 18 processes and both passes produced identical outputs and acceptance totals per fixture.

PRO4500: per-round proposal IDs and accepted-prefix counts also matched at every round in all three arms, including the second pass after reset.

[Raw requests, token IDs, hashes, confidence intervals and loader logs for PRO4500](PRO4500-verified.json).

## Reproduce

Use the [merged benchmark source](https://github.com/tjm8874/LeanSonic/commit/9ed9fa13f706636cf8f2e7cad2e07b77da33bfb8). Each arm is a directory containing the same fixed seven-level target and tokenizer and its own mtp.safetensors. Use the input .ids files adjacent to this recipe.

```sh
cp "$RECIPE/bench.rs" crates/leansonic-core/examples/dspark_three.rs
cargo build --release -p leansonic-core --example dspark_three
nvcc -Ikernels -arch=sm_120a -std=c++17 -O3 --cubin kernels/inkling2_sm120.cu -o engine.cubin
export LEANSONIC_KV=fp16 LEANSONIC_PDL=1 LEANSONIC_PDL_ALL=1 LEANSONIC_SEQS=1
export LEANSONIC_MXFP8_DEVICE_U4=1 LEANSONIC_MXFP8_DIRECT_ALL=1 LEANSONIC_CUBIN_PATH="$PWD/engine.cubin"
binary="$PWD/target/release/examples/dspark_three"
cd "$RECIPE"
"$binary" "$MODEL_DIR"
```

To record individual proposal IDs without affecting the timed binary, apply [the trace-only edit](trace.py) from a directory containing a second checkout named src, then rebuild the same example there.
