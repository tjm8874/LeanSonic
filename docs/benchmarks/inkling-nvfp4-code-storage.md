# Lossless storage of NVFP4 expert weight codes

The compact Inkling-2-Mini text package contains 10,192,158,720 bytes of packed NVFP4 expert weights. The optional `--prepack-nvfp4` flag applies the same four-lane byte rANS container used for MXFP8. The packer keeps each matrix in its current form if encoding would increase its size. The loader accepts a mixture of raw and encoded tensors, restores the original bytes and places them in VRAM or pinned host memory according to the existing budget.

| Contents | Compact baseline | With NVFP4 code compression |
|---|---:|---:|
| Safetensors shard | 15,978,753,408 B | 15,898,104,040 B |
| Tokenizer files | 3,386,494 B | 3,386,494 B |
| Text distribution | 15,982,139,902 B | **15,901,490,534 B** |

The new package saves 80,649,368 bytes (0.50%). We independently encoded every NVFP4 matrix, reopened the completed shard and decoded all 40 matrices against the original byte ranges. All 999 tensor payloads of the production-packed shard matched the independently assembled file. Two hosts produced the same shard SHA-256, `c0b42b8bf338a0690000e57668f03e9f2d27450c8187c89555165a4bf22588f4`.

Compact storage increases startup time. On RTX 5060 Ti, the first complete cycle with both files fully resident in the OS page cache took about 5.5 s to load the baseline and 9.0 s to load the compressed package. All 64 greedy outputs matched. Each arm started a fresh CLI twice per cycle and ran three decode repeats per process. The measurement file includes the page residency of each run. Only complete cycles with 100% residency before every launch enter the warm summary. [Raw timings and per-matrix byte counts](inkling-nvfp4-code-storage.json)

## Reproduce

Use the compact text model from the current packer as `BASE`. Set `NEW` to a new directory. The distribution comparison counts the shard and tokenizer and excludes build artifacts. The native loader can also read the original official weights and earlier text packages.

```bash
mkdir "$NEW"
python3 tools/pack_inkling_text.py "$BASE/model-00001-of-00001.safetensors" \
    "$NEW/model-00001-of-00001.safetensors" --prepack-nvfp4
cp -RL "$BASE/tiktoken" "$NEW/tiktoken"
sha256sum "$NEW/model-00001-of-00001.safetensors"
find "$NEW" -type f -printf '%s\n' | awk '{total += $1} END {printf "%.0f\n", total}'
cargo build --release -p leansonic-cli
nvcc -std=c++17 -O3 -arch=sm_120a --cubin kernels/inkling2_sm120.cu -o v13.cubin
export LEANSONIC_CUBIN_PATH="$PWD/v13.cubin" LEANSONIC_KV=fp16
cut -d, -f1-32 docs/benchmarks/inputs/inkling-readme1024.ids > short.ids
for model in "$BASE" "$NEW"; do
    target/release/leansonic bench --weights "$model" --tokens "@$PWD/short.ids" \
        --gen 64 --spec 0 --repeat 3
done
```
