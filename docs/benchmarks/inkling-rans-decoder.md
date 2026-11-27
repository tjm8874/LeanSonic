# Inkling rANS model load time on Blackwell hosts

The model package and the GPU code were identical across each pair. The CPU loader in commit `9183a1c` changed. With the compact text model from PR #85, loading on the Ryzen 9 9900X with RTX PRO 5000 took 2.849 s, down from 3.331 s. On the Core i5-14400 with RTX 5060 Ti it took 5.470 s, down from 5.701 s. These are fresh-process means with every model file page resident in the OS cache before each launch.

| Host and model package | Baseline load (s) | Four-lane load (s) | Reduction | Paired 95% interval |
|---|---:|---:|---:|---:|
| Ryzen 9 9900X / PRO 5000, #77 | 2.980 | 2.675 | 10.23% | 8.86–11.81% |
| Ryzen 9 9900X / PRO 5000, #85 | 3.331 | 2.849 | 14.47% | 12.66–16.28% |
| Core i5-14400 / 5060 Ti, #77 | 5.137 | 5.060 | 1.50% | 0.63–2.18% |
| Core i5-14400 / 5060 Ti, #85 | 5.701 | 5.470 | 4.05% | 3.29–4.99% |

For each of the 233 compressed tensors we compared the decoded bytes with the source. This covered 1,106,938,368 native Q8 bytes, 1,401,421,824 expert scale bytes and 4,076,863,488 MXFP8 expert weight bytes. On PRO 4500 and RTX 5070 Ti the last input position's complete 200,058 FP32 logits matched the previous loader byte-for-byte. A separate greedy generation matched its 16 output IDs.

The new loader on PRO 5000 produced a byte-identical 100-chunk BF16 score report. Across 25,500 positions, mean KL was 0.003175526, maximum KL 0.674716, top-1 agreement 24,879/25,500 and PPL 7.072582 (BF16 7.064408).

## Reproduction

Use the original Inkling-2-Mini text checkpoint and its tokenizer. The [packing commands](https://github.com/tjm8874/LeanSonic/blob/83d6e6b/docs/benchmarks/inkling-rans-q8-codes.json) and [raw timings, CPU measurements and hashes](inkling-rans-decoder.json) identify the two exact test packages. Build on the same tree to include the new packaging option in both arms.

```bash
git fetch origin pull/85/head pull/87/head
git worktree add --detach ../leansonic-load-ab 4b238be
cd ../leansonic-load-ab
cargo build --release -p leansonic-cli
cp target/release/leansonic ./cli-old
git show 9183a1c -- crates/leansonic-core/src/inkling_mxfp8.rs | git apply
cargo build --release -p leansonic-cli
cp target/release/leansonic ./cli-fast
nvcc -std=c++17 -O3 -arch=sm_120a --cubin kernels/inkling2_sm120.cu -o inkling.cubin
export LEANSONIC_CUBIN_PATH="$PWD/inkling.cubin" LEANSONIC_KV=fp16 LEANSONIC_CTX=8192
```

Set `NATIVE_DIR` to the #77 package and `COMPACT_DIR` to the #85 package, each with its own tokenizer. Set `CUDA_VISIBLE_DEVICES` to the test card. Each command below starts a new process and runs one request. The reported load time is the CLI's model loader timer. We verified cache residency with Linux `mincore` immediately before each process and compared the 16 IDs against PR #84 on each run.

```bash
for cycle in 1 2 3 4 5; do
    if (( cycle % 2 )); then
        arms=(old fast fast old)
        packages=("$COMPACT_DIR" "$NATIVE_DIR")
    else
        arms=(fast old old fast)
        packages=("$NATIVE_DIR" "$COMPACT_DIR")
    fi
    for weights in "${packages[@]}"; do
        cat "$weights/model-00001-of-00001.safetensors" > /dev/null
        for arm in "${arms[@]}"; do
            "./cli-$arm" bench --weights "$weights" \
                --tokens @docs/benchmarks/inputs/inkling-readme1024.ids \
                --gen 16 --spec 0 --repeat 1
        done
    done
done
```
