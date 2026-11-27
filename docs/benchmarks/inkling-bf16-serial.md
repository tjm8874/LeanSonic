# BF16 reference with a 256-token prefix and serial scoring

LeanSonic V13 on RTX PRO 5000 prefills the first 257 tokens of each 512-token WikiText chunk in two calls, of 256 and 1 tokens. It then teacher-forces positions 257 through 510 one by one. To measure the impact of this schedule, we ran the official BF16 GGUF using the same token grouping in llama.cpp. Each oracle scored the same next-token targets. The original 16-sequence oracle remains the canonical comparison.

| WikiText-2 test, 25,500 positions | Mean KL | Worst KL (chunk/position) | Top-1 matches | Raw output PPL |
|---|---:|---:|---:|---:|
| Serial BF16 vs canonical BF16 | 0.001862 | 0.514771 (80/452) | 25,043 | 7.073208 |
| V13 vs canonical BF16 | 0.003158 | 0.680977 (69/353) | 24,865 | 7.072956 |
| V13 vs serial BF16 | 0.003172 | 0.827418 (80/452) | 24,863 | 7.072956 |

Using the same grouping did not reduce V13's aggregate test divergence. The maximum moved to another input. The three rows use the same official model revision, WikiText IDs, FP16 KV and existing KLD scorer. Quantization of the stored reference gives target PPL 7.064408 for canonical BF16 and 7.065327 for serial BF16. The raw and encoded values differ because the KLD writer clips and quantizes log probabilities.

A trace build of the V13 engine reported `offset=0 len=256` then `offset=256 len=1`. It produced the same complete one-chunk metrics as the unmodified engine. The public generator independently checks the observed positions and batch sizes of the first BF16 chunk after successful decode. Its canonical control reproduced all 4,080 rows of the first 16 chunks byte-for-byte, including all vocabulary entries and IDs. Its separately generated serial control reproduced the first 255 rows of the 100-chunk serial reference byte-for-byte. See [builds, hashes and full numeric results](inkling-bf16-serial.json).

## Reproduce on Linux

Use the [pinned model, corpus and llama.cpp revision](../inkling-reference.md). Set `GGUF`, `WIKITEXT_DIR`, `LLAMA_SRC`, `LLAMA_BUILD`, `LEANSONIC_SRC`, and `COMPACT` to their local paths. Rebuild the llama-perplexity executable and its implementation library in the same build directory. Select a GPU with enough VRAM for the official BF16 model. Each 100-chunk encoded reference occupies 10,203,366,820 bytes.

```bash
git -C "$LLAMA_SRC" apply "$LEANSONIC_SRC/tools/inkling_bf16_serial_score.patch"
git -C "$LLAMA_SRC" apply "$LEANSONIC_SRC/tools/inkling_bf16_raw_rows.patch"
cmake --build "$LLAMA_BUILD" --target llama-perplexity -j
cargo build --release --manifest-path "$LEANSONIC_SRC/Cargo.toml" -p leansonic-cli
CLI="$LEANSONIC_SRC/target/release/leansonic"
nvcc -O3 -std=c++17 -arch=sm_120a --cubin "$LEANSONIC_SRC/kernels/inkling2_sm120.cu" -o v13.cubin
export LEANSONIC_CUBIN_PATH="$PWD/v13.cubin"

for split in test valid; do
    "$LLAMA_BUILD/bin/llama-perplexity" -m "$GGUF" -f "$WIKITEXT_DIR/wiki.$split.raw" \
        -c 512 -b 8192 -ub 8192 -ctk f16 -ctv f16 -ngl 99 -t 8 --no-warmup \
        --chunks 100 --save-all-logits "$split.canonical.kld"
    python3 "$LEANSONIC_SRC/tools/score_inkling_bf16_schedule.py" \
        --perplexity "$LLAMA_BUILD/bin/llama-perplexity" --engine "$CLI" --model "$GGUF" \
        --text "$WIKITEXT_DIR/wiki.$split.raw" --ref "$split.canonical.kld" \
        --serial --batch 512 --ubatch 512 --chunks 100 --save-kld "$split.serial.kld"
    for reference in "$split.canonical.kld" "$split.serial.kld"; do
        LEANSONIC_KV=fp16 "$CLI" eval-kld --weights "$COMPACT" --ref "$reference" --chunks 100
    done
done
```
