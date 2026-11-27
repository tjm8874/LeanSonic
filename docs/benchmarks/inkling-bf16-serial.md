# BF16 reference with a 256-token prefix and serial scoring

LeanSonic V13 on RTX PRO 5000 prefills the first 257 tokens of each 512-token WikiText chunk in two calls, of 256 and 1 tokens. It then teacher-forces positions 257 through 510 one by one. To measure the impact of this schedule, we ran the official BF16 GGUF using the same token grouping in llama.cpp. Each oracle scored the same next-token targets. The original 16-sequence oracle remains the canonical comparison.

| WikiText-2 test, 25,500 positions | Mean KL | Worst KL (chunk/position) | Top-1 matches | Raw output PPL |
|---|---:|---:|---:|---:|
| Serial BF16 vs canonical BF16 | 0.001862 | 0.514771 (80/452) | 25,043 | 7.073208 |
| V13 vs canonical BF16 | 0.003158 | 0.680977 (69/353) | 24,865 | 7.072956 |
| V13 vs serial BF16 | 0.003172 | 0.827418 (80/452) | 24,863 | 7.072956 |

Using the same grouping did not reduce V13's aggregate test divergence. The maximum moved to another input. The three rows use the same official model revision, WikiText IDs, FP16 KV and existing KLD scorer. Quantization of the stored reference gives target PPL 7.064408 for canonical BF16 and 7.065327 for serial BF16. The raw and encoded values differ because the KLD writer clips and quantizes log probabilities.

A trace build of the V13 engine reported `offset=0 len=256` then `offset=256 len=1`. It produced the same complete one-chunk metrics as the unmodified engine. The public generator independently checks the observed positions and batch sizes of the first BF16 chunk after successful decode. Its canonical control reproduced all 4,080 rows of the first 16 chunks byte-for-byte, including all vocabulary entries and IDs. Its separately generated serial control reproduced the first 255 rows of the 100-chunk serial reference byte-for-byte. See [builds, hashes and full numeric results](inkling-bf16-serial.json).

## Held-out validation and the previous kernel revision

All 100 previously held-out WikiText-2 validation chunks used the same 25,500 next-token targets and RTX PRO 5000. V12 and V13 produced identical score reports under the serial oracle on both test and validation, including the worst positions and PPL. The following rows distinguish the oracle schedule from changes to the engine.

| Validation, 25,500 positions | Mean KL | Worst KL (chunk/position) | Top-1 matches | Raw output PPL |
|---|---:|---:|---:|---:|
| Serial BF16 vs canonical BF16 | 0.001613 | 0.356536 (76/501) | 25,088 | 6.503794 |
| V13 vs canonical BF16 | 0.002761 | 0.352436 (46/379) | 24,963 | 6.504621 |
| V13 vs serial BF16 | 0.002683 | 0.515020 (30/331) | 24,985 | 6.504621 |

On validation, using the serial reference slightly lowers the mean and increases the worst KL. The largest deviation moves from 46/379 to 30/331. Keep the original canonical reference for regression decisions and use the serial reference to investigate sensitivity to execution order.

## The same engine row against both oracles

We also dumped the full V13 logit vector once per selected position through the normal greedy CLI. The raw output SHA-256 per position is in the JSON. At test 69/353 V13 predicts the observed WikiText target, and matching the schedule changes the BF16 top token to that same target. The KL remains above 0.5. At 80/452 V13 and canonical BF16 predict the observed target, while serial BF16 predicts a different token.

| Split and position | WikiText target | V13 top | Canonical BF16 top | Serial BF16 top | KL vs canonical | KL vs serial |
|---|---:|---:|---:|---:|---:|---:|
| test 29/443 | 149918 | 149918 | 149918 | 149918 | 0.175476 | 0.061269 |
| test 69/353 | 1057 | 1057 | 2952 | 1057 | 0.680977 | 0.517345 |
| test 80/452 | 197045 | 197045 | 197045 | 196977 | 0.031250 | 0.827418 |
| valid 46/379 | 220 | 17 | 170558 | 170558 | 0.352436 | 0.483306 |

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

To reproduce individual positions with NumPy, score both references against the same engine dump:

```bash
python3 "$LEANSONIC_SRC/tools/compare_inkling_oracle_rows.py" --cli "$CLI" --weights "$COMPACT" \
    --canonical test.canonical.kld --serial test.serial.kld --row 69/353 --row 80/452 --row 29/443
```
