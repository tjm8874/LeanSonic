# Inkling BF16 sensitivity to batch shape

The official BF16 model selected a different top token at 463 of 25,500 positions on the WikiText test split when its physical microbatch changed from 8,192 to 64. The logical batch stayed at 8,192 and the token IDs, weights, GPU, llama.cpp build and scorer were fixed. The worst KL on this same-model comparison reached 0.281640 on the test split and 0.357129 on validation.

| Split / BF16 batch and microbatch | Positions | Mean KL | Worst KL | Top-1 matches | Raw-logit PPL |
|---|---:|---:|---:|---:|---:|
| Test, 8192/8192, first 16 chunks | 4,080 | 0.000000141 | 0.000056 | 4,080 | 8.185805 |
| Test, 8192/64 | 25,500 | 0.001919431 | 0.281640 | 25,037 | 7.073006 |
| Validation, 8192/64 | 25,500 | 0.001692298 | 0.357129 | 25,100 | 6.501775 |

The 16-chunk control used the same schedule as the original BF16 reference. Its stored probabilities are clipped and quantized, so raw logits from the canonical run can still yield a small nonzero KL. The full test and validation comparisons each used their own 100-chunk oracle. These results establish schedule sensitivity for the BF16 implementation. Attribution of a particular LeanSonic discrepancy to schedule or weights requires a matched replay at the corresponding row. [Raw results and worst positions](inkling-bf16-schedules.json).

## Reproduction on Linux

Start with the [pinned llama.cpp build and official BF16 GGUF](../inkling-reference.md). Apply the observation patch in a disposable checkout of that commit, then rebuild the existing target. It writes the FP32 rows that upstream sends to its PPL calculation and the 512 token IDs for each chunk.

```bash
git -C "$LLAMA_SRC" apply "$LEANSONIC_SRC/tools/inkling_bf16_raw_rows.patch"
cmake --build "$LLAMA_BUILD" --target llama-perplexity -j
cargo build --release --manifest-path "$LEANSONIC_SRC/Cargo.toml" -p leansonic-cli
```

For each of `test` and `valid`, first generate the split-specific reference with the canonical schedule. Set `CUDA_VISIBLE_DEVICES` to the card with room for the 52.23 GB GGUF. Each reference needs 10.20 GB of disk space. The replay streams FP32 rows directly to the scorer and verifies all input IDs against the generated reference before reporting a result.

```bash
for split in test valid; do
    "$LLAMA_BUILD/bin/llama-perplexity" -m "$GGUF" -f "$WIKITEXT_DIR/wiki.$split.raw" \
        -c 512 -b 8192 -ub 8192 -ngl 99 -t 8 --no-warmup --chunks 100 \
        --save-all-logits "$split.kld"
    for shape in 8192/8192 8192/64 512/512; do
        python3 "$LEANSONIC_SRC/tools/score_inkling_bf16_schedule.py" \
            --engine "$LEANSONIC_SRC/target/release/leansonic" \
            --perplexity "$LLAMA_BUILD/bin/llama-perplexity" --model "$GGUF" \
            --text "$WIKITEXT_DIR/wiki.$split.raw" --ref "$split.kld" \
            --batch "${shape%/*}" --ubatch "${shape#*/}" --chunks 100
    done
done
```
