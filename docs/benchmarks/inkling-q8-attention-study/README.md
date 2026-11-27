# Attention Q8 block-scale ablation

This experiment requantizes only the 176 attention projections of Inkling-2-Mini. The attention matrices comprise 10,093,824 blocks of 64 weights. For each block, try the stock BF16 maximum and its two immediate BF16 neighbours in each direction, use the existing signed Q8 rounding rule, and retain the scale with the smallest squared weight reconstruction error. The unchanged stock option wins ties. The script verifies each original Q8 code and scale against official BF16 input before updating a separate output file.

The two Blackwell hosts independently produced the same candidate. Its scale dictionary still works because candidate scales use the same BF16-maximum grid. On the SIMT evaluation the worst KL improved from 0.663992 to 0.314452, but mean KL rose from 0.003039 to 0.003127. See `results.json` for the official shard and BF16 oracle digests, both package sizes, and the full metrics. This is an offline diagnostic.

## Reproduce on Linux

Use a new output directory on a filesystem with enough space for a complete checkpoint. `OFFICIAL` is the original checkpoint directory. `NATIVE` is the prepacked text directory with raw Q8 scale planes (head, attention, embedding, MXFP8 and entropy-scale flags). The tokenizer is shared with that native deployment. Compile with GCC and OpenMP.

```bash
g++ -O3 -fopenmp -ffp-contract=off -std=c++20 q8grid.cpp -o q8grid
OMP_NUM_THREADS=4 python3 prepare.py "$OFFICIAL" "$NATIVE" "$CANDIDATE"
sha256sum "$CANDIDATE/model-00001-of-00001.safetensors"
```

To reproduce the SIMT scores, build the CLI and its native cubin from `cc0b3e330cee64e9b8ca3e4321f64686a4cdad00`. From that checkout run the following commands for both weight directories.

```bash
cargo build --release -p leansonic-cli
nvcc -O3 -arch=sm_120a --cubin kernels/inkling2_sm120.cu -o inkling.cubin
export LEANSONIC_KV=fp16 LEANSONIC_CTX=8192 LEANSONIC_CUBIN_PATH="$PWD/inkling.cubin"
target/release/leansonic eval-kld --weights "$NATIVE" --ref "$BF16_KLD" --chunks 100
target/release/leansonic eval-kld --weights "$CANDIDATE" --ref "$BF16_KLD" --chunks 100
```

The default V12 build at `7a02b538` was also tested on PRO 5000. Use the same build commands at that revision, with FP16 KV and default Tensor Core prefill. The immutable stock dictionary gave mean KL 0.003175526 and max KL 0.674716. The grid candidate gave 0.003140554 and 0.564691, respectively. Top-1 matches increased from 24,879 to 24,910. PPL increased from 7.072582 to 7.072916. Both V12 runs missed the project quality targets.

## Output-projection ablation

To separate the 24 attention output projections (`wo_ud.weight`) from the other 152 matrices, run the following after preparing the grid candidate. The source and candidate must have the same raw native Q8 safetensors layout. The script copies the code and scale spans of the selected matrices into two fresh checkpoint copies. Each output package has the stock layout and byte count.

```bash
mkdir "$SPLIT"
python3 split.py "$NATIVE" "$CANDIDATE" "$SPLIT"
```

With the V12 CLI, evaluate `$NATIVE`, `$SPLIT/wo_only`, `$SPLIT/except_wo` and `$CANDIDATE` with `--start-chunk 69 --chunks 1` and `--start-chunk 21 --chunks 1`. The 255 scored positions per chunk give the following maxima. The maximum at chunk 69 moves from position 353 to 395 in both partial changes. The full grid restores 353 as the maximum. Each change alone reduces the chunk-69 maximum more than applying both groups together.

| V12 weight set | Max KL in chunk 21 | Max KL in chunk 69 |
| --- | ---: | ---: |
| Stock | 0.349781 | 0.674716 |
| Only 24 output projections | 0.297260 | 0.052427 |
| Other 152 attention matrices | 0.136736 | 0.062796 |
| All 176 | 0.120252 | 0.564691 |

See `split-chunks.json` for the per-chunk scores and positions. These two chunks were selected from earlier runs; the full 100-chunk result determines whether the improvement generalizes.

## Independent validation of the 24 output projections

On the 100 test chunks, refining only the output projections reduced V12 mean KL from 0.003175526 to 0.003090100, worst KL from 0.674716 to 0.380358, and PPL from 7.072582 to 7.070330. Top-1 agreement increased from 24,879 to 24,895 of 25,500. At 69/353 this reduced KL to 0.002929 and recovered the BF16 next token. The worst position moved to 90/302.

We fixed those 24 matrices before generating a separate 100-chunk oracle from `wiki.valid.raw`. Mean KL then **increased from 0.002749411 to 0.002782229**, worst KL **increased from 0.510178 to 0.625806** (91/507 to 64/494), and top-1 matches **fell from 24,948 to 24,942**. PPL improved from 6.505563 to 6.504025 (BF16 6.498589). Keep stock packing. Both configurations scored all 25,500 positions on each split. The new dictionary shard is 16,035,182,576 bytes, 80 bytes larger than stock.

Generate the separate oracle with the unmodified perplexity program at the pinned llama.cpp revision. We first reproduced all 4,080 scored rows and token IDs of the first 16 test chunks byte for byte with this executable. The validation source SHA-256 is `4cd0f6876d07a413aa911261ff6d363c72d757d47f0fdd6015702014c89cb9c7`; the resulting oracle is `b3d804824373950c9b6d7cd37e5607d132d2f3caca6f80ffd26739b2776a2ca0`.

```bash
llama-perplexity -m official-bf16.gguf -f "$WIKITEXT_DIR/wiki.valid.raw" \
  -c 512 -b 8192 -ub 8192 -ngl 99 -t 8 --no-warmup --chunks 100 --save-all-logits valid.kld
target/release/leansonic eval-kld --weights "$NATIVE" --ref valid.kld --chunks 100
target/release/leansonic eval-kld --weights "$SPLIT/wo_only" --ref valid.kld --chunks 100
```

The [one-step research converter](https://github.com/tjm8874/LeanSonic/tree/00f69a6) accepts `--refine-attention-output` with `--prepack-attention` on the original BF16 source. Its raw shard matched the independent matrix-copy experiment byte for byte. Direct dictionary conversion and repacking the raw shard also produced identical bytes. This converter remains on a research branch.
