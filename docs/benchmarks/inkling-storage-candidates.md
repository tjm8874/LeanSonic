# Storage and accuracy experiments after canonical NVFP4 zero packing

The two experiments use the 15,548,157,542-byte canonical package (#125) as the parent. All sizes below count the tokenizer.

## Retaining every normalized code

We encoded and reopened all 40 NVFP4 expert planes with two offline rANS variants. The context combines the position, the greatest magnitude already decoded in the current 16-value block, and the running average magnitude (thresholds 2, 3 and 4). Each decoded plane was compared byte-for-byte against the normalized official codes. The reference encoder reproduced the existing 40 complete frame lengths.

| Frame encoding | Total encoded frame bytes | Bytes saved over canonical | Sum of decoder-check durations (s) |
|---|---:|---:|---:|
| Canonical byte-context | 9,887,406,872 | 0 | 30.75 |
| Byte symbol with running mean | 9,884,118,088 | 3,288,784 | 47.01 |
| Nibble symbol with running mean | 9,879,709,056 | 7,697,816 | 99.15 |

The last column sums per-tensor times, including reading and validating the frames, from a single three-worker CPU run. It is not end-to-end startup latency. The smallest candidate saves only 0.050% of the complete package and takes substantially longer to decode in this experiment. The production reader retains its existing format. [Per-plane measurements](inkling-code-prefix-probe.json).

Reproduce with `c++ -O3 -std=c++17 tools/inkling_storage_probe/prefix.cpp -o prefix`. For each raw NVFP4 expert tensor in layers 0–19, call `./prefix SHARD ABSOLUTE_OFFSET RAW_BYTES NEW_FRAME nibble` (or `byte`). The absolute offset is 8 plus the little-endian safetensors header length plus the tensor's starting data offset. The program normalizes E2M1 zero signs, encodes the entire selected plane, reopens the file, and checks every decoded code and the final states.

## Reducing the precision of the MXFP8 expert weights

For each selected gate/up and down tensor, we round each finite E4M3 weight code to the nearest code with an even magnitude index. Exact midpoints select the even retained index; the original sign is preserved, including signed zero. This removes one mantissa bit while retaining the full exponent range. Each output uses the existing byte-rANS runtime format. Scales, other expert weights, the tokenizer and execution kernels stay the same. We decoded the changed frames and compared them with the rounded source, then checked the SHA-256 of every published tensor payload against its intended source.

| Research package | Total bytes | Saved vs canonical | Changed tensors |
|---|---:|---:|---:|
| Parent | 15,548,157,542 | 0 | 0 |
| Layer 23 only | 15,396,463,478 | 151,694,064 | 2 |
| Layers 20–23 | 14,941,300,742 | 606,856,800 | 8 |

These two packages change the numerical weights. Both candidates increased mean and worst KL in the 100-chunk WikiText test. [Tensor receipts and shard hashes](inkling-mxfp8-rounding.json).

Reproduce from the official checkpoint and the parent from #125. Reserve space for both complete output folders plus temporary expert frames.

```sh
cargo build --release -p leansonic-core --example inkling_mxfp8_round
python3 tools/inkling_storage_probe/pack_mxfp8.py "$OFFICIAL" "$CANONICAL" "$OUTPUT"
```

## Keeping larger-magnitude weights at full precision

The layer-23 experiment saved 151,694,064 bytes. To test whether the same byte budget can preserve more quality, we selected a cutoff for each of the eight planes in layers 20–23. Only odd codes at or below that magnitude are rounded. We counted official codes and their block scales, evaluated the actual 12-bit rANS frequency quantization, and chose between adjacent magnitude bins. Among thresholds within 100,000 predicted bytes of the closest match, we selected the lowest average per-plane relative squared weight error. Selection uses no tokens or language-model outputs.

The resulting package is **15,396,361,558 bytes** (151,795,984 fewer than the canonical parent and 101,920 fewer than the layer-23 experiment). The same reader decodes both. [Actual tensor sizes, cutoffs and SHA-256](inkling-budget-rounding.json). To reproduce using NumPy:

```sh
python3 tools/inkling_storage_probe/choose_cutoffs.py "$OFFICIAL/model-00001-of-00001.safetensors" cutoffs.json
python3 tools/inkling_storage_probe/pack_mxfp8.py "$OFFICIAL" "$CANONICAL" "$OUTPUT" --cutoffs cutoffs.json
```

## BF16 corpus scores

Same canonical BF16 oracle, context 512, greedy, FP16 KV, PDL. Each row covers 100 WikiText-2 chunks and 25,500 scored positions. The parent is the 15,548,157,542-byte canonical package and matched the previous reference measurements. The model loader and cubin were identical for the changed-weight candidates.

| Split | Model | Mean KL | Worst KL | Top-1 matches | PPL | Worst chunk, position |
|---|---|---:|---:|---:|---:|---|
| test | parent | 0.003158426 | 0.680977332 | 24,865 | 7.07295554 | 69, 353 |
| test | last | 0.003405436 | 0.691573425 | 24,822 | 7.07310191 | 69, 353 |
| test | all | 0.004220401 | 0.735487623 | 24,756 | 7.07591407 | 69, 353 |
| test | budget | 0.003175307 | 0.671736138 | 24,868 | 7.07392596 | 69, 353 |
| valid | parent | 0.002760741 | 0.352436097 | 24,963 | 6.50462099 | 46, 379 |
| valid | last | 0.003007726 | 0.379497650 | 24,942 | 6.50520427 | 46, 379 |
| valid | all | 0.003759845 | 0.375053389 | 24,865 | 6.50675857 | 46, 379 |
| valid | budget | 0.002768641 | 0.351009065 | 24,944 | 6.50412658 | 46, 379 |

Run `LEANSONIC_PDL=1 LEANSONIC_KV=fp16 LEANSONIC_CUBIN_PATH="$CUBIN" leansonic eval-kld --weights "$OUTPUT/last" --ref "$ORACLE" --chunks 100`, then repeat for `all`. Scoring uses CLI `64200ed` and cubin source `48b7ec2`. The [full metrics](inkling-rounded-corpus.json) also include the reference PPL, median and p99 KL. The layer-23-only candidate lost top-1 matches and increased mean and worst KL on test; it is not proposed as a quality-equivalent replacement.


The budget variant lowered worst KL on both splits relative to the canonical parent (test 0.680977 to 0.671736, validation 0.352436 to 0.351009). Mean KL rose by 0.00001688 and 0.00000790, respectively. Top-1 corresponded to 3 additional matches on test and 19 fewer on validation. Its test PPL rose by 0.00097042 and its validation PPL fell by 0.00049441. At the same byte budget it had lower mean and worst KL than rounding all magnitudes in layer 23 on both splits. We are retaining the candidate for broader testing; default encoding remains the canonical package. The public cutoff-selection and repacking procedure reproduced the candidate hash independently.
