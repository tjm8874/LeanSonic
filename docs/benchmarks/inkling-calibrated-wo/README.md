# Activation-calibrated attention output weights for Inkling-2-Mini

The v0.10.0 engine converts the publisher's BF16 attention output matrices to Q8_64 during loading. This experiment fits the signed Q8 codes to observed input activations and keeps every original group scale. The normal engine can read the fitted codes through its existing native Q8 reader. No runtime patch is required.

On RTX PRO 4500, 100 WikiText-2 test chunks gave the following results against the same BF16 oracle. Both arms used release aa4d429, one sequence, FP16 KV, the same SM120a cubin, and all 24 resident expert layers.

| Test metric | Official weights | Fitted weights |
| --- | ---: | ---: |
| Shard and tokenizer, bytes | 19,529,405,662 | 19,353,437,702 |
| Mean KL | 0.003204114 | 0.003125869 |
| Worst KL | 0.676308 | 0.659475 |
| BF16 top-1 matches / 25,500 | 24,896 | 24,908 |
| PPL | 7.070718 | 7.072906 |

The average KL decreased by 0.0000782. Its paired 95% whole-chunk bootstrap interval spans −0.0002185 to +0.0000633. At the baseline worst position (chunk 36, offset 375), KL fell from 0.676308 to 0.326483. At the new worst position (69, 353), KL rose from 0.608355 to 0.659475. The precision-max target is 0.0030. [Scorer output, paired effects and row hashes](test.json).

On RTX PRO 5000, the fitted candidate scored mean KL 0.002760394, worst KL 0.445438, BF16 top-1 24,976/25,500 and PPL 6.507858 on the separate validation corpus. The same-card baseline replay is queued. [Validation score](validation-fitted.json).

Calibration uses 32 disjoint 512-token windows from WikiText-2 **train**. The normal prefill supplies the signed Q8 activations and FP32 group scales entering the output projection. We sample 64 rows per visit, spanning all 48 visits to the 24 physical layers. Every fourth window is withheld. For each layer and each 64-column block, the fitter minimizes the quadratic error under a training second-moment matrix, with 5% diagonal damping and two fixed coordinate sweeps. Of 169,869,312 signed codes, 16,285,113 changed. The undamped blockwise error estimate fell 44.18% on fit windows and 35.33% on withheld train windows. These errors measure the linear projection. The table above measures end-to-end output quality.

A second clean release build replayed the saved activation capture, reproduced every per-layer fit record, and rebuilt the same shard and tokenizer byte for byte. The converter verifies decoding of every fitted code frame and FP32 scale frame, then reads back every stored tensor and the tokenizer. In the official package, 1,220 other tensor payloads are unchanged. [Distribution hashes](package.json), [calibration inputs](fixture.json), [capture fingerprint](covariance.json), [per-layer errors and code hashes](layers.json).

## Reproduction

Use a clean checkout of aa4d429, these benchmark files, CUDA with sm_120a support, Rust, Python 3.11+, NumPy and tokenizers. Use the publisher b85c3cfa shard with SHA-256 65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1. Set WORK to a new absolute directory, OFFICIAL to the publisher model directory, TRAIN to wiki.train.raw, TOKENIZER to the publisher tokenizer.json and D to the absolute path to this benchmark directory. Select the same idle GPU throughout calibration. From the release checkout:

```sh
mkdir "$WORK"
python3 "$D/windows.py" "$TRAIN" "$TOKENIZER" "$WORK/windows.ids" > "$WORK/fixture.json"
git apply "$D/capture.patch"
cp "$D"/{calibrate,dump,pack}.rs crates/leansonic-core/examples/
cargo build --release -p leansonic-core --example calibrate --example dump --example pack
BIN="$PWD/target/release/examples"
nvcc -Ikernels -arch=sm_120a -std=c++17 -O3 --cubin kernels/inkling2_sm120.cu -o "$WORK/release.cubin"
export LEANSONIC_CUBIN_PATH="$WORK/release.cubin" LEANSONIC_KV=fp16 LEANSONIC_SEQS=1
export LEANSONIC_PREFILL_CHUNK=512 LEANSONIC_PDL=1 LEANSONIC_PDL_ALL=1
cd "$WORK"
"$BIN/calibrate" "$OFFICIAL" windows.ids > baseline.ids
INKLING_TRAIN_WO=1 "$BIN/calibrate" "$OFFICIAL" windows.ids > observed.ids
cmp baseline.ids observed.ids
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python3 "$D/covariance.py" > covariance.json
python3 "$D/fit.py" "$OFFICIAL/model-00001-of-00001.safetensors" "$WORK" "$BIN/dump" "$WORK/fit"
python3 "$D/assemble.py" "$OFFICIAL" "$WORK/fit" "$BIN/pack" "$WORK/model" > package.json
```

Compare the hashes in fixture.json, covariance.json and package.json against the linked records. The optional compact overlay consumes the same 24 fitted frames and requires all 24 group-scale frames to match the compact source byte for byte. Run `python3 "$D/overlay.py" "$COMPACT" "$WORK/frames" "$WORK/compact-model"`. The #231 + #236 compact package grew by 19,584 bytes to 13,407,193,392 bytes. [Compact distribution hashes](compact-package.json).

For end-to-end scoring, use a clean release CLI with the [row observer and WikiText BF16 oracle instructions](https://github.com/tjm8874/LeanSonic/tree/fcbd52a/docs/benchmarks/inkling-mxscale-codes#reproduce). Keep the same release cubin and set LEANSONIC_G1_ATTN=1, LEANSONIC_CTX=8192, LEANSONIC_PREFILL_CHUNK=512, LEANSONIC_KV=fp16, LEANSONIC_SEQS=1, LEANSONIC_PDL=1 and LEANSONIC_PDL_ALL=1. Run eval-kld --chunks 100 once per model and oracle, setting DIAG_ROWS to a fresh file each time. The official matrix's short solo decode probe uses `bench --tokens synth:32 --spec 0 --gen 128 --repeat 2` at context 8192.

## Compact corpus

The 13,407,173,808-byte parent and 13,407,193,392-byte fitted overlay differ in the 24 output-projection code frames. All their stock scale frames match byte for byte. Each pair uses the same CLI, cubin, 512-token chunks, BF16 oracle, 24 resident expert layers and GPU (5070 Ti test, 5060 Ti validation). Each row here covers 25,500 scored predictions.

| Split | Weights | Mean KL | Worst KL | Top-1 matches | PPL |
| --- | --- | ---: | ---: | ---: | ---: |
| test | Parent | 0.0077481 | 1.462040 | 24,537 | 7.083731 |
| test | Fitted | 0.0076845 | 1.598130 | 24,525 | 7.086873 |
| valid | Parent | 0.0069527 | 0.744671 | 24,645 | 6.530797 |
| valid | Fitted | 0.0068973 | 1.625774 | 24,650 | 6.536081 |

The test top-1 count fell by 12 with the fitted overlay, and worst KL and PPL increased on both splits. On validation both arms had their largest KL at chunk 12 position 317. The fitted overlay more than doubled the divergence at that position. Retain the original compact weights for the drafter comparison. [Per-split scores, paired intervals and outliers](compact-scores.json).

## Official short solo decode probe

Fresh release CLI processes ran the bench-matrix short workload on the original and fitted precision packs. The prompt is synth:32, with 128 greedy output tokens, spec 0, context 8192 and FP16 KV. The figures below are medians of each process's warm second request, with four fresh processes per mirrored cycle. This probe uses natural greedy continuation from each model. [Every measured phase and output ID](short-decode.json).

| GPU | Complete cycles of five | Official decode tok/s | Fitted decode tok/s |
| --- | ---: | ---: | ---: |
| RTX PRO 4500 | 4 | 168.6 | 168.8 |

A [128-column calibration comparison](paired-groups.md) tested whether a lower projection-error estimate translated to lower end-to-end divergence.
