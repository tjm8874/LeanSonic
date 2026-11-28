# Lossless MXFP8 expert storage for Inkling-2-Mini

Layers 20–23 have eight physical MXFP8 gate/up and down weight tensors. The code distribution varies with the E8M0 scale shared by each block of 32 weights. The optional `.mxscale_codes_v1` frame uses that existing scale to choose one of up to eight rANS tables. The converter decodes and compares every one of the 4,076,863,488 original weight bytes. The loader reconstructs the same ordinary weights or the same 16-entry device codebook and packed nibbles.

| Four-bank distribution, including tokenizer | Parent bytes | Converted bytes | Bytes saved |
| --- | ---: | ---: | ---: |
| Released prefix embedding | 13,448,974,014 | 13,418,718,342 | 30,255,672 |
| Seven-bit rank embedding from #231 | 13,437,429,480 | 13,407,173,808 | 30,255,672 |

All 991 other tensor payloads and the tokenizer were checked on readback. A second conversion from the packed seven-bit parent reproduced all eight frames and the entire final shard. [Prefix manifest](prefix-package.json) and [rank manifest](rank-package.json).

## Quality and memory

On RTX PRO 4500, the prefix parent, converted prefix and converted seven-bit package have identical full-vocabulary fingerprints and score rows on 1,530 positions. The converted seven-bit model gave the same rows on RTX 5070 Ti and RTX 5060 Ti. Mean KL versus BF16 was 0.00859703, worst KL 0.63976597, top-1 agreement 1,469/1,530 and PPL 6.564087. Each GPU kept all 24 expert layers resident.

On the RTX 5070 Ti, the converted seven-bit package also reproduced the five-bit reference's entire 25,500-row test file on RTX PRO 4500. Test mean KL was 0.00774813, worst KL 1.46204035, top-1 agreement 24,537/25,500 and PPL 7.083731. [Score records and row hashes](quality.json).

| Converted seven-bit package | Prefill tok/s | Decode tok/s | Sampled peak GPU bytes |
| --- | ---: | ---: | ---: |
| RTX 5070 Ti | 8,388.76 | 128.37 | 15,556,673,536 |
| RTX 5060 Ti | 3,962.16 | 82.27 | 15,458,107,392 |

Each row uses one fresh process, seven warm requests, 1,024 fixed input tokens and 32 forced decode steps, one sequence, FP16 KV and speculation off. Phase timers exclude weight loading. Peaks include loading. These runs characterize the new package. [GPU settings, inputs, output IDs, samples and peaks](consumer-gpus.json).

## CPU decompression cost

Five mirrored cycles on a Ryzen 9 9900X measured the layer 20 weights in ordinary host memory. File reads precede the timer. The conditioned arm includes decoding its companion scales. The gate/up plane rose from 226.79 to 267.29 ms (+17.9%, paired 95% time-ratio interval 1.121–1.272). The down plane rose from 105.25 to 120.32 ms (+14.3%, interval 1.072–1.197). These times measure two individual planes. End-to-end GPU model startup has separate costs. [All 40 process samples](cpu-times.json) and [timer](time.rs).

## Reproduce

Use publisher revision b85c3cfa and the [four-bank prefix parent](https://github.com/tjm8874/LeanSonic/tree/588e3cd/docs/benchmarks/inkling-nvfp4-scale-split/four-banks/quartiles/cost-fit) or the [seven-bit parent](https://github.com/tjm8874/LeanSonic/tree/4d3bd11/docs/benchmarks/inkling-lossless-rank/measured-widths). The prefix variant works directly with this branch based on aa4d429. To run the rank variant, also apply the rank reader from #231. The two Python tools require Python 3.11 or newer and new output directories. From the repository root:

```sh
cargo build --release -p leansonic-core --example inkling_pack_mxscale
d=docs/benchmarks/inkling-mxscale-codes
python3 "$d/frames.py" "$PARENT" target/release/examples/inkling_pack_mxscale "$FRAMES"
python3 "$d/assemble.py" "$PARENT" "$FRAMES" "$MODEL" > manifest.json
```

For the six-chunk GPU replay, generate the reference using the [fixture and BF16 oracle instructions](https://github.com/tjm8874/LeanSonic/tree/4d3bd11/docs/benchmarks/inkling-lossless-rank#gpu-replay). In a fresh checkout with the required reader(s), build as follows. The observer records a fingerprint over all FP32 vocabulary entries after scoring. Use a fresh output file for each invocation.

```sh
python3 docs/benchmarks/inkling-mxscale-codes/observe.py crates/leansonic-cli/src/eval_kld.rs
cargo build --release -p leansonic-cli
nvcc -Ikernels -arch=sm_120a -O3 -std=c++17 --cubin kernels/inkling2_sm120.cu -o main.cubin
export LEANSONIC_CUBIN_PATH="$PWD/main.cubin" LEANSONIC_KV=fp16 LEANSONIC_SEQS=1
export LEANSONIC_PDL=1 LEANSONIC_PDL_ALL=1 LEANSONIC_CTX=8192 LEANSONIC_PREFILL_CHUNK=512
export LEANSONIC_MXFP8_DEVICE_U4=1 LEANSONIC_MXFP8_DIRECT_ALL=1
DIAG_ROWS=parent.rows target/release/leansonic eval-kld --weights "$PARENT" --ref "$SUBSET" --chunks 6 > parent.json
DIAG_ROWS=converted.rows target/release/leansonic eval-kld --weights "$MODEL" --ref "$SUBSET" --chunks 6 > converted.json
cmp parent.rows converted.rows && cmp parent.json converted.json
```

Use the respective full test or validation reference and `--chunks 100` for full-split scoring. For phase timing, use the pinned [Rust harness, 1,024-token prompt and 32-token history](https://github.com/tjm8874/LeanSonic/tree/4d40691/docs/benchmarks/inkling-class-frames/three-engines) with the combined reader, context 2,048 and prefill chunk 1,024.
