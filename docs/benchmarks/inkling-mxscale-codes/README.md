# Lossless MXFP8 expert storage for Inkling-2-Mini

Layers 20–23 have eight physical MXFP8 gate/up and down weight tensors. The code distribution varies with the E8M0 scale shared by each block of 32 weights. The optional `.mxscale_codes_v1` frame uses that existing scale to choose one of up to eight rANS tables. The converter decodes and compares every one of the 4,076,863,488 original weight bytes. The loader reconstructs the same ordinary weights or the same 16-entry device codebook and packed nibbles.

| Four-bank distribution, including tokenizer | Parent bytes | Converted bytes | Bytes saved |
| --- | ---: | ---: | ---: |
| Released prefix embedding | 13,448,974,014 | 13,418,718,342 | 30,255,672 |
| Seven-bit rank embedding from #231 | 13,437,429,480 | 13,407,173,808 | 30,255,672 |

All 991 other tensor payloads and the tokenizer were checked on readback. A second conversion from the packed seven-bit parent reproduced all eight frames and the entire final shard. [Prefix manifest](prefix-package.json) and [rank manifest](rank-package.json).

## Quality and memory

On RTX PRO 4500, the prefix parent, converted prefix and converted seven-bit package have identical full-vocabulary fingerprints and score rows on 1,530 positions. The converted seven-bit model gave the same rows on RTX 5070 Ti and RTX 5060 Ti. Mean KL versus BF16 was 0.00859703, worst KL 0.63976597, top-1 agreement 1,469/1,530 and PPL 6.564087. Each GPU kept all 24 expert layers resident. On PRO 4500 the generic full-byte device layout also matched its prefix parent across all 1,530 score rows with both direct-U4 environment switches disabled.

Both complete BF16 splits reproduced every byte of the respective five-bit score file from RTX PRO 4500, including all 51,000 full-vocabulary fingerprints.

| Split | Converted seven-bit GPU | Positions | Mean KL | Worst KL | BF16 top-1 agreement | Model PPL | BF16 PPL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Test | RTX 5070 Ti | 25,500 | 0.00774813 | 1.46204035 | 24,537 (96.22%) | 7.083731 | 7.064408 |
| Validation | RTX 5060 Ti | 25,500 | 0.00695275 | 0.74467143 | 24,645 (96.65%) | 6.530797 | 6.498589 |

Each split uses 100 chunks of 512 tokens and scores the last 255 transitions per chunk. [Score records and row hashes](quality.json).

| Converted seven-bit package | Prefill tok/s | Decode tok/s | Sampled peak GPU bytes |
| --- | ---: | ---: | ---: |
| RTX 5070 Ti | 8,388.76 | 128.37 | 15,556,673,536 |
| RTX 5060 Ti | 3,962.16 | 82.27 | 15,458,107,392 |

Each row uses one fresh process, seven warm requests, 1,024 fixed input tokens and 32 forced decode steps, one sequence, FP16 KV and speculation off. Phase timers exclude weight loading. Peaks include loading. These runs characterize the new package. [GPU settings, inputs, output IDs, samples and peaks](consumer-gpus.json).

## CPU decompression cost

Five mirrored cycles on a Ryzen 9 9900X measured the layer 20 weights in ordinary host memory. File reads precede the timer. The conditioned arm includes decoding its companion scales. The gate/up plane rose from 226.79 to 267.29 ms (+17.9%, paired 95% time-ratio interval 1.121–1.272). The down plane rose from 105.25 to 120.32 ms (+14.3%, interval 1.072–1.197). These times measure two individual planes. End-to-end GPU model startup has separate costs. [All 40 process samples](cpu-times.json) and [timer](time.rs).

## Packing the device codebook at load time

The direct-U4 loader now returns nibble indices from the rANS lookup tables and packs them on the same worker that decoded each tile. Each worker reuses its tile buffer. This removes the serial pass that remapped and packed all decoded weights. The eight physical tensors were compared with an independent NumPy packing of their original codes. Every device byte matched for the plain and scale-conditioned files, with both the original and revised loaders (32 independent decoder/oracle comparisons).

| Layer 20 tensor and file format | Original device packing, ms | Revised device packing, ms | 95% paired time ratio |
| --- | ---: | ---: | ---: |
| Gate/up, conditioned | 253.92 | 170.91 | 0.655–0.686 |
| Down, conditioned | 128.93 | 93.53 | 0.702–0.759 |
| Gate/up, plain | 221.56 | 139.85 | 0.610–0.660 |
| Down, plain | 113.30 | 71.47 | 0.617–0.643 |

Five mirrored cycles for each row on a Ryzen 9 9900X, 80 fresh processes total. Times cover production construction of the complete resident layout in host memory, including the conditioned file's companion scale decode. File reads occur before the timer. RTX 5070 Ti reproduced all 1,530 baseline rows after the packing change. The two 25,500-row GPU results above used the original packing loop. [Every process and all physical device hashes](device-load/samples.json).

To repeat the CPU experiment, use fresh checkouts of 47237bb and this version. Append [hook.rs](device-load/hook.rs) to crates/leansonic-core/src/inkling2_gpu.rs in both checkouts to expose the private production method to the timing binary. Copy [bench.rs](device-load/bench.rs) to crates/leansonic-core/examples/time_device_indices.rs, then build the release example. Run `time_device_indices mx NEW_FRAME SCALE_RANS RAW_CODE_COUNT time` or `time_device_indices plain OLD_FRAME SCALE_RANS RAW_CODE_COUNT time`. For independent parity use `dump` in place of `time`, hash stdout and compare with [oracle.py](device-load/oracle.py) on the uncompressed weight bytes. The oracle reads the original byte-frequency table from the companion .rans file.

The combined distribution also reproduced all 1,530 six-chunk score rows and full-vocabulary fingerprints on RTX PRO 5000, with all 24 expert layers resident. The same row SHA-256, 2f0536ed33e23063e251d05b2b9670644f5a9eb2dfd763bb9560921b84a10111, now covers RTX PRO 4500, RTX PRO 5000, RTX 5070 Ti and RTX 5060 Ti. [All scores and row hashes](quality.json).

## Full-engine initialization on RTX 5070 Ti

An Intel Core i5-14400 loaded the complete model in fresh processes on the same RTX 5070 Ti. Each file was read before every launch to warm the filesystem page cache. The measured interval is the engine's reported model initialization, including creation of the GPU layout, at 0.01 s resolution. All 24 layers were resident. Each completed cycle contains four processes, ordered A/B/B/A and reversed in the next cycle. Every process checked all 33 IDs for each of the eight forced and eight greedy requests. All arms reproduced their corresponding traces.

| Paired experiment | Baseline → candidate | Complete cycles | Mean, seconds | Candidate/baseline 95% interval |
| --- | --- | ---: | ---: | ---: |
| Original loader on both arms | Seven-bit parent → Conditioned codes | 5 | 14.885 → 16.036 | 1.0036–1.2120 |
| Conditioned codes on both arms | Original loader → Fused loader | 3 | 14.957 → 13.828 | pending five cycles |

The first comparison isolates the storage-format cost and the second isolates the CPU packing change. Their absolute times come from separate cohorts. Intervals resample complete mirrored cycles. Every completed leg was retained. [Individual initialization times, device and matched token histories](device-load/startup.json).

To reproduce, use the phase-timing harness below with one fresh process per leg and the same seven-bit model weights, context and flags. Warm the page cache by reading the safetensors shard and tokenizer before each launch. Parse the seconds from the final GPU initialization log. For the original loader use the combined rank and conditioned-code readers at 47237bb. For the fused loader add 3cf30b6.

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

## Rebuild the compact reader with the Q4 drafter

For the 16 GB drafter experiment in #235, build the seven-bit and conditioned-code readers on main 2f372be, which includes #237 and #239. Fetch the two PR heads into local refs to make their commits available, then use a fresh checkout of 2f372be.

```sh
git fetch origin pull/231/head:refs/remotes/origin/pr-231 pull/236/head:refs/remotes/origin/pr-236
git diff aa4d429 fcbd52a -- crates > mx.patch
git diff 4a139df 9bc4d48 -- crates/leansonic-core/src > rank.patch
git apply mx.patch
git apply rank.patch
cargo build --release -p leansonic-cli
nvcc -Ikernels -arch=sm_120a -O3 -std=c++17 --cubin kernels/inkling2_sm120.cu -o target.cubin
nvcc -Ikernels -arch=sm_120a -O3 -std=c++17 --cubin kernels/inkling2_dflash2_sm120.cu -o drafter.cubin
```

The generated patch SHA-256 values are 86add5468ae4551c0d96fefac1d4bd1d67470fa436977dd6e767d80886526255 and bfa272012c3509942dc4ec204667e5c7985b3d036caf19d8dc4c846bf3d9630b. The target and drafter cubins from the reference build hash to 96f7b35bdbceee2b8359c6e87adcd9babd0ab3368c23785c3a0ad0afabf7167b and 9c6633abffb9cb1c045ed31c6a3cdb77f26b2f45c2aca16917bbc4d14824cf1e. Point LEANSONIC_CUBIN_PATH and LEANSONIC_DF2_CUBIN_PATH at the respective cubins, set LEANSONIC_DF2_MLP=q4, and enable LEANSONIC_MXFP8_DEVICE_U4 and LEANSONIC_MXFP8_DIRECT_ALL. Select the same DFlash 2 checkpoint as the official-weight comparison using LEANSONIC_DFLASH2. The uncalibrated 13,407,173,808-byte compact package is the reference for these runs. [Handoff and device measurement protocol](https://github.com/tjm8874/LeanSonic/issues/235#issuecomment-7231158196).
