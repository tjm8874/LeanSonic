# Inkling-2-Mini compact expert weights on GeForce Blackwell

The optional loader stores each pair of MXFP8 weight codes as two four-bit indices into a 16-byte lookup table. It recovers the original weight bytes before the existing compute kernels. Only the compact model’s final eight expert planes are selected. Together they occupy 2,038,431,872 bytes in this representation, originally 4,076,863,488. One 679,477,248-byte scratch buffer is reused. The distribution on disk is identical in both arms, 13,626,745,974 bytes including the tokenizer.

## Fixed-input replay

1,024 prompt tokens, 32 forced decode steps, 8,192 context, one sequence, FP16 KV, no speculation. Five mirrored ABBA/BAAB cycles give ten fresh processes per arm. Each process performs five requests. All phase figures use the last four, giving 40 timed requests per arm. The peak is the largest per-process allocation observed with NVML at roughly 4 ms intervals during load and inference. The 95% intervals resample whole paired cycles.

| GPU | Weight storage | Resident expert layers | Prefill ms (tok/s) | Decode 32 ms (tok/s) | Peak GPU bytes |
|---|---|---:|---:|---:|---:|
| RTX 5060 Ti | Original bytes | 20/24 | 380.387 (2,692.0) | 746.732 (42.85) | 16,554,917,888 |
| RTX 5060 Ti | Four-bit indices | 24/24 | 403.469 (2,538.0) | 366.373 (87.34) | 16,598,958,080 |

RTX 5060 Ti: prefill delta +23.082 ms (95% interval +22.296 to +24.610); decode delta -380.359 ms (-380.415 to -380.277). The two modes predicted the same 32 tokens across all 100 requests. Separately, at context 2,048, all eight paired free-generation requests matched on all 64 generated IDs, the following token, and the hash of the final 200,058 finite FP32 logits. That check used both 32- and 1,024-token prompts.

## Quality and adoption

The file used here is the compact model from [PR #147](https://github.com/tjm8874/LeanSonic/pull/147). On 613 bilingual teacher-forced positions its worst KL was 0.476597, compared with 0.293811 for original Q3 and 0.095679 for canonical LeanSonic. The byte representation and model quantization are independent changes. Keep the runtime feature opt-in. The 8K-context prefill regression on 5060 Ti and the compact package’s bilingual worst case both matter for deployment.

PRO 4500, WikiText-2 test split, 25,500 positions, FP16 KV: four-bit device storage mean KL 0.0137838, worst 0.7078185 (chunk 69, position 353), top-1 24,298/25,500, PPL 7.127482. All reported metrics matched the byte-storage control on the same build. Both full scorer reports are in the receipt.

## Reproduce

Build the compact package with the [pinned lookup table and packer](https://github.com/tjm8874/LeanSonic/tree/fdede5b/docs/benchmarks/inkling-codebooks). Verify the shard SHA-256 and the two input hashes in the receipt. Use the following commands from the engine checkout. The original build used CUDA 13.0.88, SM120a, and GeForce driver 595.99.02. Choose one GPU through CUDA_VISIBLE_DEVICES. MODEL points to the compact model directory.

```sh
dir=docs/benchmarks/inkling-device-codebooks
cp "$dir/parity.rs" crates/leansonic-core/examples/device_codebook_parity.rs
cp "$dir/replay.rs" crates/leansonic-core/examples/device_codebook_speed.rs
cargo build --release -p leansonic-core --example device_codebook_parity --example device_codebook_speed
nvcc -arch=sm_120a -std=c++17 -O3 --cubin kernels/inkling2_sm120.cu -o matched.cubin
export LEANSONIC_CUBIN_PATH=$PWD/matched.cubin LEANSONIC_SEQS=1 LEANSONIC_PDL=1 LEANSONIC_KV=fp16
inputs=docs/benchmarks/inputs
for arm in 0 1; do
  export LEANSONIC_MXFP8_DEVICE_U4=$arm
  target/release/examples/device_codebook_parity "$MODEL" "$inputs/inkling-readme1024.ids"
  target/release/examples/device_codebook_speed "$MODEL" "$inputs/inkling-readme1024.ids" "$inputs/inkling-continuation32.ids"
done
```

To time both arms, repeat the speed executable in mirrored order as above. For corpus quality, run `leansonic eval-kld --weights "$MODEL" --ref "$TEST_KLD" --chunks 100` with each setting and the [same BF16 oracle](../../inkling-reference.md). The original GGUF replay entry point and model digest are in the [native timing instructions](../inkling-original-q3-timing.md).

[Phase times and sampled peaks](requests.csv) · [Build, source and fixture hashes, paired intervals, prediction traces and scorer output](receipt.json)
