# Canonical NVFP4 zero codes on Inkling-2-Mini

The official NVFP4 expert weights contain 703,609,090 negative-zero nibbles in 10,192,158,720 bytes. The existing DP4A and Tensor Core integer lookup tables decode E2M1 codes 0 and 8 to the same zero. With the new packer option, the encoder replaces code 8 with 0 before computing the existing context-rANS tables. It keeps the new frame only if the entire frame is smaller. Original zero sign bits cannot be recovered from the canonical frame. Other codes, scales, the reader and the GPU kernel are unchanged.

| Text distribution including tokenizer | Bytes |
|---|---:|
| Parent (#117) | 15,724,161,014 |
| Canonical zero | 15,548,157,542 |
| Original Unsloth IQ4_XS GGUF | 15,733,755,232 |
| Original Unsloth Q3 GGUF | 13,631,778,016 |

The 40 new frames saved 176,003,472 bytes against the parent. The other 959 tensor payloads retained their SHA-256 digests. An independent C++ encoder decoded each new frame and compared it against the normalized source plane. The public packer separately reproduced the same 15,544,771,048-byte shard from official safetensors and from the #117 package on different hosts. The full tokenizer adds 3,386,494 bytes. Shard SHA-256 is `9b242a81188f0f552625639dec98879a0090a7f9a003141631d3a836aed3fcb3`.

## Reproduce

At commit `a9872cc`, use a fresh output directory. Set `SOURCE` to the official text checkpoint directory and `OUT` to that output directory.

```sh
shard=model-00001-of-00001.safetensors
python3 tools/pack_inkling_text.py "$SOURCE/$shard" "$OUT/$shard" \
  --prepack-head --prepack-attention --compact-q8-scales --compact-q8-codes \
  --prepack-embedding --prepack-mxfp8 --canonical-nvfp4-zeros --averaged-scales --compact-controls
cp -R "$SOURCE/tiktoken" "$OUT/"
sha256sum "$OUT/$shard"
```

For an existing #117 package, pass only `--canonical-nvfp4-zeros` to the packer and copy its tokenizer. The original reversible packing options remain available.

## GPU comparison

Using the same PDL cubin, FP16 KV, 32-token prefix and 16 forced decode steps as the parent benchmark, we compared all 17 × 200,058 FP32 logits byte-for-byte in separate processes per package. Identical parent and candidate hashes were recorded on 5060Ti, 5070Ti, PRO4500. The [per-tensor receipt and GPU hashes](inkling-zero-codes.json) specify the exact outputs checked. This probe covers the tested runtime path and inputs. The full-corpus quality and repeated timing experiments are separate.

## Fixed inputs against the original Unsloth IQ4_XS

RTX PRO 4500 Blackwell, one sequence, FP16 KV, context 8192, greedy, no speculation. Both arms consumed the same 1,024 IDs and 32 teacher-forced next inputs. Five alternating ABBA/BAAB cycles used ten fresh processes and 40 warm requests per arm. Peak is the largest NVML process memory sample during load or inference, sampled approximately every 4 ms. All 25 native layers were offloaded.

| Engine | Prefill 1,024 ms / tok/s | Decode 32 ms / tok/s | Peak GPU bytes | Distribution bytes | Matched fixed IDs |
|---|---:|---:|---:|---:|---:|
| LeanSonic canonical | 122.242 / 8,376.8 | 201.482 / 158.82 | 18,173,919,232 | 15,548,157,542 | 32/32 |
| Original Unsloth IQ4_XS | 462.867 / 2,212.3 | 417.118 / 76.72 | 18,928,893,952 | 15,733,755,232 | 30/32 |

Follow the [fixed-token benchmark procedure](inkling-original-iq4.md), selecting the canonical shard identified above for LeanSonic. The 32 fixture predictions and the BF16 corpus test below answer different questions.

## Warm restart and decode versus #117

A 32-ID prefix, FP16 KV, PDL, no speculation. Five mirrored cycles (ten fresh processes per format) verified all shard pages were already resident in the OS cache before each load. Warm decode averages the last two of three 64-token requests per process. The 64 generated IDs matched for every leg.

| GPU | #117 / canonical load seconds | #117 / canonical decode64 ms |
|---|---:|---:|
| 5060Ti | 11.443 / 11.469 | 1266.50 / 1266.49 |

## BF16 WikiText-2 comparison

100 chunks per split, ctx 512, 25,500 scored positions, FP16 KV and PDL. Run `leansonic eval-kld --weights "$OUT" --ref "$ORACLE" --chunks 100` using the matching reference split. The observed mean, median, p99 and maximum KL, worst location, PPL and top-1 counts matched the prior benchmark exactly.

| GPU | Split | Mean KL | Worst KL | Top-1 matches | PPL | BF16 PPL | Worst chunk, position |
|---|---|---:|---:|---:|---:|---:|---|
| PRO5000 | test | 0.0031584260 | 0.6809773323 | 24,865/25,500 | 7.07295554 | 7.06440819 | 69, 353 |
| 5070Ti | test | 0.0031584260 | 0.6809773323 | 24,865/25,500 | 7.07295554 | 7.06440819 | 69, 353 |

[Individual timing runs, predictions, input hashes and corpus metrics](inkling-zero-measurements.json).
