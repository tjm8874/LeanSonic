# Inkling-2-Mini at the original Q3 distribution size

These two packages change only the final four layers’ eight MXFP8 expert weight planes. A contiguous scalar codebook per plane maps the original E4M3 bytes to existing finite E4M3 values. The selector minimizes the sum of block-scale-weighted relative squared error and an estimated symbol-rate penalty. It uses checkpoint weights only. The packer verifies every decoded replacement and every tensor payload in the assembled shard. The ordinary LeanSonic loader and kernels execute both files.

| Distribution | Bytes including tokenizer | Difference from original Q3 |
|---|---:|---:|
| Canonical LeanSonic | 15,548,157,542 | +1,916,379,526 |
| Original Unsloth Q3_K_XL | 13,631,778,016 | +0 |
| compact | 13,626,745,974 | -5,032,042 |
| middle | 14,590,408,254 | +958,630,238 |

## Fidelity to the official BF16 reference

| Dataset | Package | Mean KL | Worst KL | Top-1 matches / targets | PPL |
|---|---|---:|---:|---:|---:|
| test | canonical | 0.0031584 | 0.6809773 | 24,865/25,500 | 7.072956 |
| test | q3 | 0.0179920 | 6.3689020 | 24,081/25,500 | 7.134621 |
| test | compact | 0.0137735 | 0.7519345 | 24,289/25,500 | 7.131377 |
| test | middle | 0.0037406 | 0.6201534 | 24,828/25,500 | 7.078202 |
| valid | canonical | 0.0027607 | 0.3524361 | 24,963/25,500 | 6.504621 |
| valid | q3 | 0.0167140 | 4.2778210 | 24,179/25,500 | 6.565880 |
| valid | compact | 0.0131739 | 0.9835867 | 24,341/25,500 | 6.564200 |
| chat | canonical | 0.0026487 | 0.0956792 | 609/613 | 3.736971 |
| chat | q3 | 0.0135265 | 0.2938108 | 583/613 | 3.665489 |
| chat | compact | 0.0126191 | 0.4765965 | 585/613 | 3.732578 |
| chat | middle | 0.0031885 | 0.0994617 | 605/613 | 3.745258 |

Each WikiText-2 split scores 25,500 positions using a 512-token context and the same BF16 oracle. LeanSonic uses its streaming evaluator on PRO 4500 for the two codebooks. The original Unsloth GGUF uses its native batched scorer on PRO 5000. These are comparisons of the deployed weight formats and runtimes. The bilingual chat suite separately scores full-vocabulary responses at 613 supplied positions on PRO 5000.

On the test split, compact beats original Q3 on all four reported metrics while saving 5,032,042 distribution bytes. Relative to canonical LeanSonic, compact loses 576 top-1 matches and increases mean KL. Keep the canonical default.

## Matched speed on RTX PRO 4500

Both engines used context 8,192, one sequence, FP16 KV, greedy decoding, the same 1,024 README tokens and 32 forced decode inputs. Original Q3 offloaded 25/25 layers. Five mirrored ABBA/BAAB cycles used 10 fresh processes per arm, each with one excluded warm-up plus four timed requests. Peaks are process GPU allocations sampled every 4 ms across loading and inference.

| Package | Prefill ms (tok/s) | Decode 32 ms (tok/s) | Peak GPU bytes | Matches to fixed 32 IDs |
|---|---:|---:|---:|---:|
| Compact LeanSonic | 122.512 (8,358.4) | 200.621 (159.51) | 18,140,364,800 | 32/32 |
| Original Q3_K_XL | 471.488 (2,171.8) | 407.482 (78.53) | 16,999,514,112 | 31/32 |

Compact LeanSonic was 3.849× faster in prefill (paired cycle bootstrap 95% interval 3.843–3.855×) and 2.031× faster in decode (2.030–2.032×). It used 1,140,850,688 more peak GPU bytes than original Q3. These ratios apply to this card, input and pair of configurations. [Individual phase timings, token traces, model, executable, cubin and input hashes](q3-timing.json).

Build [the Lean replay driver](replay.rs) as a leansonic-core example on engine revision `1f70908` with its native SM120a cubin. Use the [original native GGUF replay instructions](../inkling-original-q3-timing.md) with the same 1,024 and 32 ID files. Run each executable with arguments MODEL PREFIX.ids CONTINUATION.ids. The drivers reset between requests and report phase milliseconds.

## Two 16 GB GeForce cards

The original native Q3 configuration offloaded all layers, then failed to allocate its 1,661,706,240-byte compute buffer at microbatch 1,024 on both cards. Reducing its internal microbatch to 512 retained all 25/25 layers on the GPU. Both packages still process the same 1,024 prompt IDs and 32 forced inputs at context 8,192. LeanSonic uses its automatic placement with 20/24 resident expert layers and four partially host-backed layers. Both arms use the PRO 4500 protocol above, including 40 warm requests per arm on each card.

| GPU and package | Prefill ms (tok/s) | Decode 32 ms (tok/s) | Peak GPU bytes | Matches to fixed 32 IDs |
|---|---:|---:|---:|---:|
| RTX 5070Ti Compact LeanSonic | 160.482 (6,380.8) | 303.962 (105.28) | 16,546,529,280 | 32/32 |
| RTX 5070Ti Original Q3_K_XL | 526.192 (1,946.1) | 402.414 (79.52) | 16,150,167,552 | 31/32 |
| RTX 5060Ti Compact LeanSonic | 380.367 (2,692.1) | 746.686 (42.86) | 16,554,917,888 | 32/32 |
| RTX 5060Ti Original Q3_K_XL | 881.696 (1,161.4) | 573.706 (55.78) | 16,030,629,888 | 30/32 |

Compact improves both phases on RTX 5070 Ti. On RTX 5060 Ti it improves prefill, but original Q3 decodes faster. Each of the eight compact MXFP8 planes uses just 9 or 13 signed byte values. A reversible four-bit representation in device memory is a candidate for reducing host-backed expert traffic. The runtime measured here expands these weights to bytes. [Every request, process peak, binary digest and paired interval](geforce-q3-timing.json).

On the 613 bilingual teacher-forced positions, compact has 585 BF16 top-1 matches versus Q3's 583 and canonical's 609. Compact's worst KL is 0.476597, exceeding original Q3's 0.293811. This independent suite reverses the two packages' WikiText ordering for worst KL.

## Assigning existing symbols by block scale

The follow-up keeps the compact codebook's output symbols and the original exponent of each 32-weight block. It refits assignments under the original per-plane cross-entropy budget. Only two planes improved the weight objective enough to change their maps. The result is 13,626,725,054 bytes, saving a further 20,920 bytes. To rebuild, append `--condition-on-scale` to the compact packing command below with a fresh output directory.

| Split | Mean KL | Worst KL | Top-1 / 25,500 | PPL |
|---|---:|---:|---:|---:|
| test | 0.0137804 | 0.7458703 | 24,278 | 7.131103 |

On test, the worst KL fell from 0.751935 to 0.745870, while top-1 matches fell from 24,289 to 24,278. Retain this as an offline experiment. [Eight frame hashes, per-plane weight distortion and corpus metrics](scale-conditioned.json).

## Reproduce

Start from the [canonical package and official checkpoint](../inkling-storage-candidates.md). Install NumPy. Use fresh output directories. To reproduce the measured byte stream directly, pass the committed lookup table to the packer. To regenerate the tables from weights use the selector first.

```sh
cargo build --release -p leansonic-core --example inkling_mxfp8_round
python3 tools/inkling_storage_probe/choose_codebooks.py "$OFFICIAL/model-00001-of-00001.safetensors" "$CANONICAL" "$MAPS" 13630729440 14600000000
python3 tools/inkling_storage_probe/pack_mxfp8.py "$OFFICIAL" "$CANONICAL" "$NEW_DIR" --encoder target/release/examples/inkling_mxfp8_round --codebooks docs/benchmarks/inkling-codebooks/compact-lut.json
LEANSONIC_KV=fp16 LEANSONIC_PDL=1 leansonic eval-kld --ref "$SPLIT_ORACLE" --weights "$NEW_DIR/codebook" --chunks 100
```

The scorer executable SHA-256 is `a49460cb9bd81ab20c5903cb1e47526f851c70283232e12871b14cca7334634a` and the quality cubin is `3714fd830d6cfedbf6954f21cad460489173be36566b0cb4b78173d42da87f0b`. See [full receipts and worst positions](results.json), [oracle recipe](../../inkling-reference.md), [original GGUF scorer](../inkling-original-gguf-quality.md), [chat scoring](../inkling-chat-quality.md) and the [free-generation fixture](../inkling-functional-chat.md).
