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

## Reproduce

Start from the [canonical package and official checkpoint](../inkling-storage-candidates.md). Install NumPy. Use fresh output directories. To reproduce the measured byte stream directly, pass the committed lookup table to the packer. To regenerate the tables from weights use the selector first.

```sh
cargo build --release -p leansonic-core --example inkling_mxfp8_round
python3 tools/inkling_storage_probe/choose_codebooks.py "$OFFICIAL/model-00001-of-00001.safetensors" "$CANONICAL" "$MAPS" 13630729440 14600000000
python3 tools/inkling_storage_probe/pack_mxfp8.py "$OFFICIAL" "$CANONICAL" "$NEW_DIR" --encoder target/release/examples/inkling_mxfp8_round --codebooks docs/benchmarks/inkling-codebooks/compact-lut.json
LEANSONIC_KV=fp16 LEANSONIC_PDL=1 leansonic eval-kld --ref "$SPLIT_ORACLE" --weights "$NEW_DIR/codebook" --chunks 100
```

The scorer executable SHA-256 is `a49460cb9bd81ab20c5903cb1e47526f851c70283232e12871b14cca7334634a` and the quality cubin is `3714fd830d6cfedbf6954f21cad460489173be36566b0cb4b78173d42da87f0b`. See [full receipts and worst positions](results.json), [oracle recipe](../../inkling-reference.md), [original GGUF scorer](../inkling-original-gguf-quality.md), [chat scoring](../inkling-chat-quality.md) and the [free-generation fixture](../inkling-functional-chat.md).
