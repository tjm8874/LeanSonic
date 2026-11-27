# Gate/up precision pilot with the original Inkling GGUF runtime

The two original Unsloth GGUF files have 734 tensors each and exactly 11,978,720 bytes each outside the tensor payloads. Gate/up expert weights account for 1,709,309,952 of the 2,101,977,216 bytes between original IQ4_XS and Q3_K_XL. In layers 1, 2 and 3 the shipped gate/up changes from NVFP4 to IQ2_S, saving 164,560,896 bytes per layer. The pilot transfers each selected gate/up tensor and its companion per-expert scale from the Q3 file into the matching original IQ4 file. Every tensor in each constructed file was reopened and matched to the expected donor by SHA-256, shape, type and byte count.

Same Unsloth `8832860` CPU runtime, two threads, FP16 KV, WikiText-2 test first 512 tokens, 255 scored positions versus the same official BF16 reference. These are measurements of actual GGUF prototypes. The new representation has not been implemented in the LeanSonic loader or decode kernels.

| Distribution | Model bytes | Mean KL | Worst KL | Top-1 / 255 | PPL |
|---|---:|---:|---:|---:|---:|
| Original IQ4 | 15,733,755,232 | 0.003003 | 0.040391 | 248 | 5.299664 |
| IQ4 + layer 1 gate/up from Q3 | 15,569,194,336 | 0.003346 | 0.052336 | 246 | 5.295852 |
| IQ4 + layers 1–3 gate/up from Q3 | 15,240,072,544 | 0.005595 | 0.141788 | 249 | 5.253679 |
| Original Q3 | 13,631,778,016 | 0.013232 | 0.280897 | 243 | 5.309377 |

Use the original GGUF artifacts and CUDA runtime pinned in [the comparator issue](https://github.com/tjm8874/LeanSonic/issues/78#issuecomment-7212689108). Set `PYTHONPATH` to the official llama.cpp checkout's `gguf-py` directory and run these commands from a scratch directory with NumPy and the package's dependencies. Each output requires a new filename. The SHA-256 digests and tensor selections are in the [run data](inkling-mixed-iq2-pilot.json).

```sh
python "$LEANSONIC/tools/inkling_q3_probe/inventory.py" "$IQ4" "$Q3"
python "$LEANSONIC/tools/inkling_q3_probe/mix.py" "$IQ4" "$Q3" layer1.gguf 1
python "$LEANSONIC/tools/inkling_q3_probe/mix.py" "$IQ4" "$Q3" layers123.gguf 1 2 3
CUDA_VISIBLE_DEVICES='' "$NATIVE/llama-perplexity" -m layers123.gguf \
  -f "$WIKITEXT_TEST" --kl-divergence --kl-divergence-base "$FIRST_CHUNK_REF" \
  -c 512 -b 512 -ub 512 -ctk f16 -ctv f16 -ngl 0 -t 2 --no-warmup --chunks 1
```

The three-layer form is worth testing over the full test and validation corpora before considering a native storage or CUDA change. The higher worst KL on this first chunk is relevant even though observed top-1 and PPL improved.
