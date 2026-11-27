# Store DSpark in the same planar Q8_64 representation used at inference

The official Inkling-2-Mini DSpark sidecar occupies 635,801,416 bytes. Its prepacked replacement occupies **386,970,072 bytes**, saving 248,831,344 bytes (39.1%). The offline packer uses the same Rust quantizer as the BF16 loader. An independent NumPy calculation compared every one of the 265,420,800 signed codes and 4,147,200 FP32 scales with the packed file. The verifier also checked the unquantized split of the Markov recurrent projection and all retained tensors, including the Markov head and hidden-output weights. All 59 output tensors passed. A second complete build produced the identical file.

| Inkling target with tokenizer | Target alone (B) | With official DSpark (B) | With prepacked DSpark (B) |
|---|---:|---:|---:|
| Direct control | 13,479,710,294 | 14,115,511,710 | 13,866,680,366 |
| Fixed seven-level | 11,609,363,902 | 12,245,165,318 | 11,996,333,974 |
| Variable palette | 11,605,970,958 | 12,241,772,374 | 11,992,941,030 |
| Ternary | 7,718,950,382 | 8,354,751,798 | 8,105,920,454 |
| Ternary with refitted scales | 7,734,665,510 | 8,370,466,926 | 8,121,635,582 |

The table sums measured files. Earlier speculation-off comparisons use the target alone. DSpark-enabled quality, accepted drafts, speed and peak memory require a compatible loader. Claude independently reproduced DSpark behavior with the official BF16 file alongside the target. [Issue discussion and replay results](https://github.com/tjm8874/LeanSonic/issues/6#issuecomment-7221258222).

## Storage layout

For each of the three layers, store five projection pairs. The prefixes are layers.{i}.self_attn.qkv_proj, layers.{i}.self_attn.kv_ctx_proj, layers.{i}.self_attn.o_proj, layers.{i}.mlp.gate_up_proj and layers.{i}.mlp.down_proj. Their orders are Q/K/V, context K/V, output, gate/up and down. Each prefix has .q8_64.codes (I8, rows × columns) and .q8_64.scales (F32, rows × columns/64). The sixteenth pair uses prefix markov_head.rnn_proj.jctx. It covers the trailing 4,608 elements of each original 5,120-element row. The leading 512 are retained as BF16 under markov_head.rnn_proj.j01.weight, shape [768,512]. The loader can expand j01 to FP32 exactly as before. Original BF16 markov_w1 and hidden_out tensors retain their names and payloads.

## Reproduce

Use the official binary named mtp.safetensors with SHA-256 `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2`. Choose a fresh output filename on the destination filesystem. Building requires the Rust workspace and Python 3.11+; independent verification also uses NumPy.

```sh
python3 tools/pack_inkling_drafter.py "$OFFICIAL_MTP" "$NEW_MTP"
python3 docs/benchmarks/inkling-dspark-package/verify.py "$OFFICIAL_MTP" "$NEW_MTP" > check.json
cmp check.json docs/benchmarks/inkling-dspark-package/verify.json
```

The packed file SHA-256 is `53f857d1a7deaa1b3e22aca83d9db42dab3c9e507c41412b8b37e998c177ca3e`. [Complete validation receipt](verify.json).
