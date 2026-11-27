# Lossless context coding of NVFP4 expert weights

The opt-in packer groups four 8-byte NVFP4 blocks. For each column, the largest absolute code in earlier bytes selects one of eight frequency tables. Each table describes all 256 possible byte values, including both signs of zero. Four rANS states encode each 128 KiB tile. The loader decodes into the same VRAM or pinned-host tier used for raw weights.

| Text distribution including tokenizer | Bytes |
|---|---:|
| Native compact with raw NVFP4 codes | 15,982,139,902 |
| Byte rANS (#102) | 15,901,490,534 |
| Block-context rANS | 15,853,391,742 |

The extra reduction against #102 is 48,098,792 bytes. The shard is 15,850,005,248 bytes and the tokenizer is 3,386,494 bytes. An independent C++ encoder reopened all forty frames and decoded all 10,192,158,720 original expert bytes successfully. The Rust production packer emitted the same forty compressed payload hashes. All 959 other tensor payloads and their shapes and types matched the input shard. The [per-tensor data](inkling-nvfp4-context.json) include encoded sizes and hashes. The single-thread decoder timings in this file include comparison with raw input and are not engine load times.

## Reproduce

Start with the compact text distribution from the [byte rANS experiment](inkling-nvfp4-code-storage.md), using its **input** with raw NVFP4 expert codes. With the tokenizer from that input in the output directory:

```sh
python3 tools/pack_inkling_text.py "$BASE/model-00001-of-00001.safetensors" "$OUT/model-00001-of-00001.safetensors" --contextual-nvfp4
sha256sum "$OUT/model-00001-of-00001.safetensors"
cargo test --workspace
```

The verified shard SHA-256 is `f33ba0663d5c3968d77a0b4fe648627d9dc298d7796863cb1db48e63f104dc92`. Files with the new `.nvctx_v1` suffix require this reader. The option only stores an encoded tensor if the full payload including tables and padding is smaller. Existing raw and byte-coded models remain readable.

Independent repacking on the second host yielded the same production shard hash. On RTX 5070 Ti and RTX 5060 Ti the flat and contextual packages both produced 13,603,944 identical FP32 logit bytes (17 positions × 200,058 entries). The common SHA-256 was `68538b83e50b2df69e032bc05740bd5f019a9dd279f7077905553fe792df3489`. This comparison used the pre-staging binary with only the new reader and FP16 KV.

## Startup and steady state

On an RTX 5060 Ti the same V13-based reader loaded #102's flat package in 9.01 s and the context package in 10.00 s (+0.98 s). Both files were fully resident in the OS page cache before every process (verified with mincore). Three mirrored cycles yielded six fresh processes per format. The final two warm 64-step decode iterations per process averaged 1331.42 and 1331.50 ms on a 32-token README prefix. All processes returned the same 64 greedy IDs. Keep the smaller encoding optional for deployments with different restart and storage budgets.
