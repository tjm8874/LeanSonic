# Reversible neighbor coding for expert scale planes

The scale directly above a value and the previous scale in its row select a probability table. The row width comes from the tensor shape. The 16-byte header stores up to 11 distinct quantile boundaries. Decoder history resets at each 128 KiB tile. Raw and flat rANS remain available, and the packer measures the three encodings before selecting a plane.

Both hosts independently built a 15,770,570,696-byte shard with SHA-256 `2f75e321d7e36ebc376548fecf2fbc3e148da5e98d1509e354e5e6f46cd90e3f`. With the same 3,386,494 tokenizer bytes the distribution is 15,773,957,190 bytes, down 79,434,552 from #109. The public IQ4_XS mainline file is 15,733,756,512 bytes. All 40 NVFP4 scale planes used neighbor coding. The 8 MXFP8 scale planes retained the flat encoding. All 999 tensors were reopened. The selected Rust frames matched an independent C++ implementation byte-for-byte and the other 959 tensor payloads matched the parent shard.

Starting from the official Inkling-2-Mini checkpoint, use the normal text packer with `--prepack-head --prepack-attention --compact-q8-scales --compact-q8-codes --prepack-embedding --prepack-mxfp8 --contextual-nvfp4 --neighbor-scales`. Include the complete `tiktoken` directory. Each tensor is measured including header, frame and padding overhead. See [per-tensor sizes, SHA-256 and observations](inkling-neighbor-scales.json).

## GPU output parity

Using the #113 cubin, PDL enabled, greedy sampling, FP16 KV and the same 32-token prompt and 16 forced continuation IDs, the previous and new representation produced the same 17 × 200,058 FP32 logits (13,603,944 bytes).
- PRO4500: `68538b83e50b2df69e032bc05740bd5f019a9dd279f7077905553fe792df3489`.
- 5060Ti: `68538b83e50b2df69e032bc05740bd5f019a9dd279f7077905553fe792df3489`.

## Loading and generation

Five mirrored cycles used fresh processes, the same CLI and cubin, and fully page-cached model files before each measurement. Each of the ten processes per arm ran three 64-step generations. Decode reports the mean of the last two runs. All 64 IDs matched in every run.

| GPU | Previous load / neighbor load (s) | Previous decode64 / neighbor decode64 (ms) |
|---|---:|---:|
| PRO4500 | 5.132 / 5.551 | 390.38 / 390.35 |
| 5060Ti | 10.458 / 11.226 | 1266.46 / 1266.48 |

## Matched BF16 quality

| WikiText-2 split | scored positions | mean KL | maximum KL | top-1 matches | PPL |
|---|---:|---:|---:|---:|---:|
| test | 25500 | 0.0031584 | 0.6809773 | 24865 | 7.0729555 |
| valid | 25500 | 0.0027607 | 0.3524361 | 24963 | 6.5046210 |
