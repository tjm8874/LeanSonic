# Smaller reversible expert scales with row and column history

The encoder evaluates the mean of up to eight previously decoded scales to the left, and separately up to eight above the current value. Each feature is an integer in 0–2040. The tensor stores separate quantile boundaries for its two axes. History resets at each 128 KiB tile, and horizontal history resets at row boundaries. The decoder chooses the same table using a rolling sum. The engine still consumes the original scale bytes.

| Text distribution including tokenizer | Bytes |
|---|---:|
| Immediate-neighbor scales and compact BF16 controllers (#116) | 15,750,591,862 |
| Eight-neighbor scales | 15,724,161,014 |
| Public Unsloth mainline IQ4_XS | 15,733,756,512 |

The two GPU hosts independently built the same 15,720,774,520-byte shard; the tokenizer adds 3,386,494 bytes. The new shard SHA-256 is `29b33880bcf7780d4a8885052ea640abedfd77b048cede15c8ee04ff4568b1d6`. Forty NVFP4 scale tensors selected the new encoding. All eight MXFP8 scale tensors retained flat rANS. The 40 Rust outputs matched an independent C++ implementation that reopened all 48 frames and reproduced all 1,401,421,824 original scale bytes. All 959 unaffected tensor payloads matched #116.

Build `64200ed` and pack the official shard with `tools/pack_inkling_text.py` using `--prepack-head --prepack-attention --compact-q8-codes --compact-q8-scales --prepack-embedding --prepack-mxfp8 --contextual-nvfp4 --compact-controls --averaged-scales`. Copy the full tokenizer directory. The packer measures raw, flat, direct neighbor and eight-neighbor frame lengths before selecting each tensor. [Per-plane and per-process results](inkling-mean-scales.json).

The public packer command above also reproduced the same complete shard hash directly from the official checkpoint.

## All-vocabulary logit parity

The #113 PDL cubin and the same reader loaded each package. The 32-token prompt prefix, 16 forced continuation steps, FP16 KV and greedy sampling yielded 17 × 200,058 FP32 values. The parent and new formats matched byte-for-byte on each listed GPU.

- PRO4500: `68538b83e50b2df69e032bc05740bd5f019a9dd279f7077905553fe792df3489`.
- 5060Ti: `68538b83e50b2df69e032bc05740bd5f019a9dd279f7077905553fe792df3489`.

## Restart cost and warm 64-token decode

Each row includes five alternating mirrored cycles, ten fresh processes per format. All model pages were resident in the OS cache before every process. Three 64-step generations per process used the same 32-token input; decode averages the last two runs. Every process emitted identical IDs.

| GPU | #116 / new load (s) | #116 / new decode64 (ms) |
|---|---:|---:|
| PRO4500 | 5.840 / 5.792 | 390.70 / 390.64 |
| 5060Ti | 11.346 / 11.499 | 1266.40 / 1266.47 |

## Test corpus on RTX 5070 Ti

The candidate matched 24,865/25,500 top tokens at PPL 7.07295554. Mean KL was 0.0031584260 and worst KL was 0.6809773323 at chunk 69, position 353. The 100-chunk BF16 reference had SHA-256 `895fac2136c06dce1854568c83eb76a0225a05fab3c37897a12fec1d56949b97` on both hosts.
