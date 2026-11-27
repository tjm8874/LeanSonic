# NVFP4 prefill tile comparison at 1,024 tokens

On RTX 5060 Ti, the 128- and 160-token dispatches produced identical full-vocabulary FP32 logits at all **132 positions** (4 inputs × 33 positions, 200,058 logits each). The 105,630,624 bytes from each run have SHA-256 a71f51347b117e25e19e3eeaa93af223e0f80807054ba878ede5cc16fd719162. Each row was checked for finite values and its argmax against the engine. [Inputs](fixtures.ids) · [Binary and output digests](parity.json).

Use the 571-slice class-frame model and replay environment described in the [parent benchmark](../README.md). Compile check.rs as a leansonic-core release example at a1592f3. The fixtures are the README prompt with native vLLM history, and three contiguous 1,056-token sequences starting at WikiText test chunks 0 and 35 and validation chunk 84. The first 1,024 tokens form the prefill. Feed the remaining 32 tokens as fixed continuations. From separate processes run:

```sh
env -u LEANSONIC_NVFP4_T160_MIN ./check MODEL fixtures.ids tile128.f32
LEANSONIC_NVFP4_T160_MIN=1024 ./check MODEL fixtures.ids tile160.f32
cmp tile128.f32 tile160.f32
```

The existing 128-token grouped NVFP4 kernel tests `wt * 64 < live` and `wt * 64 + nt * 8 < live` before its MMA work. An average of 64 routed tokens per expert does not imply 64 unused MMA rows in every 128-row tile. A 64-token tile can reduce shared-memory and idle-warp costs, but can also reread the same expert weights when an expert receives more than 64 tokens. Measuring the per-expert counts and phase timings distinguishes these effects.
