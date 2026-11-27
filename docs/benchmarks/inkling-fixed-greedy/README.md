# Buffered target-only decode on Blackwell

Both calls use the same warmed process, model, prompt and cubin, resetting before every request. The scalar API synchronizes each token. The fixed-length API uses the pinned GPU token ring in batches of at most 64. Five cycles alternate ABBA and BAAB. For each prompt/output length, the driver compares every token, the returned next ID, all 200,058 final FP32 values and one next step after the request. It also rejects a request for usize::MAX tokens before committing any state.

## 5060Ti

| Prompt + output | Scalar ms | Buffered ms | Time saved | Paired time saved, 95% interval (ms) |
|---|---:|---:|---:|---:|
| 32+1 | 9.982 | 9.982 | -0.000% | [-0.005, 0.005] |
| 32+63 | 632.814 | 632.697 | 0.018% | [0.028, 0.210] |
| 32+64 | 642.945 | 642.761 | 0.029% | [0.079, 0.307] |
| 32+65 | 653.046 | 652.933 | 0.017% | [0.021, 0.232] |
| 1024+129 | 1567.884 | 1567.420 | 0.030% | [0.319, 0.607] |
| 2046+65 | 812.653 | 812.492 | 0.020% | [0.108, 0.219] |
| 2048+64 | 800.159 | 800.009 | 0.019% | [0.095, 0.205] |

5060Ti: all 160 request checks passed, including the empty generation and the requests crossing position 2,048. [Every phase time, token sequence and final FP32 hash](5060Ti.json).

Build the [regression driver](../../../crates/leansonic-core/examples/inkling_buffered_replay.rs) with `cargo build --release -p leansonic-core --example inkling_buffered_replay`. Run it with the two-bank model directory and the comma-separated 1,024 README token IDs from [the fixed-input replay](../inkling-nvfp4-scale-split/direct/fixed-q3.md). Use FP16 KV, one sequence, PDL and global PDL, direct expert reads, compact device indices and the cubin fingerprint in the per-GPU file. The driver extends the prompt by cycling those exact IDs for the two boundary cases.
