# Qwen3.8-Flash-Next on RTX PRO 4500 32GB and PRO 5000 72GB

Two resident engines took the same token IDs over stdin, one request at a time, with a 256-token context and exactly 32 greedy output tokens. LeanSonic used the restored MLX Q4 checkpoint from the [lossless distribution](../qwen38-pack/README.md) and the pinned engine commit `9b3e4f2aabc5e33e73b34e0aea8e031b48aba0fe`. Basalt used its public v0.5.7 Linux x64 release and the `Qwen3.8-FN-UD-Q4_K_XL.basalt` checkpoint (revision `65de5a9e636e8bbe1200cbcd6eb8aa421e6a66c6`, 115,307,561,176 bytes). These checkpoints represent the same base model with different quantization and runtime arithmetic. The Q4 weights used by LeanSonic occupy 113,209,682,735 shard bytes and require 105,964,842,612 total distribution bytes using the dictionary pack.

## RTX PRO 4500 latency and sampled memory

Each arm ran one warmup pair followed by three measured pairs in the order 16 input tokens, then 128. The short input is the Japanese fixture. The long input repeats all four token fixtures and stops at 128 tokens. The table uses medians of the three measured requests. Prefill throughput includes submission through arrival of the first output token. Decode throughput covers the next 31 token intervals. The final column measures the full request through the engine's DONE line. Peak VRAM is the maximum process usage observed by NVML at a nominal 25 ms interval.

| Engine and placement | Input | First token (ms) | Prefill (tok/s) | Decode (tok/s) | Through DONE (ms) | Sampled peak (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LeanSonic, 18/48 expert layers in VRAM | 16 | 767.27 | 20.85 | 20.50 | 2279.83 | 30.07 |
| Basalt, adaptive expert placement | 16 | 172.96 | 92.50 | 92.54 | 507.50 | 30.61 |
| Basalt, fixed expert placement | 16 | 380.58 | 42.04 | 44.17 | 1082.93 | 30.60 |
| LeanSonic, 18/48 expert layers in VRAM | 128 | 6139.82 | 20.85 | 20.43 | 7657.34 | 30.07 |
| Basalt, adaptive expert placement | 128 | 357.04 | 358.50 | 74.29 | 758.09 | 30.61 |
| Basalt, fixed expert placement | 128 | 393.97 | 324.89 | 43.39 | 1107.44 | 30.60 |

All three measured repeats produced identical tokens within each arm and prompt. For the short prompt, Basalt with adaptive placement first diverged from its own warmup on output token 8. Fixed placement and LeanSonic reproduced their warmup sequences. The two Basalt modes differ in expert placement and can take different arithmetic paths. These observations come from two prompts. The measurements are for warmed resident processes on a shared host, and the VRAM sampling can miss short peaks. Exact token traces and individual times are in [pro4500.json](pro4500.json).

## RTX PRO 5000 72GB

The same binaries, fixture IDs, context size and request schedule ran on the 72GB GPU. LeanSonic kept 45/48 expert layers in VRAM on this checkout. Individual timings and output IDs are in [pro5000.json](pro5000.json).

| Engine and placement | Input | First token (ms) | Prefill (tok/s) | Decode (tok/s) | Through DONE (ms) | Sampled peak (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LeanSonic | 16 | 219.78 | 72.80 | 69.43 | 666.32 | 69.70 |
| Basalt, adaptive | 16 | 116.91 | 136.86 | 134.42 | 347.77 | 70.44 |
| Basalt, fixed | 16 | 193.54 | 82.67 | 83.68 | 564.42 | 70.44 |
| LeanSonic | 128 | 1760.95 | 72.69 | 68.76 | 2211.77 | 69.70 |
| Basalt, adaptive | 128 | 141.53 | 904.40 | 135.69 | 370.23 | 70.44 |
| Basalt, fixed | 128 | 145.14 | 881.89 | 115.69 | 413.09 | 70.44 |

## Wave 1 with native Engram dictionaries

The Wave 1 loader was subsequently tested on the same GPUs and exact input IDs with the same resident streaming runner and 32 output tokens. The Basalt values below come from the earlier v0.5.7 adaptive-placement run. LeanSonic uses the MLX Q4 weights with lossless Engram dictionaries, while Basalt uses its own UD-Q4_K_XL model. The runnable LeanSonic directory occupies 110,833,479,038 bytes including auxiliary files. The Basalt model file occupies 115,307,561,176 bytes. Output quality and quantization can differ. [Wave 1 input IDs, output IDs, peak VRAM and timing samples](../qwen38-pack/direct-results.json)

| GPU | Input | Prefill LeanSonic / Basalt (tok/s) | Decode LeanSonic / Basalt (tok/s) | Peak LeanSonic / Basalt (GiB) |
| --- | ---: | ---: | ---: | ---: |
| PRO 5000 72 GB | 16 | 173.02 / 136.86 | 162.91 / 134.42 | 70.26 / 70.44 |
| PRO 5000 72 GB | 128 | 171.77 / 904.40 | 158.84 / 135.69 | 70.26 / 70.44 |
| PRO 4500 32 GB | 16 | 22.17 / 92.50 | 21.88 / 92.54 | 29.79 / 30.61 |
| PRO 4500 32 GB | 128 | 22.15 / 358.50 | 21.84 / 74.29 | 29.79 / 30.61 |

On 72 GB, LeanSonic had higher resident decode throughput for both inputs. Basalt's 128-token prefill was 5.27× faster. On 32 GB, the tested Basalt configuration led on both metrics for both inputs.

## Teacher-forced cross-check

We also ran the stock Basalt PPL gate with fixed placement over four fixtures, truncating each to the same ten-token window. Basalt scores positions 5–8 in each window, giving 16 target distributions. Against the original LeanSonic Q4 logits, Basalt matched the top prediction at 14/16 positions. On these targets, PPL was 9.5383 for Basalt and 13.0978 for the Q4 reference. Mean/max KL as computed by the Basalt gate were 0.22557/0.92462. This gate quantizes its reference log probabilities to uint16 and drops terms with reference log probability at or below −16 when summing KL. The maximum measured export error was 0.000299 in log probability and at most 0.001722 of reference probability mass fell in that tail. The BF16 reference evaluation is a separate measurement.

## Reproduce on Linux

Use the fixtures from `docs/benchmarks/qwen38-pack`, NumPy and the Python NVML bindings. Set `CUDA_VISIBLE_DEVICES` to the UUID of a reserved GPU. Build the Rust runner as a Cargo example in a scratch checkout of the pinned engine and point `LEANSONIC_QWEN38_CUBIN_PATH` to the matching sm_120a cubin. The Basalt release archive SHA-256 was `a390c775c41d54454ca21c1e547855bf6c251732533ca387e8b890e538090525`. The Basalt checkpoint SHA-256 was `1e33bd4837ec26f5771509e3f8e2ab911af4562264e6fb8e0993d6711da00c01`.

```sh
d=docs/benchmarks/qwen38-engine-comparison
cp "$d/q38_stream.rs" crates/leansonic-core/examples/
cargo build --release -p leansonic-core --example q38_stream
ids=docs/benchmarks/qwen38-pack/fixtures.ids
python3 "$d/compare_stream.py" "$ids" lean.log lean.json target/release/examples/q38_stream "$RESTORED" 256

# Use bash for the argument array. Device 0 is the sole visible GPU.
flags=(--model "$BASALT_MODEL" --serve --gpu 0 --max-context 256 --kv fp16 --ple-io ram
       --expert-cache auto --prefill auto --spec 0 --suffix-draft 0 --prompt-cache 0
       --prompt-cache-every 0 --prompt-cache-root 0 --conversation-cache-mib 0)
python3 "$d/compare_stream.py" "$ids" basalt.log basalt.json "$BASALT" "${flags[@]}" --temperature 0
python3 "$d/compare_stream.py" "$ids" fixed.log fixed.json "$BASALT" "${flags[@]}" --greedy --adapt-every 0
```

To run the diagnostic from a scratch directory containing `fixtures.ids`, pass the original teacher output directory built by the [regression suite](../qwen38-pack/README.md) and the absolute path to the PPL driver.

```sh
python3 "$PPL_DRIVER" "$ORIGINAL_TEACHER_OUTPUT" quality.json "$BASALT" "${flags[@]}" --greedy --adapt-every 0
```
