# Original Unsloth IQ4_XS versus LeanSonic on Inkling-2-Mini

Both engines processed the same 1,024 prompt IDs, then 32 identical teacher-forced inputs with greedy sampling after each forward. Context was 8,192, one sequence, FP16 KV, speculation off. Five alternating ABBA/BAAB cycles used ten fresh processes per engine and GPU. Each process served five requests. Phase means exclude its first request (40 warm requests per engine). Peak bytes are the maximum per-process GPU memory observed during load and inference, polled via NVML every approximately 4 ms. They are sampled high-water marks.

| GPU | Engine | Prefill ms / tok/s | Decode32 ms / tok/s | Peak GPU bytes | Distribution bytes | Greedy matches / 32 |
|---|---|---:|---:|---:|---:|---:|
| PRO4500 | LeanSonic #117 | 120.611 / 8,490.1 | 200.999 / 159.21 | 18,111,004,672 | 15,724,161,014 | 32 |
| PRO4500 | Unsloth original IQ4_XS | 458.140 / 2,235.1 | 413.169 / 77.45 | 18,928,893,952 | 15,733,755,232 | 30 |

LeanSonic used the #117 shard (including its separate tokenizer for distribution size), FP16 KV, and the PDL kernels. The comparison uses the original 48-slot GGUF and its matching prebuilt CUDA backend, with 25/25 reported layers on GPU. Fixed continuations keep the two engines on exactly the same history when their predictions differ. The match count uses the committed continuation; corpus KL and perplexity need the separate BF16 reference. [Quality control on the original release](https://github.com/tjm8874/LeanSonic/issues/78#issuecomment-7212074361).

## Reproduce

Use LeanSonic engine revision `64200ed`, build `inkling_fixed_decode`, and select the #117 package with shard SHA-256 `29b33880bcf7780d4a8885052ea640abedfd77b048cede15c8ee04ff4568b1d6`. Select the same GPU for both processes. The pinned GGUF SHA-256 is `a861d21e7c6d7bc69d8cdcdfb9eb195fd37ef1cd2ab31499db61cd7d30d9b3a1`. Get the Linux x64 CUDA 13 newer binaries and matching source headers from [the publisher’s release](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8). Use source commit `883286076efe5b85fddb31d9cc7035a6b698ba9b` and [the original GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/ae9327c5fbc7333116ba32eee78bb5d2bbefae4f/Inkling-2-Mini-UD-IQ4_XS.gguf).

With `NATIVE` pointing at the extracted release binaries, `HEADERS` at its source root, and `LEAN` at the #117 benchmark executable, run these commands from the checkout containing the reporting driver. Build the SM120a PDL-capable cubin from the matching Inkling source.

```sh
c++ -std=c++17 -O2 tools/benchmark_inkling_bf16.cpp \
  -I"$HEADERS/include" -I"$HEADERS/ggml/include" -L"$NATIVE" \
  -Wl,-rpath,"$NATIVE" -l:libllama.so.1 -l:libggml.so.1 -l:libggml-base.so.1 -o native-fixed
ids=docs/benchmarks/inputs/inkling-readme1024.ids
next=docs/benchmarks/inputs/inkling-continuation32.ids
LEANSONIC_PDL=1 LEANSONIC_KV=fp16 "$LEAN" "$MODEL" "$ids" "$next"
LEANSONIC_LLAMA_BACKEND_DIR="$NATIVE" LD_LIBRARY_PATH="$NATIVE:${LD_LIBRARY_PATH:-}" \
  ./native-fixed "$GGUF" "$ids" "$next" --report-ids
```

The original timing executable resolved the backend directory from its linked libggml; the reproduction command supplies it explicitly. The [per-process samples](inkling-original-iq4.json) include the raw phase durations, predicted IDs, observed memory peaks and hashes of the timed executables and input fixtures.
