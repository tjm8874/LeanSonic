# Reproduce the bilingual chat tasks

Use a Linux GPU host and an empty scratch directory. Set `TOOLS` to this directory, `CLI` to an absolute LeanSonic binary path, `OFFICIAL` to the original Hugging Face model directory, and `MODEL` to the LeanSonic package under test. Pin one GPU. The recorded HTTP runs used `5f1c7ff` before the JSON fix and `277136d` after it, and the same SM120a cubin from `48b7ec2` (SHA-256 `3714fd830d6cfedbf6954f21cad460489173be36566b0cb4b78173d42da87f0b`). The later `inkling_chat_quality` example is an independent CPU tokenizer check. Build it with `cargo build --release -p leansonic-core --example inkling_chat_quality` and set `TOKENIZER_DRIVER` to its absolute path.

The 12 prompts and expected values are in [fixtures.json](fixtures.json). Python preparation and native scoring use Jinja2 and Hugging Face tokenizers. The HTTP runner uses the standard library and discovers its own ephemeral loopback port through Linux procfs.

```bash
python3 "$TOOLS/prepare.py" "$OFFICIAL" ids
python3 "$TOOLS/check_ids.py" "$TOKENIZER_DRIVER" "$MODEL" ids
export LEANSONIC_CTX=512 LEANSONIC_KV=fp16 LEANSONIC_SEQS=1
export LEANSONIC_PDL=1 LEANSONIC_PREFILL_CHUNK=64 LEANSONIC_CUBIN_PATH="$CUBIN"
mkdir cap128 cap384
(cd cap128 && ln -s ../ids ids && python3 "$TOOLS/suite.py" "$CLI" "$MODEL" --cap 128 --compare-encodings)
(cd cap384 && ln -s ../ids ids && python3 "$TOOLS/suite.py" "$CLI" "$MODEL" --cap 384)
```

For each GGUF, build the driver against its corresponding source headers and CUDA libraries. The BF16 reference used llama.cpp `29b52d3` with 49/49 layers offloaded, and the two original quantized models used the [Unsloth release at 8832860](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8) with 25/25 layers offloaded. Set `LLAMA` to the matching source checkout, `LIB` to the library directory and `GGUF` to its model. For the Unsloth release substitute `-l:libllama.so.1 -l:libggml.so.1 -l:libggml-base.so.1` for the three link arguments.

```bash
c++ -O2 -std=c++17 -I"$LLAMA/include" -I"$LLAMA/ggml/include" "$TOOLS/greedy.cpp" \
  -L"$LIB" -Wl,-rpath,"$LIB" -lllama -lggml -lggml-base -o greedy
mapfile -t cases < <(python3 -c 'import json,sys;print(*[r["id"] for r in json.load(open(sys.argv[1]))],sep="\n")' "$TOOLS/fixtures.json")
LEANSONIC_LLAMA_BACKEND_DIR="$LIB" LD_LIBRARY_PATH="$LIB:${LD_LIBRARY_PATH:-}" \
  ./greedy "$GGUF" ids "${cases[@]}" "${cases[0]}" >generated.ids 2>backend.log
grep -E 'offloaded (25/25|49/49) layers to GPU' backend.log
python3 "$TOOLS/score_native.py" "$OFFICIAL" generated.ids >scores.json
```

The scorer retains visible answers and hashes the complete generated sequence (decimal IDs separated by ASCII spaces, followed by a newline). It verifies that the final repetition matches the first complete sequence. Model/shard hashes, original prompt file hashes and all visible responses are in the [measurement record](../../docs/benchmarks/inkling-functional-chat.json).
