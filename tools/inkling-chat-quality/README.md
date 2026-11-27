# Inkling chat response-prefix fidelity

This fixture contains 12 English prompts from the project's benchmark suite and eight original Japanese prompts. Each has a fixed response or opening fragment. The English-prompt group includes an English-to-French translation and the Japanese-prompt group includes a Japanese-to-English translation. The groups are defined by prompt language. The observations measure the distribution that predicts the next human-specified token. They do not score the usefulness or correctness of a freely generated answer.

Each engine prefills the same message history, including the assistant text header at thinking effort zero, and receives one target token per step. We measure the distribution *before* feeding that target. There are 613 scored positions in 20 cases. At the end, one English case and one Japanese case are repeated after resetting engine state. The reports require those two repeated logit files to match their originals byte-for-byte; their positions are excluded from the aggregates.

Use the [BF16 reference build](../../docs/inkling-reference.md), the original model's tokenizer files, and the [two packages under comparison](https://github.com/tjm8874/LeanSonic/pull/130). For the reported comparison, copy `crates/leansonic-core/examples/inkling_chat_quality.rs` to the same path in a separate LeanSonic checkout at `a9872cc`. Build with `cargo build --release -p leansonic-core --example inkling_chat_quality`. Use the SM120a cubin built at `48b7ec2`. Set `LEAN` to the absolute example binary path and `TOOLS` to the absolute path of this directory. The reference writer uses little-endian Linux on NVIDIA. From a fresh working directory, run:

```bash
c++ -O2 -std=c++17 -I"$LLAMA/include" -I"$LLAMA/ggml/include" "$TOOLS/reference.cpp" \
    -L"$BUILD/bin" -Wl,-rpath,"$BUILD/bin" -lllama -lggml -lggml-base -o reference
python3 "$TOOLS/prepare.py" fixture
root=$PWD
mapfile -t cases < fixture/order.txt
"$LEAN" tokenize "$CANONICAL" fixture "${cases[@]}"
python3 "$TOOLS/check_tokens.py" "$OFFICIAL" fixture
export LEANSONIC_KV=fp16 LEANSONIC_PDL=1 LEANSONIC_SEQS=1 LEANSONIC_PREFILL_CHUNK=64
export LEANSONIC_CUBIN_PATH="$CUBIN"
mkdir bf16 parent budget
(cd bf16 && "$root/reference" "$GGUF_BF16" "$root/fixture" "${cases[@]}")
(cd parent && "$LEAN" dump "$CANONICAL" "$root/fixture" "${cases[@]}")
(cd budget && "$LEAN" dump "$BUDGET" "$root/fixture" "${cases[@]}")
for arm in parent budget; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 "$TOOLS/score.py" fixture bf16 "$arm" "$arm.json"
done
```

The independent tokenizer check uses Jinja2 and Hugging Face tokenizers. It verifies the complete user/assistant conversation against the original chat template and tokenizer.json. The scorer requires NumPy. It checks lengths and finite logits, normalizes all 200,058 FP32 logits in float64, and computes KL(BF16 || candidate), without truncating the vocabulary. PPL is the exponential of the mean target negative log-likelihood, pooled over tokens. The reports include both token hashes for each case, token offsets, top-1 IDs, KL and target loss at every position. With 22 cases including reset controls, allow approximately 1.7 GB for all three raw logit dumps.
