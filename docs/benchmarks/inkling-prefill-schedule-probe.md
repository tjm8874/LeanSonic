# Compare short prompt tails against a BF16 engine

These research tools use 10 selected prompts of lengths 63, 64, 65, 66, 71, 76, 82, 89, 91 and 95 taken from the committed README fixture. Each history receives two fixed decode inputs. Each binary writes 20 complete FP32 vocabulary distributions in slot-major order. The BF16 program uses one call per prompt and then two individual decode calls. The native program compares one prefill with manually splitting the same input at row 64 on the same engine. It requires a GPU for which the startup log confirms width 256. The selected inputs investigate #107 and do not give corpus-level perplexity.

Build the native example and the sm_120a CUDA image from this branch. Compile the BF16 probe with the headers and shared libraries from the pinned checkout in [the reference instructions](../inkling-reference.md). Set the environment before starting each program.

```sh
cargo build --release -p leansonic-core --example inkling_prefill_boundary --example inkling_tail_timing
nvcc -std=c++17 -O3 -arch=sm_120a --cubin kernels/inkling2_sm120.cu -o round.cubin
c++ -std=c++17 -O2 tools/inkling_prefill_boundary_bf16.cpp \
  -I"$LLAMA/include" -I"$LLAMA/ggml/include" -L"$LIB" -Wl,-rpath,"$LIB" \
  -lllama -lggml -lggml-base -o schedule-bf16
ids=docs/benchmarks/inputs/inkling-readme1024.ids
CUDA_VISIBLE_DEVICES='' ./schedule-bf16 "$BF16_GGUF" "$ids" cpu.f32 0
export LEANSONIC_CUBIN_PATH="$PWD/round.cubin" LEANSONIC_PDL=1 LEANSONIC_KV=fp16 LEANSONIC_PREFILL_CHUNK=256
target/release/examples/inkling_prefill_boundary "$MODEL" "$ids" default-run
LEANSONIC_TC_PREFILL=0 target/release/examples/inkling_prefill_boundary "$MODEL" "$ids" scalar-run
python3 tools/score_prefill_boundary.py cpu.f32 default-run
python3 tools/score_prefill_boundary.py cpu.f32 scalar-run
```

For the full-offload BF16 control, select the PRO 5000, supply `99` as the final argument, and score the resulting reference explicitly. For the A/B prefill timings, build the identical `inkling_tail_timing.rs` against each engine revision, select the same GPU, set `LEANSONIC_SEQS=1`, and pass the model directory and the same ID fixture. The timer emits run, input length, prefill milliseconds, eight forced decode steps in milliseconds and first predicted ID. Exclude run 0. The three process cycles use ABBA, BAAB, ABBA. The wide-workspace experiment uses automatic width; the narrow-workspace experiment sets `LEANSONIC_PREFILL_CHUNK=64`.
