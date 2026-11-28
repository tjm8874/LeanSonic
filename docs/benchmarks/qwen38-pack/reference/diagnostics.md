# Trace the error before the final projection

This diagnostic branch instruments the Wave 1 engine. The normal execution mode records the embedding output and post-MLP residual of each layer in a GPU buffer. Optional teacher forcing copies the official incremental BF16 residual into the start of each native layer on all nine prefix tokens, including tokens that populate attention and recurrent caches. Every native layer then executes its own operations on the supplied state. Each recorded boundary is a four-stream, 10,240-element FP32 vector. These binaries are for numerical analysis; their timings include diagnostic copies.

In a scratch directory, use `official-bf16` for the pinned upstream model, `restored` for the original Q4 safetensors, and `direct-engram` for the directly loadable Q4 checkpoint. Use the same `fixtures.ids`, `score-bf16`, `bf16-wave1-quality.json` and `wave1-score-direct-engram` directories/files as the main reference run. Set `ROOT` to this checkout, select an idle GPU by UUID, and set `LEANSONIC_QWEN38_CUBIN_PATH` to the Wave 1 sm_120a cubin. Each capture requires new output names.

```sh
cargo build --release --manifest-path "$ROOT/Cargo.toml" -p leansonic-core --example head_hidden --example layer_hidden
bin="$ROOT/target/release/examples"
d="$ROOT/docs/benchmarks/qwen38-pack/reference"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
"$bin/head_hidden" direct-engram fixtures.ids native-head.f32
python3 "$d/head_cross.py" > head-cross.json
python3 "$d/bf16_prefix_check.py" > bf16-prefix.json
"$bin/layer_hidden" direct-engram fixtures.ids layers-native.f32 layers-native-logits.f32
python3 "$d/layer_cross.py" > layer-cross.json
Q38_TRACE_TEACHER_STATES=layers-bf16.f32 "$bin/layer_hidden" direct-engram fixtures.ids local-layers.f32 local-logits.f32
python3 "$d/local_layer_scores.py" > local-layer-scores.json
```

The head comparison crosses the actual native head input or official BF16 head input with either the official weights or the MLX affine weights unpacked into FP32 (group size 32). All four diagnostic projections use FP32 GEMM with TF32 disabled. The original BF16 projection is also repeated using the upstream forward pass. The layer probe compares a nine-token Japanese prefix with the BF16 model's incremental cache on identical IDs. It checks that the native trace preserves the published final logits before comparing hidden states. Relative L2 divides by the BF16 residual norm; the additional block-update ratio divides by the norm of the reference output minus its input. A relative residual difference identifies a candidate for follow-up testing. A causal effect on final logits requires a separate intervention.

To split the local error, record the BF16 input to each MoE block and run the same native binary twice. The first run captures the residual immediately after the attention branch. The second additionally supplies the BF16 post-attention residual at the MoE input. Both runs use BF16 inputs to each whole layer on every prefix token. The reported MoE stage includes its hyper-connection computation and residual addition. Layer 1's attention stage also contains its Engram processing.

```sh
python3 "$d/attention_reference.py"
Q38_TRACE_TEACHER_STATES=layers-bf16.f32 Q38_TRACE_ATTENTION=1 "$bin/layer_hidden" direct-engram fixtures.ids local-attention.f32 local-attention-logits.f32
Q38_TRACE_TEACHER_STATES=layers-bf16.f32 Q38_TRACE_MLP_INPUT=attention-bf16.f32 "$bin/layer_hidden" direct-engram fixtures.ids local-mlp.f32 local-mlp-logits.f32
python3 "$d/branch_scores.py" > branch-scores.json
```
