# Staging selected NVFP4 weights during decode

On the 16 GB cards tested here, three layers keep their gate/up expert weights in pinned host memory. Once the router selects eight experts, a gather kernel can copy their weights into GPU memory in contiguous spans. A second kernel calculates the outputs from the copy, using the original expert IDs for scales. Each pass covers 192 output channels per expert. Six passes cover 1,152 channels. During a singleton visit the existing prefill workspace supplies 3,538,944 bytes of temporary storage. Both operations run on the compute stream.

| GPU, 32 input IDs and 64 decode steps | Existing mapped reads | Always stage | Automatic selection | Choice in auto processes |
|---|---:|---:|---:|---|
| RTX 5070 Ti | 663.16 ms | 599.56 ms | 599.45 ms | Stage, 6 of 6 |
| RTX 5060 Ti | 1,329.09 ms | 1,345.39 ms | 1,329.19 ms | Mapped reads, 6 of 6 |

The RTX 5070 Ti saved 9.61% of elapsed decode time. The RTX 5060 Ti took 1.23% longer when forced to stage. Three balanced cycles ran each arm in a fresh CLI twice per cycle, rotating the first arm and reversing the order. Each visit contributed the median of warm repeats 2–4. All three arms generated identical IDs. The model, KV format, GPU residency policy and cubin stayed fixed. [Per-run timings, activation decisions and logits hashes](inkling-selected-host-weights.json)

The decision uses an actual host-backed matrix when the engine loads. It times both paths on two sets of eight distinct experts, reversing their order for the second set. Staging must save at least 3% in the probe. The loader skips the measurement if the model fits in VRAM, the scratch is too small, or the cubin lacks either staging kernel. The decision is made before decode graphs are captured.

## Reproduce

Build the CLI and CUDA module and prepare the fixed 32 input and 16 teacher-forced IDs using the [gate/up benchmark instructions](inkling-nvfp4-half-warp.md). Select one physical GPU with `CUDA_VISIBLE_DEVICES`. Use the compact model with FP16 KV, context 8192, greedy decoding and speculation disabled.

```bash
export LEANSONIC_CUBIN_PATH="$PWD/auto.cubin" LEANSONIC_KV=fp16
nvcc -O3 -std=c++17 -arch=sm_120a --cubin kernels/inkling2_sm120.cu -o "$LEANSONIC_CUBIN_PATH"
for mode in 0 1 auto; do
    LEANSONIC_STAGE_HOST_UP="$mode" "$CLI" bench --weights "$COMPACT" \
        --tokens "@$PWD/short.ids" --spec 0 --gen 64 --repeat 4
done
for mode in 0 auto; do
    LEANSONIC_STAGE_HOST_UP="$mode" LEANSONIC_EVAL_FORCE="$(cat force.ids)" \
        LEANSONIC_EVAL_DUMP="$mode.f32" "$CLI" bench --weights "$COMPACT" \
        --tokens "@$PWD/short.ids" --spec 0 --gen 0 --repeat 1
done
cmp 0.f32 auto.f32 && sha256sum auto.f32
```

The default is automatic selection. A value of `0` keeps mapped reads and `1` forces staging when the necessary buffer and kernels are available. A microbenchmark of all eight selected host matrices found 1,215.6 → 886.8 µs per packed up operation on 5070 Ti, versus 3,193.5 → 3,225.8 µs on 5060 Ti. The real decoder uses six smaller tiles to avoid allocating another 20.25 MiB.
