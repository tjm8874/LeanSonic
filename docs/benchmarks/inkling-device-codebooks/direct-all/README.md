# Scratch-free four-bit MXFP8 expert storage

On RTX PRO 5000, each of four unequal sequences produced the same 32 generated token IDs across sequential, interleaved, and four-token batched verification. All four sequence hashes also matched across the eight configurations below. This checks that shared state, scale packing and speculative verification preserve these continuations.

Use the compact model and matched SM120a build described in the [parent instructions](../README.md). Build the release CLI at 24184f4. Run the following with each value of DIRECT (0, 1), SPEC (0, 3), and each row of the scale/kernel table.

| Scale setting | LEANSONIC_TC_MXFP8 | LEANSONIC_MOE_DOWN_XW |
|---|---:|---:|
| packed | 1 | 1 |
| raw | 0 | 0 |

```sh
export LEANSONIC_KV=fp16 LEANSONIC_CTX=2048 LEANSONIC_PREFILL_CHUNK=512
export LEANSONIC_PDL=1 LEANSONIC_MXFP8_DEVICE_U4=1
export LEANSONIC_MXFP8_DIRECT_ALL="$DIRECT" LEANSONIC_GPU_SCALES="$SCALES"
export LEANSONIC_TC_MXFP8="$TC" LEANSONIC_MOE_DOWN_XW="$XW"
target/release/leansonic multiseq --weights "$MODEL" --tokens "@$PWD/docs/benchmarks/inputs/inkling-readme1024.ids" --seqs 4 --round 4 --gen 32 --spec "$SPEC"
```

[CLI, cubin, input and per-sequence hashes](multiseq.json)
