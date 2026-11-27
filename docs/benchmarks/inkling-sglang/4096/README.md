# Inkling-2-Mini at 4,096 prompt tokens and 32 decode steps

On RTX PRO 5000 72 GB, LeanSonic 6aa971d with the 13,448,974,014-byte four-bank model processed this prompt 1.031× as fast as original SGLang and decoded 1.629× as fast. Five alternating ABBA/BAAB cycles used ten fresh processes per engine. Each process ran one warmup and seven timed requests per mode. All 33 predictions matched across both engines, all repetitions, and both forced and free LeanSonic generation.

| Engine and history | Prefill ms | Prefill tok/s | Decode 32 ms | Decode tok/s | Sampled peak process-tree bytes |
|---|---:|---:|---:|---:|---:|
| LeanSonic, fixed SGLang history | 271.902 | 15,064.3 | 152.855 | 209.35 | 17,123,246,080 |
| LeanSonic, free greedy | 271.892 | 15,064.8 | 152.862 | 209.34 | 17,123,246,080 |
| Original SGLang, free greedy | 280.431 | 14,606.1 | 249.048 | 128.49 | 26,543,652,864 |

Paired across cycles, LeanSonic fixed minus SGLang was −8.529 ms for prefill (95% interval −8.760 to −8.282) and −96.194 ms for decode (−96.346 to −96.041). Both used context 8,192, one sequence and no speculation. SGLang used BF16 KV, FA4, metrics, disabled radix caching, and a fixed static memory fraction. LeanSonic used FP16 KV, PDL_ALL and direct device codebooks. The phase counters are native to each engine. SGLang's sampled process-tree peak includes its configured memory pool.

The complete downloaded SGLang distribution is 20,191,643,333 bytes including an unused 635,801,416-byte MTP file. With exactly the main weight shard and tiktoken/tokenizer.model counted on each side, the original is 19,529,405,662 bytes and LeanSonic is 13,448,974,014 bytes. This byte comparison does not imply those two files alone can start the SGLang server. [Original manifest](../distribution.json) · [LeanSonic package receipt](../../inkling-nvfp4-scale-split/four-banks/quartiles/cost-fit/second.json).

## Quality on separate BF16 corpus runs

These quality runs score 25,500 positions per split on the same WikiText references. The LeanSonic scores come from the same physical package in its existing four-bank corpus evaluation. The exact timing revision/cubin is undergoing its own corpus replay on PRO 5000. Single-prompt token agreement does not establish corpus equivalence.

| Split | Package | Mean KL | Worst KL | Top-1 / 25,500 | PPL |
|---|---|---:|---:|---:|---:|
| Test | LeanSonic | 0.0076817 | 1.5215775 | 24,542 | 7.079444 |
| Test | SGLang | 0.0103728 | 3.7027736 | 24,375 | 7.084607 |
| Validation | LeanSonic | 0.0068862 | 1.1576402 | 24,655 | 6.532434 |
| Validation | SGLang | 0.0090103 | 1.0163396 | 24,484 | 6.518670 |

LeanSonic has lower mean KL on both splits. SGLang has lower validation worst KL and PPL. [LeanSonic corpus outputs](../../inkling-nvfp4-scale-split/four-banks/quartiles/cost-fit/bf16.json) · [Native SGLang scores and tracing procedure](../quality/wiki.md).

## Reproduce

The [original comparison](../README.md) pins the SGLang build, model revision and CUDA configuration. At LeanSonic 6aa971db373df0611caf684f9785e5be3c325817 copy lean.rs to crates/leansonic-core/examples/long_sglang_replay.rs and build the release example. Compile kernels/inkling2_sm120.cu with nvcc -Ikernels -arch=sm_120a -std=c++17 -O3 --cubin. Use the files in this directory in a fresh working directory. Set OFFICIAL and LEAN_MODEL to the measured model directories and SGL_PYTHON to the pinned environment. Set LEANSONIC_CUBIN_PATH to the absolute cubin path and enable LEANSONIC_PDL, LEANSONIC_PDL_ALL, LEANSONIC_MXFP8_DEVICE_U4 and LEANSONIC_MXFP8_DIRECT_ALL. Set LEANSONIC_KV=fp16, LEANSONIC_SEQS=1, LEANSONIC_CTX=8192, LEANSONIC_PREFILL_CHUNK=4096. Place the example at ./bench, create runs/, and use leg.py anchor sglang before alternating leg.py runs/CYCLE-LEG-ARM ARM in the five cycles above. The native SGLang counters and exact request IDs, GPU, binary/cubin digests and both confidence intervals are recorded in [verified.json](verified.json).
