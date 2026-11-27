# Capped Inkling-2-Mini versus original SGLang

On one RTX PRO 5000 72 GB, SGLang processed the 1,024 input tokens faster. LeanSonic generated the next 32 tokens 1.434 times as fast and used 7,983,857,664 fewer sampled GPU bytes. These results cover a single fixed token history. The timing used the 13,624,461,934-byte [fifteen-symbol model](../) and the unmodified [official SGLang checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/b85c3cfa615fc413cc9af3379f491b963f3b586a).

| Engine | Prefill ms | Prefill tok/s | Decode 32 ms | Decode tok/s | Peak GPU bytes | Distributed model bytes |
|---|---:|---:|---:|---:|---:|---:|
| LeanSonic bc526e3 | 89.578 | 11,431.43 | 171.397 | 186.70 | 16,028,532,736 | 13,624,461,934 |
| SGLang 9c64454 | 84.837 | 12,070.23 | 245.738 | 130.22 | 24,012,390,400 | 20,191,643,333 |

Five cycles alternated ABBA and BAAB order. Each of ten fresh processes per engine made eight requests; we excluded the first request per process. Rates use the reciprocal of the mean time across the remaining 70 requests. The central 95% paired cycle bootstrap intervals for LeanSonic/SGLang throughput are 0.9452–0.9493 for prefill and 1.4329–1.4345 for decode. The process-tree NVML peak includes loading and inference and uses a nominal 4 ms interval once allocations appear. Runtime and CUDA libraries are excluded from package sizes. The SGLang snapshot contains a 635,801,416-byte unused speculative checkpoint. Comparing only the target shard plus the same tokenizer gives 19,529,405,662 bytes for SGLang and 13,624,461,934 for LeanSonic. The official shard may contain additional model components. [Official file manifest](../../../../inkling-sglang/distribution.json)

Each engine used one sequence, 2,048 context capacity, no speculation or cross-request prefix reuse, and the same 1,024 prompt IDs. SGLang generated 33 greedy IDs and used its native prefill latency and decode throughput. LeanSonic timed synchronized engine calls using FP16 KV and replayed the first 32 SGLang IDs to keep the full history fixed. SGLang used BF16 KV and FA4. LeanSonic matched 31/32 continuation IDs and the final successor. The disagreement was at zero-based index 30 (SGLang 196973, LeanSonic 197005). This fixed-history test measures neither free-generation agreement beyond that point nor general output quality.

## BF16 reference

Earlier quality runs used these same weight distributions and the same WikiText-2 oracle, with a separate LeanSonic scorer build on RTX PRO 4500. Each split covers 25,500 positions. The capped model has lower mean KL on both splits and higher worst KL on validation. [Capped package quality and worst positions](../quality.md) · [SGLang corpus setup and scores](https://github.com/tjm8874/LeanSonic/blob/828b19f/docs/benchmarks/inkling-sglang/quality/wiki.md).

| Split / engine | Mean KL | Worst KL | BF16 top-1 matches / 25,500 | PPL |
|---|---:|---:|---:|---:|
| test / LeanSonic | 0.0081320 | 0.6864531 | 24,476 | 7.085819 |
| test / SGLang | 0.0103728 | 3.7027736 | 24,375 | 7.084607 |
| valid / LeanSonic | 0.0076511 | 1.9394759 | 24,588 | 6.530274 |
| valid / SGLang | 0.0090103 | 1.0163396 | 24,484 | 6.518670 |

## Reproduce

Use a clean Linux work directory on a dedicated GPU. Pin LeanSonic bc526e3553524cc2bdd6f1707a65fd592f75b546 in ENGINE. Install [SGLang at the measured revision](https://github.com/sgl-project/sglang/tree/9c64454bba553e475ce7b7b790015d2f941e2807). Set OFFICIAL to the official snapshot, LEAN_MODEL to the capped package, and FIXTURE to this directory, all as absolute paths. Use the SGLang Python environment. Set CUDA_VISIBLE_DEVICES to the full GPU UUID; our device was GPU-4636fd80-68e6-909c-cea4-60f0f4f5601e, driver 595.99.02, power cap 350 W.

```sh
cp "$FIXTURE/replay.rs" "$ENGINE/crates/leansonic-core/examples/probe_sglang.rs"
(cd "$ENGINE" && cargo build --release -p leansonic-core --example probe_sglang && nvcc -arch=sm_120a -std=c++17 -O3 --cubin kernels/inkling2_sm120.cu -o engine.cubin)
export REPLAY="$ENGINE/target/release/examples/probe_sglang" LEANSONIC_CUBIN_PATH="$ENGINE/engine.cubin"
export OFFICIAL LEAN_MODEL CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID
export LEANSONIC_PDL=1 LEANSONIC_PDL_ALL=1 LEANSONIC_MXFP8_DEVICE_U4=1 LEANSONIC_MXFP8_DIRECT_ALL=1 LEANSONIC_SEQS=1 LEANSONIC_KV=fp16 LEANSONIC_CTX=8192
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cp "$FIXTURE"/{prompt.ids,continuation.ids,peak.py,run.py,sgl.py} .
for c in 1 2 3 4 5; do python run.py "$c" || exit 1; done
python "$FIXTURE/summarize.py" runs
```

The replay executable SHA-256 was `d5aaa96b27c3401bd44fed529e51259c159254e97e0bd1bfe7864ed1889ee16a` and the cubin SHA-256 was `65fb004d7d44c3d5450b7146dc9e3cb302d728ac35f1bcda2cf162a51c6dfb6d`. The measured target shard and tokenizer matched the digests in the [capped build receipt](../build.json). Run `python summarize.py` in this directory to check [all captured requests and process peaks](verified.json) without a GPU.
