# LeanSonic

**Lean 4-Verified Minimalist NVIDIA Blackwell Inference Engine & Zero-Human-Review Autonomous Optimization Pipeline** for next-generation Looped & Hybrid Linear/Sparse MoE LLMs (`Inkling-2-mini` → `Inkling-2-Mid` and `Qwen3.8-Flash-Next` → `Qwen4`).

[English](./README.md) | **[日本語ドキュメント](./README.ja.md)**

---

## Why `LeanSonic`?

Conventional LLM inference engines struggle with two opposing failure modes:
1. **Unverified Manual Kernel Tuning**: Hand-written CUDA kernels routinely hit discrete hardware cliffs on NVIDIA Blackwell (`sm_120`)—such as crossing the `64 KiB → 100 KiB` Shared Memory carveout boundary (shrinking L1 from `64 KiB` to `28 KiB`), silent `union Smem` stage aliasing, register spills (`STACK > 0`), or `1-ULP` reduction tree mismatches that cascade across 48 looped layers.
2. **Untrusted Autonomous Code Generation**: Letting AI coding agents optimize kernels without formal boundaries leads to *reward hacking* (loosening KLD tolerances, injecting `sorry` into proofs, or regressing tail concurrency) and exposes self-hosted GPU runners to untrusted code execution.

**`LeanSonic`** solves both by combining a **Three-Layer Hybrid Architecture** with an **Immutable Trust Core** designed from day one for **autonomous `Issue → PR → 5 Machine Gates → Merge → Release` operation**:

* **Target Hardware**: Exclusively **NVIDIA Blackwell (`sm_120` / `sm_100`)** — zero legacy architecture baggage.
* **Target Model Lineage**:
  * **`Inkling-2-mini` → `Inkling-2-Mid`**: `24` physical layers executed across `48` looped visits (`6` Prelude + `3 × 12` Loop Core + `6` Coda), `12` Global Full Attention + `36` Sliding-Window Attention (`window = 1024`), `128` experts (`top-8`), `NVFP4` / `MXFP8` / `Q8` Planar SoA, `W = 16` hybrid speculative verification.
  * **`Qwen3.8-Flash-Next` → `Qwen4`**: `3:1` macro-blocks (`3×` **Gated DeltaNet** linear recurrent layers + `1×` **Qwen Sparse Attention** block-selected layer), **4-Branch Gated Residual (GR)**, `512` routed experts (`top-10`) + shared expert, and **51B `Engram` host N-gram table** asynchronous PCIe prefetching.
* **Zero Third-Party Rust Dependencies**: Pure Rust standard library workspace (`leansonic-core`, `leansonic-tuner`, `leansonic-cli`)—sub-second incremental builds and minimal supply-chain surface.

---

## Three-Layer Hybrid Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ Layer 1: Lean 4 Formal Specification & Verified Codegen (`spec/`)            │
│  • Constructive `sm_120` laws: {0,8,16,32,64,100} KiB carveouts & 65,536 regs│
│  • Stage-disjoint `union Smem` non-aliasing & 128B TMA alignment proofs      │
│  • Exhaustive 16-code NVFP4 (`E2M1`) -> `2x int8` lossless `__byte_perm` LUT │
│  • Non-associative IEEE-754 `0-ULP` SIMT <-> `16Bx2` MMA butterfly identity  │
│  • `lake exe leansonic-gen` -> `generated_spec.rs` & `leansonic_verified.cuh`│
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Layer 2: Discrete Cliff Pruner & Whole-Model Roofline (`leansonic-tuner`)    │
│  • Rejects L1-starving carveouts, register spills & wide cluster tail traps  │
│  • Ranks valid tile knobs by whole-model streaming & speculative ceilings    │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Layer 3: Minimalist Rust Runtime & Blackwell PTX Templates (`crates/`,`kernels/`)
│  • `VisitBytecode`     : Unified 48-visit Looped & 3:1 GDN/QSA + 4-branch GR │
│  • `HybridSlotPool`    : 64-key KV/SWA ring + `C*(1+W)` O(1) GDN rollback    │
│  • `attn_blocklist_tc` : 22-bit (`hi+lo/2048`) FP16 TC Split-KV (Full/SWA/QSA)│
│  • `moe_dual_regime`   : `REG:64` SIMT (<128 pairs) <-> `2xE2M1` MMA (>=128) │
│  • `Async HtoD`        : Late-wait staging for Spilled Experts & 51B Engram  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Zero-Human-Review 5-Gate Autonomous CI/CD

To run **`Issue → PR → Merge → Release` without human code review** safely on a public GitHub repository, `LeanSonic` physically separates its **Immutable Trust Core** (frozen to human owners via `CODEOWNERS` and Gate 0) from its **Mutable Engine & Proofs**:

| Gate | Runner | Script / Tool | Enforced Machine Invariant |
| :--- | :--- | :--- | :--- |
| **Gate 0** | GitHub CPU | `ci/check_immutable_core.py` | Blocks autonomous PRs from touching `Axioms.lean`, `ci/`, `.github/`, `CODEOWNERS`, or `benches/goldens/` (prevents reward hacking). |
| **Gate 1** | GitHub CPU | `lake build` + `ci/check_lean_axioms.py` | Zero `sorry`, zero `sorryAx`, zero `native_decide`, zero custom `axiom` across all 10+ theorems; `git diff --exit-code` on generated `.rs`/`.cuh` headers. |
| **Gate 2** | CPU / CUDA | `ci/audit_sass.py` + `clippy -D warnings` | `STACK: 0` (zero local-memory spills), zero `CALL.REL.NOINC` in hot loops, zero unvectorized 8-bit `LDG`, strict Rust formatting & lint cleanliness. |
| **Gate 3** | CPU & GPU | `leansonic verify-parity` | **48-visit `0-ULP BATCH IDENTICAL` parity** (`C = 1` `SimtReg64` bit-exact with `C = 64` `GroupedMmaInt8`), or strict 100-chunk KLD bounds on quantized PRs. |
| **Gate 4** | Ephemeral Blackwell | `ci/abba_stat_gate.py` | Same-session interleaved `ABBA × N` paired bootstrap 95% CI check (never triggered on external fork PRs; egress-isolated runner). |
| **Gate 5** | GitHub Merge Queue | `.github/workflows/autonomous_loop.yml` | Serialized merge queue validation + automated SemVer binary packaging and GitHub Release publication. |

---

## Repository Layout

```text
LeanSonic/
├── spec/
│   ├── LeanSonic/Axioms.lean               # [Phase 0 Immutable Core] SM120, 0-ULP & OpenAI SSE/Serve laws
│   ├── LeanSonic/Proofs/Inkling2Mini.lean  # [Phase 1 Verified Proofs] Inkling-2 & Qwen3.8 schedules + SM120 suite
│   ├── LeanSonic/Proofs/ServingAndPrompt.lean # [Phase 1 Verified Proofs] Inductive SSE completeness & `-p`==`serve` 0-ULP
│   ├── LeanSonic/Proofs/SpilledAndEngram.lean # [Phase 1 Verified Proofs] Spilled Ping-Pong & Qwen3.8 Engram 0-ULP
│   └── LeanSonic/Emit.lean                 # Verified constant codegen -> Rust & CUDA headers
├── kernels/
│   ├── include/leansonic_verified.cuh      # Auto-generated from Lean 4 proofs
│   ├── include/sm120_ptx.cuh               # PTX: cp.async.cg, mma.m16n8k16/k32, PDL, 3-op E2M1 LUT
│   ├── attn_blocklist_tc.cu                # Unified Full + SWA + QSA Multi-Row Tensor Core Split-KV
│   ├── moe_dual_regime.cu                  # Dual-regime SIMT <-> Grouped MMA 0-ULP MoE
│   └── loop_residual_blend.cu              # Fused RMSNorm + Loop / 4-Branch GR Blend + Planar Q8
├── crates/
│   ├── leansonic-core/                     # Bytecode, HybridSlotPool, Planar SoA, `-p` & OpenAI `serve`
│   ├── leansonic-tuner/                    # Layer 2 SM120 discrete cliff pruner & roofline ranker
│   └── leansonic-cli/                      # `leansonic` CLI (`-p`, `serve`, `info`, `schedules`, `verify-parity`, `tune`)
└── ci/                                     # [Phase 0 Immutable Core] Gates 0, 1, 2 & 4 verification scripts
```

---

## Quick Start

### 1. Build & Run the Minimalist Rust Engine (Zero Dependencies)

```bash
git clone https://github.com/tjm8874/LeanSonic.git
cd LeanSonic

# Run all workspace unit tests & 48-visit 0-ULP batch-invariance checks
cargo test --workspace

# Inspect Lean-verified SM120 & Hybrid Slot Pool constants
cargo run --release -p leansonic-cli -- info

# Validate Inkling-2-Mini (24->48 looped) & Qwen3.8-Flash-Next (3:1 GDN+QSA+Engram) schedules
cargo run --release -p leansonic-cli -- schedules

# Execute Gate 3 (48-visit C=1 == C=64 AND `-p` == `serve` 0-ULP check)
cargo run --release -p leansonic-cli -- verify-parity

# Single-shot prompt (`-p` / `--prompt`) — streaming stdout or structured `--json`
cargo run --release -p leansonic-cli -- -p "Explain Looped Transformers on Blackwell" --max-tokens 16
cargo run --release -p leansonic-cli -- -p "Hello Qwen4 hybrid GDN+QSA" --model qwen3.8-flash-next --max-tokens 12 --json

# Launch zero-dependency OpenAI-compatible HTTP + SSE server (`serve`)
cargo run --release -p leansonic-cli -- serve --host 127.0.0.1 --port 8080 --model inkling-2-mini

# Run Layer 2 Blackwell RTX PRO 5000 cliff pruning & streaming ceiling estimator
cargo run --release -p leansonic-cli -- tune
```

To compare real Inkling-2-Mini logits with a llama.cpp `_logits_` reference on a Blackwell GPU, run:

```bash
leansonic eval-kld --weights "$MODEL_DIR" --ref inkling2-bf16-base.kld
```

The command prints JSON with model and reference perplexity, mean/median/p99/maximum per-position KL divergence, and top-1 agreement. It resets the GPU engine for each reference chunk, prefills through its midpoint, and scores each remaining next token using the reference prefix. KL uses the reference log-probability cutoff of −16. Use `--chunks N` to score the first N chunks. Use `--start-chunk I --chunks 1` to replay a zero-based chunk. The `ref_ppl` field uses the probabilities stored in the reference file. Its lowest code covers logits at least 16 below the row maximum when the row is clipped. The `ref_floor_targets` field counts scored next tokens with code zero. See the [reference build and checksums](docs/inkling-reference.md) to reproduce the 100-chunk file and compare with the generator’s perplexity.

Inkling chooses its prefill chunk at load time from 64, 128, 256, 512 and 1024 tokens. It widens the chunk when scratch space and the local KV rings fit alongside the expert matrices selected for GPU memory. The startup log reports the selected width. Set `LEANSONIC_PREFILL_CHUNK=64` to use the smallest scratch allocation, or set it to one of the larger widths to cap automatic selection. External cubins without wide grouping use 64.

For a single Inkling prompt, `-p` and `bench` use its token count to limit the workspace. For example, a 65-token prompt needs at most 128 rows. The HTTP server sizes its workspace for future requests. An explicit `LEANSONIC_PREFILL_CHUNK` takes precedence in all three modes.

For Inkling-2-Mini on Blackwell, set `LEANSONIC_GPU_SCALES=packed` to keep eligible expert scale planes compressed in VRAM. This works with the official checkpoint and with a shard produced by `--prepack-scales`. The MoE kernels reconstruct the original scale bytes as they read them. Each plane falls back to raw storage if its values exceed the supported range. A cubin without the packed-scale kernels uses raw storage throughout. The default is `packed` (set `LEANSONIC_GPU_SCALES=raw` to keep uncompressed scales). On a 16 GB card, the loader can use the released memory for a wider prompt chunk and additional resident weights. Benchmark your prompt and completion lengths together when choosing a chunk cap.

Inkling stores KV in FP16 by default. `LEANSONIC_KV=int8` stores the four full-attention (global) layers as INT8 with one FP32 scale per row and KV head, and `LEANSONIC_KV=int8-all` also stores the sliding-window rings as INT8. On GPUs with at most 16.5 GiB of memory, a context above 8,192 tokens (the CLI default) selects `int8` automatically when `LEANSONIC_KV` is unset, because the FP16 global KV would otherwise displace resident expert layers. The startup log shows `int8 (auto, ...)` in that case. Set `LEANSONIC_KV=fp16` to keep FP16.

#### Prepare a smaller text-only Inkling checkpoint

For a text-only deployment, omit the vision tower and keep the language tensors at their original precision. The converter needs Python 3.9+ and about 19 GB of free disk space for the output.

```bash
src=/path/to/official/Inkling-2-Mini
dst=inkling-text
shard=model-00001-of-00001.safetensors
mkdir "$dst" && python3 tools/pack_inkling_text.py "$src/$shard" "$dst/$shard" && cp -R "$src/tiktoken" "$dst/"
```

Pass the new directory to `--weights`. To use the DSpark drafter, also copy `mtp.safetensors` from the official model into that directory. Image inference requires the original checkpoint.

Add `--prepack-head` to the packing command to save another 436,576,496 bytes with the official checkpoint. The packer uses Cargo on the CPU to store the vocabulary projection as the same Q8 codes and FP32 scales the Inkling GPU loader normally computes at startup. This package requires a LeanSonic version that supports the native output head. Keep the original BF16 weights for other engines or future quantization experiments.

To prepack the 176 attention projections as well, run the same command with `--prepack-head --prepack-attention`. The official text shard then occupies 17,847,272,800 bytes. If the head is already packed, pass that shard as the source and use `--prepack-attention` alone. Both options require Cargo while building the package.

Add `--prepack-scales` to store the 48 expert scale tensors losslessly. With all three packing flags, the official text shard occupies 17,240,610,424 bytes. You can also run `--prepack-scales` on the head-and-attention package. The loader restores the original scale bytes in 1 MiB tiles before inference. This option reduces on-disk size and requires a reader that supports the packed scales.

Use `--prepack-embedding` to store the BF16 input embedding losslessly. You can pass all four flags when converting the official checkpoint or add just this flag when converting the three-flag package. The resulting official text shard occupies **16,932,881,536 bytes**. The loader restores the original BF16 bytes directly into pinned host memory at startup. This format requires a LeanSonic reader with input embedding support.

Add `--prepack-mxfp8` to losslessly encode the eight MXFP8 expert weight tensors. It can be applied to the previous package or combined with the other flags on the official checkpoint. With all five flags the text shard occupies **16,195,210,184 bytes**. This format requires a compatible reader. At startup the loader places the restored weights using their original byte counts. Weights assigned to host memory decode into their pinned allocation. GPU-resident weights decode in batches of up to 16 MiB before upload.

Use `--entropy-scales` in place of `--prepack-scales` when converting the official checkpoint with `--prepack-head --prepack-attention --prepack-embedding --prepack-mxfp8`. This stores the same expert scale bytes in rANS frames and produces a **16,086,645,336-byte** shard. It requires a reader with entropy-scale support. In five warm-file ABBA rounds, mean load time changed from 2.73 to 2.61 seconds on the Ryzen 9 9900X / RTX PRO 4500 and from 4.65 to 4.86 seconds on the Core i5-14400 / RTX 5060 Ti. Both formats restore the original GPU scale values.

Add `--compact-q8-scales` to that command to store the attention and output-head Q8 scales in per-matrix dictionaries. You can also pass this flag alone when repacking the native Q8 shard. The loader restores every F32 bit before GPU upload. On Inkling-2-Mini this reduces the shard to **16,035,182,496 bytes**. The tokenizer is distributed separately. This format requires a reader with Q8 scale dictionary support.

To also compress the signed Q8 code bytes, add `--compact-q8-codes` or apply it to the dictionary package. Each matrix falls back to raw storage if rANS would expand it. With both Q8 options and the earlier packing flags the text shard is **15,978,753,408 bytes**. Including the original tokenizer brings the distribution to **15,982,139,902 bytes**. A compatible loader restores the same native Q8 codes before inference, staging at most 16 MiB of decoded codes at once. On an RTX 5060 Ti, this saved 56,429,088 distribution bytes and increased mean model load time from 5.181 to 5.785 seconds across 20 fresh processes with warm files. See [timings and reproduction commands](docs/benchmarks/inkling-rans-q8-codes.json).

### 2. Query the OpenAI-Compatible Server (`GET /v1/models`, `POST /v1/chat/completions`, SSE)

```bash
# List verified models
curl -s http://127.0.0.1:8080/v1/models

# Non-streaming Chat Completions (`0-ULP` bit-exact with `leansonic -p`)
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"Hello Blackwell"}],"max_tokens":12}'

# Streaming SSE (`text/event-stream`, guarded by Lean-verified `SseProtocolGuard`)
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Stream GDN+QSA"}],"max_tokens":12,"stream":true}'
```

### 3. Benchmarking & Quality Verification Guide (`leansonic bench`)

`leansonic bench` measures real GPU prefill throughput (`tok/s`), top-5 vocabulary logits + logit checksum, and autoregressive / speculative decode throughput (`tok/s` & `ms/tok`) across both **`Inkling-2-Mini`** (`16g` Speed Pack / `32g` Precision Pack) and **`Qwen3.8-Flash-Next`** (`106 GB` 22-shard checkpoint). Model architecture is auto-detected from `config.json`.

#### `leansonic bench` Command Reference

```bash
leansonic bench \
  --weights <model_dir> \
  --tokens <synth:N | chat:<text> | chat:@prompt.txt | id1,id2,... | @ids.csv> \
  [--gen N] \
  [--repeat R] \
  [--spec 0..7] \
  [--no-graph]
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-w`, `--weights <dir>` | *(required)* | Model directory containing `.safetensors` shards and tokenizer (`tiktoken/` for Inkling, `tokenizer.json` for Qwen3.8). |
| `-t`, `--tokens <spec>` | *(required)* | Prompt input specification (see the 4 formats below). |
| `-g`, `--gen <N>` | `32` | Number of greedy completion tokens to generate (`0` = prefill + first-step logits only). |
| `-r`, `--repeat <R>` | `1` | Repeat `reset() -> prefill -> decode` `R` times in the same process (`run 0` includes cold graph/buffer warmup; `run >= 1` measures warm steady-state). |
| `--spec <K>` | `0` | Speculative draft depth (`0` = Solo Decode; `1..7` = `0-ULP` verified speculative decode via `DFlash 2` / `DSpark` / `Qwen3.8 MTP`). |
| `--no-graph` | *(enabled)* | Disable CUDA Graph replay for single-step decode debugging / `nsys` kernel profiling. |

#### `--tokens` Input Formats
1. **`synth:N`** (e.g. `--tokens synth:2048`): Generates `N` deterministic synthetic token IDs (`(i * 7 + 100) % 50000`) for standardized prefill and KV-scaling benchmarks without external prompt files.
2. **`chat:<text>`** (e.g. `--tokens "chat:Explain Gated DeltaNet in three bullet points."`): Formats a single user turn with the checkpoint's native chat template (`effort = none`) and tokenizes on the fly.
3. **`chat:@file.txt`** (e.g. `--tokens chat:@benches/prompts/code.txt`): Reads UTF-8 prompt text from `file.txt` and formats it as a single user chat turn.
4. **`id1,id2,...` or `@ids.csv`** (e.g. `--tokens 100,107,114` or `--tokens @fixture.ids`): Feeds exact comma-separated token IDs directly into `prefill_tokens`.

#### Practical `leansonic bench` Examples

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# 1. Inkling-2-Mini (32g Precision Pack or 16g Speed Pack): 2K prefill + 128-token Solo vs DFlash 2 Speculative Decode
leansonic bench --weights /path/to/inkling-32g --tokens synth:2048 --gen 128 --repeat 3
leansonic bench --weights /path/to/inkling-32g --tokens "chat:Write an iterative quicksort in Rust." --spec 7 --gen 128 --repeat 3

# 2. Qwen3.8-Flash-Next (106 GB Hybrid GDN + QSA + 512-Expert MoE + 30 GiB Engram): Batched Prefill & Solo Decode
leansonic bench --weights /path/to/qwen3.8-flash-next --tokens synth:512 --gen 64 --repeat 3
leansonic bench --weights /path/to/qwen3.8-flash-next --tokens synth:2048 --gen 64 --repeat 3

# 3. Long-context decode timing at 32K KV depth without waiting to prefill 32K tokens (timing-only jump)
LEANSONIC_CTX=36864 LEANSONIC_BENCH_DECODE_AT=32768 \
  leansonic bench --weights /path/to/qwen3.8-flash-next --tokens synth:64 --gen 64 --repeat 3
```

#### Environment Variables for `bench`, `-p`, and `serve`

| Variable | Applies To | Purpose |
| :--- | :--- | :--- |
| `CUDA_DEVICE_ORDER=PCI_BUS_ID` | All GPU runs | Ensures `CUDA_VISIBLE_DEVICES` matches `nvidia-smi` PCIe indices on multi-GPU hosts. |
| `LEANSONIC_CTX=<tokens>` | Inkling & Qwen3.8 | Sets maximum KV/QSA context capacity (auto-grows on Qwen3.8 up to `LEANSONIC_CTX`). |
| `LEANSONIC_BENCH_DECODE_AT=<pos>` | `bench` | Jumps decode position to `<pos>` after prefill to benchmark long-context decode latency directly. |
| `LEANSONIC_EVAL_DUMP=<file.f32>` | `bench` | Writes raw full-vocabulary FP32 logits (`[1 + forced.len(), vocab]`) for offline KLD / `0-ULP` parity audits (pair with `LEANSONIC_EVAL_FORCE=id1,id2,...`). |
| `LEANSONIC_PREFILL_CHUNK=<64..1024>` | Inkling | Caps or forces the batched prefill chunk width. |
| `LEANSONIC_KV=<fp16\|int8\|int8-all>` | Inkling | Selects KV cache precision (`int8` auto-activates above `8K` context on `<= 16.5 GiB` GPUs). |
| `LEANSONIC_DFLASH2_DIR=<dir>` | Inkling `--spec` | Explicit path to the 5-layer `DFlash 2` block-diffusion drafter directory. |
| `LEANSONIC_CUBIN_PATH` / `LEANSONIC_QWEN38_CUBIN_PATH` | Inkling / Qwen3.8 | Overrides automatic `.cubin` discovery next to the `leansonic` binary. |

#### Companion Suite Commands (`bench-matrix`, `multiseq`, `canary`, `eval-kld`)

```bash
# Multi-GPU bootstrap benchmark matrix (scores median + 95% CI against docs/benchmarks/targets.json)
leansonic bench-matrix --weights "$MODEL_DIR" --gpus all --n 5 --out host_results.json

# Multi-sequence continuous batching & C=1 == C=N state-isolation verification (Inkling-2-Mini)
leansonic multiseq --weights "$INKLING_DIR" --tokens synth:256 --seqs 16 --gen 64

# Deterministic factual ("Tokyo") + 16K needle-in-a-haystack ("RAVEN-731") + --spec 0 == --spec 7 0-ULP gate
leansonic canary --weights "$MODEL_DIR"
leansonic canary --weights "$INKLING_DIR" --long   # Adds 128K ("MARSH-527" / "CORAL-841") retrieval gate

# Teacher-forced 100-chunk KL-divergence & Perplexity audit against BF16 reference
leansonic eval-kld --weights "$MODEL_DIR" --ref "$REF_KLD"
```

### 4. Re-Verify Layer 1 Lean 4 Proofs & Regenerate Headers

```bash
cd spec
lake build
lake exe leansonic-gen ..
cd ..
python3 ci/check_lean_axioms.py
```

---

## Contributing & AI Agent Skill

Contributions from both human engineers and autonomous coding agents follow the 5-Gate Immutable Trust Core workflow:
- **[CONTRIBUTING.md](./CONTRIBUTING.md)**: Full guide to the 5 Machine Gates (`Gates 0–4`), SM120 discrete resource laws, mandatory exact-size `compute-sanitizer --padding 65536` + planted-fault red controls, shared GPU `flock` etiquette, and public-repo privacy rules.
- **[`.agents/skills/leansonic_contribute/SKILL.md`](./.agents/skills/leansonic_contribute/SKILL.md)** (`leansonic_contribute`): Ready-to-use workspace skill for Antigravity, Claude Code, and Codex.

---

## Release Reports (`docs/releases/`)

Detailed engineering & hardware verification reports are published in **both English and Japanese** under [`docs/releases/`](./docs/releases/):

- **`v0.1.0` – `v0.4.0` Retrospective**: [English](./docs/releases/v0.1.0_to_v0.4.0_retrospective.md) | [日本語](./docs/releases/v0.1.0_to_v0.4.0_retrospective.ja.md) — Immutable Trust Core, Batched Prefill, `DSpark` Speculative Decode, and Initial `106 GB` `Qwen3.8-Flash-Next` Engine
- **`v0.5.0`**: [English](./docs/releases/v0.5.0_report.md) | [日本語](./docs/releases/v0.5.0_report.ja.md) — `16 GB` Spilled VRAM Budgeting & Lossless `Inkling-2-Mini` Shard Packing
- **`v0.6.0`**: [English](./docs/releases/v0.6.0_report.md) | [日本語](./docs/releases/v0.6.0_report.ja.md) — Multi-Slot Continuous Batching (`C = 1..64`) & `128K` Long-Context Split-KV
- **`v0.7.0`**: [English](./docs/releases/v0.7.0_report.md) | [日本語](./docs/releases/v0.7.0_report.ja.md) — `INT8` Global KV Cache & Wide `B = 1024` Grouped MoE Prefill
- **`v0.8.0`**: [English](./docs/releases/v0.8.0_report.md) | [日本語](./docs/releases/v0.8.0_report.ja.md) — `DFlash 2` Block-Diffusion Speculative Decoding & Exact-Size Sanitizer Gate
- **`v0.9.0`**: [English](./docs/releases/v0.9.0_report.md) | [日本語](./docs/releases/v0.9.0_report.ja.md) — `PDL` Overlap, Packed GPU Expert Scales & `4`-Card Blackwell Matrix
- **`v0.10.0`**: [English](./docs/releases/v0.10.0_report.md) | [日本語](./docs/releases/v0.10.0_report.ja.md) — Adaptive Spilled `spec_k` Governor & Unified `DFlash 2` Default Drafter
- **`v0.11.0` (`Inkling-2-Mini` Final Edition)**: [English](./docs/releases/v0.11.0_report.md) | [日本語](./docs/releases/v0.11.0_report.ja.md) — Two-Tier `16g` Speed (`10,222 tok/s` Prefill / `235.9 tok/s` Spec, `0` Spill) & `32g` Precision (`15,057 tok/s` Prefill / `313.9 tok/s` Spec / `mean_kld = 0.002989`) Packs
- **`v1.0.0` (General Availability — `Qwen3.8-Flash-Next` & `Inkling-2-Mini`)**: [English](./docs/releases/v1.0.0_report.md) | [日本語](./docs/releases/v1.0.0_report.ja.md) — Full `106 GB` `Qwen3.8-Flash-Next` (`48/48` `Z24` VRAM, `7.9s` Load, `504.4 tok/s` Batched Prefill, `179.6–200.8 tok/s` Full-VRAM `MTP` `--spec 1`, `68.7 tok/s` `PRO 4500` `CalibratedMixed`, `212` Lean 4 Theorems)

---

## License

MIT License — see [LICENSE](./LICENSE).
