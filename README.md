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

For the smallest tested text shard, use `--entropy-scales` in place of `--prepack-scales` when converting the official checkpoint with `--prepack-head --prepack-attention --prepack-embedding --prepack-mxfp8`. This stores the same expert scale bytes in rANS frames and produces a **16,086,645,336-byte** shard. It requires a reader with entropy-scale support. In five warm-file ABBA rounds, mean load time changed from 2.73 to 2.61 seconds on the Ryzen 9 9900X / RTX PRO 4500 and from 4.65 to 4.86 seconds on the Core i5-14400 / RTX 5060 Ti. Both formats restore the original GPU scale values.

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

### 3. Re-Verify Layer 1 Lean 4 Proofs & Regenerate Headers

```bash
cd spec
lake build
lake exe leansonic-gen ..
cd ..
python3 ci/check_lean_axioms.py
```

---

## Roadmap & Releases

- [x] **Phase 0 & 1 (`v0.1.0`) — Immutable Trust Core, 5-Gate CI & `Inkling-2-Mini` GPU Engine**: Constructive Lean 4 `sm_120` laws, `VisitBytecode` (`24 → 48` looped visits), full-GPU `NVFP4`/`MXFP8`/`Q8_64` `Inkling-2-Mini` (`131.3 tok/s` on `RTX PRO 4500`, `65.9 tok/s` Spilled on `RTX 5070 Ti`).
- [x] **Cycle 1 (`v0.2.0`) — Multi-Token Batched Prefill**: Planar `Q8_64` `__dp4a` GEMM + Zero-Atomic Grouped MoE (`794.1 tok/s` = `7.14×` on `RTX PRO 4500`, `337.6 tok/s` = `5.55×` on `RTX 5070 Ti`).
- [x] **Cycle 2 (`v0.3.0`) — Lossless `DSpark` Full-GPU Speculative Decoding**: `mtp.safetensors` GPU drafter + `O(1)` Multi-Token Rewind + `0.0000`-ULP hardware FMA alignment (`326.8 tok/s` = `2.58×` on `RTX PRO 4500`, `132.0 tok/s` = `2.02×` on `RTX 5070 Ti`, `0` Trust Core diff).
- [x] **Cycle 3 (`v0.4.0`) — Real `106 GB` `Qwen3.8-Flash-Next` / `Qwen4Exp` GPU Engine**: 22-shard `safetensors` `mmap`, `36` Gated DeltaNet + `12` QSA Sparse Attention layers, `4`-Stream Hyper-Connection, `512`-Expert `Top-10` Group-32 4-Bit MoE (3-Tier VRAM / HostPinned / MmapCompact), `29.80 GiB` `320M`-row 3-Gram `Engram`/`PLE` Async Offload, 48-layer CUDA Graph, and `61` machine-checked Lean 4 theorems (`0` Trust Core diff).

---

## License

MIT License — see [LICENSE](./LICENSE).
