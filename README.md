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
│   ├── LeanSonic/Axioms.lean             # [Phase 0 Immutable Core] SM120 laws, 0-ULP & pool specs
│   ├── LeanSonic/Proofs/Inkling2Mini.lean# [Phase 1 Verified Proofs] Inkling-2 & Qwen3.8 schedules + SM120 suite
│   └── LeanSonic/Emit.lean               # Verified constant codegen -> Rust & CUDA headers
├── kernels/
│   ├── include/leansonic_verified.cuh    # Auto-generated from Lean 4 proofs
│   ├── include/sm120_ptx.cuh             # PTX: cp.async.cg, mma.m16n8k16/k32, PDL, 3-op E2M1 LUT
│   ├── attn_blocklist_tc.cu              # Unified Full + SWA + QSA Multi-Row Tensor Core Split-KV
│   ├── moe_dual_regime.cu                # Dual-regime SIMT <-> Grouped MMA 0-ULP MoE
│   └── loop_residual_blend.cu            # Fused RMSNorm + Loop / 4-Branch GR Blend + Planar Q8
├── crates/
│   ├── leansonic-core/                   # Bytecode, HybridSlotPool, Planar SoA weights, 0-ULP engine
│   ├── leansonic-tuner/                  # Layer 2 SM120 discrete cliff pruner & roofline ranker
│   └── leansonic-cli/                    # `leansonic` CLI (`info`, `schedules`, `verify-parity`, `tune`)
└── ci/                                   # [Phase 0 Immutable Core] Gates 0, 1, 2 & 4 verification scripts
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

# Execute Gate 3 (48-visit C=1 SimtReg64 == C=64 GroupedMmaInt8 0-ULP check)
cargo run --release -p leansonic-cli -- verify-parity

# Run Layer 2 Blackwell RTX PRO 5000 cliff pruning & streaming ceiling estimator
cargo run --release -p leansonic-cli -- tune
```

### 2. Re-Verify Layer 1 Lean 4 Proofs & Regenerate Headers

```bash
cd spec
lake build
lake exe leansonic-gen ..
cd ..
python3 ci/check_lean_axioms.py
```

---

## Roadmap

- [x] **Phase 0 — Immutable Trust Core & 5-Gate CI Harness**: Constructive Lean 4 `sm_120` hardware & `0-ULP` axioms, verified `.rs`/`.cuh` header emitter, Gate 0–4 auditors, and GitHub Actions workflows.
- [x] **Phase 1 — `Inkling-2-mini` Vertical Slice & `Qwen3.8` Schedule Foundation**: `VisitBytecode` (`24 → 48` looped visits + `3:1` GDN/QSA), `HybridSlotPool` (`64`-aligned SWA ring + `O(1)` GDN speculative rollback), `128B` Planar SoA weights, `0-ULP` dual-regime MoE, and parameterized `sm_120` CUDA/PTX templates.
- [ ] **Phase 2 — Live Safetensors Checkpoint Loader, `libcuda` Graph Executor & Full Autonomous Loop**: Direct `mmap` weight streaming for `Inkling-2-mini` and `Qwen3.8-Flash-Next`, QSA micro-block indexer kernel, 51B `Engram` host DMA ring, and unattended autonomous optimization PR cycling on Blackwell runners.

---

## License

MIT License — see [LICENSE](./LICENSE).
