# Contributing to LeanSonic

Thank you for contributing to **LeanSonic** — the Lean 4-verified minimalist NVIDIA Blackwell (`sm_120` / `sm_100`) inference engine for Looped (`Inkling-2-Mini`) and Hybrid Linear/Sparse MoE (`Qwen3.8-Flash-Next`) LLMs.

LeanSonic is architected for **Zero-Human-Review Autonomous & Multi-Agent Engineering** (`Issue → Focused Branch → PR → 5 Machine Gates → Merge`). Every contribution — whether authored by a human developer or an autonomous AI agent (Antigravity / Gemini, Claude Code, Codex) — is governed by the exact same formal, static, numerical, and statistical verification gates.

> **AI Agent Skill**: If your coding agent supports workspace skills (`SKILL.md`), load [`.agents/skills/leansonic_contribute/SKILL.md`](./.agents/skills/leansonic_contribute/SKILL.md) (`leansonic_contribute`) before editing code, CUDA kernels, or Lean 4 proofs.

---

## 1. Immutable Trust Core (`Gate 0` — Do Not Modify)

To prevent *reward hacking* (weakening axioms, loosening KLD/canary tolerances, or altering CI workflows), the following paths form the **Immutable Trust Core** and are blocked from autonomous PRs by `ci/check_immutable_core.py` and `CODEOWNERS`:

- `spec/LeanSonic/Axioms.lean`
- `ci/**`
- `.github/**`
- `CODEOWNERS`
- `benches/goldens/**`

### Generated Headers & Baselines
- **Never hand-edit `@generated` files**:
  - `crates/leansonic-core/src/generated_spec.rs`
  - `kernels/include/leansonic_verified.cuh`
  Both are generated deterministically by `lake exe leansonic-gen ..` from `spec/LeanSonic/Emit.lean`, and CI enforces `git diff --exit-code` after regeneration.
- **Preserve historical benchmark baselines**: Never alter `initial_basalt_targets` in `docs/benchmarks/targets.json`.
- **Zero external Rust crate dependencies**: Keep `crates/leansonic-core`, `crates/leansonic-tuner`, and `crates/leansonic-cli` pure Rust `std` (`0` third-party crates in `Cargo.toml`).

---

## 2. The 5 Verification Gates (Local Checklist Before Opening a PR)

Run every applicable gate locally before pushing your branch:

### Gate 0 — Mutable Scope Check
```bash
python3 ci/check_immutable_core.py --base origin/main
```

### Gate 1 — Lean 4 Constructive Proofs & Header Sync
Every new kernel (`KernelResourceSpec`) or lossless compression / state / scheduling transformation must be registered in `spec/LeanSonic/Proofs/*.lean` with **zero `sorry`**, **zero `sorryAx`**, **zero `native_decide`**, and **zero custom `axiom`**:
```bash
cd spec
lake build
lake exe leansonic-gen ..
cd ..
python3 ci/check_lean_axioms.py
git diff --exit-code crates/leansonic-core/src/generated_spec.rs kernels/include/leansonic_verified.cuh
```

### Gate 2 — CUDA/PTX/SASS Static Audit & Rust Lint Cleanliness
All `sm_120a` kernels must compile with **`0` bytes stack frame (`STACK: 0`)**, **`0` spill stores/loads**, **`0` `CALL.REL.NOINC`** in hot loops, **`0` scalar 8-bit global loads (`LDG.E.U8`)**, and preserve **`>= 64 KiB` L1 cache** unless explicitly justified by `minRequiredL1KiB`:
```bash
# On a CUDA 12.8+ / 13.x host (Blackwell sm_120a):
nvcc -O3 -arch=sm_120a -Xptxas -v -I kernels -cubin kernels/leansonic_sm120.cu -o /tmp/leansonic_sm120a.cubin
python3 ci/audit_sass.py --cubin /tmp/leansonic_sm120a.cubin

nvcc -O3 -arch=sm_120a -Xptxas -v -I kernels -cubin kernels/qwen38_sm120.cu -o /tmp/qwen38_sm120a.cubin
python3 ci/audit_sass.py --cubin /tmp/qwen38_sm120a.cubin

# Template & Rust lint audit (runs on any host):
python3 ci/audit_sass.py
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
```

### Gate 3 — `0-ULP` Parity, Unit Tests & Exact-Size Memory Safety
```bash
cargo test --workspace
cargo run --release -p leansonic-cli -- verify-parity
```

#### Mandatory Exact-Size `compute-sanitizer` + Planted-Fault Red Controls
Whenever you add or modify a CUDA kernel (`kernels/*.cu`), you **must** add it to the corresponding exact-size harness (`inkling2_gpu`, `qwen38_gpu::tests::exact_size`, or `qwen38_mtp::tests::exact_size`) with exact byte-sized `DeviceBuf` allocations (`DeviceBuf` pads allocations `< 16 B` to `16 B`, so scalar pointer faults must step by `>= 16 B`) and prove **both**:
1. **Green run (`0` errors)** across all kernels under `65,536`-byte padding:
   ```bash
   compute-sanitizer --tool memcheck --padding 65536 --error-exitcode 77 \
     "$TEST_BIN" qwen38_gpu::tests::exact_size::exact_size_qwen38_kernels --exact --ignored --nocapture
   ```
2. **Planted-fault Red Control (`rc == 77`)** per added/modified kernel (`short = true` drops the final row/element so an over-read or over-write is proven detectable):
   ```bash
   LEANSONIC_EXACT_ONLY="<kernel_name>" compute-sanitizer --tool memcheck --padding 65536 --error-exitcode 77 \
     "$TEST_BIN" qwen38_gpu::tests::exact_size::exact_size_qwen38_red_control --exact --ignored
   # Must exit with status 77!
   ```

### Gate 4 — Blackwell Hardware Performance, Canary & KLD Quality Gates
- **End-to-End Canary Gate** (verifies `"Tokyo"` factual answer, `0` non-finite logits, `--spec 0 == --spec 7` `0-ULP` speculative identity when drafter is active, and `16K` `"RAVEN-731"` long-context retrieval):
  ```bash
  leansonic canary --weights "$MODEL_DIR"
  ```
- **Teacher-Forced KL-Divergence Gate** (required whenever touching quantization, scale packing, KV precision, or attention math):
  ```bash
  leansonic eval-kld --weights "$MODEL_DIR" --ref "$REF_KLD"
  ```
- **Paired `ABBA × N` Statistical Performance Gate** (eliminates GPU clock/thermal drift when claiming speedups):
  ```bash
  python3 ci/abba_stat_gate.py --help
  ```

---

## 3. Shared Blackwell GPU Etiquette (`flock` & `PCI_BUS_ID`)

Multi-GPU development hosts are shared concurrently across agents and benchmark suites. **Never launch a GPU benchmark, test, or `compute-sanitizer` job without holding the per-GPU advisory `flock` and pinning `CUDA_DEVICE_ORDER=PCI_BUS_ID`:**

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0   # Physical PCIe index on this host

exec 8>>"/tmp/leansonic-gpu-$(hostname)-0.lock"
flock 8

# Run leansonic bench / canary / compute-sanitizer inside the lock
./target/release/leansonic bench --weights "$MODEL_DIR" --tokens synth:512 --gen 64 --repeat 3
```

- Clean up temporary `/tmp/leansonic-*` build trees and scratch artifacts when your verification session finishes.
- When archiving source trees on macOS to copy to Linux GPU hosts, suppress AppleDouble extended attributes:
  ```bash
  COPYFILE_DISABLE=1 tar --no-xattrs -czf - Cargo.toml Cargo.lock crates kernels ci
  ```

---

## 4. Public-Repository Privacy & Hygiene Rules

LeanSonic is a public repository. Before every `git commit`, verify that your diff contains **zero**:
- Private LAN IPv4 addresses (`RFC 1918`: `10.x.x.x`, `172.16-31.x.x`, `192.168.x.x`) or internal SSH aliases/usernames
- Local workstation home paths (e.g. `/Users/<name>/...`)
- Raw hardware `GPU-<UUID>` serial identifiers
- Uncommitted model shards, `.cubin` binaries, or temporary `.nsys-rep` / `.sqlite` traces

Quick pre-commit check:
```bash
! git diff --cached | grep -E "Users/|192\.168\.|10\.[0-9]+\.|172\.(1[6-9]|2[0-9]|3[01])\.|GPU-[0-9a-fA-F-]{8,}"
```

---

## 5. Multi-Agent PR Workflow

1. **Coordinate on GitHub Issues**: Claim a clear, non-overlapping track or kernel slice on the active tracking Issue before editing shared hot files (`inkling2_gpu.rs`, `qwen38_gpu.rs`, `kernels/*.cu`).
2. **One Focused Hypothesis per PR**: Keep branches small and reviewable (`perf(...)`, `feat(...)`, `fix(...)`, `test(...)`, `docs(...)`).
3. **Include Reproducible Evidence in the PR Body**:
   - Exact `ptxas -v` register, SMEM, and spill numbers for touched CUDA kernels
   - Exact-size `compute-sanitizer` green (`0` errors) + red-control (`rc = 77`) counts
   - Before/after `leansonic bench` / `canary` / `eval-kld` tables with GPU SKU, resident/spilled layer counts, prompt lengths, and repeat counts
4. **Squash-Merge Only When All Gates Pass**: Wait for GitHub Actions `Gates 0-3` (and hardware `Gate 4` where applicable) to turn green before squash-merging into `main`.
