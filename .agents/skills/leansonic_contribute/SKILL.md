---
name: leansonic_contribute
description: >-
  Workflow, 5-Gate Immutable Trust Core invariants, SM120 Blackwell kernel & Lean 4 proof checklist,
  exact-size compute-sanitizer + red-control verification, shared GPU flock etiquette, and privacy
  redaction rules for contributing PRs to LeanSonic. Activate whenever modifying LeanSonic Rust crates,
  CUDA/PTX kernels, Lean 4 proofs, model packs, benchmarks, or opening/reviewing LeanSonic Issues and PRs.
---

# LeanSonic Contributor Skill (`leansonic_contribute`)

Follow this mandatory workflow whenever developing features, optimizing NVIDIA Blackwell (`sm_120` / `sm_120a`) CUDA kernels, authoring Lean 4 proofs, or submitting PRs to **LeanSonic** (see [CONTRIBUTING.md](../../../CONTRIBUTING.md)).

---

## 1. Non-Negotiable Architecture & Trust Boundaries

1. **Immutable Trust Core (`Gate 0` — ZERO diff allowed)**:
   Never modify `spec/LeanSonic/Axioms.lean`, `ci/**`, `.github/**`, `CODEOWNERS`, or `benches/goldens/**`.
2. **Never Manually Edit `@generated` Files**:
   `crates/leansonic-core/src/generated_spec.rs` and `kernels/include/leansonic_verified.cuh` are emitted strictly by `cd spec && lake exe leansonic-gen ..`. For NVRTC compatibility, strip `#include <stdint.h>` at runtime in Rust or guard with `#ifndef __CUDACC_RTC__` inside `.cu` files — never touch `leansonic_verified.cuh`.
3. **Zero Third-Party Rust Crates**:
   All workspace crates (`leansonic-core`, `leansonic-tuner`, `leansonic-cli`) must remain 100% pure Rust `std`.
4. **Preserve Historical Benchmark Targets**:
   Never modify `initial_basalt_targets` in `docs/benchmarks/targets.json`.

---

## 2. NVIDIA Blackwell (`sm_120` / `sm_120a`) Discrete Cliff Laws

Every warp-sized kernel in `kernels/*.cu` must have a matching `KernelResourceSpec` proven `ValidSm120Kernel` in `spec/LeanSonic/Proofs/*.lean`:

| Resource | `sm_120` Hardware Law | Practical Rule |
| :--- | :--- | :--- |
| **Combined L1 + SMEM** | `128 KiB` / SM | Carveouts are discrete: `{0, 8, 16, 32, 64, 100} KiB`. |
| **System Reserved SMEM** | `1,024 B` / CTA | `totalSmem = ctasPerSm * (userSmem + pad + 1024)`. |
| **`64 KiB` L1 Cliff** | `totalSmem <= 65,536 B` | Crossing `64 KiB` jumps to the `100 KiB` carveout, shrinking L1 from `64 KiB` to `28 KiB`! For `2` CTAs/SM under `64 KiB`, keep `userSmem <= 31,744 B`; for `3` CTAs/SM, `<= 20,821 B`; for `5` CTAs/SM, `<= 12,083 B`. |
| **Register File** | `65,536` 32-bit regs / SM | Allocated in multiples of `8` regs/thread (`alignedRegs * threads * ctasPerSm <= 65,536`). |
| **Spills & Calls** | `STACK: 0`, `0` `CALL.REL.NOINC` | Enforced on compiled `.cubin` by `python3 ci/audit_sass.py --cubin <file.cubin>`. |
| **Vectorized Loads** | `0` `LDG.E.U8` | Use `uint4` (`LDG.E.128`) or `uint32_t` (`LDG.E.U32`) + SWAR/`__byte_perm`; never issue scalar byte global loads. |

---

## 3. Mandatory Pre-PR Verification Sequence

Execute all steps before opening or updating a PR:

### Step A — Gate 0 & Gate 1 (Lean 4 Formal Proofs)
```bash
python3 ci/check_immutable_core.py --base origin/main
cd spec && lake build && lake exe leansonic-gen .. && cd ..
python3 ci/check_lean_axioms.py
git diff --exit-code crates/leansonic-core/src/generated_spec.rs kernels/include/leansonic_verified.cuh
```
- Forbidden tokens in `spec/`: `sorry`, `sorryAx`, `native_decide`, custom `axiom`. Use `decide`, `bv_decide`, `omega`, and inductive structural proofs.

### Step B — Gate 2 (CUDA `ptxas` + SASS Audit + Rust Clippy)
```bash
# Compile target cubin(s) with -Xptxas -v and audit SASS:
nvcc -O3 -arch=sm_120a -Xptxas -v -I kernels -cubin kernels/leansonic_sm120.cu -o target/release/leansonic_sm120a.cubin
python3 ci/audit_sass.py --cubin target/release/leansonic_sm120a.cubin

nvcc -O3 -arch=sm_120a -Xptxas -v -I kernels -cubin kernels/qwen38_sm120.cu -o target/release/qwen38_sm120a.cubin
python3 ci/audit_sass.py --cubin target/release/qwen38_sm120a.cubin

# Formatting & strict lints:
python3 ci/audit_sass.py
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
```

### Step C — Gate 3 (`0-ULP` Parity & Exact-Size `compute-sanitizer` + Red Controls)
```bash
cargo test --workspace
cargo run --release -p leansonic-cli -- verify-parity
```
When adding or changing any CUDA kernel, acquire the GPU `flock` (Section 4) and run **both** the green exact-size pass (`0` errors) and per-kernel planted-fault red controls (`rc == 77`):
```bash
# Remember: DeviceBuf::alloc_uninit clamps <16B to 16B; plant >=16B faults for scalar buffers!
compute-sanitizer --tool memcheck --padding 65536 --error-exitcode 77 \
  "$TEST_BIN" <module>::tests::exact_size::<green_test_name> --exact --ignored --nocapture

for k in <touched_kernels...>; do
  set +e
  LEANSONIC_EXACT_ONLY="$k" compute-sanitizer --tool memcheck --padding 65536 --error-exitcode 77 \
    "$TEST_BIN" <module>::tests::exact_size::<red_control_test_name> --exact --ignored
  rc=$?
  set -e
  test "$rc" -eq 77 || { echo "RED CONTROL FAILED for $k (rc=$rc)"; exit 1; }
done
```

### Step D — Gate 4 (Hardware Canary, KLD & Throughput Verification)
```bash
# Always rebuild BOTH --bins and --tests before benchmarking!
cargo build --release --workspace --bins --tests

# Factual + 16K needle-in-a-haystack + speculative 0-ULP check:
./target/release/leansonic canary --weights "$MODEL_DIR"

# Throughput measurement (repeat >= 3 so warm steady-state is captured):
./target/release/leansonic bench --weights "$MODEL_DIR" --tokens synth:512 --gen 64 --repeat 3
```

---

## 4. Shared GPU Locking & Shell Discipline

1. **Always Lock the Physical GPU**:
   ```bash
   export CUDA_DEVICE_ORDER=PCI_BUS_ID
   export CUDA_VISIBLE_DEVICES=0
   exec 8>>"/tmp/leansonic-gpu-$(hostname)-0.lock"
   flock 8
   ```
2. **Safe GitHub CLI Markdown**:
   Always write PR/Issue bodies to a scratch file and pass `--body-file <file.md>` to `gh pr create` / `gh issue comment` (never inline backticks inside double-quoted shell strings).
3. **macOS Archive Hygiene**:
   Use `COPYFILE_DISABLE=1 tar --no-xattrs` so `._*` AppleDouble files never pollute remote build trees.
4. **Public-Repo Privacy Check Before Commit**:
   ```bash
   ! git diff --cached | grep -E "Users/|192\.168\.|10\.[0-9]+\.|172\.(1[6-9]|2[0-9]|3[01])\.|GPU-[0-9a-fA-F-]{8,}"
   ```
