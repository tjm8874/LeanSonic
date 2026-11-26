#!/usr/bin/env python3
"""Gate 1 — Lean 4 Axiom & `sorry` Auditor (`ci/check_lean_axioms.py`).

Enforces two independent formal-verification invariants:
1. Lexical/AST Hygiene: Strips comments across all `.lean` files under `spec/` and
   rejects any occurrence of `sorry`, `admit`, `axiom`, or `unsafe`.
2. Kernel Environment Audit: Invokes `lake env lean` on `#print axioms` for every
   Phase 1 theorem and ensures `sorryAx` or custom axioms never appear.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tempfile

FORBIDDEN_TOKENS = re.compile(r"\b(sorry|admit|axiom|unsafe)\b")

# Standard Lean 4 foundational kernel axioms ONLY.
# Explicitly blocks `sorryAx`, `Lean.ofReduceBool` (`native_decide`), `Lean.trustCompiler`,
# and any user-declared `axiom`.
ALLOWED_KERNEL_AXIOMS = {
    "propext",
    "Classical.choice",
    "Quot.sound",
}

REQUIRED_THEOREMS = [
    "LeanSonic.Proofs.inkling2_schedule_valid",
    "LeanSonic.Proofs.inkling2_visit_count_eq_48",
    "LeanSonic.Proofs.inkling2_full_attn_visits_eq_12",
    "LeanSonic.Proofs.inkling2_swa_visits_eq_36",
    "LeanSonic.Proofs.qwen38_schedule_valid",
    "LeanSonic.Proofs.phase1_hybrid_pool_safe",
    "LeanSonic.Proofs.phase1_all_kernels_valid",
    "LeanSonic.Proofs.attn_blocklist_smem_safe",
    "LeanSonic.Proofs.e2m1_all_16_codes_exact",
    "LeanSonic.Proofs.simt_and_mma_16bx2_0ulp_eq",
    "LeanSonic.Proofs.phase1_serving_scheduler_valid",
    "LeanSonic.Proofs.phase1_slot_bands_disjoint",
    "LeanSonic.Proofs.canonical_sse_trace_always_valid",
    "LeanSonic.Proofs.sse_without_done_is_rejected",
    "LeanSonic.Proofs.sse_frame_after_done_is_rejected",
    "LeanSonic.Proofs.solo_prompt_eq_serve_batch_slot",
    "LeanSonic.Proofs.openai_usage_total_exact",
    "LeanSonic.Proofs.spilledInkling2On5070Ti_valid",
    "LeanSonic.Proofs.residentInkling2OnPro5000_zero_spill_valid",
    "LeanSonic.Proofs.inkling2_48visit_ping_pong_safe",
    "LeanSonic.Proofs.tiered_fetch_eq_golden",
    "LeanSonic.Proofs.resident_eq_spilled_moe_0ulp",
    "LeanSonic.Proofs.qwen38_engram_51b_offload_valid",
    "LeanSonic.Proofs.scanTrigrams_append_exact",
]


def strip_lean_comments(src: str) -> str:
    """Remove nested `/- ... -/` block comments and `-- ...` line comments."""
    out: list[str] = []
    i = 0
    depth = 0
    n = len(src)
    while i < n:
        if i + 1 < n and src[i : i + 2] == "/-":
            depth += 1
            i += 2
        elif depth > 0 and i + 1 < n and src[i : i + 2] == "-/":
            depth -= 1
            i += 2
        elif depth == 0 and i + 1 < n and src[i : i + 2] == "--":
            while i < n and src[i] != "\n":
                i += 1
        else:
            if depth == 0:
                out.append(src[i])
            i += 1
    return "".join(out)


def audit_sources(spec_dir: pathlib.Path) -> list[str]:
    errors: list[str] = []
    for lean_file in sorted(spec_dir.rglob("*.lean")):
        if ".lake" in lean_file.parts:
            continue
        cleaned = strip_lean_comments(lean_file.read_text(encoding="utf-8"))
        for match in FORBIDDEN_TOKENS.finditer(cleaned):
            errors.append(f"{lean_file}: forbidden keyword `{match.group(1)}` detected")
    return errors


def audit_kernel_axioms(spec_dir: pathlib.Path) -> list[str]:
    probe_lines = [
        "import LeanSonic.Proofs.Inkling2Mini",
        "import LeanSonic.Proofs.ServingAndPrompt",
        "import LeanSonic.Proofs.SpilledAndEngram",
    ]
    for thm in REQUIRED_THEOREMS:
        probe_lines.append(f"#print axioms {thm}")
    probe_code = "\n".join(probe_lines) + "\n"

    with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False, encoding="utf-8") as tmp:
        tmp.write(probe_code)
        tmp_path = pathlib.Path(tmp.name)

    try:
        proc = subprocess.run(
            ["lake", "env", "lean", str(tmp_path)],
            cwd=str(spec_dir),
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        return [f"`lake env lean` failed:\n{proc.stdout}\n{proc.stderr}"]

    errors: list[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if "does not depend on any axioms" in line:
            continue
        if "depends on axioms:" in line:
            _, rhs = line.split("depends on axioms:", 1)
            axioms = [a.strip(" []") for a in rhs.split(",") if a.strip(" []")]
            for ax in axioms:
                if ax not in ALLOWED_KERNEL_AXIOMS:
                    errors.append(f"Disallowed axiom `{ax}` in: {line}")
        else:
            errors.append(f"Unexpected `#print axioms` output: {line}")
    return errors


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    spec_dir = repo_root / "spec"

    src_errors = audit_sources(spec_dir)
    if src_errors:
        for e in src_errors:
            print(f"[FAIL] {e}", file=sys.stderr)
        return 1

    ax_errors = audit_kernel_axioms(spec_dir)
    if ax_errors:
        for e in ax_errors:
            print(f"[FAIL] {e}", file=sys.stderr)
        return 1

    print(
        f"[PASS] Gate 1: Verified {len(REQUIRED_THEOREMS)} Lean 4 theorems "
        "(0 `sorry`, 0 custom `axiom`, kernel environment clean)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
