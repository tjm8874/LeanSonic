#!/usr/bin/env python3
"""Gate 0 — Immutable Trust Core Guard (`ci/check_immutable_core.py`).

Prevents autonomous coding agents from modifying the formal specification core,
CI verification gates, GitHub workflows, or benchmark golden baselines
(Reward-Hacking & Self-Hosted Runner Escalation Defense).
"""

from __future__ import annotations

import argparse
import subprocess
import sys

IMMUTABLE_PREFIXES = (
    "CODEOWNERS",
    "spec/LeanSonic/Axioms.lean",
    "ci/",
    ".github/",
    "benches/goldens/",
)


def is_immutable_path(path: str) -> bool:
    norm = path.strip().removeprefix("./")
    return any(norm == p.rstrip("/") or norm.startswith(p) for p in IMMUTABLE_PREFIXES)


def changed_files(base_ref: str, head_ref: str) -> list[str]:
    out = subprocess.check_output(
        ["git", "diff", "--name-only", f"{base_ref}...{head_ref}"],
        text=True,
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def run_self_test() -> int:
    assert is_immutable_path("spec/LeanSonic/Axioms.lean")
    assert is_immutable_path("ci/audit_sass.py")
    assert is_immutable_path(".github/workflows/gate_blackwell_gpu.yml")
    assert is_immutable_path("benches/goldens/inkling2_mini_baseline.json")
    assert not is_immutable_path("spec/LeanSonic/Proofs/Inkling2Mini.lean")
    assert not is_immutable_path("crates/leansonic-core/src/kernels.rs")
    assert not is_immutable_path("kernels/moe_dual_regime.cu")
    print("[PASS] Gate 0 self-test: Immutable Trust Core path classifier verified.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 0 Immutable Trust Core Guard")
    parser.add_argument("--base", help="Base git ref/SHA")
    parser.add_argument("--head", default="HEAD", help="Head git ref/SHA")
    parser.add_argument(
        "--human-override",
        action="store_true",
        help="Explicit human owner override flag",
    )
    parser.add_argument("--self-test", action="store_true", help="Run unit self-test")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    if not args.base:
        print("[SKIP] No --base ref supplied; run with --self-test or --base <sha>.")
        return 0

    touched = [p for p in changed_files(args.base, args.head) if is_immutable_path(p)]
    if touched and not args.human_override:
        print("[FAIL] Gate 0: Autonomous PR modified Immutable Trust Core files:", file=sys.stderr)
        for p in touched:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print("[PASS] Gate 0: Mutable scope respected (0 Immutable Trust Core files modified).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
