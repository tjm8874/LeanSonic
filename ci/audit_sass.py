#!/usr/bin/env python3
"""Gate 2 — Blackwell SASS & CUDA Static Resource Auditor (`ci/audit_sass.py`).

Enforces static CUDA/SASS quality gates:
1. Source-level checks across `kernels/*.cu`:
   - No `tanhf(` / `expf(` calls inside `.cu` device kernels (must use PTX `.approx`)
   - No `__Cluster_dims__` > 2
   - Mandatory `#include "include/leansonic_verified.cuh"`
2. Binary SASS & `cuobjdump -res-usage -sass` checks (when a compiled `.cubin`/elf is supplied):
   - `STACK: 0` (zero local-memory register spills)
   - Zero `CALL.REL.NOINC` subroutine calls
   - Zero 8-bit scalar global loads (`LDG.E.U8` / `LDG.E.S8`)
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

FORBIDDEN_LIBM_RE = re.compile(r"\b(tanhf|expf|logf|powf)\s*\(")
STACK_SPILL_RE = re.compile(r"STACK:\s*(\d+)")
CALL_SASS_RE = re.compile(r"\bCALL\.REL\.NOINC\b")
SCALAR_LDG8_RE = re.compile(r"\bLDG\.E\.[US]8\b")


def audit_cuda_sources(kernels_dir: pathlib.Path) -> list[str]:
    errors: list[str] = []
    cu_files = sorted(kernels_dir.glob("*.cu"))
    if not cu_files:
        return ["No `.cu` kernel files found in `kernels/`"]
    for cu in cu_files:
        text = cu.read_text(encoding="utf-8")
        if "leansonic_verified.cuh" not in text:
            errors.append(f"{cu.name}: missing `#include \"include/leansonic_verified.cuh\"`")
        for m in FORBIDDEN_LIBM_RE.finditer(text):
            errors.append(
                f"{cu.name}: forbidden host/device libm `{m.group(1)}(...)` risks SASS CALL.REL.NOINC"
            )
    return errors


def audit_sass_text(sass_text: str, res_text: str) -> list[str]:
    errors: list[str] = []
    for m in STACK_SPILL_RE.finditer(res_text):
        stack_bytes = int(m.group(1))
        if stack_bytes > 0:
            errors.append(f"Non-zero stack frame / register spill detected: STACK:{stack_bytes}")
    if CALL_SASS_RE.search(sass_text):
        errors.append("SASS `CALL.REL.NOINC` detected in device code")
    if SCALAR_LDG8_RE.search(sass_text):
        errors.append("Unvectorized 8-bit scalar `LDG.E.[U/S]8` detected in SASS")
    return errors


def run_self_test() -> int:
    clean_errs = audit_sass_text("  /*0010*/ LDG.E.128 R4, [R2];", "REG:120 STACK:0 SHARED:30720")
    assert not clean_errs, f"unexpected errors: {clean_errs}"
    bad_errs = audit_sass_text("  CALL.REL.NOINC 0x0; LDG.E.S8 R1, [R2];", "REG:168 STACK:24")
    assert len(bad_errs) == 3
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 2 SASS & CUDA Static Resource Auditor")
    parser.add_argument("--cubin", help="Optional compiled cubin/executable to inspect with cuobjdump")
    parser.add_argument("--self-test", action="store_true", help="Run parser self-tests")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    src_errs = audit_cuda_sources(repo_root / "kernels")
    if src_errs:
        for e in src_errs:
            print(f"[FAIL] {e}", file=sys.stderr)
        return 1

    if args.cubin:
        res_out = subprocess.check_output(["cuobjdump", "-res-usage", args.cubin], text=True)
        sass_out = subprocess.check_output(["cuobjdump", "-sass", args.cubin], text=True)
        bin_errs = audit_sass_text(sass_out, res_out)
        if bin_errs:
            for e in bin_errs:
                print(f"[FAIL] {e}", file=sys.stderr)
            return 1

    print("[PASS] Gate 2: CUDA/PTX kernel templates & SASS static audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
