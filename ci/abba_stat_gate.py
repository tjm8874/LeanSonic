#!/usr/bin/env python3
"""Gate 4 — Same-Session Interleaved ABBA Statistical Benchmark Gate (`ci/abba_stat_gate.py`).

Cancels GPU thermal & boost clock drift by interleaving Baseline (`A`) and Candidate (`B`)
in `A -> B -> B -> A` rounds within the same session, then computing a deterministic
paired bootstrap 95% confidence interval on `speedup_pct = (A - B) / A * 100`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import sys


def evaluate_abba_rounds(
    rounds: list[dict[str, float]],
    max_regression_pct: float = 0.5,
    min_speedup_pct: float = 0.0,
    bootstrap_iters: int = 2000,
) -> tuple[bool, dict[str, float]]:
    if len(rounds) < 4:
        raise ValueError("At least 4 interleaved ABBA rounds are required for Gate 4")

    rel_gains: list[float] = []
    for r in rounds:
        a_mean = 0.5 * (r["a1_ms"] + r["a2_ms"])
        b_mean = 0.5 * (r["b1_ms"] + r["b2_ms"])
        rel_gains.append(((a_mean - b_mean) / a_mean) * 100.0)

    rng = random.Random(20261125)
    n = len(rel_gains)
    boot_means = sorted(
        statistics.fmean(rel_gains[rng.randrange(n)] for _ in range(n))
        for _ in range(bootstrap_iters)
    )
    ci95_lo = boot_means[int(0.025 * bootstrap_iters)]
    ci95_hi = boot_means[int(0.975 * bootstrap_iters)]
    mean_gain = statistics.fmean(rel_gains)

    non_regressing = ci95_lo >= -max_regression_pct
    meets_target = mean_gain >= min_speedup_pct and (min_speedup_pct == 0.0 or ci95_lo > 0.0)
    passed = non_regressing and meets_target

    return passed, {
        "mean_speedup_pct": round(mean_gain, 4),
        "ci95_lo_pct": round(ci95_lo, 4),
        "ci95_hi_pct": round(ci95_hi, 4),
        "rounds": float(n),
    }


def run_self_test() -> int:
    win_rounds = [
        {"a1_ms": 10.02, "b1_ms": 9.61, "b2_ms": 9.63, "a2_ms": 10.00},
        {"a1_ms": 10.05, "b1_ms": 9.64, "b2_ms": 9.62, "a2_ms": 10.03},
        {"a1_ms": 10.01, "b1_ms": 9.60, "b2_ms": 9.61, "a2_ms": 10.02},
        {"a1_ms": 10.04, "b1_ms": 9.63, "b2_ms": 9.65, "a2_ms": 10.01},
        {"a1_ms": 10.03, "b1_ms": 9.62, "b2_ms": 9.60, "a2_ms": 10.04},
    ]
    ok, stats = evaluate_abba_rounds(win_rounds, min_speedup_pct=1.0)
    assert ok and stats["ci95_lo_pct"] > 3.0, f"expected win, got {stats}"

    reg_rounds = [
        {"a1_ms": 8.59, "b1_ms": 8.95, "b2_ms": 8.92, "a2_ms": 8.60},
        {"a1_ms": 8.61, "b1_ms": 8.94, "b2_ms": 8.96, "a2_ms": 8.58},
        {"a1_ms": 8.60, "b1_ms": 8.93, "b2_ms": 8.95, "a2_ms": 8.59},
        {"a1_ms": 8.58, "b1_ms": 8.91, "b2_ms": 8.94, "a2_ms": 8.61},
    ]
    ok_reg, _ = evaluate_abba_rounds(reg_rounds)
    assert not ok_reg, "expected regression to fail Gate 4"
    print("[PASS] Gate 4 self-test: Paired bootstrap ABBA evaluator verified.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 4 Same-Session ABBA Statistical Gate")
    parser.add_argument("--json", help="Path to ABBA rounds JSON file")
    parser.add_argument("--max-regression-pct", type=float, default=0.5)
    parser.add_argument("--min-speedup-pct", type=float, default=0.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    if not args.json:
        print("[SKIP] Supply --json <abba_results.json> or --self-test.")
        return 0

    payload = json.loads(pathlib.Path(args.json).read_text(encoding="utf-8"))
    passed, summary = evaluate_abba_rounds(
        payload["rounds"],
        max_regression_pct=args.max_regression_pct,
        min_speedup_pct=args.min_speedup_pct,
    )
    print(json.dumps(summary, indent=2))
    if not passed:
        print("[FAIL] Gate 4: ABBA statistical performance gate rejected candidate.", file=sys.stderr)
        return 1
    print("[PASS] Gate 4: Candidate passed ABBA statistical performance gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
