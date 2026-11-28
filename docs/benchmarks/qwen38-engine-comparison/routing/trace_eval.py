"""Evaluate expert residency on held-out prompts at the stock 18-layer byte budget."""

import ast
import json
from collections import Counter
from pathlib import Path

d = Path.cwd()
lengths = [len(s.split()) for s in (d / "route-fixtures.ids").read_text().splitlines()]
cases = [{} for _ in lengths]
case = -1
for line in (d / "route-output.log").open():
    if not line.startswith("ROUTE "):
        continue
    _, p, l, tail = line.split(" ", 3)
    p, l = int(p), int(l)
    if p == 0 and l == 0:
        case += 1
    ids = ast.literal_eval(tail)
    if (
        not 0 <= case < len(lengths)
        or len(ids) != 10
        or len(set(ids)) != 10
        or min(ids) < 0
        or max(ids) >= 512
        or not 0 <= l < 48
    ):
        raise ValueError("invalid routed IDs")
    if (p, l) in cases[case]:
        raise ValueError("duplicate route")
    cases[case][p, l] = ids
if case != len(lengths) - 1:
    raise ValueError("missing cases")
for n, case in zip(lengths, cases):
    if set(case) != {(p, l) for p in range(n + 31) for l in range(48)}:
        raise ValueError("incomplete route trace")
if (d / "route-golden.txt").read_bytes() != (d / "route-output.txt").read_bytes():
    raise ValueError("trace altered generation")

# Match the measured stock residency: 18 * 512 identical-sized expert slots.
capacity = 18 * 512
counts = Counter(
    (l, e)
    for i, case in enumerate(cases[:4])
    for (pos, l), ids in case.items()
    if pos >= lengths[i]
    for e in ids
)
# Deterministic ties on unseen experts. Calibrate only on the 31 decode steps per training case.
rank = sorted(
    ((l, e) for l in range(48) for e in range(512)), key=lambda x: (-counts[x], x)
)
chosen = set(rank[:capacity])
fixed = {(l, e) for l in range(18) for e in range(512)}
# Fixed equal capacity per layer protects underrepresented layers.
equal = set()
for layer in range(48):
    tier = sorted(range(512), key=lambda e: (-counts[layer, e], e))
    equal.update((layer, e) for e in tier[: capacity // 48])
assert len(chosen) == len(fixed) == len(equal) == capacity
samples = []
for i, case in enumerate(cases[4:], 4):
    selected = [
        (l, e) for (pos, l), ids in case.items() if pos >= lengths[i] for e in ids
    ]
    samples.append(
        {
            "case": i,
            "total": len(selected),
            "stock": sum(x in fixed for x in selected),
            "calibrated": sum(x in chosen for x in selected),
            "equal_layer_quota": sum(x in equal for x in selected),
        }
    )
res = {
    "slot_bytes": 3072000,
    "capacity": capacity,
    "calibration_cases": 4,
    "holdout_cases": 4,
    "generated_tokens_per_case": 32,
    "decode_steps_scored_per_case": 31,
    "train_observed_distinct": len(counts),
    "samples": samples,
}
for key in ("stock", "calibrated", "equal_layer_quota"):
    res[key + "_holdout_hit_fraction"] = sum(r[key] for r in samples) / sum(
        r["total"] for r in samples
    )
print(json.dumps(res, indent=2))
