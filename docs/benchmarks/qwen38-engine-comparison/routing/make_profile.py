"""Build per-layer expert profiles using only the first four completed decode traces."""

import ast
import random
import sys
from collections import Counter
from pathlib import Path

n = int(sys.argv[1])
assert 0 < n < 512
lengths = [len(s.split()) for s in Path("route-fixtures.ids").read_text().splitlines()]
counts = [Counter() for _ in range(48)]
case = -1
steps = 0
with open("route-output.log") as f:
    for line in f:
        if not line.startswith("ROUTE "):
            continue
        _, p, l, tail = line.split(" ", 3)
        p, l = int(p), int(l)
        if p == 0 and l == 0:
            case += 1
        if case == 4:
            break
        if p < lengths[case]:
            continue
        ids = ast.literal_eval(tail)
        assert len(set(ids)) == len(ids) == 10
        counts[l].update(ids)
        steps += 1
assert steps == 4 * 31 * 48
for kind in ("calibrated", "random"):
    rng = random.Random(20261128)
    with open(f"profile-{kind}-{n}.txt", "x") as out:
        for c in counts:
            ranked = sorted(range(512), key=lambda i: (-c[i], i))
            if kind == "random":
                rng.shuffle(ranked)
            print(*sorted(ranked[:n]), file=out)
