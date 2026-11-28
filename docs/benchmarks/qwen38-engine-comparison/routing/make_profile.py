"""Build per-layer expert profiles using only the first four completed decode traces."""

import argparse
import ast
import random
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
capacity = parser.add_mutually_exclusive_group(required=True)
capacity.add_argument("per_layer", type=int, nargs="?", help="equal expert count in each of 48 layers")
capacity.add_argument("--resident-layers", type=int, help="match the number of records in N full layers")
args = parser.parse_args()
if args.per_layer is not None:
    if not 0 < args.per_layer < 512:
        parser.error("per-layer expert count must be from 1 to 511")
    quotas = [args.per_layer] * 48
    label = str(args.per_layer)
else:
    if not 0 < args.resident_layers < 48:
        parser.error("resident layer count must be from 1 to 47")
    count, extra = divmod(512 * args.resident_layers, 48)
    quotas = [count + (layer < extra) for layer in range(48)]
    label = f"{args.resident_layers}layers"
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
    with open(f"profile-{kind}-{label}.txt", "x") as out:
        for c, n in zip(counts, quotas, strict=True):
            ranked = sorted(range(512), key=lambda i: (-c[i], i))
            if kind == "random":
                rng.shuffle(ranked)
            print(*sorted(ranked[:n]), file=out)
