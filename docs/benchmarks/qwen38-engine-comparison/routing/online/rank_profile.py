"""Rank the fixed 19-layer-equivalent profile by calibration decode frequency."""
import argparse
import ast
from collections import Counter
from pathlib import Path


def rank(trace, profile, destination):
    lengths = [len(line.split()) for line in (trace / "route-fixtures.ids").read_text().splitlines()]
    counts = [Counter() for _ in range(48)]
    case = -1
    records = 0
    with (trace / "route-output.log").open() as log:
        for line in log:
            if not line.startswith("ROUTE "):
                continue
            _, pos, layer, ids = line.split(" ", 3)
            pos, layer = int(pos), int(layer)
            if pos == layer == 0:
                case += 1
            if case == 4:
                break
            if case >= 0 and pos >= lengths[case]:
                counts[layer].update(ast.literal_eval(ids))
                records += 1
    if records != 4 * 48 * 31:
        raise ValueError(f"expected four full calibration traces, found {records} layer records")
    with destination.open("x") as out:
        for count, line in zip(counts, profile.read_text().splitlines(), strict=True):
            print(*sorted(map(int, line.split()), key=lambda expert: (-count[expert], expert)), file=out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trace", "profile", "destination"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    rank(args.trace, args.profile, args.destination)
