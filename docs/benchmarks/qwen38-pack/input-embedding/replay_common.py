"""Select the same teacher rows as Basalt's common-position PPL interface."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def replay(teacher, fixtures, manifest):
    spec = json.loads(manifest.read_bytes())
    rows = np.loadtxt(teacher / "rows.tsv", dtype=np.int64, ndmin=2)
    vocab = spec["vocabulary"]
    logits = np.memmap(teacher / "logits.f32", dtype="<f4", mode="r").reshape(len(rows), vocab)
    positions = {(int(case), int(pos)): index for index, (case, pos, *_) in enumerate(rows)}
    ids = [list(map(int, s.split())) for s in fixtures.read_text().splitlines()]
    ctx = min(map(len, ids))
    expected = [(i, j) for i in range(len(ids)) for j in range(ctx // 2, ctx - 1)]
    if len(rows) != len(positions) or spec["context"] != ctx or expected != [tuple(r[:2]) for r in spec["rows"]]:
        raise ValueError("reference context or positions differ")
    for case, pos, target, _ in spec["rows"]:
        i = positions[case, pos]
        if rows[i, 2:4].tolist() != ids[case][pos:pos+2] or target != ids[case][pos+1]:
            raise ValueError("prefix and target mismatch")
        sys.stdout.buffer.write(logits[i].tobytes())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("teacher", "fixtures", "manifest"):
        p.add_argument(key, type=Path)
    args = p.parse_args()
    replay(args.teacher, args.fixtures, args.manifest)
