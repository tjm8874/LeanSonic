"""Compare teacher-forced FP32 vocabulary logits on exactly the same prefixes."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

VOCAB = 248320


def run(ref, trial):
    rows = [np.loadtxt(p / "rows.tsv", dtype=np.int64, ndmin=2) for p in (ref, trial)]
    if (
        rows[0].shape != rows[1].shape
        or rows[0].shape[1] != 5
        or not np.array_equal(rows[0][:, :4], rows[1][:, :4])
    ):
        raise ValueError("unaligned or invalid teacher-forced positions")
    matrices = [
        np.memmap(p / "logits.f32", dtype="<f4", mode="r") for p in (ref, trial)
    ]
    count = len(rows[0])
    if not count or any(m.size != count * VOCAB for m in matrices):
        raise ValueError("unexpected vocabulary or row count")
    a, b = [m.reshape(count, VOCAB) for m in matrices]
    kl = []
    matches = 0
    exact = 0
    nll = [0.0, 0.0]
    for i, (x, y) in enumerate(zip(a, b)):
        target = rows[0][i, 3]
        if not 0 <= target < VOCAB:
            raise ValueError("target out of range")
        logs = []
        for arm, v in enumerate((x, y)):
            if not np.all(np.isfinite(v)):
                raise ValueError("nonfinite logits")
            shift = v.astype(np.float64) - float(v.max())
            lp = shift - np.log(np.exp(shift).sum())
            logs.append(lp)
            nll[arm] -= lp[target]
            if v.argmax() != rows[arm][i, 4]:
                raise ValueError("recorded top token differs from logits")
        kl.append(float(np.dot(np.exp(logs[0]), logs[0] - logs[1])))
        exact += int(x.tobytes() == y.tobytes())
        matches += int(x.argmax() == y.argmax())
    hashes = []
    for p in (ref, trial):
        with (p / "logits.f32").open("rb") as f:
            hashes.append(hashlib.file_digest(f, "sha256").hexdigest())
    canaries = [(p / "canaries.txt").read_text().splitlines() for p in (ref, trial)]
    if any(
        len(c) != len(set(rows[0][:, 0])) or any(len(s.split()) != 8 for s in c)
        for c in canaries
    ):
        raise ValueError("incomplete canary sequences")
    worst = int(np.argmax(kl))
    return {
        "rows": count,
        "vocabulary": VOCAB,
        "mean_kl": sum(kl) / count,
        "max_kl": kl[worst],
        "worst_case_position": rows[0][worst, :2].tolist(),
        "top1_matches": matches,
        "byte_equal_rows": exact,
        "reference_ppl": math.exp(nll[0] / count),
        "candidate_ppl": math.exp(nll[1] / count),
        "reference_sha256": hashes[0],
        "candidate_sha256": hashes[1],
        "canary_cases": len(canaries[0]),
        "matching_canaries": sum(x == y for x, y in zip(*canaries)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    result = run(args.reference, args.candidate)
    print(json.dumps(result, indent=2))
    if args.require_exact and (
        result["reference_sha256"] != result["candidate_sha256"]
        or result["matching_canaries"] != result["canary_cases"]
    ):
        raise SystemExit("lossless inference parity failed")
