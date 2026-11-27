"""Score the 20 forced distributions emitted by the Inkling prefill boundary probes."""
import argparse
import json
from pathlib import Path

import numpy as np


def read_rows(path):
    rows = np.fromfile(path, dtype="<f4").reshape(20, 200_058)
    if not np.isfinite(rows).all():
        raise ValueError(f"nonfinite logits in {path}")
    return rows


def log_probabilities(rows):
    stable = rows.astype(np.float64)
    stable -= stable.max(axis=1, keepdims=True)
    return stable - np.log(np.exp(stable).sum(axis=1, keepdims=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("observations", type=Path)
    args = parser.parse_args()
    reference = read_rows(args.reference)
    log_ref = log_probabilities(reference)
    wide, narrow = [read_rows(args.observations / name) for name in ("w256.f32", "w64.f32")]
    result = {
        "unequal_values": int(np.count_nonzero(wide.view("u4") != narrow.view("u4"))),
        "top_changes": int(np.count_nonzero(wide.argmax(1) != narrow.argmax(1))),
    }
    for width, rows in [(256, wide), (64, narrow)]:
        divergence = (np.exp(log_ref) * (log_ref - log_probabilities(rows))).sum(axis=1)
        result[width] = {
            "mean_kl": float(divergence.mean()),
            "worst_kl": float(divergence.max()),
            "top1_matches": int(np.count_nonzero(rows.argmax(1) == reference.argmax(1))),
            "position_kl": divergence.tolist(),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
