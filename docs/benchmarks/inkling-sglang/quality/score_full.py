"""Compare deployed vocabularies, including the mass of additional SGLang output IDs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group(rows):
    return {
        "rows": len(rows),
        "mean_kl": float(np.mean([r["kl"] for r in rows])),
        "worst": max(rows, key=lambda r: r["kl"]),
        "matches": sum(r["top"] == r["bf16_top"] for r in rows),
        "ppl": float(np.exp(np.mean([r["nll"] for r in rows]))),
        "bf16_ppl": float(np.exp(np.mean([r["bf16_nll"] for r in rows]))),
        "max_extra_probability": max(r["extra_probability"] for r in rows),
        "extra_top_count": sum(r["top"] >= 200058 for r in rows),
    }


def logprob(row):
    x = row.astype(np.float64)
    assert np.isfinite(x[:200058]).all()
    assert not np.isnan(x[200058:]).any() and not np.isposinf(x[200058:]).any()
    x -= x.max()
    return x - np.log(np.exp(x).sum())


def run(fixtures, root, reference, observed):
    rows = []
    hashes = {}
    for c in json.loads(fixtures.read_text()):
        name = c["id"]
        answer = list(map(int, (root / name / "answer.ids").read_text().split()))
        x = np.memmap(reference / (name + ".f32"), dtype="<f4", mode="r").reshape(
            -1, 200058
        )
        y = np.memmap(observed / (name + ".f32"), dtype="<f4", mode="r").reshape(
            -1, 201024
        )
        assert len(x) == len(y) == len(answer)
        hashes[name] = {
            key: digest(root / name / (key + ".ids")) for key in ("prefix", "answer")
        }
        for i, target in enumerate(answer):
            p, q = logprob(x[i]), logprob(y[i])
            kl = float(np.dot(np.exp(p), p - q[:200058]))
            assert kl >= -1e-12 and 0 <= target < 200058
            rows.append(
                {
                    "case": name,
                    "language": c["language"],
                    "offset": i,
                    "target": target,
                    "kl": kl,
                    "top": int(q.argmax()),
                    "bf16_top": int(p.argmax()),
                    "nll": float(-q[target]),
                    "bf16_nll": float(-p[target]),
                    "extra_probability": float(np.exp(q[200058:]).sum()),
                }
            )
    for repeat in root.glob("*_repeat"):
        for folder in (reference, observed):
            assert digest(folder / (repeat.name + ".f32")) == digest(
                folder / (repeat.name.removesuffix("_repeat") + ".f32")
            )
    assert len(rows) == 613
    return {
        "all": group(rows),
        "by_language": {
            k: group([r for r in rows if r["language"] == k]) for k in ("en", "ja")
        },
        "input_sha256": hashes,
        "observations": rows,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("fixtures", "root", "reference", "observed", "output"):
        p.add_argument(key, type=Path)
    a = p.parse_args()
    result = run(a.fixtures, a.root, a.reference, a.observed)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
