"""Compare raw, little-endian FP32 rows. Accumulate probabilities in float64."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

VOCAB = 200_058


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(directory, case, count):
    path = directory / (case + ".f32")
    if path.stat().st_size != count * VOCAB * 4:
        raise ValueError(f"incorrect logit file length: {path}")
    return np.memmap(path, dtype="<f4", mode="r", shape=(count, VOCAB))


def log_probs(row):
    x = row.astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError("non-finite logit")
    x -= np.max(x)
    return x - np.log(np.exp(x).sum())


def summarize(observations):
    values = [r["kl"] for r in observations]
    worst = max(observations, key=lambda r: r["kl"])
    return dict(rows=len(values), mean_kl=float(np.mean(values)), worst=worst,
                p99_kl=float(np.percentile(values, 99)),
                matches=sum(r["top"] == r["bf16_top"] for r in observations),
                ppl=float(np.exp(np.mean([r["nll"] for r in observations]))),
                bf16_ppl=float(np.exp(np.mean([r["bf16_nll"] for r in observations]))))


def evaluate(fixture_root, reference, candidate):
    fixtures = json.loads(Path(__file__).with_name("fixtures.json").read_text())
    observations = []
    manifest = {}
    for fixture in fixtures:
        case = fixture["id"]
        directory = fixture_root / case
        for field in ["prompt", "answer"]:
            if (directory / (field + ".txt")).read_text() != fixture[field]:
                raise ValueError("fixture text changed: " + case)
        manifest[case] = {key: digest(directory / (key + ".ids")) for key in ("prefix", "answer")}
        targets = list(map(int, (directory / "answer.ids").read_text().split()))
        ref = rows(reference, case, len(targets))
        obs = rows(candidate, case, len(targets))
        for i, target in enumerate(targets):
            if not 0 <= target < VOCAB:
                raise ValueError("invalid target")
            p = log_probs(ref[i])
            q = log_probs(obs[i])
            kl = float(np.dot(np.exp(p), p - q))
            if kl < -1e-12:
                raise ValueError("negative KL")
            observations.append(dict(case=case, language=fixture["language"], offset=i,
                                     target=target, kl=kl, top=int(q.argmax()), bf16_top=int(p.argmax()),
                                     nll=float(-q[target]), bf16_nll=float(-p[target])))
    for f in (fixtures[0], fixtures[-1]):
        for directory in (reference, candidate):
            case = f["id"]
            if digest(directory / (case + ".f32")) != digest(directory / (case + "_repeat.f32")):
                raise ValueError("reset control diverged: " + case)
    groups = {lang: summarize([r for r in observations if r["language"] == lang])
              for lang in sorted({r["language"] for r in observations})}
    per_case = {f["id"]: summarize([r for r in observations if r["case"] == f["id"]]) for f in fixtures}
    return dict(all=summarize(observations), by_prompt_language=groups, cases=per_case,
                token_sha256=manifest, observations=observations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("fixture", "reference", "candidate", "output"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    report = evaluate(args.fixture, args.reference, args.candidate)
    args.output.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
