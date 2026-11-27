"""Validate a spec pricing replay directory and summarize its five paired cycles."""
import argparse
import csv
import itertools
import json
import statistics
from pathlib import Path

FIXTURES = ["q00_code_merge", "q11_json_schema", "ja_reasoning", "readme1024",
            "en_boolean", "en_sort", "ja_stock", "ja_name",
            "en_count", "en_extract", "ja_json", "ja_filter"]


def read_run(path):
    with path.open() as f:
        rows = list(csv.reader(f))
    assert [(int(r[0]), r[1]) for r in rows] == list(itertools.product(range(2), FIXTURES))
    assert all(len(r) == 12 and len(r[-1].split()) == 128 for r in rows)
    peak = json.loads(path.with_suffix(".peak").read_text())
    loader = path.with_suffix(".log").read_text().splitlines()[-1]
    assert peak["polls"] > 100 and peak["bytes"] > 0
    assert "24/24 (Spilled: 0)" in loader and "KV: fp16" in loader
    return dict(rows=rows, peak=peak, loader=loader)


def analyze(card):
    scalar_path = card / "scalar.csv"
    if not scalar_path.exists():
        return None
    scalar = read_run(scalar_path)
    assert "Spec: off" in scalar["loader"]
    gold = {r[1]: (r[2], r[5], r[6], r[-1]) for r in scalar["rows"]}
    assert all(gold[r[1]] == (r[2], r[5], r[6], r[-1]) for r in scalar["rows"])
    runs = []
    for path in sorted(card.glob("*-*-*.csv")):
        cycle, leg, arm = path.stem.split("-")
        cycle, leg = int(cycle), int(leg)
        expected = "price" if (leg in (1, 2)) ^ bool(cycle % 2) else "stock"
        assert 0 <= cycle < 5 and 0 <= leg < 4 and arm == expected
        run = read_run(path)
        assert "DSpark(K=7)" in run["loader"]
        for r in run["rows"]:
            assert gold[r[1]] == (r[2], r[5], r[6], r[-1]), (path, r[0:2])
            assert arm == "price" or r[10] == "0"
        runs.append(dict(cycle=cycle, leg=leg, arm=arm, **run))
    print(card.name, "completed", len(runs), flush=True)
    if len(runs) != 20:
        return None
    summary = {}
    for name in FIXTURES:
        paired = []
        by_arm = {}
        for arm in ["stock", "price"]:
            group = [(run["cycle"], r) for run in runs if run["arm"] == arm
                     for r in run["rows"] if r[0:2] == ["1", name]]
            assert len(group) == 10
            by_arm[arm] = dict(decode_tps=128000/statistics.mean(float(r[4]) for _, r in group),
                               prefill_ms=statistics.mean(float(r[3]) for _, r in group),
                               skips=[int(r[10]) for _, r in group])
            per_cycle = [statistics.mean(float(r[4]) for c, r in group if c == i) for i in range(5)]
            paired.append(per_cycle)
        ratios = sorted(sum(paired[0][i] for i in sample) / sum(paired[1][i] for i in sample)
                        for sample in itertools.product(range(5), repeat=5))
        by_arm["decode_ratio_ci95"] = [ratios[78], ratios[-79]]
        summary[name] = by_arm
    return dict(card=card.name, scalar=scalar, runs=runs, summary=summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("card", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.card)
    if result:
        args.output.write_text(json.dumps(result, indent=2)+"\n")
