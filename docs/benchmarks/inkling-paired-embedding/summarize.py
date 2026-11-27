"""Paired cycle bootstrap of percent elapsed-time change (candidate/base - 1)."""
import csv
import json
import random
import statistics
import sys
from pathlib import Path

def summarize(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 44
    cycles = []
    for cycle in range(11):
        legs = [(int(r['leg']), int(r['arm']), int(r['ns'])) for r in rows if int(r['cycle']) == cycle]
        assert [arm for _, arm, _ in legs] == ([0, 1, 1, 0] if cycle % 2 else [1, 0, 0, 1])
        assert [leg for leg, _, _ in legs] == list(range(4))
        if cycle:
            cycles.append([statistics.mean(ns / 1e6 for _, a, ns in legs if a == arm) for arm in (0, 1)])
    def stats(c):
        base, trial = map(statistics.mean, zip(*c))
        return base, trial, 100 * (trial / base - 1)
    base, trial, pct = stats(cycles)
    rng = random.Random(8874)
    samples = sorted(stats(rng.choices(cycles, k=len(cycles)))[2] for _ in range(20000))
    return dict(baseline_ms=base, paired_ms=trial, elapsed_change_pct=pct,
                paired_cycle_bootstrap_95pct=[samples[499], samples[19499]],
                scored_cycles=len(cycles), matching_full_decodes=len(rows))

if __name__ == '__main__':
    print(json.dumps({Path(f).stem: summarize(Path(f)) for f in sys.argv[1:]}, indent=2))
