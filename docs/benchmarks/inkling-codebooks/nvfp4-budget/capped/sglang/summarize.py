"""Summarize 20 mirrored process runs, keeping cycles as the bootstrap unit."""
import hashlib
import itertools
import json
import statistics
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
if len(sys.argv) == 2:
    runs = [json.loads(p.read_text()) for p in Path(sys.argv[1]).glob("*.json")]
else:
    runs = json.loads((root / "verified.json").read_text())["runs"]
assert len(runs) == 20
for filename, digest in [
    ("prompt.ids", "e93129d009d923541cd0b1ee39a6f121c11e000638ceb868258665ec974ed843"),
    ("continuation.ids", "ff73215c1c28d70806277f87faf7109c76a47414d49dabf94c883fd2562bf169"),
]:
    assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == digest
reference = list(map(int, (root / "continuation.ids").read_text().strip().split(",")))
for cycle in range(1, 6):
    legs = sorted((r for r in runs if r["cycle"] == cycle), key=lambda r: r["leg"])
    assert [r["leg"] for r in legs] == list(range(4))
    arms = ["lean", "sglang", "sglang", "lean"]
    assert [r["arm"] for r in legs] == (arms if cycle % 2 else ["sglang", "lean", "lean", "sglang"])
for arm in ("lean", "sglang"):
    group = [r for r in runs if r["arm"] == arm]
    assert all(r["peak"]["positive_polls"] > 100 and r["peak"]["peak_bytes"] > 0 for r in group)
    assert all([x["request"] for x in r["requests"]] == list(range(8)) for r in group)
    traces = {tuple(x["ids"]) for r in group for x in r["requests"]}
    trace, = traces
    assert len(trace) == 33
    if arm == "sglang":
        assert list(trace[:32]) == reference
    print(arm, "sampled peak bytes", max(r["peak"]["peak_bytes"] for r in group),
          "prediction mismatches (0-based)", [i for i, (a,b) in enumerate(zip(trace, reference)) if a != b])
for key, count in (("prefill_ms", 1024), ("decode32_ms", 32)):
    times = {}
    for arm in ("lean", "sglang"):
        times[arm] = [statistics.mean(x[key] for r in runs if r["cycle"] == c and r["arm"] == arm
                                       for x in r["requests"][1:]) for c in range(1, 6)]
        average = statistics.mean(times[arm])
        print(arm, key, average, "tok/s", count * 1000 / average)
    boot = sorted(sum(times["sglang"][i] for i in sample) / sum(times["lean"][i] for i in sample)
                  for sample in itertools.product(range(5), repeat=5))
    print(key, "Lean/SGLang throughput ratio, paired cycle percentile interval", boot[78], boot[-79])
