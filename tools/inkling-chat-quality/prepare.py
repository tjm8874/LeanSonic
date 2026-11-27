"""Materialize fixed, UTF-8 prompt/answer fixtures and independent reset controls."""
import json
import sys
from pathlib import Path

fixtures = json.loads(Path(__file__).with_name("fixtures.json").read_text())
root = Path(sys.argv[1])
root.mkdir(exist_ok=False)
cases = []
for f in fixtures + [dict(f, id=f["id"] + "_repeat") for f in (fixtures[0], fixtures[-1])]:
    case = f["id"]
    (root / case).mkdir()
    (root / case / "prompt.txt").write_text(f["prompt"], encoding="utf-8")
    (root / case / "answer.txt").write_text(f["answer"], encoding="utf-8")
    cases.append(case)
(root / "order.txt").write_text("\n".join(cases) + "\n")
