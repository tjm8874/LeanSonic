"""Score fixed common prefix positions with Basalt's stock PPL interface."""

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from kld_fixture import write_base

reference = Path(sys.argv[1])
args = sys.argv[3:]
root = Path.cwd()
result = Path(sys.argv[2])
inputs = [
    list(map(int, line.split()))
    for line in Path("fixtures.ids").read_text().splitlines()
]
with tempfile.TemporaryDirectory(dir=root) as tmp:
    work = Path(tmp)
    base = work / "ref.kld"
    rows = work / "rows.tsv"
    rec = write_base(Path("fixtures.ids"), reference, base)
    ctx = rec["context"]
    first = ctx // 2
    with open(str(result) + ".log", "a") as log:
        p = subprocess.Popen(
            [*args, "--ppl-base", str(base), "--ppl-out", str(rows)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
        )
        try:
            for line in p.stdout:
                if line.startswith("READY"):
                    break
            else:
                raise RuntimeError("engine quit before ready")
            for i, seq in enumerate(inputs):
                ids = ",".join(map(str, seq[: first + 1]))
                p.stdin.write(f"GEN {ctx - 1 - first} ppl={i} ppl_t=1 {ids}\n")
                p.stdin.flush()
                for line in p.stdout:
                    if line.startswith("ERR"):
                        raise RuntimeError(line)
                    if line.startswith("DONE"):
                        break
                else:
                    raise RuntimeError("engine quit before PPL")
            p.stdin.write("QUIT\n")
            p.stdin.flush()
            if p.wait(timeout=60) != 0:
                raise RuntimeError("engine failed")
        finally:
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=30)
    scored = []
    for row in rows.read_text().splitlines():
        a = row.split()
        if len(a) != 10:
            raise ValueError("invalid PPL output")
        scored.append(
            [
                int(a[0]),
                int(a[1]),
                int(a[2]),
                float(a[3]),
                float(a[4]),
                float(a[5]),
                int(a[6]),
                int(a[7]),
                int(a[8]),
                float(a[9]),
            ]
        )
    if [[r[0], r[1], r[2], r[8]] for r in scored] != rec["rows"]:
        raise ValueError("unaligned PPL rows")
    n = len(scored)
    rec.update(
        reference=str(reference),
        candidate_ppl=math.exp(sum(r[3] for r in scored) / n),
        mean_truncated_kl=sum(r[5] for r in scored) / n,
        max_truncated_kl=max(r[5] for r in scored),
        top1_matches=sum(r[6] for r in scored),
        scored=scored,
    )
    with result.open("x") as f:
        json.dump(rec, f, indent=2)
