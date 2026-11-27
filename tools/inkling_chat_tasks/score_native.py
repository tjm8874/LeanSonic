"""Score decoded assistant text at two caps without revealing hidden reasoning."""

import hashlib
import json
import sys
from pathlib import Path

from suite import passes
from tokenizers import Tokenizer

weights, trace = map(Path, sys.argv[1:])
tokenizer = Tokenizer.from_file(str(weights / "tokenizer.json"))
fixtures = json.loads(Path(__file__).with_name("fixtures.json").read_text())
cases = {r["id"]: r for r in fixtures}


def visible(ids):
    state = "none"
    spans = []
    for tok in ids:
        if tok == 200001:
            state = "model"
        elif tok == 200004 and state == "model":
            state = "text"
            spans.append([])
        elif tok == 200008 and state == "model":
            state = "thinking"
        elif tok >= 199998:
            state = "none"
        elif state == "text":
            spans[-1].append(tok)
    return "".join(tokenizer.decode(x, skip_special_tokens=True) for x in spans)


reports = []
for line in trace.read_text().splitlines():
    case, *xs = line.split()
    seq = list(map(int, xs))
    assert (
        0 < len(seq) <= 384
        and (seq[-1] == 200006 or len(seq) == 384)
        and all(0 <= t < 200058 for t in seq)
    )
    outputs = {}
    for cap in [128, 384]:
        ids = seq[:cap]
        stopped = ids[-1] == 200006
        if stopped:
            ids = ids[:-1]
        text = visible(ids)
        outputs[cap] = {
            "text": text,
            "finish": "stop" if stopped else "length",
            "tokens": len(ids),
            "passed": stopped and passes(cases[case], text),
        }
    digest = hashlib.sha256((" ".join(xs) + "\n").encode("ascii")).hexdigest()
    reports.append({"case": case, "generated": outputs, "token_sha256": digest})
assert [r["case"] for r in reports] == [r["id"] for r in fixtures] + [fixtures[0]["id"]]
assert (
    reports[-1] == reports[0]
    and trace.read_text().splitlines()[0] == trace.read_text().splitlines()[-1]
)
print(json.dumps(reports, indent=2, ensure_ascii=False))
