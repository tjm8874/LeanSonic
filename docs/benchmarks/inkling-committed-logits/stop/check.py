"""Compare all terminal and resumed rows from spec_eos with scalar decode."""
import csv
from pathlib import Path


def records(arm):
    with Path(f"{arm}.csv").open() as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 10
    result = {name: (token, top, ids, resume) for name, token, top, _, _, ids, resume in rows}
    assert len(result) == len(rows)
    return result


scalar = records("scalar")
assert all(token == top for token, top, _, _ in scalar.values())
assert records("after") == scalar
for case in scalar:
    for suffix in ("terminal", "resume"):
        filename = f"{case}.{suffix}"
        reference = (Path("scalar-rows") / filename).read_bytes()
        assert len(reference) == 200058 * 4
        assert (Path("after-rows") / filename).read_bytes() == reference, filename
print("10 cases matched, including all 20 full FP32 rows")
