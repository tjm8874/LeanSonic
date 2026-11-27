import hashlib
import json
import struct
import sys
from pathlib import Path


def chunks(file):
    with file.open("rb") as f:
        assert struct.unpack("<8sIII", f.read(20)) == (b"_logits_", 512, 200058, 100)
        result = [f.read(512 * 4) for _ in range(100)]
        assert all(len(c) == 512 * 4 for c in result)
        return result


new, test, valid, train = map(Path, sys.argv[1:])
observed = chunks(new)
earlier = chunks(test) + chunks(valid)
for line in train.read_text().splitlines():
    earlier.append(struct.pack("<512I", *map(int, line.split(","))))
assert len(earlier) == 232
# Compare 255 consecutive scored input tokens inside each old/new window,
# so resetting or inserting a BOS at a chunk boundary cannot hide duplication.
old_sequences = {r[i * 4 : (i + 255) * 4] for r in earlier for i in range(258)}
assert all(r[256 * 4 : 511 * 4] not in old_sequences for r in observed)
assert len({r[256 * 4 : 511 * 4] for r in observed}) == 100
assert new.stat().st_size == 10203366820
print(
    json.dumps(
        dict(
            chunks=100,
            rows=25500,
            compared_windows=len(earlier),
            scored_input_duplicates=0,
            ids_sha256=hashlib.sha256(b"".join(observed)).hexdigest(),
        )
    )
)
