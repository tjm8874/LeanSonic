"""Select a fresh sequence of complete articles from the second half of WikiText-2 valid."""

import hashlib
import json
import re
import sys
from pathlib import Path

source, output = map(Path, sys.argv[1:3])
raw = source.read_bytes()
heading = re.compile(rb"^ = [^=\r\n]+ = *$", re.MULTILINE)
match = heading.search(raw, len(raw) // 2)
if match is None:
    raise ValueError("missing article boundary in the second half")
start = match.start()
data = raw[start:]
with output.open("xb") as f:
    f.write(data)
print(
    json.dumps(
        dict(
            dataset="WikiText-2 raw validation",
            byte_offset=start,
            original_bytes=len(raw),
            original_sha256=hashlib.sha256(raw).hexdigest(),
            suffix_bytes=len(data),
            suffix_sha256=hashlib.sha256(data).hexdigest(),
        )
    )
)
