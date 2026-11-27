"""Restore intermediate-package NVFP4 weights for a bounded subset of physical layers."""

import hashlib
import json
import re
import shutil
import struct
import sys
from pathlib import Path
from choose_codebooks import header

candidate, middle, output = map(Path, sys.argv[1:4])
first, stop = map(int, sys.argv[4:6])
assert 0 <= first < stop <= 20
shard = "model-00001-of-00001.safetensors"
selected = set()
h = {}
entries = []
offset = 0
with (candidate / shard).open("rb") as source, (middle / shard).open("rb") as reference:
    a, a0 = header(source)
    b, b0 = header(reference)
    assert a.keys() == b.keys()
    for key, spec in a.items():
        if key == "__metadata__":
            h[key] = spec
            continue
        m = re.fullmatch(
            r"model\.llm\.layers\.(\d+)\.mlp\.experts\.w(?:13|2)_weight\.nvctx_v1", key
        )
        restore = m is not None and first <= int(m[1]) < stop
        if restore:
            selected.add(key)
        item, stream, base = (b[key], reference, b0) if restore else (spec, source, a0)
        lo, hi = item["data_offsets"]
        h[key] = dict(item, data_offsets=[offset, offset + hi - lo])
        offset += hi - lo
        entries.append((stream, base + lo, hi - lo))
    assert len(selected) == 2 * (stop - first)
    raw = json.dumps(h, separators=(",", ":"), ensure_ascii=False).encode()
    raw += b" " * (-len(raw) % 8)
    output.mkdir()
    with (output / shard).open("xb") as dest:
        dest.write(struct.pack("<Q", len(raw)) + raw)
        for stream, start, length in entries:
            stream.seek(start)
            while length:
                block = stream.read(min(length, 16 << 20))
                assert block
                dest.write(block)
                length -= len(block)
        assert dest.tell() == 8 + len(raw) + offset
shutil.copytree(candidate / "tiktoken", output / "tiktoken")
with (output / shard).open("rb") as f:
    digest = hashlib.file_digest(f, "sha256").hexdigest()
print(
    json.dumps(
        dict(
            restore=sorted(selected),
            shard_sha256=digest,
            package_bytes=sum(
                f.stat().st_size for f in output.rglob("*") if f.is_file()
            ),
        )
    )
)
