"""Replace the 24 native W_o code frames while checking every old scale frame."""

import json
import shutil
import struct
import sys
from pathlib import Path

from io_utils import SHARD, manifest, read_header, transfer

parent, frames, dest = map(Path, sys.argv[1:])
dest.mkdir()
with (parent / SHARD).open("rb") as source:
    base, header = read_header(source)
    keys = sorted(
        (k for k in header if k != "__metadata__"),
        key=lambda k: header[k]["data_offsets"][0],
    )
    new = {"__metadata__": header["__metadata__"]} if "__metadata__" in header else {}
    position = 0
    replacements = {}
    for i in range(24):
        name = f"model.llm.layers.{i}.attn.wo_ud.weight.q8_64"
        a, b = header[name + ".scales.dict_v1"]["data_offsets"]
        source.seek(base + a)
        assert source.read(b - a) == (frames / f"{i}.dict").read_bytes(), (
            f"scale mismatch at {i}"
        )
        replacements[name + ".codes.rans_v1"] = frames / f"{i}.rans"
    for k in keys:
        t = header[k]
        a, b = t["data_offsets"]
        size = b - a
        assert position % 8 == a % 8
        if k in replacements:
            size = replacements[k].stat().st_size
            assert size % 8 == 0 and t["dtype"] == "U8"
            t = dict(t, shape=[size])
        new[k] = dict(t, data_offsets=[position, position + size])
        position += size
    raw = json.dumps(new, separators=(",", ":"), ensure_ascii=False).encode()
    raw += b" " * (-len(raw) % 8)
    hashes = {}
    with (dest / SHARD).open("xb") as out:
        out.write(struct.pack("<Q", len(raw)))
        out.write(raw)
        for k in keys:
            if k in replacements:
                f = replacements[k]
                with f.open("rb") as src:
                    hashes[k] = transfer(src, f.stat().st_size, out)
            else:
                a, b = header[k]["data_offsets"]
                source.seek(base + a)
                hashes[k] = transfer(source, b - a, out)
with (dest / SHARD).open("rb") as src:
    offset, h = read_header(src)
    assert h == new
    for k in keys:
        a, b = h[k]["data_offsets"]
        src.seek(offset + a)
        assert transfer(src, b - a) == hashes[k]
tok = Path("tiktoken/tokenizer.model")
(dest / tok.parent).mkdir()
shutil.copyfile(parent / tok, dest / tok)
assert (parent / tok).read_bytes() == (dest / tok).read_bytes()
files = manifest(dest)
print(
    json.dumps(
        {
            "files": files,
            "size": sum(v["bytes"] for v in files.values()),
            "retained_tensors": len(keys) - 24,
            "changed_codes": 24,
            "identical_scale_frames": 24,
        },
        indent=2,
    )
)
