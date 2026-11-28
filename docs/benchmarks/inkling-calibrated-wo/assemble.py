"""Build the precision candidate by replacing just the 24 W_o matrices."""

import hashlib
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from io_utils import SHARD, manifest, read_header, transfer

model, fit, packer, dest = map(Path, sys.argv[1:])
rows = json.loads((fit / "layers.json").read_bytes())
assert len(rows) == 24
frames = dest.parent / "frames"
frames.mkdir()
dest.mkdir()
replace = {}
for row in rows:
    i = row["layer"]
    a, b = fit / f"{i}-fit.i8", fit / f"{i}-scales.f32"
    assert hashlib.sha256(a.read_bytes()).hexdigest() == row["fitted_codes"]
    assert hashlib.sha256(b.read_bytes()).hexdigest() == row["scales"]
    c, s = frames / f"{i}.rans", frames / f"{i}.dict"
    subprocess.run([packer, a, b, c, s], check=True)
    name = f"model.llm.layers.{i}.attn.wo_ud.weight"
    replace[name] = [
        (name + ".q8_64.codes.rans_v1", c),
        (name + ".q8_64.scales.dict_v1", s),
    ]
with (model / SHARD).open("rb") as source:
    base, header = read_header(source)
    assert replace.keys() <= header.keys()
    new = {"__metadata__": header["__metadata__"]} if "__metadata__" in header else {}
    offset = 0
    streams = []
    ordered = sorted(
        (k for k in header if k != "__metadata__"),
        key=lambda k: header[k]["data_offsets"][0],
    )
    for name in ordered:
        t = header[name]
        start, end = t["data_offsets"]
        if offset % 8 != start % 8:
            raise ValueError("changed tensor alignment")
        if name in replace:
            assert t["shape"] == [2304, 3072] and t["dtype"] == "BF16"
            for key, path in replace[name]:
                size = path.stat().st_size
                assert key not in header and size % 8 == 0
                new[key] = {
                    "dtype": "U8",
                    "shape": [size],
                    "data_offsets": [offset, offset + size],
                }
                offset += size
                streams.append((key, path, 0, size))
        else:
            size = end - start
            new[name] = dict(t, data_offsets=[offset, offset + size])
            offset += size
            streams.append((name, None, base + start, size))
    text = json.dumps(new, ensure_ascii=False, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    hashes = {}
    with (dest / SHARD).open("xb") as out:
        out.write(struct.pack("<Q", len(text)))
        out.write(text)
        for key, path, start, size in streams:
            if path:
                with path.open("rb") as f:
                    hashes[key] = transfer(f, size, out)
            else:
                source.seek(start)
                hashes[key] = transfer(source, size, out)
with (dest / SHARD).open("rb") as f:
    base2, actual = read_header(f)
    assert actual == new
    for name, expected in hashes.items():
        a, b = actual[name]["data_offsets"]
        f.seek(base2 + a)
        assert transfer(f, b - a) == expected
tok = Path("tiktoken/tokenizer.model")
(dest / tok.parent).mkdir()
shutil.copyfile(model / tok, dest / tok)
assert (dest / tok).read_bytes() == (model / tok).read_bytes()
files = manifest(dest)
print(
    json.dumps(
        {
            "parent_shard_plus_tokenizer_bytes": sum(
                (model / k).stat().st_size for k in (tok, SHARD)
            ),
            "package_bytes": sum(v["bytes"] for v in files.values()),
            "files": files,
            "unchanged_tensor_count": len(ordered) - 24,
            "changed_w_o_planes": 24,
            "fitted_codes_changed": sum(x["changed"] for x in rows),
            "train_loss_ratio": sum(x["fit_trial"] for x in rows)
            / sum(x["fit_parent"] for x in rows),
            "heldout_train_loss_ratio": sum(x["holdout_trial"] for x in rows)
            / sum(x["holdout_parent"] for x in rows),
        },
        indent=2,
    )
)
