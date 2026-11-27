"""Expand compact signed codes through the fitted group tables for a stock runtime."""

import hashlib, json, shutil, struct, subprocess, sys
from pathlib import Path

compact, luts, exe, out = map(Path, sys.argv[1:])
out.mkdir()
name = "model-00001-of-00001.safetensors"
src = compact / name
with src.open("rb") as f:
    if (
        hashlib.file_digest(f, "sha256").hexdigest()
        != "a60c22b6626a50e1380abbfb80b69a73ff96d6404fb5b60c750d5bff7ff4cd02"
    ):
        raise ValueError("wrong compact model")
    f.seek(0)
    n = struct.unpack("<Q", f.read(8))[0]
    meta = json.loads(f.read(n))
    base = 8 + n
fit = json.loads((luts / "fit.json").read_text())
material = {}
for k, p in fit.items():
    lut = luts / (k + ".lut")
    assert hashlib.sha256(lut.read_bytes()).hexdigest() == p["sha256"]
    a, b = meta[k + ".rans_v1"]["data_offsets"]
    dest = out / (k + ".frame")
    subprocess.run(
        [
            exe,
            src,
            str(base + a),
            str(b - a),
            str(p["columns"]),
            str(p["parity"]),
            lut,
            dest,
        ],
        check=True,
    )
    material[k + ".rans_v1"] = dest
assert len(material) == 8
new = {}
end = 0
for k, v in meta.items():
    if k == "__metadata__":
        new[k] = v
        continue
    a, b = v["data_offsets"]
    size = material[k].stat().st_size if k in material else b - a
    new[k] = {**v, "data_offsets": [end, end + size]}
    end += size
    if k in material:
        new[k]["shape"] = [size]
head = json.dumps(new, separators=(",", ":"), ensure_ascii=False).encode()
head += b" " * (-len(head) % 8)
with src.open("rb") as orig, (out / name).open("xb") as target:
    target.write(struct.pack("<Q", len(head)) + head)
    for k, v in meta.items():
        if k == "__metadata__":
            continue
        if k in material:
            with material[k].open("rb") as f:
                shutil.copyfileobj(f, target, 8 << 20)
            material[k].unlink()
        else:
            a, b = v["data_offsets"]
            orig.seek(base + a)
            while b > a:
                part = orig.read(min(b - a, 8 << 20))
                assert part
                target.write(part)
                a += len(part)
    assert target.tell() == 8 + len(head) + end
shutil.copytree(compact / "tiktoken", out / "tiktoken")
with (out / name).open("rb") as f:
    digest = hashlib.file_digest(f, "sha256").hexdigest()
logical = lambda d: sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
base_bytes = src.stat().st_size + logical(compact / "tiktoken")
print(
    json.dumps(
        dict(
            base_bytes=base_bytes,
            unpacked_table_bytes=logical(luts),
            materialized_bytes=logical(out),
            shard_sha256=digest,
        )
    )
)
