import hashlib, json, shutil, struct, sys, subprocess
from pathlib import Path
from choose_codebooks import header

official, base, maps, exe, out = map(Path, sys.argv[1:])
shard = "model-00001-of-00001.safetensors"
out.mkdir()
with official.open("rb") as f:
    official_meta, off = header(f)
with (base / shard).open("rb") as f:
    old, begin = header(f)
selection = json.loads(maps.read_text())
frames = {}
stats = {}
for name, cut in sorted(selection["cutoffs"].items()):
    key = name + ".nvctx_v1"
    assert key in old
    a, b = official_meta[name]["data_offsets"]
    s, t = official_meta[name + "_scale"]["data_offsets"]
    assert b - a == 8 * (t - s)
    lut = out / (name + ".limits")
    lut.write_bytes(bytes(x if x >= 0 else 255 for x in cut))
    frame = out / (name + ".frame")
    vals = subprocess.check_output(
        [exe, official, str(off + a), str(off + s), str(b - a), lut, frame], text=True
    ).split()
    lut.unlink()
    size, nchanged = map(int, vals)
    assert frame.stat().st_size == size
    with frame.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    stats[key] = dict(size=size, changed=nchanged, sha256=digest)
    frames[key] = frame
assert len(frames) == 40
meta = {}
end = 0
for key, v in old.items():
    if key == "__metadata__":
        meta[key] = v
        continue
    a, b = v["data_offsets"]
    size = frames[key].stat().st_size if key in frames else b - a
    meta[key] = {**v, "data_offsets": [end, end + size]}
    end += size
    if key in frames:
        meta[key]["shape"] = [size]
h = json.dumps(meta, separators=(",", ":"), ensure_ascii=False).encode()
h += b" " * (-len(h) % 8)
with (base / shard).open("rb") as oldfile, (out / shard).open("xb") as dest:
    dest.write(struct.pack("<Q", len(h)) + h)
    for key, v in old.items():
        if key == "__metadata__":
            continue
        if key in frames:
            with frames[key].open("rb") as f:
                shutil.copyfileobj(f, dest, 8 << 20)
            frames[key].unlink()
        else:
            a, b = v["data_offsets"]
            oldfile.seek(begin + a)
            while b > a:
                buf = oldfile.read(min(8 << 20, b - a))
                assert buf
                dest.write(buf)
                a += len(buf)
    assert dest.tell() == 8 + len(h) + end
shutil.copytree(base / "tiktoken", out / "tiktoken")
with (out / shard).open("rb") as f:
    sha = hashlib.file_digest(f, "sha256").hexdigest()
size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
print(
    json.dumps(
        dict(
            package_bytes=size,
            shard_sha256=sha,
            target=selection["target_size"],
            estimated=selection["estimated_size"],
            planes=stats,
        )
    )
)
