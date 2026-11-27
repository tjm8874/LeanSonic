"""Rebuild only the canonical package's Q8 attention frames from original BF16."""

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

SHARD = "model-00001-of-00001.safetensors"
PARENT_HASH = "9b242a81188f0f552625639dec98879a0090a7f9a003141631d3a836aed3fcb3"


def header(path):
    with path.open("rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(length)), 8 + length


def transfer(stream, start, count, out=None):
    stream.seek(start)
    digest = hashlib.sha256()
    while count:
        chunk = stream.read(min(count, 8 << 20))
        if not chunk:
            raise ValueError("truncated tensor")
        digest.update(chunk)
        if out is not None:
            out.write(chunk)
        count -= len(chunk)
    return digest.hexdigest()


p = argparse.ArgumentParser(description=__doc__)
for arg in ("encoder", "official", "parent", "output"):
    p.add_argument(arg, type=Path)
p.add_argument("--mode", choices=["refine", "rate"], required=True)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
original = a.official / SHARD
parent = a.parent / SHARD
with parent.open("rb") as f:
    if hashlib.file_digest(f, "sha256").hexdigest() != PARENT_HASH:
        raise ValueError("expected canonical parent")
raw, raw_offset = header(original)
meta, base_offset = header(parent)
names = sorted(
    k.removesuffix(".q8_64.codes.rans_v1")
    for k in meta
    if ".attn." in k and k.endswith(".q8_64.codes.rans_v1")
)
if len(names) != 24 * 7 + 4 * 2:
    raise ValueError(f"unexpected attention matrix count: {len(names)}")

with tempfile.TemporaryDirectory(dir=a.output.parent) as tmp:
    root = Path(tmp)

    def measure(name):
        code = name + ".q8_64.codes.rans_v1"
        scale = name + ".q8_64.scales.dict_v1"
        tensor = raw[name]
        start, end = tensor["data_offsets"]
        size = (end - start) // 2
        rows, cols = tensor["shape"]
        assert tensor["dtype"] == "BF16" and cols % 64 == 0 and size == rows * cols
        info = {}
        for mode in ("control", a.mode):
            c = root / (code + "." + mode)
            s = root / (scale + "." + mode)
            cmd = [
                str(a.encoder.resolve()),
                mode,
                str(original),
                str(raw_offset + start),
                str(size),
                str(c),
                str(s),
            ]
            stat = json.loads(subprocess.check_output(cmd, text=True))
            if mode == "control":
                with parent.open("rb") as f:
                    for key, path in [(code, c), (scale, s)]:
                        lo, hi = meta[key]["data_offsets"]
                        assert (
                            transfer(f, base_offset + lo, hi - lo)
                            == hashlib.sha256(path.read_bytes()).hexdigest()
                        ), key
                        path.unlink()
            else:
                assert (
                    stat["new_sse"] <= stat["old_sse"]
                    and sum(stat["cap_histogram"]) == size // 64
                )
                info = stat
        return name, info

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        stats = dict(pool.map(measure, names))
    frames = {
        key: root / (key + "." + a.mode)
        for key in meta
        if (root / (key + "." + a.mode)).exists()
    }
    assert len(frames) == 2 * len(names)
    new = {}
    offset = 0
    for key, old in meta.items():
        if key == "__metadata__":
            new[key] = old
            continue
        lo, hi = old["data_offsets"]
        length = frames[key].stat().st_size if key in frames else hi - lo
        new[key] = {**old, "data_offsets": [offset, offset + length]}
        if key in frames:
            new[key]["shape"] = [length]
        offset += length
    package = root / "package"
    package.mkdir()
    folder = package / SHARD
    buf = json.dumps(new, separators=(",", ":"), ensure_ascii=False).encode()
    buf += b" " * (-len(buf) % 8)
    hashes = {}
    with parent.open("rb") as f, folder.open("xb") as out:
        out.write(struct.pack("<Q", len(buf)) + buf)
        for key in meta:
            if key == "__metadata__":
                continue
            lo, hi = meta[key]["data_offsets"]
            if key in frames:
                with frames[key].open("rb") as replacement:
                    digest = transfer(replacement, 0, frames[key].stat().st_size, out)
            else:
                digest = transfer(f, base_offset + lo, hi - lo, out)
            hashes[key] = digest
    check, base = header(folder)
    assert check == new
    with folder.open("rb") as f:
        for key, digest in hashes.items():
            lo, hi = check[key]["data_offsets"]
            assert transfer(f, base + lo, hi - lo) == digest, key
        f.seek(0)
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    shutil.copytree(a.parent / "tiktoken", package / "tiktoken")
    receipt = {
        "mode": a.mode,
        "matrices": stats,
        "shard_sha256": sha,
        "package_bytes": sum(
            x.stat().st_size for x in package.rglob("*") if x.is_file()
        ),
        "parent_sha256": PARENT_HASH,
        "changed_frames": len(frames),
    }
    package.rename(a.output)
    print(json.dumps(receipt, indent=2))
