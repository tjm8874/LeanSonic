"""Spend the lossless compression savings on the same frozen training rank."""
import hashlib
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

def header(file):
    with file.open("rb") as f:
        n, = struct.unpack("<Q", f.read(8))
        return 8 + n, json.loads(f.read(n))

def transfer(source, dest, n):
    digest = hashlib.sha256()
    while n:
        chunk = source.read(min(n, 4 << 20))
        assert chunk
        if dest is not None:
            dest.write(chunk)
        digest.update(chunk)
        n -= len(chunk)
    return digest.hexdigest()

parent, rank_receipt, fitter, target = map(Path, sys.argv[1:])
r = json.loads(rank_receipt.read_text())
key = r["plane"]
shard = "model-00001-of-00001.safetensors"
base, meta = header(parent / shard)
prior_size = sum(p.stat().st_size for p in parent.rglob("*") if p.is_file())
assert prior_size == 7747846286 and bytes(r["rank"]) == Path("rank.bin").read_bytes()
cap = 7764291776
swaps = {key + ".nvscale_four_v1": Path("fit.weights"), key + "_scale.mean8_v1": Path("fit.scales")}
old_pair = sum(meta[k]["data_offsets"][1] - meta[k]["data_offsets"][0] for k in swaps)
width = 72 if key.endswith(".w2_weight") else 144
report = subprocess.check_output([fitter.resolve(), Path.cwd(), str(width), str(old_pair + cap - prior_size - 4096)], text=True).splitlines()
count = int(report[-1].split()[1])
assert len(r["selected"]) < count < 128
end = 0
new = {}
for name, entry in meta.items():
    if name == "__metadata__":
        new[name] = entry
        continue
    lo, hi = entry["data_offsets"]
    size = swaps[name].stat().st_size if name in swaps else hi - lo
    new[name] = dict(entry, data_offsets=[end, end + size])
    if name in swaps: new[name]["shape"] = [size]
    end += size
raw = json.dumps(new, ensure_ascii=False, separators=(",", ":")).encode()
raw += b" " * (-len(raw) % 8)
target.mkdir()
checks = {}
with (parent/shard).open("rb") as src, (target/shard).open("xb") as out:
    out.write(struct.pack("<Q", len(raw)))
    out.write(raw)
    for k, entry in meta.items():
        if k == "__metadata__": continue
        if k in swaps:
            f = swaps[k]
            with f.open("rb") as src2: checks[k] = transfer(src2, out, f.stat().st_size)
        else:
            lo, hi = entry["data_offsets"]
            src.seek(base + lo)
            checks[k] = transfer(src, out, hi - lo)
shutil.copytree(parent/"tiktoken", target/"tiktoken")
start, actual = header(target/shard)
assert actual == new and len(checks) == 999
with (target/shard).open("rb") as out:
    for name, expected in checks.items():
        lo, hi = actual[name]["data_offsets"]
        out.seek(start + lo)
        assert transfer(out, None, hi - lo) == expected
assert (parent/"tiktoken/tokenizer.model").read_bytes() == (target/"tiktoken/tokenizer.model").read_bytes()
files = {}
for path in sorted(target.rglob("*")):
    if path.is_file():
        with path.open("rb") as fp:
            files[str(path.relative_to(target))] = dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(fp,"sha256").hexdigest())
size = sum(v["bytes"] for v in files.values())
assert prior_size < size <= cap
print(json.dumps(dict(plane=key, rank=r["rank"], selected=r["rank"][:count], trials=report,
                      package_bytes=size, files=files, unchanged_tensors=997, verified_tensors=999,
                      parent_rank_sha256=hashlib.sha256(rank_receipt.read_bytes()).hexdigest()), indent=2))
