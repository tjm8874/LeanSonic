"""Compare a shipped direct-coded package to a mixed two-/four-bank repack."""
import hashlib, json, struct, subprocess, sys
from pathlib import Path

def header(folder):
    path = folder / "model-00001-of-00001.safetensors"
    with path.open("rb") as f:
        size, = struct.unpack("<Q", f.read(8))
        return path, 8 + size, json.loads(f.read(size))

def digest(f, pos, size):
    f.seek(pos)
    sha = hashlib.sha256()
    while size:
        block = f.read(min(4 << 20, size))
        assert block
        sha.update(block)
        size -= len(block)
    return sha.digest()

old, new, decoder, receipt = map(Path, sys.argv[1:])
a, ao, am = header(old)
b, bo, bm = header(new)
r = json.loads(receipt.read_text())
direct, four = ".nvscale_direct_v1", ".nvscale_four_v1"
assert am.get("__metadata__") == bm.get("__metadata__")
same = 0
checked = {}
seen = set()
with a.open("rb") as af, b.open("rb") as bf:
    for key, info in am.items():
        if key == "__metadata__": continue
        target = key
        stem = key.removesuffix(direct)
        if key.endswith(direct) and stem + four in bm: target = stem + four
        actual = bm[target]
        seen.add(target)
        assert actual["dtype"] == info["dtype"]
        start, end = info["data_offsets"]
        first, last = actual["data_offsets"]
        if not key.endswith(direct):
            assert actual["shape"] == info["shape"] and last-first == end-start
            assert digest(af, ao+start, end-start) == digest(bf, bo+first, last-first)
            same += 1
            continue
        trial = r["planes"][stem]
        assert end-start == trial["old_bytes"] and last-first == min(end-start,trial["trial_bytes"])
        assert actual["shape"] == [last-first]
        sl, sr = am[stem+"_scale.mean8_v1"]["data_offsets"]
        width = 144 if stem.endswith(".w13_weight") else 72
        codec = "direct" if target == key else "four"
        command = [decoder, a, b, ao+start, end-start, bo+first, last-first, ao+sl, sr-sl, width, codec]
        n = int(subprocess.check_output([str(x) for x in command], text=True))
        checked[stem] = dict(codec=codec, decoded=n, saved=(end-start)-(last-first))
assert seen == set(bm)-{"__metadata__"}
assert same == 959 and len(checked) == 40 and sum(x["decoded"] for x in checked.values()) == 10192158720
assert (old/"tiktoken/tokenizer.model").read_bytes() == (new/"tiktoken/tokenizer.model").read_bytes()
size = sum(p.stat().st_size for p in new.rglob("*") if p.is_file())
assert size == r["package_bytes"]
print(json.dumps(dict(package_bytes=size, unchanged_tensors=same, verified_weights=checked),indent=2))
