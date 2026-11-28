"""Reread a direct checkpoint and compare every tensor and auxiliary file to its source."""
import argparse
import hashlib
import json
import struct
import subprocess
from collections import Counter
from pathlib import Path

from transport import ENGRAM, EXPERT, exact

INDEX = "model.safetensors.index.json"


def header(f):
    size, = struct.unpack("<Q", exact(f, 8))
    if size > 32 << 20:
        raise ValueError("oversized shard header")
    tensors = json.loads(exact(f, size))
    tensors.pop("__metadata__", None)
    return tensors, 8 + size


def file_map(root):
    return {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}


def verify(source, packed, codec):
    files, output = file_map(source), file_map(packed)
    if files.keys() != output.keys():
        raise ValueError("distribution file names differ")
    original, candidate = (json.loads(paths[INDEX].read_bytes())["weight_map"] for paths in (files, output))
    shards = set(original.values())
    if len(original) != len(candidate) or shards != set(candidate.values()):
        raise ValueError("model index mismatch")
    counts = Counter()
    for name in sorted(shards):
        with files[name].open("rb") as src, output[name].open("rb") as dst:
            left, left_base = header(src)
            right, right_base = header(dst)
            for tensors, mapping in ((left, original), (right, candidate)):
                if tensors.keys() != {k for k, shard in mapping.items() if shard == name}:
                    raise ValueError(f"shard index mismatch in {name}")
            for key, tensor in left.items():
                target = key if key in candidate else key + ".q38dict1"
                other = right[target]
                lo, hi = tensor["data_offsets"]
                start, end = other["data_offsets"]
                src.seek(left_base + lo)
                dst.seek(right_base + start)
                if target != key:
                    if not (ENGRAM.fullmatch(key) or EXPERT.fullmatch(key)) or other["dtype"] != "U8" or other["shape"] != [end-start]:
                        raise ValueError(f"unexpected dictionary tensor {target}")
                    decoded = subprocess.check_output([codec, "restore", str((hi-lo)//2)], input=exact(dst, end-start))
                    if decoded != exact(src, hi-lo):
                        raise ValueError(f"decoded mismatch in {key}")
                    counts["expert_frames" if EXPERT.fullmatch(key) else "engram_frames"] += 1
                else:
                    if tensor["shape"] != other["shape"] or tensor["dtype"] != other["dtype"] or hi-lo != end-start:
                        raise ValueError(f"shape or type mismatch for {key}")
                    for pos in range(lo, hi, 8 << 20):
                        chunk = min(hi-pos, 8 << 20)
                        if exact(src, chunk) != exact(dst, chunk):
                            raise ValueError(f"payload mismatch for {key}")
                    counts["plain_tensors"] += 1
    for name in files.keys() - shards - {INDEX}:
        if files[name].read_bytes() != output[name].read_bytes():
            raise ValueError(f"auxiliary file mismatch in {name}")
        counts["auxiliary_files"] += 1
    return dict(counts, shards=len(shards), files=len(files),
                original_bytes=sum(p.stat().st_size for p in files.values()),
                direct_bytes=sum(p.stat().st_size for p in output.values()),
                direct_index_sha256=hashlib.sha256(output[INDEX].read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("original", "packed", "codec"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.original, args.packed, args.codec.resolve()), indent=2))
