"""Restore official decoder norms without changing any safetensors or package lengths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile

INDEX = "model.safetensors.index.json"
NORMS = {"hc_norm", "k_layernorm", "q_layernorm", "k_norm", "q_norm",
         "norm_conv", "norm_key", "norm_query"}


def header(stream):
    length, = struct.unpack("<Q", stream.read(8))
    if length > 64 << 20:
        raise ValueError("unexpected safetensors header size")
    return json.loads(stream.read(length)), 8 + length


def locate(directory, mapping, key):
    with (directory / mapping[key]).open("rb") as f:
        tensors, base = header(f)
        info = tensors[key]
        first, last = info["data_offsets"]
        if info["dtype"] != "BF16" or len(info["shape"]) != 1 or last - first != 2 * info["shape"][0]:
            raise ValueError(f"invalid norm vector {key}")
        f.seek(base + first)
        data = f.read(last - first)
        if len(data) != last - first:
            raise EOFError(key)
        return base + first, data


def patches(base, official):
    manifest = json.loads((base / INDEX).read_bytes())
    if manifest["metadata"].get("leansonic_dense_format") != "symmetric-q8-group32-bf16":
        raise ValueError("expected the group-32 Q8 target package")
    source = manifest["weight_map"]
    upstream = json.loads((official / INDEX).read_bytes())["weight_map"]
    result = {}
    for key, filename in source.items():
        if not key.startswith("language_model.model.") or not key.endswith(".weight") or key.split(".")[-2] not in NORMS:
            continue
        offset, before = locate(base, source, key)
        other = key.replace("language_model.model.", "model.language_model.", 1)
        _, after = locate(official, upstream, other)
        if len(before) != len(after):
            raise ValueError(key)
        if before != after:
            result.setdefault(filename, []).append((offset, key, after))
    if sum(map(len, result.values())) != 148:
        raise ValueError("expected 148 differing norm tensors in the published Q8 source")
    return result


def audit_file(source, replacement, patches):
    # Check complete files, including their original headers and all unmodified payloads.
    digest = hashlib.sha256()
    size = source.stat().st_size
    if replacement.stat().st_size != size:
        raise ValueError(f"changed size: {source.name}")
    with source.open("rb") as src, replacement.open("rb") as dst:
        for offset, key, official in sorted(patches) + [(size, None, b"")]:
            if offset < src.tell():
                raise ValueError("overlapping payloads")
            while src.tell() < offset:
                original = src.read(min(offset - src.tell(), 8 << 20))
                actual = dst.read(len(original))
                if not original or original != actual:
                    raise ValueError(f"changed bytes outside norm data: {source.name}")
                digest.update(actual)
            if key is not None:
                src.seek(len(official), 1)
                actual = dst.read(len(official))
                if actual != official:
                    raise ValueError(f"official bytes differ: {key}")
                digest.update(actual)
    return digest.hexdigest()


def restore(base, official, dest):
    if dest.exists() or dest.resolve().is_relative_to(base.resolve()):
        raise ValueError("use a fresh destination outside the source")
    by_shard = patches(base, official)
    count = total = 0
    hashes = {}
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        staged = Path(tmp) / "model"
        staged.mkdir()
        for source in sorted(base.iterdir()):
            if not source.is_file():
                raise ValueError(f"unexpected source entry: {source.name}")
            target = staged / source.name
            ranges = by_shard.get(source.name, [])
            if not ranges and source.stat().st_dev == staged.stat().st_dev:
                os.link(source, target)
            else:
                shutil.copyfile(source, target)
                if ranges:
                    with target.open("r+b") as out:
                        for offset, _, official_bytes in ranges:
                            out.seek(offset)
                            out.write(official_bytes)
            hashes[source.name] = audit_file(source, target, ranges)
            count += len(ranges)
            total += target.stat().st_size
            print(f"verified {source.name}", file=sys.stderr, flush=True)
        staged.rename(dest)
    return {"restored_tensors": count, "restored_payload_bytes": sum(len(data) for entries in by_shard.values() for _, _, data in entries),
            "distribution_bytes": total, "changed_shards": len(by_shard), "sha256": hashes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("base", "official", "dest"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    print(json.dumps(restore(args.base, args.official, args.dest), indent=2))
