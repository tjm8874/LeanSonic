"""Pack or restore the indexed Qwen model in a new directory, preserving every shard byte."""

import argparse
import hashlib
import json
import re
import shutil
import struct
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

MAGIC = b"Q38DIST1"
MANIFEST = "q38-pack-v1.json"
ENGRAM = re.compile(
    r"language_model\.model\.layers\.1\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.(scales|biases)$"
)
EXPERT = re.compile(
    r"language_model\.model\.layers\.\d+\.mlp\.switch_mlp\.(gate|up|down)_proj\.(scales|biases)$"
)


def exact(f, length):
    block = f.read(length)
    if len(block) != length:
        raise ValueError("truncated input")
    return block


def shard(src, dst, restore, codec):
    digest = hashlib.sha256()
    counts = Counter()
    with src.open("rb") as inp, dst.open("xb") as out:
        if restore:
            if exact(inp, 8) != MAGIC:
                raise ValueError("unexpected shard format")
        else:
            out.write(MAGIC)
        prefix = exact(inp, 8)
        (size,) = struct.unpack("<Q", prefix)
        if size > 32 << 20:
            raise ValueError("unexpected safetensors header size")
        header = exact(inp, size)
        out.write(prefix + header)
        digest.update(prefix + header)
        tensors = [
            (t["data_offsets"], k, t)
            for k, t in json.loads(header).items()
            if k != "__metadata__"
        ]
        cursor = 0
        for (start, end), key, tensor in sorted(tensors):
            if start != cursor or end < start:
                raise ValueError(f"invalid tensor offsets for {key}")
            cursor = end
            length = end - start
            group = (
                "engram"
                if ENGRAM.fullmatch(key)
                else "expert"
                if EXPERT.fullmatch(key)
                else None
            )
            if group:
                if (
                    tensor["dtype"] != "BF16"
                    or length % 2
                    or not 0 < length <= 256 << 20
                ):
                    raise ValueError(f"unexpected scale/bias plane {key}")
                if restore:
                    (packed,) = struct.unpack("<Q", exact(inp, 8))
                    if packed > length:
                        raise ValueError("oversized frame")
                    raw = (
                        subprocess.run(
                            [codec, "restore", str(length // 2)],
                            input=exact(inp, packed),
                            stdout=subprocess.PIPE,
                            check=True,
                        ).stdout
                        if packed
                        else exact(inp, length)
                    )
                    if len(raw) != length:
                        raise ValueError("unexpected decoded size")
                    out.write(raw)
                else:
                    raw = exact(inp, length)
                    frame = subprocess.run(
                        [codec, "pack"], input=raw, stdout=subprocess.PIPE, check=True
                    ).stdout
                    packed = len(frame) if len(frame) < length else 0
                    out.write(struct.pack("<Q", packed))
                    out.write(frame if packed else raw)
                digest.update(raw)
                counts[group + "_raw"] += length
                counts[group + "_stored"] += 8 + (packed or length)
                counts[group + "_planes"] += 1
            else:
                while length:
                    block = exact(inp, min(length, 8 << 20))
                    digest.update(block)
                    out.write(block)
                    length -= len(block)
        if inp.read(1):
            raise ValueError("unexpected trailing data")
    return digest.hexdigest(), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["pack", "restore"])
    parser.add_argument("codec", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    dest = args.destination.resolve()
    if dest.exists() or dest.is_relative_to(source):
        raise ValueError("choose a new directory outside the source")
    restore = args.action == "restore"
    index = json.loads((source / "model.safetensors.index.json").read_bytes())
    names = sorted(set(index["weight_map"].values()))
    if not names or any(
        not re.fullmatch(r"model-[0-9]+-of-[0-9]+\.safetensors", n) for n in names
    ):
        raise ValueError("unexpected shard names")
    record = json.loads((source / MANIFEST).read_bytes()) if restore else {}
    if not restore and (source / MANIFEST).exists():
        raise ValueError("source is already a pack")
    if restore and (
        record.get("format") != "Q38DIST1" or set(record["sha256"]) != set(names)
    ):
        raise ValueError("index and digest list differ")
    totals = Counter()
    hashes = {}
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".q38-build-") as tmp:
        build = Path(tmp) / "model"
        build.mkdir()
        originals = set(names)
        packed_names = {n + ".q38" for n in names}
        expected = packed_names | {MANIFEST} if restore else originals
        reserved = originals if restore else packed_names
        for path in source.iterdir():
            if path.name in reserved:
                raise ValueError(f"ambiguous shard layout at {path.name}")
            if path.name in expected:
                continue
            if path.is_dir():
                shutil.copytree(path, build / path.name)
            else:
                shutil.copy2(path, build / path.name)
        for n in names:
            original, compact = n, n + ".q38"
            src = source / (compact if restore else original)
            dst = build / (original if restore else compact)
            sha, counts = shard(src, dst, restore, str(args.codec.resolve()))
            if restore and sha != record["sha256"][n]:
                raise ValueError(f"{n} failed SHA-256 verification")
            hashes[n] = sha
            totals.update(counts)
            totals["source_shard_bytes"] += src.stat().st_size
            totals["output_shard_bytes"] += dst.stat().st_size
            print(n, sha, flush=True)
        if not restore:
            record = {"format": "Q38DIST1", "sha256": hashes, "counts": dict(totals)}
            (build / MANIFEST).write_text(json.dumps(record, indent=2) + "\n")
        build.rename(dest)
    print(json.dumps(dict(totals)), flush=True)


if __name__ == "__main__":
    main()
