"""Measure warm expert-weight projection reads using the linked libzstd version."""

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path
from statistics import median

import zstandard
from weight_blocks import payloads

BLOCK = 256 << 10


def pack_plane(path, offset, size, packed):
    if size % BLOCK:
        raise ValueError("expected a block-aligned expert projection")
    entries = []
    codec = zstandard.ZstdCompressor(level=1)
    with path.open("rb") as src, packed.open("w+b") as dst:
        src.seek(offset)
        dst.seek(size // BLOCK * 8)
        for _ in range(size // BLOCK):
            data = src.read(BLOCK)
            if len(data) != BLOCK:
                raise ValueError("short projection read")
            frame = codec.compress(data)
            if len(frame) < BLOCK:
                entries.append(len(frame))
                dst.write(frame)
            else:
                entries.append((1 << 63) | BLOCK)
                dst.write(data)
        dst.seek(0)
        for entry in entries:
            dst.write(struct.pack("<Q", entry))


def bench(model, binary, work):
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        packed = Path(tmp) / "plane.blocks"
        for path, group, key, offset, size in payloads(model):
            if group != "expert" or not any(
                f".layers.{i}." in key for i in (0, 23, 47)
            ):
                continue
            pack_plane(path, offset, size, packed)
            lines = subprocess.check_output(
                [binary, path, str(offset), str(size), packed], text=True
            ).splitlines()
            samples = [list(map(int, line.split(","))) for line in lines]
            rows = []
            for threads in (1, 2, 4, 8):
                for arm in (0, 1):
                    ns = [n for w, p, n in samples if w == threads and p == arm]
                    if len(ns) != 4:
                        raise ValueError("incomplete benchmark")
                    rows.append(
                        {
                            "threads": threads,
                            "packed": bool(arm),
                            "median_ms": median(ns) / 1e6,
                            "sample_ns": ns,
                        }
                    )
            print(
                json.dumps(
                    {
                        "key": key,
                        "raw_bytes": size,
                        "packed_bytes": packed.stat().st_size,
                        "rows": rows,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for opt in ("model", "binary", "work"):
        parser.add_argument(opt, type=Path)
    args = parser.parse_args()
    bench(args.model, args.binary.resolve(), args.work)
