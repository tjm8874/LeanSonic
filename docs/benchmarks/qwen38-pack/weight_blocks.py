"""Screen seekable Zstandard block sizes, then measure and verify the winner on every Q4 payload."""

import argparse
import json
import re
import struct
import time
from collections import defaultdict
from pathlib import Path

import zstandard

EXPERT = re.compile(
    r"language_model\.model\.layers\.\d+\.mlp\.switch_mlp\.(gate|up|down)_proj\.weight"
)
ENGRAM = re.compile(
    r"language_model\.model\.layers\.1\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight"
)
BLOCKS = (16 << 10, 64 << 10, 256 << 10)
LEVELS = (1, 3, 6)


def payloads(model):
    mapping = json.loads((model / "model.safetensors.index.json").read_bytes())[
        "weight_map"
    ]
    wanted = defaultdict(list)
    for key, name in mapping.items():
        group = (
            "expert"
            if EXPERT.fullmatch(key)
            else "engram"
            if ENGRAM.fullmatch(key)
            else None
        )
        if group:
            if Path(name).name != name:
                raise ValueError("invalid shard name")
            wanted[name].append((key, group))
    for name, entries in sorted(wanted.items()):
        with (model / name).open("rb") as f:
            (size,) = struct.unpack("<Q", f.read(8))
            if size > 32 << 20:
                raise ValueError("oversized header")
            header = json.loads(f.read(size))
            for key, group in entries:
                t = header[key]
                a, b = t["data_offsets"]
                if t["dtype"] not in ("U32", "I32") or a >= b:
                    raise ValueError(f"unexpected Q4 payload: {key}")
                yield model / name, group, key, 8 + size + a, b - a


def measure(data, size, encoder, decoder):
    total = 0
    for i in range(0, len(data), size):
        block = data[i : i + size]
        coded = encoder.compress(block)
        # Each block has an 8-byte seek-table entry. Raw fallback uses a length flag.
        if len(coded) < len(block):
            if decoder.decompress(coded, max_output_size=len(block)) != block:
                raise ValueError("round trip differs")
            total += 8 + len(coded)
        else:
            total += 8 + len(block)
    return total


def main(model):
    planes = list(payloads(model))
    if any(
        sum(g == kind for _, g, *_ in planes) != expected
        for kind, expected in (("expert", 144), ("engram", 128))
    ):
        raise ValueError("expected 144 expert and 128 Engram Q4 weight planes")
    codecs = {level: zstandard.ZstdCompressor(level=level) for level in LEVELS}
    decoder = zstandard.ZstdDecompressor()
    samples = defaultdict(int)
    for path, kind, _, offset, length in planes:
        with path.open("rb") as f:
            for eighth in (1, 4, 7):
                pos = (length * eighth // 8) // max(BLOCKS) * max(BLOCKS)
                f.seek(offset + pos)
                block = f.read(max(BLOCKS))
                if len(block) != max(BLOCKS) or pos + len(block) > length:
                    raise ValueError("sample exceeds plane")
                for n in BLOCKS:
                    for level in LEVELS:
                        samples[kind, n, level] += measure(
                            block, n, codecs[level], decoder
                        )
    winners = {}
    for kind in ("expert", "engram"):
        sampled, n, level = min(
            (v, n, level) for (g, n, level), v in samples.items() if g == kind
        )
        winners[kind] = (n, level)
        print(
            json.dumps(
                {
                    "stage": "sample",
                    "group": kind,
                    "block_bytes": n,
                    "level": level,
                    "sample_stored_bytes": sampled,
                    "candidates": [
                        {"block_bytes": b, "level": l, "stored_bytes": v}
                        for (g, b, l), v in samples.items()
                        if g == kind
                    ],
                }
            ),
            flush=True,
        )
    totals = defaultdict(int)
    for path, kind, key, offset, length in planes:
        n, level = winners[kind]
        stored = 0
        start = time.monotonic()
        with path.open("rb") as f:
            f.seek(offset)
            remaining = length
            while remaining:
                chunk = f.read(min(4 << 20, remaining))
                if not chunk:
                    raise ValueError("short read")
                stored += measure(chunk, n, codecs[level], decoder)
                remaining -= len(chunk)
        totals[kind] += min(length, stored)
        print(
            json.dumps(
                {
                    "stage": "full",
                    "plane": key,
                    "group": kind,
                    "raw": length,
                    "stored": stored,
                    "seconds": time.monotonic() - start,
                }
            ),
            flush=True,
        )
    print(
        json.dumps({"stage": "done", "raw_or_blocked_bytes": dict(totals)}), flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    main(parser.parse_args().original)
