"""Read Engram rows directly from their frames and compare against the original planes."""

import argparse
import json
import struct
import subprocess
from pathlib import Path
from statistics import median

from transport import ENGRAM, EXPERT, MAGIC, exact


def timing(samples, label):
    return median(int(ns) / int(n) for _, flag, ns, n, _ in samples if flag == label)


def scan(model, packed, binary):
    index = json.loads((model / "model.safetensors.index.json").read_bytes())[
        "weight_map"
    ]
    names = sorted({v for k, v in index.items() if ENGRAM.fullmatch(k)})
    scanned = set()
    for name in names:
        original = model / name
        compact = packed / (name + ".q38")
        with original.open("rb") as raw, compact.open("rb") as frame:
            if exact(frame, 8) != MAGIC:
                raise ValueError("invalid packed shard")
            prefix = exact(raw, 8)
            (size,) = struct.unpack("<Q", prefix)
            if size > 32 << 20:
                raise ValueError("oversized header")
            header = exact(raw, size)
            if exact(frame, 8 + size) != prefix + header:
                raise ValueError("original and packed shard headers differ")
            tensors = sorted(
                (t["data_offsets"], k, t)
                for k, t in json.loads(header).items()
                if k != "__metadata__"
            )
            for (start, end), key, tensor in tensors:
                length = end - start
                stored = 0
                if ENGRAM.fullmatch(key) or EXPERT.fullmatch(key):
                    (stored,) = struct.unpack("<Q", exact(frame, 8))
                    if stored > length:
                        raise ValueError("invalid packed size")
                if ENGRAM.fullmatch(key):
                    if (
                        not stored
                        or tensor["dtype"] != "BF16"
                        or tensor["shape"][1:] != [5]
                    ):
                        raise ValueError(
                            f"expected a dictionary frame with five BF16 words per row: {key}"
                        )
                    if key in scanned or index[key] != name:
                        raise ValueError("index mismatch")
                    scanned.add(key)
                    command = [
                        str(binary),
                        str(original),
                        str(size + 8 + start),
                        str(compact),
                        str(frame.tell()),
                        str(stored),
                        str(tensor["shape"][0]),
                    ]
                    samples = [
                        line.split(",")
                        for line in subprocess.check_output(
                            command, text=True
                        ).splitlines()
                    ]
                    if len(samples) != 20 or len({r[4] for r in samples}) != 1:
                        raise ValueError("incomplete timing run")

                    yield {
                        "plane": key,
                        "raw_ns_per_row": timing(samples, "false"),
                        "packed_ns_per_row": timing(samples, "true"),
                        "random_draws": 65536,
                        "edge_rows": 2,
                        "samples": samples,
                    }
                frame.seek(stored or length, 1)
            if frame.tell() != compact.stat().st_size:
                raise ValueError("shard length mismatch")
    if scanned != {k for k in index if ENGRAM.fullmatch(k)}:
        raise ValueError("missing Engram planes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("original", "packed", "row_bench"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    for item in scan(args.original, args.packed, args.row_bench.resolve()):
        print(json.dumps(item), flush=True)
