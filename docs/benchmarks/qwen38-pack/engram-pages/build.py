"""Build private conventional and 4 KiB row-page fixtures from one direct Q4 Engram shard."""

import argparse
import json
import os
from pathlib import Path
import struct

import numpy as np


def tensor(base, mapping, key):
    path = base / mapping[key]
    with path.open("rb") as stream:
        (size,) = struct.unpack("<Q", stream.read(8))
        spec = json.loads(stream.read(size))[key]
    lo, hi = spec["data_offsets"]
    return np.memmap(
        path, dtype="u1", mode="r", offset=8 + size + lo, shape=(hi - lo,)
    ), spec


def indices(base, mapping, key, rows):
    data, spec = tensor(base, mapping, key + ".q38dict1")
    magic, count, entries = struct.unpack("<8sQI", data[:20])
    if magic != b"Q38DICT1" or count != rows * 5 or (entries - 1).bit_length() != 10:
        raise ValueError(f"unsupported dictionary shape in {key}")
    n = (count * 10 + 7) // 8
    begin = 20 + 2 * entries
    if spec["dtype"] != "U8" or len(data) != begin + n + 2 or any(data[-2:]):
        raise ValueError(f"invalid dictionary byte layout in {key}")
    return data[begin : begin + n], data[20:begin]


def write_blocks(stream, weights, scales, biases, rows, per_block, block_bytes):
    groups = (rows + per_block - 1) // per_block
    widths = (80 * per_block, 50 * per_block // 8, 50 * per_block // 8)
    for p in range(0, groups, 4096):
        n = min(4096, groups - p)
        block = np.zeros((n, block_bytes), dtype="u1")
        start = 0
        for width, src in zip(widths, (weights, scales, biases)):
            piece = src[p * width : (p + n) * width]
            whole, tail = divmod(len(piece), width)
            block[:whole, start : start + width] = piece[: whole * width].reshape(
                whole, width
            )
            if tail:
                block[whole, start : start + tail] = piece[whole * width :]
            start += width
        stream.write(block)


def write(base, out, shard):
    index = json.loads((base / "model.safetensors.index.json").read_bytes())[
        "weight_map"
    ]
    prefix = f"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}."
    weights, spec = tensor(base, index, prefix + "weight")
    rows, columns = spec["shape"]
    if columns != 20 or spec["dtype"] != "U32" or len(weights) != rows * 80:
        raise ValueError("unexpected Engram Q4 weights")
    scales, sd = indices(base, index, prefix + "scales", rows)
    biases, bd = indices(base, index, prefix + "biases", rows)
    out.mkdir()
    # The conventional fixture also keeps its dictionaries in RAM. Only row data is timed.
    for name, data in (
        ("weights", weights),
        ("scales", scales),
        ("biases", biases),
        ("scale-dict", sd),
        ("bias-dict", bd),
    ):
        with (out / name).open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    for name, per_block, block_bytes in (("pages", 44, 4096), ("tiles", 4, 370)):
        with (out / name).open("xb") as stream:
            write_blocks(stream, weights, scales, biases, rows, per_block, block_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    return dict(
        shard=shard,
        rows=rows,
        direct_bytes=sum(map(len, (weights, scales, biases, sd, bd))) + 44,
        page_bytes=(out / "pages").stat().st_size + len(sd) + len(bd),
        tile_bytes=(out / "tiles").stat().st_size + len(sd) + len(bd),
        dictionary_entries=[len(sd) // 2, len(bd) // 2],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--shard", type=int, required=True)
    a = p.parse_args()
    print(json.dumps(write(a.base, a.out, a.shard)))
