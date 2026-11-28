"""Re-read every tile byte against the direct checkpoint and verify all other tensor payloads."""

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from build import indices, tensor


def file_header(path):
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        d = json.loads(f.read(n))
        d.pop("__metadata__", None)
    return 8 + n, d


def block_hash(stream, start, size):
    stream.seek(start)
    h = hashlib.sha256()
    while size:
        chunk = stream.read(min(8 << 20, size))
        if not chunk:
            raise EOFError
        h.update(chunk)
        size -= len(chunk)
    return h.digest()


def audit(base, test):
    def index(p):
        return json.loads((p / "model.safetensors.index.json").read_bytes())[
            "weight_map"
        ]

    m, n = index(base), index(test)
    old = set()
    new = set()
    rows = 2_500_012
    for i in range(128):
        p = f"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_{i}."
        weight, _ = tensor(base, m, p + "weight")
        sp, sd = indices(base, m, p + "scales", rows)
        bp, bd = indices(base, m, p + "biases", rows)
        content, spec = tensor(test, n, p + "q38tile4")
        magic, number, sc, bc = struct.unpack("<8sIHH", content[:16])
        if (magic, number, sc, bc) != (b"Q38TILE4", rows, len(sd) // 2, len(bd) // 2):
            raise ValueError(p)
        if spec["dtype"] != "U8" or spec["shape"] != [len(content)]:
            raise ValueError(p)
        start = 16
        for dictionary in [sd, bd]:
            if not np.array_equal(content[start : start + len(dictionary)], dictionary):
                raise ValueError(p)
            start += len(dictionary)
        tiles = content[start:].reshape(rows // 4, 370)
        for begin, end, data in ((0, 320, weight), (320, 345, sp), (345, 370, bp)):
            original = data.reshape(rows // 4, end - begin)
            for low in range(0, len(tiles), 32768):
                if not np.array_equal(
                    tiles[low : low + 32768, begin:end], original[low : low + 32768]
                ):
                    raise ValueError(p)
        old.update(p + s for s in ("weight", "scales.q38dict1", "biases.q38dict1"))
        new.add(p + "q38tile4")
    if set(m) - old != set(n) - new:
        raise ValueError("unexpected keys")
    checked = 0
    for shard in sorted(set(m.values())):
        left, right = base / shard, test / shard
        if left.samefile(right):
            checked += sum(k not in old and sh == shard for k, sh in m.items())
            continue
        lo, h = file_header(left)
        ro, g = file_header(right)
        with left.open("rb") as x, right.open("rb") as y:
            for key in set(h) - old:
                a, b = h[key], g[key]
                start, end = a["data_offsets"]
                other, stop = b["data_offsets"]
                if (
                    a["dtype"] != b["dtype"]
                    or a["shape"] != b["shape"]
                    or end - start != stop - other
                ):
                    raise ValueError(key)
                if block_hash(x, lo + start, end - start) != block_hash(
                    y, ro + other, stop - other
                ):
                    raise ValueError(key)
                checked += 1
    if checked != len(set(m) - old):
        raise ValueError("incomplete audit")
    return {
        "engram_rows_exact": 128 * rows,
        "other_tensors_exact": checked,
        "source_file_bytes": sum(p.stat().st_size for p in base.iterdir()),
        "result_file_bytes": sum(p.stat().st_size for p in test.iterdir()),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("test", type=Path)
    a = p.parse_args()
    print(json.dumps(audit(a.base, a.test)))
