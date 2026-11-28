"""Independently verify every Q8 code, BF16 scale and unchanged tensor from the compact pack."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pack import INDEX, projections, read_header, source_key


def headers(directory, mapping):
    out = {}
    for shard in set(mapping.values()):
        with (directory / shard).open("rb") as f:
            h, start = read_header(f)
        h.pop("__metadata__", None)
        if set(h) != {k for k, v in mapping.items() if v == shard}:
            raise ValueError(f"tensor index differs from header: {shard}")
        out[shard] = start, h
    return out


def plane(directory, mapping, info, name, dtype, dims):
    sh = mapping[name]
    base, tensors = info[sh]
    t = tensors[name]
    if t["dtype"] != dtype or t["shape"] != list(dims):
        raise ValueError(name)
    types = {"I8": "i1", "BF16": "<u2"}
    return np.memmap(directory / sh, dtype=types[dtype], mode="r",
                     offset=base + t["data_offsets"][0], shape=dims)


def digest(path, off, count):
    h = hashlib.sha256()
    with path.open("rb") as f:
        f.seek(off)
        while count:
            buf = f.read(min(count, 8 << 20))
            if not buf:
                raise EOFError(path)
            h.update(buf)
            count -= len(buf)
    return h.digest()


def bf16_bits(f32):
    bits = f32.view(np.uint32)
    return ((bits + 32767 + ((bits >> 16) & 1)) >> 16).astype("<u2")


def f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def audit(base, official, model):
    mappings = [json.loads((p / INDEX).read_bytes())["weight_map"] for p in (base, official, model)]
    m, u, v = mappings
    old = headers(base, m)
    src = headers(official, u)
    new = headers(model, v)
    names = projections(m)
    removed = {p + ".biases" for p in names}
    if {k: sh for k, sh in m.items() if k not in removed} != v:
        raise ValueError("unexpected tensor mapping")
    changed = {p + suffix for p in names for suffix in (".weight", ".scales")}
    count = elements = 0
    for p in sorted(names):
        key = source_key(p)
        rows, cols = src[u[key]][1][key]["shape"]
        ref = plane(official, u, src, key, "BF16", (rows, cols))
        codes = plane(model, v, new, p + ".weight", "I8", (rows, cols))
        scales = plane(model, v, new, p + ".scales", "BF16", (rows, cols // 32))
        for i in range(0, rows, 256):
            sl = slice(i, i + 256)
            x = f32(ref[sl]).reshape(-1, 32)
            if not np.isfinite(x).all():
                raise ValueError(f"nonfinite {p}")
            scale_bits = bf16_bits(np.abs(x).max(axis=1, keepdims=True) / np.float32(127))
            scale = f32(scale_bits)
            expected = np.clip(np.rint(x / np.where(scale == 0, 1, scale)), -127, 127).astype(np.int8)
            if not np.array_equal(scale_bits.ravel(), scales[sl].ravel()) or not np.array_equal(expected.ravel(), codes[sl].ravel()):
                raise ValueError(f"Q8 encoding differs at {p} row block {i}")
        elements += rows * cols
    for key, sh in v.items():
        if key in changed:
            continue
        a, old_header = old[sh]
        b, new_header = new[sh]
        x, y = old_header[key], new_header[key]
        lo, hi = x["data_offsets"]
        low, high = y["data_offsets"]
        if x["shape"] != y["shape"] or x["dtype"] != y["dtype"] or hi - lo != high - low:
            raise ValueError(key)
        if not (base / sh).samefile(model / sh):
            if digest(base / sh, a + lo, hi - lo) != digest(model / sh, b + low, high - low):
                raise ValueError(f"changed tensor {key}")
        count += 1
    return dict(q8_matrices=len(names), codes_exact=elements, scales_exact=elements // 32,
                unchanged_tensors=count, distribution_bytes=sum(f.stat().st_size for f in model.iterdir()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "official", "model"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    print(json.dumps(audit(args.base, args.official, args.model), indent=2))
