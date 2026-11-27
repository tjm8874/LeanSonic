"""Choose a byte-matched MXFP8 candidate from weights and scale codes only."""

from pathlib import Path
import json, math, os, struct
import numpy as np
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("official_shard", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
p = args.official_shard
target = 151_694_064  # Measured saving from rounding every code in layer 23.


def number(m):
    return (m & 7) * 2**-9 if m < 8 else (8 + (m & 7)) * 2 ** ((m >> 3) - 10)


def rounded(m):
    return m if m % 2 == 0 else m + 1 if m & 2 else m - 1


def bits(hist):
    # Same 12-bit frequency normalization and tie order as the actual rANS encoder.
    total = hist.sum()
    f = np.where(hist > 0, np.maximum(1, hist * 4096 // total), 0)
    err = hist * 4096 - f * total
    while f.sum() != 4096:
        if f.sum() < 4096:
            j = np.argmax(np.where(f > 0, err, -(2**62)))
            f[j] += 1
            err[j] -= total
        else:
            j = np.argmin(np.where(f > 1, err, 2**62))
            f[j] -= 1
            err[j] += total
    mask = hist > 0
    return float((hist[mask] * np.log2(4096 / f[mask])).sum())


def costs(h):
    codes = h.sum(axis=0)
    weight = np.exp2(2 * (np.arange(256, dtype="f8") - 127)) @ h
    level = np.array([number(i & 127) for i in range(256)])
    power = np.dot(weight, level**2)
    tables = {}
    for threshold in range(-1, 127, 2):
        ids = np.arange(256)
        m = ids & 127
        to = np.array(
            [(i & 128) | rounded(i & 127) if (i & 127) <= threshold else i for i in ids]
        )
        tr = np.zeros(256, dtype=np.int64)
        np.add.at(tr, to, codes)
        tables[threshold] = dict(
            bits=bits(tr), error=float(np.dot(weight, (level - level[to]) ** 2) / power)
        )
    base = tables[-1]["bits"]
    for t in tables.values():
        t["saved"] = (base - t["bits"]) / 8
    return tables


with p.open("rb") as f, p.open("rb") as scales:
    (n,) = struct.unpack("<Q", f.read(8))
    hdr = json.loads(f.read(n))
    off = 8 + n
    curves = {}
    for il in range(20, 24):
        for which in ("w13_weight", "w2_weight"):
            k = f"model.llm.layers.{il}.mlp.experts.{which}"
            a, b = hdr[k]["data_offsets"]
            s, e = hdr[k + "_scale"]["data_offsets"]
            assert b - a == 32 * (e - s)
            f.seek(off + a)
            scales.seek(off + s)
            h = np.zeros((256, 256), dtype="i8")
            for row in range(0, e - s, 1 << 16):
                nn = min(1 << 16, e - s - row)
                sc = np.frombuffer(scales.read(nn), np.uint8)
                raw = np.frombuffer(f.read(32 * nn), np.uint8).reshape(-1, 32)
                h += np.bincount(
                    ((sc.astype("i4")[:, None] << 8) | raw).ravel(), minlength=65536
                ).reshape(256, 256)
            assert (
                h.sum() == b - a
                and h[255].sum() == h[:, 127].sum() == h[:, 255].sum() == 0
            )
            curves[k] = costs(h)
# Start with the last common cutoff below the target; try all 2^8 ways to
# raise each plane by one E4M3 bin. Minimize relative error among near-ties.
import itertools

lev = [
    t
    for t in range(-1, 127, 2)
    if sum(c[t]["saved"] for c in curves.values()) <= target
][-1]
assert lev + 2 <= 125
options = []
for steps in itertools.product([0, 2], repeat=8):
    thresholds = dict(zip(curves, [lev + i for i in steps], strict=True))
    saved = sum(curves[k][t]["saved"] for k, t in thresholds.items())
    err = sum(curves[k][t]["error"] for k, t in thresholds.items()) / len(curves)
    options.append((abs(saved - target), err, saved, thresholds))
nearest = min(o[0] for o in options)
delta, err, saved, ts = min(
    (o for o in options if o[0] <= nearest + 100000), key=lambda o: o[1]
)
report = dict(
    target_bytes=target,
    predicted_saved=saved,
    mean_per_plane_relative_weight_mse=err,
    full_rounding_mean_relative_weight_mse=sum(c[125]["error"] for c in curves.values())
    / len(curves),
    thresholds=ts,
    curves=curves,
)
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(saved, err, ts, flush=True)
