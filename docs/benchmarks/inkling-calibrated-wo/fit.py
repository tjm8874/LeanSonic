"""Reassign Q8_64 attention-output weights using fixed train-only block covariances."""

import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

shard, covariance, dump, dest = map(Path, sys.argv[1:])
dest.mkdir()
with shard.open("rb") as f:
    (length,) = struct.unpack("<Q", f.read(8))
    header = json.loads(f.read(length))
base = 8 + length
fit = np.load(covariance / "fit.npy")
hold = np.load(covariance / "heldout.npy")
assert fit.shape == hold.shape == (24, 48, 64, 64)


def loss(error, h):
    return float(((error @ h) * error).sum())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


rows = []
for il in range(24):
    key = f"model.llm.layers.{il}.attn.wo_ud.weight"
    t = header[key]
    start, end = t["data_offsets"]
    assert (
        t["dtype"] == "BF16"
        and t["shape"] == [2304, 3072]
        and end - start == 2304 * 3072 * 2
    )
    c = dest / f"{il}-parent.i8"
    s = dest / f"{il}-scales.f32"
    subprocess.run([dump, shard, str(base + start), c, s], check=True)
    bf = np.memmap(
        shard, mode="r", dtype="<u2", offset=base + start, shape=(2304, 48, 64)
    )
    weights = (bf.astype("<u4") << 16).view("<f4")
    scales = np.fromfile(s, dtype="<f4").reshape(2304, 48)
    original = np.fromfile(c, dtype="i1").reshape(2304, 48, 64)
    updated = original.copy()
    totals = np.zeros((2, 2))
    for block in range(48):
        weight = np.ascontiguousarray(weights[:, block, :], dtype="f8")
        scale = scales[:, block : block + 1].astype("f8")
        code = original[:, block, :].astype("f8")
        h = fit[il, block].copy()
        # Damping and the two traversal orders are fixed across every layer.
        h.flat[::65] += max(1e-14, 0.05 * np.trace(h) / 64)
        error = code - np.divide(
            weight, scale, out=np.zeros_like(weight), where=scale != 0
        )
        delta = error @ h
        for order in (range(64), reversed(range(64))):
            for col in order:
                old = code[:, col]
                new = np.clip(np.rint(old - delta[:, col] / h[col, col]), -127, 127)
                update = new - old
                delta += update[:, None] * h[col]
                code[:, col] = new
        updated[:, block] = code.astype("i1")
        for arm, q in enumerate((original[:, block], updated[:, block])):
            err = q * scale - weight
            totals[arm, 0] += loss(err, fit[il, block])
            totals[arm, 1] += loss(err, hold[il, block])
    out = dest / f"{il}-fit.i8"
    updated.tofile(out)
    assert np.isfinite(totals).all()
    rows.append(
        {
            "layer": il,
            "changed": int(np.count_nonzero(original != updated)),
            "counts": original.size,
            "fit_parent": totals[0, 0],
            "fit_trial": totals[1, 0],
            "holdout_parent": totals[0, 1],
            "holdout_trial": totals[1, 1],
            "parent_codes": sha(c),
            "fitted_codes": sha(out),
            "scales": sha(s),
        }
    )
    (dest / "layers.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(il, rows[-1]["changed"], totals[1, 1] / totals[0, 1], flush=True)
assert len(rows) == 24
