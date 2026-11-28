"""Estimate per-Q8-block W_o second moments from 24 fit and eight withheld train windows."""

import hashlib
import json
from pathlib import Path

import numpy as np

src = Path("activations.bin")
with src.open("rb") as f:
    sha = hashlib.file_digest(f, "sha256").hexdigest()
record = 4 + 64 * (3072 + 192)
assert src.stat().st_size == 32 * 48 * record
h = np.zeros((2, 24, 48, 64, 64), dtype="f8")
counts = np.zeros((2, 24), dtype=np.int64)
with src.open("rb") as f:
    for window in range(32):
        fold = int(window % 4 == 3)
        visits = []
        for step in range(48):
            il = int.from_bytes(f.read(4), "little")
            visits.append(il)
            data = np.frombuffer(f.read(record - 4), dtype="u1").reshape(64, -1)
            codes = data[:, :3072].view("i1").reshape(64, 48, 64)
            scale = data[:, 3072:].copy().view("<f4").reshape(64, 48)
            assert np.isfinite(scale).all() and (scale >= 0).all() and np.any(codes)
            act = (codes.astype("f8") * scale[:, :, None]).transpose(1, 0, 2)
            h[fold, il] += act.transpose(0, 2, 1) @ act
            counts[fold, il] += 64
        assert sorted(visits) == sorted(
            [*range(6), *range(6, 18), *range(6, 18), *range(6, 18), *range(18, 24)]
        )
h /= counts[:, :, None, None, None]
for fold, name in enumerate(("fit", "heldout")):
    np.save(name + ".npy", h[fold])
eig = np.linalg.eigvalsh(h)
assert np.isfinite(h).all() and eig.min() > -1e-8
print(
    json.dumps(
        {
            "source": "WikiText-2 train, 32 disjoint 512-token windows, rows 3,11,...,507",
            "fit_windows": 24,
            "holdout_windows": 8,
            "physical_layers": 24,
            "visits": 1536,
            "source_bytes": src.stat().st_size,
            "source_sha256": sha,
            "positions_per_layer": counts.tolist(),
            "trace_range": [
                float(np.trace(h, axis1=-1, axis2=-2).min()),
                float(np.trace(h, axis1=-1, axis2=-2).max()),
            ],
        },
        indent=2,
    )
)
