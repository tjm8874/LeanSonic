import hashlib
import json
import struct
from pathlib import Path
import numpy as np

data = Path("routes.bin").read_bytes()
size = 8 + 12 * 128
layers = list(range(6)) + list(range(6, 18)) * 3 + list(range(18, 20))
assert len(data) == 32 * len(layers) * size
hits = np.zeros((20, 128), "i8")
energy = np.zeros((20, 128), "f8")
slots = {}
for i in range(len(data) // size):
    at = i * size
    layer, slot = struct.unpack_from("<II", data, at)
    assert layer == layers[i % len(layers)]
    assert slots.setdefault(i % len(layers), slot) == slot
    h = np.frombuffer(data, "<u4", 128, at + 8)
    e = np.frombuffer(data, "<f8", 128, at + 8 + 4 * 128)
    assert (
        int(h.sum()) == 512 * 8
        and np.all(np.isfinite(e))
        and np.all((e > 0) == (h > 0))
    )
    hits[layer] += h
    energy[layer] += e
# Half of the impact estimate keeps the original uniform prior. Normalize
# within each physical layer; the existing fitter separately counts visits.
factor = 0.5 + 0.5 * energy / energy.mean(axis=1)[:, None]
assert np.allclose(factor.mean(axis=1), 1) and factor.min() >= 0.5
p = dict(
    source_sha256=hashlib.sha256(data).hexdigest(),
    windows=32,
    visits_per_window=len(layers),
    visits_by_layer=[layers.count(i) for i in range(20)],
    hits=hits.tolist(),
    sum_squared_route_weights=energy.tolist(),
    factor=factor.tolist(),
    min_factor=float(factor.min()),
    max_factor=float(factor.max()),
)
Path("routing.json").write_text(json.dumps(p, separators=(",", ":")) + "\n")
print(p["min_factor"], p["max_factor"])
