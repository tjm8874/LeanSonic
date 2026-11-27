"""Fit alternate E4M3 reproduction values for fixed compact codes, using weights only."""

import hashlib, json, struct, sys
from pathlib import Path
import numpy as np
from choose_codebooks import header, magnitude

official, compact, maps, out = map(Path, sys.argv[1:])
out.mkdir()
with official.open("rb") as f:
    meta, base = header(f)
with compact.open("rb") as f:
    packed, packed_base = header(f)
values = np.array([magnitude(x & 127) * (-1 if x & 128 else 1) for x in range(256)])
powers = np.exp2(2 * (np.arange(256, dtype=np.float64) - 127))
maps = json.loads(maps.read_text())["maps"]
summary = {}
for name, lut in sorted(maps.items()):
    row_groups = 16
    parity = 2 if name.endswith("w13_weight") else 1
    E, R, C = meta[name]["shape"]
    assert E == 128 and R == 2304 and C in (1152, 2304)
    weights = np.memmap(
        official,
        dtype="u1",
        mode="r",
        offset=base + meta[name]["data_offsets"][0],
        shape=(E * R, C),
    )
    scales = np.memmap(
        official,
        dtype="u1",
        mode="r",
        offset=base + meta[name + "_scale"]["data_offsets"][0],
        shape=(E * R, C // 32),
    )
    with compact.open("rb") as f:
        f.seek(packed_base + packed[name + ".rans_v1"]["data_offsets"][0])
        symbols = np.flatnonzero(np.frombuffer(f.read(512), dtype="<u2")).astype("u1")
    assert len(symbols) <= 16 and symbols[0] == 0
    table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if (i & 127) < 127 and lut[i & 127]:
            table[i] = (i & 128) | lut[i & 127]
    assert set(table) == set(symbols)
    index = np.zeros(256, dtype=np.int32)
    index[symbols] = np.arange(len(symbols))
    file = out / (name + ".lut")
    length = changes = 0
    prior_error = new_error = signal = 0.0
    with file.open("xb") as f:
        for first in range(0, E * R, 128):
            a = weights[first : first + 128]
            s = scales[first : first + 128]
            assert not np.any((a & 127) == 127) and np.all(s < 255)
            row = np.arange(len(a), dtype=np.int32)
            group = (row // row_groups) * parity + row % parity
            groups = int(group.max() + 1)
            bins = (group[:, None] * 16 + index[table[a]]).ravel()
            power = np.repeat(powers[s], 32, axis=1).ravel()
            v = values[a].ravel()
            total = np.bincount(bins, weights=power, minlength=groups * 16).reshape(
                groups, 16
            )
            sums = np.bincount(bins, weights=power * v, minlength=groups * 16).reshape(
                groups, 16
            )
            centers = np.zeros((groups, 16), dtype="u1")
            centers[:, : len(symbols)] = symbols
            for k, code in enumerate(symbols):
                if code == 0:
                    continue
                active = total[:, k] > 0
                mu = np.abs(sums[active, k] / total[active, k])
                mag = np.argmin(np.abs(mu[:, None] - values[:127]), axis=1).astype("u1")
                centers[active, k] = np.where(mag > 0, mag | (code & 128), 0)
            orig = values[table[a]].ravel()
            new = values[centers.ravel()[bins]]
            changes += int(np.count_nonzero(orig != new))
            length += a.size
            prior_error += float(np.dot(power, (orig - v) ** 2))
            new_error += float(np.dot(power, (new - v) ** 2))
            signal += float(np.dot(power, v * v))
            f.write(centers.tobytes())
    assert new_error <= prior_error * (1 + 1e-10)
    summary[name] = dict(
        rows_per_group=row_groups,
        parity=parity,
        columns=C,
        table_bytes=file.stat().st_size,
        codes=length,
        changed_weights=changes,
        original_relative_mse=prior_error / signal,
        fitted_relative_mse=new_error / signal,
        sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
    )
    print(name, summary[name], flush=True)
(out / "fit.json").write_text(json.dumps(summary, indent=2) + "\n")
