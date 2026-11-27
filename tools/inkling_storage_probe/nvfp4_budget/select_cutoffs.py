"""Spend the middle package's additional MXFP8 precision within the original Q3 byte budget."""

import json, struct, sys
from pathlib import Path
import numpy as np
from choose_codebooks import header, magnitude

official, base, maps = map(Path, sys.argv[1:4])
shard = "model-00001-of-00001.safetensors"
with official.open("rb") as f:
    original, start = header(f)
with (base / shard).open("rb") as f:
    model, _ = header(f)
levels = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
mag = np.arange(256, dtype=np.uint8)
low = mag & 15
high = mag >> 4
normalize = ((low * (low != 8)) | ((high * (high != 8)) << 4)).astype("u1")


def trim_nibble(v):
    # 0.5 -> 0 and 1.5 -> 2 are both half-unit shifts. Preserve sign for nonzero results.
    m = v & 7
    return np.where(m <= 1, 0, np.where(m == 3, (v & 8) | 4, v))


trim = (trim_nibble(low) | (trim_nibble(high) << 4)).astype("u1")
maximum = np.maximum(low & 7, high & 7).astype(np.int32)
power = levels[low & 7] ** 2 + levels[high & 7] ** 2
e4 = np.array([magnitude(x) for x in range(127)])


def arr(k, dt):
    m = original[k]
    return np.memmap(
        official,
        dtype=dt,
        mode="r",
        offset=start + m["data_offsets"][0],
        shape=tuple(m["shape"]),
    )


def rate(blocks):
    prev = np.zeros(len(blocks), dtype=np.int32)
    bits = 0.0
    for x in blocks.T:
        h = np.bincount(prev * 256 + x, minlength=2048).reshape(8, 256)
        tot = h.sum(axis=1)
        rr, cc = np.nonzero(h)
        count = h[rr, cc]
        bits += np.dot(count, np.log2(tot[rr] / count))
        prev = np.maximum(prev, maximum[x])
    return bits / 8


planes = {}
for layer in range(20):
    for plane in ["w13", "w2"]:
        name = f"model.llm.layers.{layer}.mlp.experts.{plane}_weight"
        weights = arr(name, "u1")
        scales = arr(name + "_scale", "u1")
        amax = arr(name + "_amax", "<f4").astype("f8")
        denom = arr(name + "_outer_scale_denominator", "<f4").item()
        assert weights.shape[0] == 128 and np.all(np.isfinite(amax)) and denom > 0
        # Every expert is represented equally. Rotate the deterministic sample origin across experts.
        codes = []
        scale = []
        ex = []
        for i in range(128):
            pos = (19 * i + 7 * layer) % 128
            group = weights[i].reshape(-1, 8)[pos::128]
            sv = scales[i].ravel()[pos::128]
            assert np.all(sv < 127) and len(sv) == len(group)
            codes.append(normalize[group])
            scale.append(sv)
            ex.append(np.full(len(sv), i, dtype=np.int32))
        codes = np.concatenate(codes)
        scale = np.concatenate(scale)
        ex = np.concatenate(ex)
        actual = (amax[:, None] / denom * e4[None, :]) ** 2
        signal = np.mean(power[codes].sum(axis=1) * actual[ex, scale]) / 16
        assert signal > 0
        # Three visits to each recurrent layer. Compare impact per changed weight.
        risk = actual * (3 if 6 <= layer < 18 else 1) / signal
        planes[name] = (
            codes,
            trim[codes],
            ex,
            scale,
            risk,
            rate(codes),
            weights.size / codes.size,
        )
size = sum(f.stat().st_size for f in base.rglob("*") if f.is_file())
target = int(sys.argv[4])


def estimate(th):
    savings = 0.0
    luts = {}
    for key, (codes, cut, ex, scale, risk, old, factor) in planes.items():
        # Monotone mapping from E4M3 magnitude to impact for each expert.
        thresholds = (np.sum(risk <= th, axis=1) - 1).tolist()
        mask = scale <= np.array(thresholds)[ex]
        result = np.where(mask[:, None], cut, codes)
        savings += (old - rate(result)) * factor
        luts[key] = thresholds
    return round(size - savings), luts


lo = 0.0
hi = max(row[4].max() for row in planes.values())
endpoint, _ = estimate(hi)
if endpoint > target:
    raise ValueError(f"even all blocks would occupy {endpoint} > {target}")
for i in range(23):
    mid = (lo + hi) / 2
    est, _ = estimate(mid)
    if est > target:
        lo = mid
    else:
        hi = mid
est, lut = estimate(hi)
maps.write_text(
    json.dumps(
        dict(
            parent_size=size,
            target_size=target,
            estimated_size=est,
            all_blocks_size=endpoint,
            threshold=hi,
            cutoffs=lut,
        ),
        indent=2,
    )
    + "\n"
)
print(size, target, est, endpoint)
