"""Spend the middle package's additional MXFP8 precision within the original Q3 byte budget."""

import argparse
import json
from pathlib import Path
import numpy as np
from choose_codebooks import header, magnitude

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("official", type=Path)
parser.add_argument("base", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("target", type=int)
parser.add_argument("routing", type=Path, nargs="?")
parser.add_argument("--trim-baseline", type=Path, help="cutoff JSON for an already measured trim-context package")
parser.add_argument("--trim-bytes", type=int, help="measured bytes of that entire package")
args = parser.parse_args()
if (args.trim_baseline is None) != (args.trim_bytes is None):
    parser.error("--trim-baseline and --trim-bytes must be supplied together")
if args.trim_bytes is not None and args.trim_bytes <= 0:
    parser.error("--trim-bytes must be positive")
baseline = None
if args.trim_baseline is not None:
    baseline = json.loads(args.trim_baseline.read_text())["cutoffs"]
factors = np.ones((20, 128))
if args.routing is not None:
    factors = np.asarray(json.loads(args.routing.read_text())["factor"], dtype=np.float64)
    if (
        factors.shape != (20, 128)
        or not np.isfinite(factors).all()
        or np.any(factors < 0.5)
        or not np.allclose(factors.mean(axis=1), 1)
    ):
        raise ValueError(
            "expected 20 x 128 finite expert factors >= 0.5 with per-layer mean 1"
        )
official, base, maps = args.official, args.base, args.output
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


def trim_rate(blocks):
    maximum = np.zeros(len(blocks), dtype=np.int32)
    seen = np.zeros(len(blocks), dtype=bool)
    total = 0.0
    for column in blocks.T:
        context = 2 * maximum + seen
        hist = np.bincount(context * 256 + column, minlength=4096).reshape(16, 256)
        rows, symbols = np.nonzero(hist)
        counts = hist[rows, symbols]
        total += np.dot(counts, np.log2(hist.sum(axis=1)[rows] / counts))
        lo, hi = column & 7, (column >> 4) & 7
        maximum = np.maximum(maximum, np.maximum(lo, hi))
        seen |= (lo == 1) | (lo == 3) | (hi == 1) | (hi == 3)
    return total / 8


frame_rate = rate if baseline is None else trim_rate
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
        risk *= factors[layer, :, None]
        anchor = codes
        if baseline is not None:
            cutoffs = np.asarray(baseline[name])
            if cutoffs.shape != (128,) or np.any((cutoffs < -1) | (cutoffs > 126)):
                raise ValueError(f"invalid baseline cutoffs for {name}")
            anchor = np.where((scale <= cutoffs[ex])[:, None], trim[codes], codes)
        planes[name] = (
            codes,
            trim[codes],
            ex,
            scale,
            risk,
            frame_rate(anchor),
            weights.size / codes.size,
        )
size = args.trim_bytes if baseline is not None else sum(
    f.stat().st_size for f in base.rglob("*") if f.is_file()
)
target = args.target


def estimate(th):
    savings = 0.0
    luts = {}
    for key, (codes, cut, ex, scale, risk, old, factor) in planes.items():
        # Monotone mapping from E4M3 magnitude to impact for each expert.
        thresholds = (np.sum(risk <= th, axis=1) - 1).tolist()
        mask = scale <= np.array(thresholds)[ex]
        result = np.where(mask[:, None], cut, codes)
        savings += (old - frame_rate(result)) * factor
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
