"""Choose contiguous E4M3 magnitude bins for a complete distribution byte budget."""

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np


def header(stream):
    (size,) = struct.unpack("<Q", stream.read(8))
    return json.loads(stream.read(size)), 8 + size


def count_weights(shard, meta, offset, key):
    lo, hi = meta[key]["data_offsets"]
    begin, end = meta[key + "_scale"]["data_offsets"]
    assert meta[key]["dtype"] == "F8_E4M3" and meta[key + "_scale"]["dtype"] == "U8"
    assert hi - lo == 32 * (end - begin)
    counts = np.zeros((256, 256), dtype=np.int64)
    with shard.open("rb") as weights, shard.open("rb") as scales:
        weights.seek(offset + lo)
        scales.seek(offset + begin)
        for first in range(0, end - begin, 1 << 16):
            n = min(1 << 16, end - begin - first)
            scale = np.frombuffer(scales.read(n), dtype=np.uint8).astype(np.int32)
            code = np.frombuffer(weights.read(32 * n), dtype=np.uint8).reshape(n, 32)
            counts += np.bincount(
                ((scale[:, None] << 8) | code).ravel(), minlength=65536
            ).reshape(256, 256)
    assert (
        counts.sum() == hi - lo
        and not counts[:, [127, 255]].any()
        and not counts[255].any()
    )
    return counts


def magnitude(code):
    if code < 8:
        return code * 2.0**-9
    return (8 + (code & 7)) * 2.0 ** ((code >> 3) - 10)


class Plane:
    def __init__(self, counts):
        codes = counts.sum(axis=0)
        power_by_code = np.exp2(2 * (np.arange(256, dtype=np.float64) - 127)) @ counts
        weight = power_by_code[:127] + power_by_code[128:255]
        levels = np.array([magnitude(i) for i in range(127)])
        signal = np.dot(weight, levels**2)
        self.total = int(codes.sum())
        self.error = np.full((127, 128), np.inf)
        self.bits = np.zeros((127, 128))
        self.center = np.zeros((127, 128), dtype=np.uint8)
        positive = np.r_[0, codes[:127].cumsum()]
        negative = np.r_[0, codes[128:255].cumsum()]
        for end in range(1, 128):
            for start in range(end):
                w = weight[start:end]
                values = levels[start:end]
                # Reserve exact zero. Other bins cannot use another bin's reproduction value.
                if start == 0:
                    center = 0
                elif w.sum():
                    center = start + int(
                        np.argmin(np.abs(values - np.average(values, weights=w)))
                    )
                else:
                    center = start
                self.center[start, end] = center
                self.error[start, end] = (
                    np.dot(w, (values - levels[center]) ** 2) / signal
                )
                pos, neg = (
                    int(positive[end] - positive[start]),
                    int(negative[end] - negative[start]),
                )
                frequencies = (pos, neg) if center else (pos + neg,)
                self.bits[start, end] = sum(
                    n * math.log2(self.total / n) for n in frequencies if n
                )

    def solve(self, penalty):
        # Use bits in gigabits to keep the multiplier independent of tensor order.
        cost = self.error + penalty * self.bits / 1e9
        best = np.zeros(128)
        previous = np.zeros(128, dtype=int)
        for end in range(1, 128):
            alternatives = best[:end] + cost[:end, end]
            start = int(np.argmin(alternatives))
            previous[end] = start
            best[end] = alternatives[start]
        lut = [0] * 127
        bits = distortion = 0.0
        end = 127
        while end:
            start = previous[end]
            lut[start:end] = [int(self.center[start, end])] * (end - start)
            bits += self.bits[start, end]
            distortion += self.error[start, end]
            end = start
        return dict(map=lut, bits=bits, relative_error=distortion)


def choose(planes, fixed, target):
    # Upper estimate for the per-tile state and size fields; actual byte counts come from the packer.
    overhead = sum(512 + 20 * math.ceil(p.total / 131072) + 7 for p in planes.values())

    def evaluate(penalty):
        solutions = {k: p.solve(penalty) for k, p in planes.items()}
        return fixed + overhead + sum(
            v["bits"] for v in solutions.values()
        ) / 8, solutions

    low, high = 0.0, 1.0
    while evaluate(high)[0] > target:
        high *= 2
        if high > 1e12:
            raise ValueError("unreachable byte budget")
    for _ in range(42):
        mid = (low + high) / 2
        if evaluate(mid)[0] > target:
            low = mid
        else:
            high = mid
    size, results = evaluate(high)
    return dict(
        target_bytes=target,
        estimated_bytes=size,
        penalty=high,
        maps={k: v["map"] for k, v in results.items()},
        plane_relative_weight_error={
            k: v["relative_error"] for k, v in results.items()
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official_shard", type=Path)
    parser.add_argument("canonical_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("target_bytes", type=int, nargs="+")
    args = parser.parse_args()
    with args.official_shard.open("rb") as f:
        raw, offset = header(f)
    with (args.canonical_directory / args.official_shard.name).open("rb") as f:
        parent, _ = header(f)
    fixed = sum(
        f.stat().st_size for f in args.canonical_directory.rglob("*") if f.is_file()
    )
    planes = {}
    for layer in range(20, 24):
        for matrix in ("w13_weight", "w2_weight"):
            key = f"model.llm.layers.{layer}.mlp.experts.{matrix}"
            lo, hi = parent[key + ".rans_v1"]["data_offsets"]
            fixed -= hi - lo
            planes[key] = Plane(count_weights(args.official_shard, raw, offset, key))
    args.output.mkdir()
    for target in args.target_bytes:
        result = choose(planes, fixed, target)
        (args.output / f"{target}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(target, result["estimated_bytes"], flush=True)
