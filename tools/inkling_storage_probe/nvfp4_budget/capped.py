"""Minimize checkpoint-weighted squared error with at most eight E4M3 magnitudes."""

import argparse
import json
from pathlib import Path

import numpy as np
from choose_codebooks import Plane, count_weights, header


def solve(plane, max_bins):
    # best[k,e] covers [0,e) with k nonempty, contiguous magnitude bins.
    best = np.full((max_bins + 1, 128), np.inf)
    start_at = np.zeros(best.shape, dtype=np.int32)
    best[0, 0] = 0.0
    for k in range(1, max_bins + 1):
        for end in range(k, 128):
            starts = slice(k - 1, end)
            costs = best[k - 1, starts] + plane.error[starts, end]
            i = int(np.argmin(costs))
            best[k, end] = costs[i]
            start_at[k, end] = k - 1 + i
    k = 1 + int(np.argmin(best[1:, 127]))
    error = float(best[k, 127])
    mapping = [0] * 127
    end = 127
    while k:
        start = int(start_at[k, end])
        mapping[start:end] = [int(plane.center[start, end])] * (end - start)
        end = start
        k -= 1
    assert end == 0 and mapping[0] == 0 and mapping == sorted(mapping)
    assert 2 * len(set(mapping)) - 1 <= 2 * max_bins - 1
    return mapping, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.official.open("rb") as stream:
        meta, start = header(stream)
    maps, errors = {}, {}
    for layer in range(20, 24):
        for plane in ("w13_weight", "w2_weight"):
            name = f"model.llm.layers.{layer}.mlp.experts.{plane}"
            counts = count_weights(args.official, meta, start, name)
            maps[name], errors[name] = solve(Plane(counts), 8)
    with args.output.open("x") as f:
        json.dump(
            dict(max_magnitude_bins=8, maps=maps, plane_relative_weight_error=errors),
            f,
            indent=2,
        )
        f.write("\n")


if __name__ == "__main__":
    main()
