"""Minimize summed per-plane relative weight MSE in a matched estimated-byte window."""
import argparse
import bisect
import itertools
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("curves", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
base = json.loads(args.curves.read_text())
curves = base["curves"]
keys = sorted(curves)
assert len(keys) == 8
low = base["predicted_saved"]
high = low + 100_000
levels = tuple(range(85, 112, 2))  # includes all eight original cutoffs


def choices(names):
    for thresholds in itertools.product(levels, repeat=len(names)):
        cells = [curves[k][str(t)] for k, t in zip(names, thresholds, strict=True)]
        yield sum(c["saved"] for c in cells), sum(c["error"] for c in cells), thresholds


right = sorted(choices(keys[4:]))
sizes = [r[0] for r in right]
# Sparse table for the lowest distortion among combinations in any indexed interval.
index = [list(range(len(right)))]
def best(x, y):
    return x if (right[x][1], x) <= (right[y][1], y) else y

while (1 << len(index)) <= len(right):
    last = index[-1]
    half = 1 << (len(index) - 1)
    index.append([best(a, b) for a, b in zip(last, last[half:])])
winner = None
for size, loss, thresholds in choices(keys[:4]):
    start = bisect.bisect_left(sizes, low - size)
    stop = bisect.bisect_right(sizes, high - size)
    if start == stop:
        continue
    level = (stop - start).bit_length() - 1
    r = right[best(index[level][start], index[level][stop - (1 << level)])]
    item = (loss + r[1], size + r[0], thresholds + r[2])
    if winner is None or item < winner:
        winner = item
if winner is None:
    raise ValueError("no feasible combination")
loss, saved, thresholds = winner
result = dict(thresholds=dict(zip(keys, thresholds, strict=True)), predicted_saved=saved,
              mean_per_plane_relative_weight_mse=loss/8, previous_mean=base["mean_per_plane_relative_weight_mse"],
              predicted_byte_window=[low, high], search_levels=levels)
assert low <= saved <= high and loss/8 < result["previous_mean"]
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
