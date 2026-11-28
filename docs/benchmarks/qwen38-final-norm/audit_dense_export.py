"""Audit decoder tensors stored without Q4 codes in the MLX export against official values."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from safetensors import safe_open


def audit(official, mlx):
    index = "model.safetensors.index.json"
    mapping = [json.loads((p / index).read_bytes())["weight_map"] for p in (official, mlx)]
    groups = Counter()
    differences = []
    elements = 0
    for src, source_file in sorted(mapping[0].items()):
        if src == "lm_head.weight":
            dest = "language_model." + src
        elif src.startswith("model.language_model."):
            dest = src.replace("model.language_model.", "language_model.model.", 1)
        else:
            continue
        if ".mlp.experts." in src:
            groups["routed"] += 1
            continue
        if dest.removesuffix(".weight") + ".scales" in mapping[1]:
            groups["quantized"] += 1
            continue
        if dest not in mapping[1]:
            raise ValueError(f"missing export tensor {dest}")
        with safe_open(official / source_file, framework="pt") as a, safe_open(
            mlx / mapping[1][dest], framework="pt"
        ) as b:
            left, right = a.get_tensor(src), b.get_tensor(dest)
        if src.endswith(".conv1d.weight"):
            right = right.transpose(1, 2)
        if left.shape != right.shape:
            raise ValueError(f"shape differs for {dest}: {left.shape}, {right.shape}")
        elements += left.numel()
        if left.dtype == right.dtype and torch.equal(left, right):
            groups["equal"] += 1
            continue
        x, y = left.to(torch.float64), right.to(torch.float64)
        delta = (x - y).abs()
        groups["different"] += 1
        differences.append(dict(tensor=dest, source_dtype=str(left.dtype), export_dtype=str(right.dtype),
                                elements=x.numel(), unequal=int(torch.count_nonzero(delta)),
                                max_abs=float(delta.max()), mean_abs=float(delta.mean())))
    return dict(counts=dict(groups), compared_elements=elements, differences=differences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official", type=Path)
    parser.add_argument("mlx", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.official, args.mlx), indent=2))
