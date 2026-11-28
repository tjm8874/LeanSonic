"""Overlay official BF16 value/output projections on an otherwise unchanged MLX Q4 model."""
import argparse
import json
import os
import tempfile
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
import torch


def build(base, official, dest, layers, projections):
    index = "model.safetensors.index.json"
    manifest = json.loads((base / index).read_bytes())
    upstream = json.loads((official / index).read_bytes())["weight_map"]
    extra = "attention-bf16-probe.safetensors"
    if dest.exists() or (base / extra).exists():
        raise ValueError("choose an unchanged base and a fresh destination")
    added = {}
    for i in layers:
        for proj in projections:
            key = f"model.language_model.layers.{i}.self_attn.{proj}_proj.weight"
            with safe_open(official / upstream[key], framework="pt", device="cpu") as f:
                t = f.get_tensor(key)
            shape = (512, 2560) if proj == "v" else (2560, 6144)
            if tuple(t.shape) != shape or t.dtype != torch.bfloat16:
                raise ValueError(key)
            name = f"language_model.model.layers.{i}.self_attn.{proj}_proj.probe_bf16"
            added[name] = t
            manifest["weight_map"][name] = extra
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        out = Path(tmp) / "model"
        out.mkdir()
        for f in base.iterdir():
            if f.name != index:
                os.link(f, out / f.name)
        save_file(added, out / extra)
        manifest["metadata"]["total_size"] += sum(t.numel() * t.element_size() for t in added.values())
        (out / index).write_text(json.dumps(manifest, indent=2) + "\n")
        out.rename(dest)
    print(json.dumps({"layers": list(layers), "projections": projections, "file_bytes": sum(f.stat().st_size for f in dest.iterdir())}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for p in ("base", "official", "destination"):
        ap.add_argument(p, type=Path)
    ap.add_argument("--layers", choices=("four", "twelve"), required=True)
    ap.add_argument("--projections", choices=("v","o","vo"), required=True)
    a = ap.parse_args()
    build(a.base, a.official, a.destination, [15,31,35,39] if a.layers == "four" else range(3,48,4), a.projections)
