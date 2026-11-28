"""Restore both official BF16 PLE projections on the native Q4 checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from build_attention import save_overlay

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "official", "dest"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    index = "model.safetensors.index.json"
    extra = "ple-bf16-probe.safetensors"
    if a.dest.exists() or (a.base / extra).exists():
        raise FileExistsError("choose a fresh overlay")
    manifest = json.loads((a.base / index).read_bytes())
    src = json.loads((a.official / index).read_bytes())["weight_map"]
    values = []
    for name, rows in (("key_proj", 10240), ("value_proj", 2560)):
        key = f"model.language_model.layers.1.ple.{name}.weight"
        with safe_open(a.official / src[key], framework="pt", device="cpu") as f:
            t = f.get_tensor(key)
            if t.shape != (rows, 2560) or t.dtype != torch.bfloat16:
                raise ValueError(key)
            values.append(t)
    data = {"language_model.model.layers.1.ple.kv_proj.probe_bf16": torch.cat(values)}
    save_overlay(a.base, a.dest, manifest, extra, data)
    print(
        json.dumps(
            {
                "sidecar_bytes": (a.dest / extra).stat().st_size,
                "distribution_bytes": sum(f.stat().st_size for f in a.dest.iterdir()),
            }
        )
    )
