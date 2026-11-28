"""Build complete-vocabulary precision probes without modifying either source checkpoint."""
import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def refit_q4(base, table):
    """Fit group-32 affine parameters, retaining the lowest measured SSE for every group."""
    prefix = "language_model.model.embed_tokens."
    mapping = json.loads((base / "model.safetensors.index.json").read_bytes())["weight_map"]
    planes = {}
    for name in ("weight", "scales", "biases"):
        key = prefix + name
        with safe_open(base / mapping[key], framework="pt", device="cpu") as f:
            planes[name] = f.get_tensor(key)
    out = {name: torch.empty_like(t) for name, t in planes.items()}
    shifts = torch.arange(0, 32, 4, dtype=torch.int64)
    for start in range(0, len(table), 256):
        sl = slice(start, start + 256)
        shape = planes["weight"][sl].shape
        codes = ((planes["weight"][sl].to(torch.int64).unsqueeze(-1) >> shifts) & 15).reshape(-1, 32).float()
        w = table[sl].reshape_as(codes).float()
        scale = planes["scales"][sl].float().reshape(-1, 1)
        bias = planes["biases"][sl].float().reshape(-1, 1)
        best = (codes * scale + bias - w).square().sum(1, keepdim=True)
        best_codes, best_scale, best_bias = codes.clone(), scale.clone(), bias.clone()
        for _ in range(5):
            centered = codes - codes.mean(1, keepdim=True)
            denom = centered.square().sum(1, keepdim=True)
            scale = torch.where(denom > 0, (centered * w).sum(1, keepdim=True) / denom.clamp_min(1e-30), scale).bfloat16().float()
            bias = (w - scale * codes).mean(1, keepdim=True).bfloat16().float()
            codes = torch.where(scale != 0, ((w - bias) / scale).round().clamp(0, 15), codes)
            loss = (codes * scale + bias - w).square().sum(1, keepdim=True)
            take = loss < best
            best_codes = torch.where(take, codes, best_codes)
            best_scale = torch.where(take, scale, best_scale)
            best_bias = torch.where(take, bias, best_bias)
            best = torch.minimum(best, loss)
        out["weight"][sl] = (best_codes.reshape(-1, 8).to(torch.int64) << shifts).sum(1).reshape(shape).to(torch.uint32)
        out["scales"][sl] = best_scale.reshape_as(out["scales"][sl])
        out["biases"][sl] = best_bias.reshape_as(out["biases"][sl])
    return {prefix + name: t for name, t in out.items()}


def build(base, official, destination, mode):
    if destination.exists():
        raise FileExistsError(destination)
    index = "model.safetensors.index.json"
    key = "model.language_model.embed_tokens.weight"
    shard = json.loads((official / index).read_bytes())["weight_map"][key]
    prefix = "language_model.model.embed_tokens.probe_"
    with safe_open(official / shard, framework="pt", device="cpu") as f:
        table = f.get_tensor(key)
    if table.shape != (248320, 2560) or table.dtype != torch.bfloat16:
        raise ValueError("unexpected upstream input embedding")

    if mode == "refit_q4":
        data = refit_q4(base, table)
    elif mode == "bf16":
        data = {prefix + "bf16": table}
    else:
        codes = torch.empty_like(table, dtype=torch.int8)
        scales = torch.empty((len(table), 80), dtype=torch.bfloat16)
        for start in range(0, len(table), 512):
            chunk = slice(start, start + 512)
            values = table[chunk].float().reshape(-1, 80, 32)
            scale = (values.abs().amax(-1) / 127).bfloat16()
            divisor = torch.where(scale == 0, 1, scale).float()
            codes[chunk] = (values / divisor.unsqueeze(-1)).round().clamp(-127, 127).reshape(-1, 2560).to(torch.int8)
            scales[chunk] = scale
        data = {prefix + "q8": codes, prefix + "q8_scales": scales}
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        output = Path(scratch) / "model"
        output.mkdir()
        manifest = json.loads((base / index).read_bytes())
        extra = "input-embedding-probe.safetensors"
        for path in base.iterdir():
            if path.name != index:
                os.link(path, output / path.name)
        save_file(data, output / extra)
        for name in data:
            manifest["weight_map"][name] = extra
        if mode != "refit_q4":
            manifest["metadata"]["total_size"] += sum(x.numel() * x.element_size() for x in data.values())
        (output / index).write_text(json.dumps(manifest, indent=2) + "\n")
        output.rename(destination)
    print(json.dumps({"mode": mode, "probe_shard_bytes": (destination / extra).stat().st_size,
                      "package_bytes": sum(p.stat().st_size for p in destination.iterdir() if p.is_file())}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "official", "destination"):
        ap.add_argument(name, type=Path)
    ap.add_argument("mode", choices=("refit_q4", "q8", "bf16"))
    args = ap.parse_args()
    build(args.base, args.official, args.destination, args.mode)
