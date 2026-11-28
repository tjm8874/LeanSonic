"""Fit MLX affine Q4 values to the official BF16 weights for selected attention layers."""

import json
import os
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

layers = (15, 31, 35, 39)
base = Path("direct-engram")
official = Path("official-bf16")
dest = Path("fit-vo-q4")
if dest.exists():
    raise FileExistsError(dest)


def index(p):
    return json.loads((p / "model.safetensors.index.json").read_bytes())


q = index(base)
b = index(official)["weight_map"]


def read(root, mapping, key):
    with safe_open(root / mapping[key], framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def fit(words, scales, biases, target):
    shift = torch.arange(0, 32, 4)
    codes = (
        ((words.to(torch.int64).unsqueeze(-1) >> shift) & 15)
        .flatten(-2)
        .reshape(-1, 32)
        .float()
    )
    w = target.float().reshape(-1, 32)
    s = scales.float().reshape(-1, 1)
    b = biases.float().reshape(-1, 1)
    err = ((codes * s + b - w) ** 2).sum(1, keepdim=True)
    baseline = err.sum(dtype=torch.float64).item()
    best = err.clone()
    bs = s.clone()
    bb = b.clone()
    bc = codes.clone()
    for _ in range(5):
        x = codes - codes.mean(1, keepdim=True)
        den = (x * x).sum(1, keepdim=True)
        s = (
            torch.where(den > 0, (x * w).sum(1, keepdim=True) / den, s)
            .bfloat16()
            .float()
        )
        b = (w - codes * s).mean(1, keepdim=True).bfloat16().float()
        codes = torch.where(s != 0, torch.round((w - b) / s).clamp(0, 15), codes)
        err = ((codes * s + b - w) ** 2).sum(1, keepdim=True)
        use = err < best
        bs = torch.where(use, s, bs)
        bb = torch.where(use, b, bb)
        bc = torch.where(use, codes, bc)
        best = torch.minimum(best, err)
    words = (
        (bc.reshape(-1, 8).to(torch.int64) << shift)
        .sum(-1)
        .to(torch.uint32)
        .reshape_as(words)
    )
    return (
        words,
        bs.bfloat16().reshape_as(scales),
        bb.bfloat16().reshape_as(biases),
    ), (baseline, best.sum(dtype=torch.float64).item())


stats = {}
planes = {}
for i in layers:
    for proj in ["v_proj", "o_proj"]:
        name = f"layers.{i}.self_attn.{proj}"
        p = "language_model.model." + name
        orig = read(official, b, "model.language_model." + name + ".weight")
        keys = [p + "." + v for v in ("weight", "scales", "biases")]
        arrays = [read(base, q["weight_map"], k) for k in keys]
        output = [torch.empty_like(z) for z in arrays]
        before = after = 0
        for start in range(0, len(orig), 64):
            sl = slice(start, start + 64)
            values, (err0, err1) = fit(*(x[sl] for x in arrays), orig[sl])
            before += err0
            after += err1
            for dst, src in zip(output, values):
                dst[sl].copy_(src)
        for k, v in zip(keys, output):
            planes[k] = v.contiguous()
            q["weight_map"][k] = "attention-fit-q4.safetensors"
        stats[name] = {
            "before_sse": before,
            "after_sse": after,
            "sse_ratio": after / before,
        }
        print(name, stats[name], flush=True)
with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
    out = Path(tmp) / "candidate"
    out.mkdir()
    for f in base.iterdir():
        if f.name != "model.safetensors.index.json":
            os.link(f, out / f.name)
    save_file(planes, out / "attention-fit-q4.safetensors")
    (out / "model.safetensors.index.json").write_text(json.dumps(q, indent=2) + "\n")
    out.rename(dest)
Path("fit-vo-weights.json").write_text(json.dumps(stats, indent=2) + "\n")
