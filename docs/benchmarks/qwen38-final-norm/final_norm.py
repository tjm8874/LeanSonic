"""Isolate final hyper-connection norm encoding without changing the other 48 layers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def read(directory, key):
    mapping = json.loads((directory / "model.safetensors.index.json").read_bytes())["weight_map"]
    with safe_open(directory / mapping[key], framework="pt", device="cpu") as data:
        value = data.get_tensor(key)
    if value.dtype != torch.bfloat16 or value.shape != (10240,):
        raise ValueError(f"unexpected final norm {value.dtype}, {value.shape}")
    return value


def digest(x):
    return hashlib.sha256(x.view(torch.uint16).numpy().tobytes()).hexdigest()


def run(args):
    key = "language_model.model.hyper_connection_mixer.hc_norm.weight"
    base, dest = args.base, args.dest
    official = read(args.official, key.replace("language_model.model.", "model.language_model."))
    exported = read(base, key)
    folded = (official.float() + 1).bfloat16()
    reconstructed = (exported.float() - 1).bfloat16()
    replacement = official if args.mode == "official" else reconstructed
    manifest_name = "model.safetensors.index.json"
    sidecar = "final-norm-ablation.safetensors"
    if dest.exists() or (base / sidecar).exists():
        raise FileExistsError("choose a fresh destination and base")
    manifest = json.loads((base / manifest_name).read_bytes())
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        model = Path(tmp) / "model"
        model.mkdir()
        for p in base.iterdir():
            if p.name != manifest_name:
                os.link(p, model / p.name)
        save_file({key: replacement}, model / sidecar)
        manifest["weight_map"][key] = sidecar
        (model / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n")
        model.rename(dest)
    return dict(mode=args.mode, elements=len(exported), raw_mean=float(exported.float().mean()),
                official_mean=float(official.float().mean()), exported_sha256=digest(exported),
                official_sha256=digest(official), installed_sha256=digest(replacement),
                equals_bf16_one_plus_official=int((exported == folded).sum()),
                subtract_add_returns_raw=int(((reconstructed.float() + 1) == exported.float()).sum()),
                sidecar_bytes=(dest / sidecar).stat().st_size,
                distribution_bytes=sum(p.stat().st_size for p in dest.iterdir()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("base", "official", "dest"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--mode", choices=("official", "reconstructed"), required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))
