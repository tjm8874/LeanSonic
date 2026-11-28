"""Build complete-vocabulary precision probes without modifying either source checkpoint."""
import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


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

    if mode == "bf16":
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
        manifest["metadata"]["total_size"] += sum(x.numel() * x.element_size() for x in data.values())
        (output / index).write_text(json.dumps(manifest, indent=2) + "\n")
        output.rename(destination)
    print(json.dumps({"mode": mode, "probe_shard_bytes": (destination / extra).stat().st_size,
                      "package_bytes": sum(p.stat().st_size for p in destination.iterdir() if p.is_file())}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "official", "destination"):
        ap.add_argument(name, type=Path)
    ap.add_argument("mode", choices=("q8", "bf16"))
    args = ap.parse_args()
    build(args.base, args.official, args.destination, args.mode)
