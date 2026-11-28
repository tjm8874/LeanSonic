"""Refit the 290 target-model hyper-connection projections without changing their Q4 layout."""
import argparse
import json
import os
from pathlib import Path
import tempfile

from safetensors import safe_open
from safetensors.torch import save_file

from build import refit_q4


def build(base, official, dest):
    index = "model.safetensors.index.json"
    shard = "hyper-connection-refit.safetensors"
    manifest = json.loads((base / index).read_text())
    upstream = json.loads((official / index).read_text())["weight_map"]
    if dest.exists() or (base / shard).exists():
        raise FileExistsError("choose a fresh destination and unchanged base")
    keys = sorted(k.removesuffix("scales") for k in manifest["weight_map"]
                  if k.startswith("language_model.model.") and "hyper_connection" in k and k.endswith(".scales"))
    if len(keys) != 290:
        raise ValueError(f"unexpected hyper-connection projection count {len(keys)}")
    planes = {}
    for prefix in keys:
        key = prefix.replace("language_model.model.", "model.language_model.", 1) + "weight"
        with safe_open(official / upstream[key], framework="pt", device="cpu") as f:
            planes.update(refit_q4(base, f.get_tensor(key), prefix))
    with tempfile.TemporaryDirectory(dir=dest.parent) as scratch:
        out = Path(scratch) / "model"
        out.mkdir()
        for p in base.iterdir():
            if p.name != index:
                os.link(p, out / p.name)
        save_file(planes, out / shard)
        for key in planes:
            manifest["weight_map"][key] = shard
        (out / index).write_text(json.dumps(manifest, indent=2) + "\n")
        out.rename(dest)
    print(json.dumps({"projections": len(keys), "replacement_payload_bytes":
                      sum(t.numel() * t.element_size() for t in planes.values()),
                      "overlay_file_bytes": sum(f.stat().st_size for f in dest.iterdir())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "official", "dest"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    build(args.base, args.official, args.dest)
