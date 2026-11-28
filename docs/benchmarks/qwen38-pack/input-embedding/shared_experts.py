"""Refit shared expert Q4 groups on the original BF16 weights and optionally stack an input probe."""
import argparse
import json
import os
import tempfile
from pathlib import Path

from build import refit_q4
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("base", "official", "destination"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    index = "model.safetensors.index.json"
    manifest = json.loads((args.base / index).read_bytes())
    upstream = json.loads((args.official / index).read_bytes())["weight_map"]
    shard = "shared-expert-refit-q4.safetensors"
    if args.destination.exists() or (args.base / shard).exists():
        raise FileExistsError("choose an original base and a fresh destination")
    planes = {}
    for layer in range(48):
        for proj in ("shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj", "shared_expert_gate"):
            suffix = f"layers.{layer}.mlp.{proj}."
            key = "model.language_model." + suffix + "weight"
            with safe_open(args.official / upstream[key], framework="pt", device="cpu") as f:
                table = f.get_tensor(key)
            prefix = "language_model.model." + suffix
            planes.update(refit_q4(args.base, table, prefix))
    with tempfile.TemporaryDirectory(dir=args.destination.parent) as scratch:
        out = Path(scratch) / "model"
        out.mkdir()
        for p in args.base.iterdir():
            if p.name != index:
                os.link(p, out / p.name)
        save_file(planes, out / shard)
        for name in planes:
            manifest["weight_map"][name] = shard
        (out / index).write_text(json.dumps(manifest, indent=2) + "\n")
        out.rename(args.destination)
    print(json.dumps({"parameters_updated":len(planes), "tensor_payload_bytes":sum(t.numel()*t.element_size() for t in planes.values()),
                      "shard_bytes":(args.destination/shard).stat().st_size,"fixture_bytes":sum(p.stat().st_size for p in args.destination.iterdir())}))


if __name__ == "__main__":
    main()
