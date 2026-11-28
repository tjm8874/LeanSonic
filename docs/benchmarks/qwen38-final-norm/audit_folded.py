"""Verify all folded target/drafter gains from checkpoint bytes without a GPU."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


def tensor(root, mapping, name):
    with safe_open(root / mapping[name], framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def audit(official, exported):
    index = "model.safetensors.index.json"
    upstream, mlx = [json.loads((root / index).read_bytes())["weight_map"] for root in (official, exported)]
    final = "model.language_model.hyper_connection_mixer.hc_norm.weight"
    names = [final] + sorted(k for k in upstream if k.startswith("mtp.") and "norm" in k and k.endswith(".weight"))
    if len(names) != 10:
        raise ValueError(f"unexpected number of checkpoint norms: {len(names)}")
    records = []
    for name in names:
        target = name.replace("model.language_model.", "language_model.model.", 1) if name == final else "language_model." + name
        reference = tensor(official, upstream, name)
        stored = tensor(exported, mlx, target)
        if stored.shape != reference.shape or any(t.dtype != torch.bfloat16 for t in (reference, stored)):
            raise ValueError(target)
        folded = (reference.float() + 1).bfloat16()
        records.append(dict(tensor=target, elements=reference.numel(),
                            equal_to_official=int((stored == reference).sum()),
                            equal_to_folded=int((stored == folded).sum()),
                            sha256=hashlib.sha256(stored.view(torch.uint16).numpy().tobytes()).hexdigest()))
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official", type=Path)
    parser.add_argument("exported", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.official, args.exported), indent=2))
