"""Create a diagnostic mixed-precision index from matching Q4 and Q8 packages."""
import argparse
import json
import os
from pathlib import Path
import tempfile

INDEX = "model.safetensors.index.json"
FAMILIES = {
    "attention": (".self_attn.", ".linear_attn.", ".ple.key_proj.", ".ple.value_proj."),
    "mixer": (".attn_hyper_connection.", ".mlp_hyper_connection.", ".hyper_connection_mixer."),
    "shared": (".mlp.shared_expert.", ".mlp.shared_expert_gate."),
}
COUNTS = {"attention": 242, "mixer": 290, "shared": 192}


def overlay(q4, q8, output, families):
    if output.exists():
        raise ValueError("choose a fresh output directory")
    base = json.loads((q4 / INDEX).read_bytes())
    candidate = json.loads((q8 / INDEX).read_bytes())
    if candidate["metadata"].get("leansonic_dense_format") != "symmetric-q8-group32-bf16":
        raise ValueError("unexpected mixed package format")
    original = base["weight_map"]
    mixed = candidate["weight_map"]
    if len(set(families)) != len(families):
        raise ValueError("repeated projection family")
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        model = Path(tmp) / "model"
        model.mkdir()
        for file in q4.iterdir():
            if file.name != INDEX:
                os.link(file, model / file.name)
        for family in families:
            prefixes = sorted(k.removesuffix(".scales") for k in original
                              if k.startswith("language_model.model.") and k.endswith(".scales")
                              and any(tag in k for tag in FAMILIES[family]))
            if len(prefixes) != COUNTS[family]:
                raise ValueError(f"unexpected {family} count {len(prefixes)}")
            for prefix in prefixes:
                if prefix + ".biases" in mixed:
                    raise ValueError(f"unexpected affine weights {prefix}")
                for suffix in (".weight", ".scales"):
                    key = prefix + suffix
                    filename = mixed[key]
                    target = model / ("q8-" + filename)
                    if not target.exists():
                        os.link(q8 / filename, target)
                    original[key] = target.name
                del original[prefix + ".biases"]
        (model / INDEX).write_text(json.dumps(base, indent=2) + "\n")
        model.rename(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("q4", "q8", "output"):
        p.add_argument(arg, type=Path)
    p.add_argument("families", choices=list(FAMILIES), nargs="+")
    args = p.parse_args()
    overlay(args.q4, args.q8, args.output, args.families)
