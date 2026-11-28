"""Record BF16 per-channel input energy for the attention V/O Q4 fit."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from bf16_teacher import load


@torch.inference_mode()
def capture(official, fixtures, destination):
    sequences = [list(map(int, line.split())) for line in fixtures.read_text().splitlines()]
    if not sequences or any(len(sequence) < 2 for sequence in sequences):
        raise ValueError("expected tokenized calibration prompts")
    model = load(official, "cuda")
    energy, counts = {}, {}

    def collect(name):
        def hook(module, inputs):
            x = inputs[0].detach().reshape(-1, module.in_features).float()
            energy[name] = energy.get(name, 0) + x.square().sum(dim=0).cpu()
            counts[name] = counts.get(name, 0) + x.shape[0]
        return hook

    for layer in (15, 31, 35, 39):
        for proj in ("v_proj", "o_proj"):
            name = f"layers.{layer}.self_attn.{proj}"
            model.model.get_submodule(name).register_forward_pre_hook(collect(name))
    for seq in sequences:
        model(input_ids=torch.tensor([seq], device="cuda"), use_cache=False, return_dict=True)
    expected = sum(map(len, sequences))
    if len(counts) != 8 or any(n != expected for n in counts.values()):
        raise ValueError(f"capture counts differ from {expected}: {counts}")
    for name, values in energy.items():
        if not torch.isfinite(values).all() or values.sum() <= 0:
            raise ValueError(f"invalid energy in {name}")
        values.div_(expected)
    save_file(energy, destination, metadata={"tokens": str(expected)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("official", "fixtures", "destination"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    capture(args.official, args.fixtures, args.destination)
