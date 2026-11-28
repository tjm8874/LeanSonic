"""Evaluate stock Q4 attention V/O projections in an otherwise official BF16 forward pass."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
from bf16_teacher import Weights, evaluate, load


@torch.inference_mode()
def run(official, q4, fixtures, output):
    source = Weights(q4)
    model = load(official, "cuda")
    shifts = torch.arange(0, 32, 4, device="cuda", dtype=torch.int32)
    targets = {i for i, layer in enumerate(model.model.layers) if hasattr(layer, "self_attn") and
               f"language_model.model.layers.{i}.self_attn.v_proj.scales" in source.mapping}
    if targets != set(range(3, 48, 4)):
        raise ValueError(f"unexpected full-attention layers {targets}")
    original = {15, 31, 35, 39}
    output.mkdir()
    for arm, additions in [("official", set()), ("four-vo", original), ("twelve-vo", targets - original)]:
        for i in sorted(additions):
            for name in ("v_proj", "o_proj"):
                key = f"language_model.model.layers.{i}.self_attn.{name}"
                weight, scale, bias = [source.dense(f"{key}.{field}") for field in ("weight", "scales", "biases")]
                target = model.model.layers[i].self_attn.get_submodule(name).weight
                for lo in range(0, len(weight), 256):
                    sl = slice(lo, lo + 256)
                    codes = ((weight[sl].to("cuda", torch.int32).unsqueeze(-1) >> shifts) & 15)
                    codes = codes.reshape(len(codes), -1, 32).float()
                    values = codes * scale[sl].cuda().float()[:, :, None] + bias[sl].cuda().float()[:, :, None]
                    target[sl].copy_(values.reshape_as(target[sl]))
        path = output / arm
        path.mkdir()
        evaluate(model, fixtures, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("official", "q4", "fixtures", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.official, args.q4, args.fixtures, args.output)
