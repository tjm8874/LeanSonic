"""Capture the official pre-MLP residual on the nine-token Japanese prefix."""

import json
from pathlib import Path

import numpy as np
import torch
from bf16_teacher import load


@torch.inference_mode()
def run():
    ids = json.loads(Path("layer-cross.json").read_text())["ids"]
    model = load(Path("official-bf16"), "cuda")
    traces = []

    def capture(module, args):
        traces.append(args[0].flatten().float().cpu().numpy().copy())

    handles = [
        layer.mlp_hyper_connection.register_forward_pre_hook(capture)
        for layer in model.model.layers
    ]
    past = None
    for token in ids:
        output = model(
            input_ids=torch.tensor([[token]], device="cuda"),
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
    for hook in handles:
        hook.remove()
    states = np.stack(traces).reshape(len(ids), 48, 10240)
    if not np.isfinite(states).all():
        raise ValueError("nonfinite pre-MLP state")
    with open("attention-bf16.f32", "xb") as f:
        states.astype("<f4").tofile(f)


run()
