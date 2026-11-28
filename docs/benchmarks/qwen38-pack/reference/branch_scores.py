"""Measure attention and MoE branch errors with matched BF16 inputs."""

import json
from pathlib import Path

import numpy as np

ref = (
    np.fromfile("layers-bf16.f32", dtype="<f4").reshape(9, 49, 10240).astype(np.float64)
)
mid = (
    np.fromfile("attention-bf16.f32", dtype="<f4")
    .reshape(9, 48, 10240)
    .astype(np.float64)
)


def score(arm, base, target):
    measured = (
        np.fromfile(arm, dtype="<f4").reshape(ref.shape)[:, 1:].astype(np.float64)
    )
    err = np.linalg.norm(measured - target, axis=-1)
    return {
        "relative_residual_l2": (err / np.linalg.norm(target, axis=-1)).tolist(),
        "relative_branch_update_l2": (
            err / np.linalg.norm(target - base, axis=-1)
        ).tolist(),
    }


print(
    json.dumps(
        {
            "ids": json.loads(Path("layer-cross.json").read_text())["ids"],
            "attention": score("local-attention.f32", ref[:, :-1], mid),
            "mlp": score("local-mlp.f32", mid, ref[:, 1:]),
        },
        indent=2,
    )
)
