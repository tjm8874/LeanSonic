"""Repeat the worst published prefix on the official BF16 model in two execution modes."""

import json
from pathlib import Path

import numpy as np
import torch
from bf16_teacher import load


def distribution(x):
    x = x.astype(np.float64)
    x -= x.max()
    return x - np.log(np.exp(x).sum())


def summary(a, b):
    x, y = distribution(a), distribution(b)
    return {
        "forward_kl": float(np.dot(np.exp(x), x - y)),
        "max_abs_logit_diff": float(np.abs(a - b).max()),
        "top_ids": [int(a.argmax()), int(b.argmax())],
    }


record = json.loads(Path("bf16-wave1-quality.json").read_text())
case, pos = record["worst_case_position"]
seq = list(map(int, Path("fixtures.ids").read_text().splitlines()[case].split()))[
    : pos + 1
]


def read_row(p):
    rows = np.loadtxt(p / "rows.tsv", dtype=np.int64, ndmin=2)
    (idx,) = np.flatnonzero((rows[:, 0] == case) & (rows[:, 1] == pos))
    return (
        np.memmap(p / "logits.f32", mode="r", dtype="<f4")
        .reshape(-1, 248320)[idx]
        .copy()
    )


a = read_row(Path("score-bf16"))
q = read_row(Path("wave1-score-direct-engram"))
model = load(Path("official-bf16"), "cuda")
with torch.inference_mode():
    ids = torch.tensor([seq], device="cuda")
    prefix = (
        model(input_ids=ids, use_cache=True, return_dict=True)
        .logits[0, -1]
        .float()
        .cpu()
        .numpy()
    )
    past = None
    for i in range(len(seq)):
        pred = model(
            input_ids=ids[:, i : i + 1],
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = pred.past_key_values
    step = pred.logits[0, -1].float().cpu().numpy()
print(
    json.dumps(
        {
            "case": case,
            "position": pos,
            "input_ids": seq,
            "full_bf16_to_prefix": summary(a, prefix),
            "full_bf16_to_incremental": summary(a, step),
            "incremental_bf16_to_wave1_q4": summary(step, q),
            "full_bf16_to_wave1_q4": summary(a, q),
        },
        indent=2,
    )
)
