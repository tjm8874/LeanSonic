"""Compare the post-layer residual streams on the worst Japanese prefix."""

import json
from pathlib import Path

import numpy as np
import torch
from bf16_teacher import load


@torch.inference_mode()
def run():
    ids = list(map(int, Path("fixtures.ids").read_text().splitlines()[1].split()))[:9]
    native = np.fromfile("layers-native.f32", dtype="<f4").reshape(len(ids), 49, -1)
    rows = np.loadtxt("wave1-score-direct-engram/rows.tsv", dtype=np.int64)
    (idx,) = np.flatnonzero((rows[:, 0] == 1) & (rows[:, 1] == 8))
    old = np.memmap(
        "wave1-score-direct-engram/logits.f32", mode="r", dtype="<f4"
    ).reshape(-1, 248320)[idx]
    trace = np.fromfile("layers-native-logits.f32", dtype="<f4")
    if not np.array_equal(trace, old):
        raise ValueError("native instrumentation changed final logits")
    model = load(Path("official-bf16"), "cuda")
    values = []

    def first(mod, args, kwargs):
        value = args[0] if args else kwargs["hidden_states"]
        values.append(value.reshape(-1).float().cpu().numpy())

    def post(mod, args, output):
        values.append(output.reshape(-1).float().cpu().numpy())

    model.model.layers[0].register_forward_pre_hook(first, with_kwargs=True)
    for layer in model.model.layers:
        layer.register_forward_hook(post)
    past = None
    records = []
    refstates = []
    for i, token in enumerate(ids):
        values.clear()
        o = model(
            input_ids=torch.tensor([[token]], device="cuda"),
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = o.past_key_values
        refs = np.stack(values)
        assert (
            refs.shape == native[i].shape
            and np.isfinite(refs).all()
            and np.isfinite(native[i]).all()
        )
        refstates.append(refs)
        results = []
        for s, (a, b) in enumerate(zip(refs, native[i])):
            a, b = a.astype(np.float64), b.astype(np.float64)
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            results.append(
                [
                    float(np.linalg.norm(a - b) / na),
                    float(np.dot(a, b) / (na * nb)),
                    float(nb / na),
                ]
            )
        records.append(results)
    np.asarray(refstates, dtype="<f4").tofile("layers-bf16.f32")
    return {
        "fixture": 1,
        "ids": ids,
        "slots": ["embedding"] + list(range(48)),
        "fields": ["relative_l2", "cosine", "norm_ratio_native_over_bf16"],
        "native_logits_unchanged": True,
        "values": records,
    }


print(json.dumps(run(), indent=2))
