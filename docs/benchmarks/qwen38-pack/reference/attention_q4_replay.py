"""Run the upstream attention blocks with the original MLX weights decoded to BF16."""

import json
from pathlib import Path

import numpy as np
import torch
from bf16_teacher import Weights, load


@torch.inference_mode()
def main():
    ids = json.loads(Path("layer-cross.json").read_text())["ids"]
    ref = np.fromfile("layers-bf16.f32", dtype="<f4").reshape(9, 49, 10240)
    mids = np.fromfile("attention-bf16.f32", dtype="<f4").reshape(9, 48, 10240)
    native = np.fromfile("local-attention.f32", dtype="<f4").reshape(9, 49, 10240)[
        :, 1:
    ]
    source = Weights(Path("restored"))
    model = load(Path("official-bf16"), "cuda")
    chosen = (15, 31, 35, 39)
    current = 0
    pos = []

    def supply(idx):
        def pre(mod, args, kwargs):
            x = args[0] if args else kwargs["hidden_states"]
            h = torch.as_tensor(
                ref[current, idx], device=x.device, dtype=x.dtype
            ).view_as(x)
            if args:
                return (h, *args[1:]), kwargs
            return args, {**kwargs, "hidden_states": h}

        return pre

    def capture(mod, args):
        pos.append(args[0].detach().float().cpu().numpy().reshape(10240).copy())

    for i in chosen:
        model.model.layers[i].register_forward_pre_hook(supply(i), with_kwargs=True)
        model.model.layers[i].mlp_hyper_connection.register_forward_pre_hook(capture)
    ptr = torch.arange(0, 32, 4, device="cuda", dtype=torch.int32)

    def assign(i, name):
        module = model.model.layers[i].get_submodule(name)
        k = f"language_model.model.layers.{i}.{name}"
        if k + ".scales" in source.mapping:
            matrix = source.dense(k + ".weight")
            scale = source.dense(k + ".scales")
            bias = source.dense(k + ".biases")
            for start in range(0, matrix.shape[0], 256):
                sl = slice(start, start + 256)
                u = (
                    ((matrix[sl].to("cuda", torch.int32).unsqueeze(-1) >> ptr) & 15)
                    .reshape(matrix[sl].shape[0], -1, 32)
                    .float()
                )
                val = (
                    u * scale[sl].cuda().float()[:, :, None]
                    + bias[sl].cuda().float()[:, :, None]
                )
                module.weight[sl].copy_(val.reshape_as(module.weight[sl]))
        else:
            module.weight.copy_(source.dense(k + ".weight"))

    def metric(target, candidate):
        return (
            np.linalg.norm(
                target.astype(np.float64) - candidate.astype(np.float64), axis=-1
            )
            / np.linalg.norm(target.astype(np.float64), axis=-1)
        ).tolist()

    results = {}
    for arm, names in [
        ("official", ()),
        (
            "qk",
            (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.q_norm",
                "self_attn.k_norm",
            ),
        ),
        ("qkvo", ("self_attn.v_proj", "self_attn.o_proj")),
        (
            "qkvo_hc",
            (
                "attn_hyper_connection.hc_norm",
                "attn_hyper_connection.input_mix_weight_down",
                "attn_hyper_connection.input_mix_weight_up",
                "attn_hyper_connection.block_inject_weight",
            ),
        ),
    ]:
        for i in chosen:
            for name in names:
                assign(i, name)
        pos.clear()
        past = None
        for current, token in enumerate(ids):
            out = model(
                input_ids=torch.tensor([[token]], device="cuda"),
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = out.past_key_values
        vals = np.asarray(pos).reshape(len(ids), len(chosen), 10240)
        if arm == "official" and not np.array_equal(vals, mids[:, chosen]):
            raise ValueError(
                "upstream control did not reproduce the official attention result"
            )
        results[arm] = {
            "rel_l2_to_official": metric(mids[:, chosen], vals),
            "rel_l2_to_native": metric(native[:, chosen], vals),
        }
    return {
        "layers": list(chosen),
        "ids": ids,
        "arms": results,
        "native_rel_l2_to_official": metric(mids[:, chosen], native[:, chosen]),
    }


print(json.dumps(main(), indent=2))
