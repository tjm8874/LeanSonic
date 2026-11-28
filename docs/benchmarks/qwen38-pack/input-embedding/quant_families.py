"""Measure independent MLX Q4 tensor families in the official BF16 forward pass."""
import argparse
from pathlib import Path

import torch

from upstream_q4 import EngramQ4, Q4Source, Weights, evaluate, load


def family(name):
    if "hyper_connection" in name:
        return "hyper-connection"
    if ".linear_attn." in name:
        return "gdn"
    if ".self_attn." in name:
        return "attention"
    if ".mlp.shared_expert" in name:
        return "shared-expert"
    if ".ple." in name:
        return "engram"
    if name in ("lm_head.weight", "model.embed_tokens.weight"):
        return "vocabulary"
    raise ValueError(f"unclassified quantized parameter {name}")


@torch.inference_mode()
def run(args):
    model = load(args.official, "cuda")
    original = Weights(args.official)
    q4 = Q4Source(args.q4)
    groups = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        key = ("language_model." + name).removesuffix(".weight")
        if key + ".scales" in q4.weights.mapping:
            groups.setdefault(family(name), []).append((name, parameter, key))
    table_path = "model.layers.1.ple.ple_embedding.ngram_embedding"
    bf16_table = model.get_submodule(table_path)
    args.dest.mkdir()
    for group in ("hyper-connection", "gdn", "attention", "shared-expert", "engram", "vocabulary"):
        selected = groups[group]
        for name, parameter, key in selected:
            for first in range(0, len(parameter), 256):
                rows = slice(first, first + 256)
                weights = q4.unpack(key, rows)
                if weights.shape != parameter[rows].shape:
                    raise ValueError(name)
                parameter[rows].copy_(weights)
        if group == "engram":
            model.set_submodule(table_path, EngramQ4(bf16_table, q4))
        print(group, len(selected), flush=True)
        dest = args.dest / group
        dest.mkdir()
        evaluate(model, args.ids, dest)
        if group == "engram":
            model.set_submodule(table_path, bf16_table)
        for name, parameter, _ in selected:
            official_key = name.replace("model.", "model.language_model.", 1)
            parameter.copy_(original.dense(official_key))
    dest = args.dest / "restored-bf16"
    dest.mkdir()
    evaluate(model, args.ids, dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("official", "q4", "ids", "dest"):
        parser.add_argument(argument, type=Path)
    run(parser.parse_args())
