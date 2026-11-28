"""Separate the two PLE projections from the Engram lookup table on identical token IDs."""
import argparse
from pathlib import Path

import torch

from upstream_q4 import EngramQ4, Q4Source, RoutedQ4, Weights, evaluate, install_dense, load


@torch.inference_mode()
def run(args):
    model = load(args.official, "cuda")
    q4 = Q4Source(args.q4)
    original = Weights(args.official)
    args.dest.mkdir()
    ple = model.model.layers[1].ple
    bf16_table = ple.ple_embedding.ngram_embedding
    q4_table = EngramQ4(bf16_table, q4)

    def projections(quantized):
        for name in ("key_proj", "value_proj"):
            prefix = "layers.1.ple." + name
            target = getattr(ple, name).weight
            if quantized:
                for first in range(0, len(target), 256):
                    rows = slice(first, first + 256)
                    value = q4.unpack("language_model.model." + prefix, rows)
                    if value.shape != target[rows].shape:
                        raise ValueError(name)
                    target[rows].copy_(value)
            else:
                target.copy_(original.dense("model.language_model." + prefix + ".weight"))

    if args.background == "q4":
        for i, layer in enumerate(model.model.layers):
            layer.mlp.experts = RoutedQ4(layer.mlp.experts, q4, f"language_model.model.layers.{i}.mlp.switch_mlp")
        install_dense(model, q4)
    for arm, table, use_q4_proj in (
        ("table-q4", q4_table, False), ("projections-q4", bf16_table, True),
        ("both-q4", q4_table, True), ("both-bf16", bf16_table, False),
    ):
        ple.ple_embedding.ngram_embedding = table
        projections(use_q4_proj)
        dest = args.dest / arm
        dest.mkdir()
        evaluate(model, args.ids, dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("official", "q4", "ids", "dest"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--background", required=True, choices=("bf16", "q4"))
    run(parser.parse_args())
