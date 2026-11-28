"""Keep routed experts and the Engram table in MLX Q4; vary other Q4-eligible matrices."""
import argparse
import json
from pathlib import Path

import torch

from upstream_q4 import EngramQ4, Q4Source, RoutedQ4, Weights, evaluate, load


def install(model, source, original, bits):
    tensors = 0
    elements = 0
    peak_error = 0.0
    for name, target in model.named_parameters(remove_duplicate=False):
        prefix = ("language_model." + name).removesuffix(".weight")
        if prefix + ".scales" not in source.weights.mapping:
            continue
        if target.ndim != 2 or target.shape[1] % 32 or target.dtype != torch.bfloat16:
            raise ValueError(f"ineligible matrix {name}")
        key = name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name
        with original.tensor(key) as view:
            if view.get_shape() != list(target.shape) or view.get_dtype() != "BF16":
                raise ValueError(f"official matrix shape/dtype mismatch {key}")
            for start in range(0, len(target), 256):
                sl = slice(start, start + 256)
                w = view[sl].to(device=target.device, dtype=torch.float32).reshape(-1, 32)
                limit = (1 << (bits - 1)) - 1
                scale = (w.abs().amax(dim=1, keepdim=True) / limit).bfloat16().float()
                denom = torch.where(scale == 0, 1.0, scale)
                codes = (w / denom).round().clamp(-limit, limit)
                restored = (codes * scale).bfloat16()
                peak_error = max(peak_error, (restored.float() - w).abs().max().item())
                target[sl].copy_(restored.reshape_as(target[sl]))
        tensors += 1
        elements += target.numel()
    if (tensors, elements) != (726, 4882227200):
        raise ValueError(f"incomplete dense coverage: {tensors}, {elements}")
    return dict(matrices=tensors, elements=elements, bits=bits, group_size=32,
                code_payload_bytes=elements * bits // 8, bf16_scale_bytes=elements // 32 * 2,
                max_abs_weight_error=peak_error)


@torch.inference_mode()
def run(args):
    model = load(args.official, "cuda")
    q4 = Q4Source(args.q4)
    original = Weights(args.official)
    for i, layer in enumerate(model.model.layers):
        layer.mlp.experts = RoutedQ4(layer.mlp.experts, q4, f"language_model.model.layers.{i}.mlp.switch_mlp")
    table = model.model.layers[1].ple.ple_embedding
    table.ngram_embedding = EngramQ4(table.ngram_embedding, q4)
    args.dest.mkdir()
    for bits in (16, 8, 6):
        arm = args.dest / f"dense-{bits}"
        arm.mkdir()
        if bits != 16:
            (arm / "weights.json").write_text(json.dumps(install(model, q4, original, bits), indent=2))
        for name, ids in (("original", args.original_ids), ("new", args.new_ids)):
            output = arm / name
            output.mkdir()
            evaluate(model, ids, output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("official", "q4", "original_ids", "new_ids", "dest"):
        p.add_argument(key, type=Path)
    run(p.parse_args())
