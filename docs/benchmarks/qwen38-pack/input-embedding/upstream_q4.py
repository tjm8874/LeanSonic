"""Run the official eager model on the MLX Q4 codes, decoded to BF16 at their point of use."""
import argparse
import sys
from bisect import bisect_right
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
from bf16_teacher import ExpertRows, ShardedEmbedding, Weights, load, evaluate


class Q4Source:
    def __init__(self, directory):
        self.weights = Weights(directory)
        self.shifts = torch.arange(0, 32, 4, device="cuda", dtype=torch.int32)

    def unpack(self, key, select):
        planes = []
        for name in ("weight", "scales", "biases"):
            with self.weights.tensor(f"{key}.{name}") as view:
                values = torch.stack([view[i] for i in select]) if isinstance(select, list) else view[select]
                planes.append(values.cuda())
        codes, scales, biases = planes
        if codes.dtype != torch.uint32 or scales.dtype != torch.bfloat16 or biases.dtype != scales.dtype:
            raise ValueError(f"unexpected Q4 tensor types for {key}")
        q = ((codes.int().unsqueeze(-1) >> self.shifts) & 15).reshape(*scales.shape, 32)
        return (q.float() * scales.float().unsqueeze(-1) + biases.float().unsqueeze(-1)).flatten(-2).bfloat16()


class RoutedQ4(nn.Module):
    def __init__(self, previous, source, prefix):
        super().__init__()
        self.experts = previous.experts
        self.source = source
        self.prefix = prefix

    def forward(self, hidden, ids, scores):
        chosen, local = torch.unique(ids, sorted=True, return_inverse=True)
        chosen = chosen.cpu().tolist()
        if not chosen or chosen[0] < 0 or chosen[-1] >= self.experts.num_experts:
            raise ValueError("invalid selected expert IDs")
        data = {}
        for name in ("gate", "up", "down"):
            data[name] = self.source.unpack(f"{self.prefix}.{name}_proj", chosen)
        params = {"gate_up_proj": torch.cat((data.pop("gate"), data.pop("up")), dim=1),
                  "down_proj": data.pop("down")}
        for name, t in params.items():
            if t.shape[1:] != getattr(self.experts, name).shape[1:]:
                raise ValueError(f"expert orientation mismatch {name}")
        return torch.func.functional_call(self.experts, params, (hidden, local.reshape_as(ids), scores), strict=True)


class EngramQ4(nn.Module):
    def __init__(self, previous, source):
        super().__init__()
        self.source = source
        self.keys = [k.replace("model.language_model.", "language_model.model.", 1).removesuffix(".weight")
                     for k in previous.keys]
        self.ends = previous.ends
        self.register_buffer("weight", previous.weight)

    def forward(self, ids):
        unique, order = torch.unique(ids.cpu(), sorted=True, return_inverse=True)
        rows = []
        for i in unique.tolist():
            if i < 0 or i >= self.ends[-1]:
                raise ValueError("Engram row outside checkpoint")
            shard = bisect_right(self.ends, i)
            local = i - (self.ends[shard-1] if shard else 0)
            rows.append(self.source.unpack(self.keys[shard], slice(local, local+1))[0].cpu())
        return torch.stack(rows)[order.reshape(ids.shape)]


def install_dense(model, source):
    count = 0
    for name, param in model.named_parameters(remove_duplicate=False):
        if ".mlp.experts.experts." in name:
            continue
        key = "language_model." + name
        prefix = key.removesuffix(".weight")
        if prefix + ".scales" in source.weights.mapping:
            for start in range(0, len(param), 256):
                sl = slice(start, start + 256)
                converted = source.unpack(prefix, sl)
                if converted.shape != param[sl].shape:
                    raise ValueError(f"shape mismatch: {name}")
                param[sl].copy_(converted)
        else:
            tensor = source.weights.dense(key)
            if name.endswith(".conv1d.weight") and tensor.ndim == 3:
                tensor = tensor.transpose(1, 2)
            if tensor.shape != param.shape:
                raise ValueError(f"shape mismatch: {name}")
            param.copy_(tensor)
        count += 1
    for name, buf in model.named_buffers():
        key = "language_model." + name
        if key in source.weights.mapping:
            tensor = source.weights.dense(key)
            if tensor.shape != buf.shape:
                raise ValueError(f"buffer shape mismatch: {name}")
            buf.copy_(tensor)
    return count


@torch.inference_mode()
def run(a):
    model = load(a.official, "cuda")
    source = Q4Source(a.q4)
    a.dest.mkdir()
    for i, layer in enumerate(model.model.layers):
        if not isinstance(layer.mlp.experts, ExpertRows):
            raise ValueError("expected the official expert-row wrapper")
        layer.mlp.experts = RoutedQ4(layer.mlp.experts, source, f"language_model.model.layers.{i}.mlp.switch_mlp")
    for arm in ("only-routed", "all-q4"):
        if arm == "all-q4":
            eng = "model.layers.1.ple.ple_embedding.ngram_embedding"
            previous = model.get_submodule(eng)
            if not isinstance(previous, ShardedEmbedding):
                raise ValueError("unexpected Engram layout")
            model.set_submodule(eng, EngramQ4(previous, source))
            print(f"loaded {install_dense(model, source)} dense tensors", flush=True)
        out = a.dest / arm
        out.mkdir()
        evaluate(model, a.ids, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("official", "q4", "ids", "dest"):
        ap.add_argument(name, type=Path)
    run(ap.parse_args())
