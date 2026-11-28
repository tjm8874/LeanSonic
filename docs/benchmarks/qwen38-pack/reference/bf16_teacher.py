"""Emit a text-only teacher and greedy reference from the upstream BF16 checkpoint."""

import argparse
import itertools
import json
import tempfile
from bisect import bisect_right
from contextlib import contextmanager
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM


class Weights:
    def __init__(self, directory):
        self.directory = directory
        self.mapping = json.loads(
            (directory / "model.safetensors.index.json").read_bytes()
        )["weight_map"]

    @contextmanager
    def tensor(self, key):
        with safe_open(
            self.directory / self.mapping[key], framework="pt", device="cpu"
        ) as file:
            yield file.get_slice(key)

    def dense(self, key):
        with self.tensor(key) as view:
            return view[:]


class ExpertRows(nn.Module):
    """Keep the upstream eager MoE function and transfer only the selected BF16 matrices."""

    def __init__(self, module, weights, prefix):
        super().__init__()
        self.experts = module
        self.weights = weights
        self.prefix = prefix
        for name in ("gate_up_proj", "down_proj"):
            with self.weights.tensor(prefix + "." + name) as view:
                if (
                    tuple(view.get_shape()) != tuple(getattr(module, name).shape)
                    or view.get_dtype() != "BF16"
                ):
                    raise ValueError("expert dimension or dtype differs")

    def forward(self, hidden, ids, scores):
        chosen, local = torch.unique(ids, sorted=True, return_inverse=True)
        chosen = chosen.cpu().tolist()
        if chosen[0] < 0 or chosen[-1] >= self.experts.num_experts:
            raise ValueError("out-of-range routed expert")
        params = {}
        for name in ("gate_up_proj", "down_proj"):
            with self.weights.tensor(self.prefix + "." + name) as weight:
                params[name] = torch.stack([weight[i] for i in chosen]).to(
                    hidden.device
                )
        return torch.func.functional_call(
            self.experts, params, (hidden, local.reshape_as(ids), scores), strict=True
        )


class ShardedEmbedding(nn.Module):
    """Gather rows in the upstream converter's numerical shard order without concatenating 95 GiB."""

    def __init__(self, weights, prefix, shape, parts):
        super().__init__()
        self.weights = weights
        self.keys = [f"{prefix}.shard_{i}.weight" for i in range(parts)]
        self.ends = []
        total = 0
        for key in self.keys:
            with weights.tensor(key) as view:
                n, dim = view.get_shape()
                if dim != shape[1] or view.get_dtype() != "BF16":
                    raise ValueError("incompatible Engram shard")
                total += n
                self.ends.append(total)
        if total != shape[0]:
            raise ValueError("Engram row count differs")
        # The upstream PLE module uses .weight.device to choose the gather device.
        self.register_buffer("weight", torch.empty(0, dtype=torch.bfloat16))

    def forward(self, ids):
        unique, order = torch.unique(ids.cpu(), sorted=True, return_inverse=True)
        rows = []
        for i in unique.tolist():
            if i < 0 or i >= self.ends[-1]:
                raise ValueError("Engram row out of range")
            shard = bisect_right(self.ends, i)
            with self.weights.tensor(self.keys[shard]) as view:
                rows.append(view[i - (self.ends[shard - 1] if shard else 0)])
        return torch.stack(rows)[order.reshape(ids.shape)]


def load(directory, device):
    cfg = AutoConfig.from_pretrained(directory, local_files_only=True).text_config
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    torch.set_default_dtype(torch.bfloat16)
    with init_empty_weights(include_buffers=False):
        model = Qwen4ExpForCausalLM(cfg)
    weights = Weights(directory)

    def source(name):
        return (
            name.replace("model.", "model.language_model.", 1)
            if name.startswith("model.")
            else name
        )

    eng = "model.layers.1.ple.ple_embedding.ngram_embedding"
    used = set()
    for name, p in list(model.named_parameters(remove_duplicate=False)):
        module, attr = name.rsplit(".", 1)
        parent = model.get_submodule(module)
        if module == eng:
            table = ShardedEmbedding(
                weights, source(eng), p.shape, cfg.split_ngram_parts
            )
            model.set_submodule(eng, table)
            used.update(table.keys)
        elif module.endswith(".mlp.experts"):
            if attr == "gate_up_proj":
                model.set_submodule(module, ExpertRows(parent, weights, source(module)))
            used.add(source(name))
        else:
            v = weights.dense(source(name))
            if p.shape != v.shape:
                raise ValueError(name)
            setattr(
                parent, attr, nn.Parameter(v.to(device, p.dtype), requires_grad=False)
            )
            used.add(source(name))
    for name, buf in list(model.named_buffers()):
        if name == eng + ".weight":
            continue
        key = source(name)
        v = weights.dense(key) if key in weights.mapping else buf
        if key in weights.mapping:
            used.add(key)
        if v.shape != buf.shape:
            raise ValueError(name)
        module, attr = name.rsplit(".", 1)
        setattr(model.get_submodule(module), attr, v.to(device, buf.dtype))
    expected = {
        k
        for k in weights.mapping
        if k.startswith("model.language_model.") or k == "lm_head.weight"
    }
    if expected != used:
        raise ValueError(
            f"unused/missing checkpoint keys: {sorted(expected ^ used)[:10]}"
        )
    return model.eval()


@torch.inference_mode()
def evaluate(model, ids_path, out):
    seqs = [list(map(int, s.split())) for s in ids_path.read_text().splitlines()]
    with (
        (out / "logits.f32").open("xb") as logits,
        (out / "rows.tsv").open("x") as rows,
        (out / "canaries.txt").open("x") as canaries,
    ):
        for case, seq in enumerate(seqs):
            x = torch.tensor([seq], dtype=torch.long, device="cuda")
            # Causal prefill supplies every prefix distribution for this sequence and the first greedy token.
            pred = model(input_ids=x, use_cache=True, return_dict=True)
            values = pred.logits[0].float().cpu()
            if values.shape != (len(seq), 248320) or not torch.isfinite(values).all():
                raise ValueError("invalid logits")
            values[:-1].contiguous().numpy().astype("<f4", copy=False).tofile(logits)
            for pos, (input_id, target) in enumerate(itertools.pairwise(seq)):
                print(
                    case,
                    pos,
                    input_id,
                    target,
                    values[pos].argmax().item(),
                    sep="\t",
                    file=rows,
                )
            ids = [int(values[-1].argmax())]
            for _ in range(7):
                pred = model(
                    input_ids=torch.tensor([[ids[-1]]], device="cuda"),
                    past_key_values=pred.past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                if not torch.isfinite(pred.logits).all():
                    raise ValueError("invalid greedy logits")
                ids.append(int(pred.logits[0, -1].argmax()))
            print(*ids, file=canaries)
            print("fixture", case, "positions", len(seq) - 1, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("model", "ids", "dest"):
        p.add_argument(k, type=Path)
    args = p.parse_args()
    if args.dest.exists():
        raise ValueError("destination already exists")
    with tempfile.TemporaryDirectory(dir=args.dest.parent) as tmp:
        result = Path(tmp) / "score"
        result.mkdir()
        evaluate(load(args.model, "cuda"), args.ids, result)
        result.rename(args.dest)
