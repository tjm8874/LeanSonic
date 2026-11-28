"""Run the upstream eager teacher with every stored parameter read from the mixed package."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np
from safetensors import safe_open
import torch
from torch import nn
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
from bf16_teacher import AutoConfig, evaluate, init_empty_weights, Qwen4ExpForCausalLM

from pack import INDEX, read_header

FINAL = "language_model.model.hyper_connection_mixer.hc_norm.weight"
FOLDED = "8224a458ee9fb93bbf6360f04c7448fb97ca235419fa4a2eae31d5894d919525"


def dictionary_values(stream, dictionary, width, first, count):
    offsets = (first + np.arange(count, dtype=np.int64)) * width
    byte = offsets >> 3
    words = stream[byte].astype(np.uint32) | (stream[byte + 1].astype(np.uint32) << 8)
    if width > 9:
        words |= stream[byte + 2].astype(np.uint32) << 16
    codes = (words >> (offsets & 7).astype(np.uint32)) & ((1 << width) - 1)
    return dictionary[codes]


class Store:
    def __init__(self, directory):
        self.directory = directory
        manifest = json.loads((directory / INDEX).read_text())
        if manifest["metadata"].get("leansonic_dense_format") != "symmetric-q8-group32-bf16":
            raise ValueError("expected the mixed Q8 package")
        self.mapping = manifest["weight_map"]
        self.used = set()
        self.decoder = zstandard.ZstdDecompressor()

    @lru_cache(maxsize=None)
    def shard(self, filename):
        path = self.directory / filename
        with path.open("rb") as f:
            h, start = read_header(f)
        return h, start, np.memmap(path, mode="r", dtype="u1")

    def payload(self, key):
        self.used.add(key)
        h, start, data = self.shard(self.mapping[key])
        spec = h[key]
        lo, hi = spec["data_offsets"]
        return data[start + lo:start + hi], spec

    def dense(self, key):
        self.used.add(key)
        with safe_open(self.directory / self.mapping[key], framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    @lru_cache(maxsize=None)
    def expert_planes(self, prefix):
        def parameters(key):
            encoded, _ = self.payload(key + ".q38dict1")
            magic, count, entries = struct.unpack_from("<8sQI", encoded)
            if magic != b"Q38DICT1" or count != 512 * 51200:
                raise ValueError(key)
            width = (entries - 1).bit_length()
            start = 20 + 2 * entries
            if not 1 <= width <= 16 or len(encoded) != start + (count * width + 7) // 8 + 2:
                raise ValueError(key)
            return encoded[start:], encoded[20:start].view("<u2"), width
        weights, _ = self.payload(prefix + ".weight.q38wz1")
        if struct.unpack_from("<8sII", weights) != (b"Q38WZ001", 512, 819200):
            raise ValueError(prefix)
        offsets = weights[16:16 + 513 * 4].view("<u4")
        frames = weights[16 + 513 * 4:]
        if offsets[0] or offsets[-1] != len(frames) or np.any(offsets[1:] <= offsets[:-1]):
            raise ValueError(prefix)
        return frames, offsets, parameters(prefix + ".scales"), parameters(prefix + ".biases")

    @lru_cache(maxsize=2048)
    def expert(self, prefix, expert):
        frames, offsets, scales, biases = self.expert_planes(prefix)
        data = frames[offsets[expert]:offsets[expert + 1]]
        if len(data) != 819200:
            data = self.decoder.decompress(data, max_output_size=819200)
        if len(data) != 819200:
            raise ValueError("invalid decoded expert size")
        codes = torch.from_numpy(np.frombuffer(data, dtype="<u4").copy())
        params = [
            torch.from_numpy(dictionary_values(*plane, expert * 51200, 51200)).view(torch.bfloat16)
            for plane in (scales, biases)
        ]
        return codes, *params


class PackedExperts(nn.Module):
    def __init__(self, module, store, prefix):
        super().__init__()
        self.module = module
        self.store = store
        self.prefix = prefix
        for projection in ("gate", "up", "down"):
            store.expert_planes(prefix + "." + projection + "_proj")

    def forward(self, hidden, ids, scores):
        chosen, local = torch.unique(ids, sorted=True, return_inverse=True)
        indices = chosen.tolist()
        if not indices or indices[0] < 0 or indices[-1] >= 512:
            raise ValueError("invalid expert indices")
        shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=hidden.device)
        matrices = {}
        for name in ("gate", "up", "down"):
            prefix = self.prefix + "." + name + "_proj"
            records = [self.store.expert(prefix, i) for i in indices]
            w, scale, bias = [torch.stack(p).to(hidden.device) for p in zip(*records)]
            n = len(indices)
            q = ((w.int().unsqueeze(-1) >> shifts) & 15).reshape(n, -1, 32).float()
            values = (q * scale.float().unsqueeze(-1) + bias.float().unsqueeze(-1)).bfloat16()
            shape = (2560, 640) if name == "down" else (640, 2560)
            matrices[name] = values.reshape(n, *shape)
        params = {"gate_up_proj": torch.cat((matrices["gate"], matrices["up"]), dim=1),
                  "down_proj": matrices["down"]}
        return torch.func.functional_call(self.module, params, (hidden, local.reshape_as(ids), scores), strict=True)


class PackedEngram(nn.Module):
    def __init__(self, store, prefix, shape):
        super().__init__()
        self.register_buffer("weight", torch.empty(0, dtype=torch.bfloat16))
        self.shards = []
        for i in range(128):
            data, _ = store.payload(f"{prefix}.shard_{i}.q38tile4")
            magic, rows, ns, nb = struct.unpack_from("<8sIHH", data)
            if magic != b"Q38TILE4" or rows != 2500012:
                raise ValueError(prefix)
            middle = 16 + 2 * ns
            start = middle + 2 * nb
            self.shards.append((data[start:].reshape(rows // 4, 370),
                                data[16:middle].view("<u2"), data[middle:start].view("<u2")))
        if tuple(shape) != (128 * rows, 160):
            raise ValueError("unexpected Engram shape")

    def forward(self, ids):
        unique, order = torch.unique(ids.cpu(), sorted=True, return_inverse=True)
        rows = []
        for index in unique.tolist():
            shard, local = divmod(index, 2500012)
            if not 0 <= shard < len(self.shards):
                raise ValueError("invalid Engram row")
            tiles, sd, bd = self.shards[shard]
            t, r = divmod(local, 4)
            tile = tiles[t]
            # Add two zero bytes for the terminal 10-bit parameter in a four-row tile.
            def params(offset, table):
                bits = np.pad(tile[offset:offset + 25], (0, 2))
                return torch.from_numpy(dictionary_values(bits, table, 10, r * 5, 5)).view(torch.bfloat16).float()
            weights = tile[80 * r:80 * (r + 1)]
            nibbles = np.stack((weights & 15, weights >> 4), axis=-1).reshape(5, 32)
            s, b = params(320, sd), params(345, bd)
            rows.append((torch.from_numpy(nibbles).float() * s[:, None] + b[:, None]).bfloat16().flatten())
        return torch.stack(rows)[order.reshape(ids.shape)]


def load(directory):
    store = Store(directory)
    cfg = AutoConfig.from_pretrained(directory, local_files_only=True).text_config
    cfg._attn_implementation = cfg._experts_implementation = "eager"
    torch.set_default_dtype(torch.bfloat16)
    with init_empty_weights(include_buffers=False):
        model = Qwen4ExpForCausalLM(cfg)
    eng = "model.layers.1.ple.ple_embedding.ngram_embedding"
    for name, param in list(model.named_parameters(remove_duplicate=False)):
        module, attr = name.rsplit(".", 1)
        key = "language_model." + name
        if module == eng:
            model.set_submodule(eng, PackedEngram(store, "language_model." + eng, param.shape))
            continue
        if module.endswith(".mlp.experts"):
            if attr == "gate_up_proj":
                prefix = "language_model." + module.removesuffix(".experts") + ".switch_mlp"
                model.set_submodule(module, PackedExperts(model.get_submodule(module), store, prefix))
            continue
        src = store.dense(key)
        if src.dtype == torch.int8:
            scales = store.dense(key.removesuffix(".weight") + ".scales")
            rows, cols = param.shape
            if tuple(src.shape) != (rows, cols) or scales.shape != (rows, cols // 32) or scales.dtype != torch.bfloat16:
                raise ValueError(key)
            value = torch.empty_like(param, device="cuda")
            for first in range(0, rows, 256):
                sl = slice(first, first + 256)
                w = src[sl].cuda().float().reshape(-1, 32)
                value[sl] = (w * scales[sl].cuda().float().reshape(-1, 1)).reshape_as(value[sl])
        else:
            if name.endswith(".conv1d.weight"):
                src = src.transpose(1, 2)
            if key == FINAL and hashlib.sha256(src.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest() == FOLDED:
                src = (src.float() - 1).bfloat16()
            if src.shape != param.shape:
                raise ValueError(key)
            value = src.to("cuda", param.dtype)
        setattr(model.get_submodule(module), attr, nn.Parameter(value, requires_grad=False))
    for name, buf in list(model.named_buffers()):
        if name == eng + ".weight":
            continue
        key = "language_model." + name
        value = store.dense(key) if key in store.mapping else buf
        module, attr = name.rsplit(".", 1)
        setattr(model.get_submodule(module), attr, value.to("cuda", buf.dtype))
    expected = {k for k in store.mapping if k.startswith("language_model.") and not k.startswith("language_model.mtp.")}
    if expected != store.used:
        raise ValueError(f"unaccounted decoder tensors: {sorted(expected ^ store.used)[:12]}")
    return model.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("ids", type=Path)
    parser.add_argument("dest", type=Path)
    args = parser.parse_args()
    with torch.inference_mode():
        target = load(args.model)
        args.dest.mkdir()
        evaluate(target, args.ids, args.dest)
