"""Replace dense decoder Q4 planes with symmetric group-32 Q8 from official BF16 weights."""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile

from safetensors import safe_open
import torch

INDEX = "model.safetensors.index.json"


def projections(mapping):
    names = {name.removesuffix(".scales") for name in mapping
             if name.startswith("language_model.") and name.endswith(".scales")
             and not any(s in name for s in (".mtp.", ".switch_mlp.", ".ngram_embedding."))}
    if len(names) != 726:
        raise ValueError(f"expected 726 decoder matrices, found {len(names)}")
    return names


def source_key(prefix):
    if prefix == "language_model.lm_head":
        return "lm_head.weight"
    if not prefix.startswith("language_model.model."):
        raise ValueError(prefix)
    return prefix.replace("language_model.model.", "model.language_model.", 1) + ".weight"


def read_header(stream):
    (length,) = struct.unpack("<Q", stream.read(8))
    return json.loads(stream.read(length)), 8 + length


def encode(official, index, prefix, scratch):
    key = source_key(prefix)
    codes = scratch / (prefix + ".i8")
    scales = scratch / (prefix + ".bf16")
    with safe_open(official / index[key], framework="pt", device="cpu") as tensors:
        matrix = tensors.get_slice(key)
        rows, cols = matrix.get_shape()
        if matrix.get_dtype() != "BF16" or cols % 32:
            raise ValueError(key)
        with codes.open("xb") as c, scales.open("xb") as s:
            for first in range(0, rows, 256):
                weights = matrix[first:first + 256].float().reshape(-1, 32)
                scale = (weights.abs().amax(dim=1, keepdim=True) / 127).bfloat16()
                divisor = torch.where(scale == 0, 1.0, scale.float())
                quant = (weights / divisor).round().clamp(-127, 127).to(torch.int8)
                c.write(quant.numpy().tobytes())
                s.write(scale.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes())
    return {prefix + ".weight": ("I8", [rows, cols], codes),
            prefix + ".scales": ("BF16", [rows, cols // 32], scales)}


def copy_range(src, out, count):
    while count:
        chunk = src.read(min(count, 8 << 20))
        if not chunk:
            raise EOFError("short tensor payload")
        out.write(chunk)
        count -= len(chunk)


def rewrite(source, dest, updated, removed):
    with source.open("rb") as src:
        header, start = read_header(src)
        metadata = header.pop("__metadata__", None)
        new = {} if metadata is None else {"__metadata__": metadata}
        offset = 0
        for key, info in sorted(header.items(), key=lambda item: item[1]["data_offsets"]):
            if key in removed:
                continue
            lo, hi = info["data_offsets"]
            dtype, shape = info["dtype"], info["shape"]
            size = hi - lo
            if key in updated:
                dtype, shape, path = updated[key]
                size = path.stat().st_size
                expected = ([shape[0], shape[1] // 8], "U32") if dtype == "I8" else (shape, "BF16")
                if (info["shape"], info["dtype"]) != expected:
                    raise ValueError(f"unexpected original layout for {key}")
            new[key] = dict(dtype=dtype, shape=shape, data_offsets=[offset, offset + size])
            offset += size
        if (updated.keys() | removed) - header.keys():
            raise ValueError(f"missing tensor in {source.name}")
        data = json.dumps(new, separators=(",", ":")).encode()
        data += b" " * (-len(data) % 64)
        with dest.open("xb") as out:
            out.write(struct.pack("<Q", len(data)) + data)
            for key in new:
                if key == "__metadata__":
                    continue
                if key in updated:
                    with updated[key][2].open("rb") as plane:
                        shutil.copyfileobj(plane, out, 8 << 20)
                else:
                    lo, hi = header[key]["data_offsets"]
                    src.seek(start + lo)
                    copy_range(src, out, hi - lo)
        return offset - sum(v["data_offsets"][1] - v["data_offsets"][0] for v in header.values())


def pack(base, official, dest):
    if dest.exists() or dest.resolve().is_relative_to(base.resolve()):
        raise ValueError("choose a fresh destination alongside the source")
    manifest = json.loads((base / INDEX).read_bytes())
    mapping = manifest["weight_map"]
    upstream = json.loads((official / INDEX).read_bytes())["weight_map"]
    names = projections(mapping)
    by_shard = {}
    for name in sorted(names):
        shard = mapping[name + ".weight"]
        if any(mapping[name + p] != shard for p in (".scales", ".biases")):
            raise ValueError(f"expected three colocated planes for {name}")
        by_shard.setdefault(shard, []).append(name)
    delta = 0
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        model = Path(tmp) / "model"
        model.mkdir()
        for f in sorted(base.iterdir()):
            if f.name == INDEX:
                continue
            if f.name not in by_shard:
                os.link(f, model / f.name)
                continue
            with tempfile.TemporaryDirectory(dir=tmp) as work:
                updated = {}
                removed = set()
                for name in by_shard[f.name]:
                    updated.update(encode(official, upstream, name, Path(work)))
                    removed.add(name + ".biases")
                delta += rewrite(f, model / f.name, updated, removed)
                for key in removed:
                    del mapping[key]
            print(f"converted {f.name}", file=sys.stderr, flush=True)
        manifest["metadata"]["total_size"] += delta
        manifest["metadata"]["leansonic_dense_format"] = "symmetric-q8-group32-bf16"
        (model / INDEX).write_text(json.dumps(manifest, indent=2) + "\n")
        model.rename(dest)
    return dict(matrices=len(names), rewritten_shards=len(by_shard), tensor_payload_delta=delta,
                distribution_bytes=sum(p.stat().st_size for p in dest.iterdir()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("base", "official", "dest"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    print(json.dumps(pack(args.base, args.official, args.dest), indent=2))
