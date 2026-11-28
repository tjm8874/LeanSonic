"""Create a runnable Qwen3.8 model with directly addressable dictionary-coded Engram parameters."""

import argparse
import json
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

from transport import ENGRAM, EXPERT, exact

ROWS = 2_500_012
SUFFIX = ".q38dict1"


def copy_range(inp, out, offset, size):
    inp.seek(offset)
    while size:
        data = exact(inp, min(8 << 20, size))
        out.write(data)
        size -= len(data)


def convert(src, dest, names, codec, rows=ROWS, experts=False):
    """Append dictionary frames after the untouched tensors to preserve typed alignment."""
    with src.open("rb") as inp, tempfile.TemporaryFile(dir=dest.parent) as spool:
        (n,) = struct.unpack("<Q", exact(inp, 8))
        if n > 32 << 20:
            raise ValueError("unexpected header size")
        header = json.loads(exact(inp, n))
        meta = header.pop("__metadata__", None)
        if set(header) != names or any(k.endswith(SUFFIX) for k in names):
            raise ValueError(f"unexpected source tensors in {src.name}")
        plain = []
        encoded = []
        changed = []
        original = 0
        for key, tensor in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"]):
            start, end = tensor["data_offsets"]
            if start != original or end < start:
                raise ValueError(f"invalid offsets for {key}")
            original = end
            length = end - start
            engram = ENGRAM.fullmatch(key)
            if engram or (experts and EXPERT.fullmatch(key)):
                shape = [rows, 5] if engram else (
                    [512, 2560, 20] if ".down_proj." in key else [512, 640, 80]
                )
                words = shape[0] * shape[1] if engram else 512 * 51200
                if (
                    tensor["dtype"] != "BF16"
                    or tensor["shape"] != shape
                    or length != 2 * words
                ):
                    raise ValueError(f"invalid BF16 parameters for {key}")
                inp.seek(8 + n + start)
                data = exact(inp, length)
                frame = subprocess.run(
                    [codec, "pack"], input=data, capture_output=True, check=True
                ).stdout
                if len(frame) < length:
                    check = subprocess.run(
                        [codec, "restore", str(words)],
                        input=frame,
                        capture_output=True,
                        check=True,
                    ).stdout
                    if data != check:
                        raise ValueError(f"frame failed round trip: {key}")
                    encoded.append(
                        (
                            key + SUFFIX,
                            {"dtype": "U8", "shape": [len(frame)]},
                            len(frame),
                        )
                    )
                    spool.write(frame)
                    changed.append(key)
                    continue
            plain.append((key, tensor, length))
        if src.stat().st_size != 8 + n + original:
            raise ValueError("shard length mismatch")
        if not changed:
            shutil.copy2(src, dest)
            return [], original, original
        new_header = {} if meta is None else {"__metadata__": meta}
        cursor = 0
        for key, tensor, length in plain + encoded:
            new_header[key] = {**tensor, "data_offsets": [cursor, cursor + length]}
            cursor += length
        raw = json.dumps(new_header, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 64)
        with dest.open("xb") as out:
            out.write(struct.pack("<Q", len(raw)))
            out.write(raw)
            for _, tensor, length in plain:
                copy_range(inp, out, 8 + n + tensor["data_offsets"][0], length)
            spool.seek(0)
            shutil.copyfileobj(spool, out, 8 << 20)
        return changed, original, cursor


def pack(source, dest, codec, experts=False):
    source = source.resolve()
    dest = dest.resolve()
    if dest.exists() or dest.is_relative_to(source):
        raise ValueError("choose a new directory outside the model")
    index_name = "model.safetensors.index.json"
    index = json.loads((source / index_name).read_bytes())
    mapping = index["weight_map"]
    shards = {}
    for key, name in mapping.items():
        if key.endswith(SUFFIX) or not re.fullmatch(
            r"model-[0-9]+-of-[0-9]+\.safetensors", name
        ):
            raise ValueError("unexpected index")
        shards.setdefault(name, set()).add(key)
    if sum(bool(ENGRAM.fullmatch(k)) for k in mapping) != 256:
        raise ValueError("expected 256 original Engram scale/bias planes")
    if experts and sum(bool(EXPERT.fullmatch(k)) for k in mapping) != 288:
        raise ValueError("expected 288 original routed expert scale/bias planes")
    count = original = stored = 0
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".q38-engram-") as tmp:
        build = Path(tmp) / "model"
        build.mkdir()
        for name, keys in sorted(shards.items()):
            changed, before, after = convert(source / name, build / name, keys, codec, experts=experts)
            for key in changed:
                mapping[key + SUFFIX] = mapping.pop(key)
            original += before
            stored += after
            count += len(changed)
            print(name, len(changed), before - after, flush=True)
        for path in source.iterdir():
            if path.name in shards or path.name == index_name:
                continue
            if path.is_dir():
                shutil.copytree(path, build / path.name)
            else:
                shutil.copy2(path, build / path.name)
        index.setdefault("metadata", {})["total_size"] = stored
        (build / index_name).write_text(json.dumps(index, indent=2) + "\n")
        build.rename(dest)
    print(
        json.dumps(
            {"dictionary_planes": count, "tensor_bytes_saved": original - stored}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("original", "destination", "codec"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--experts", action="store_true", help="also encode the routed expert parameters")
    args = parser.parse_args()
    pack(args.original, args.destination, args.codec.resolve(), args.experts)
