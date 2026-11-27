"""Repack only the NVFP4 expert planes that become smaller after exact decoding."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

SHARD = "model-00001-of-00001.safetensors"
OLD = ".nvctx_v1"
NEW = ".nvctx_trim_v1"


def header(f):
    (size,) = struct.unpack("<Q", f.read(8))
    return json.loads(f.read(size)), 8 + size


def copy_verified(source, offset, length, destination):
    source.seek(offset)
    h = hashlib.sha256()
    while length:
        part = source.read(min(8 << 20, length))
        if not part:
            raise ValueError("incomplete tensor payload")
        h.update(part)
        if destination is not None:
            destination.write(part)
        length -= len(part)
    return h.hexdigest()


def repack(src, official, encoder, dest):
    if dest.exists():
        raise FileExistsError(dest)
    with official.open("rb") as f:
        base, _ = header(f)
    with (src / SHARD).open("rb") as f:
        meta, start = header(f)
    expected = {
        f"model.llm.layers.{i}.mlp.experts.{p}_weight"
        for i in range(20)
        for p in ("w13", "w2")
    }
    old_keys = {k.removesuffix(OLD) for k in meta if k.endswith(OLD)}
    if old_keys != expected:
        raise ValueError("expected 40 original contextual NVFP4 planes")
    if {p.relative_to(src).as_posix() for p in src.rglob("*") if p.is_file()} != {
        SHARD,
        "tiktoken/tokenizer.model",
    }:
        raise ValueError("expected a single Inkling shard and its tokenizer")
    frames = {}
    rebuilt = {}
    positions = []
    comparisons = {}
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        tmp = Path(tmp)
        model = tmp / "model"
        model.mkdir()
        end = 0
        for key, item in meta.items():
            if key == "__metadata__":
                rebuilt[key] = item
                continue
            a, b = item["data_offsets"]
            new_key, size = key, b - a
            if key.endswith(OLD):
                name = key.removesuffix(OLD)
                oa, ob = base[name]["data_offsets"]
                frame = tmp / (name + ".frame")
                size2 = int(
                    subprocess.check_output(
                        [
                            encoder,
                            src / SHARD,
                            str(start + a),
                            str(size),
                            str(ob - oa),
                            frame,
                        ],
                        text=True,
                    )
                )
                comparisons[name] = dict(
                    old_bytes=size, trial_bytes=size2, decoded_bytes=ob - oa
                )
                if size2 < size:
                    if frame.stat().st_size != size2:
                        raise ValueError("incomplete encoded frame")
                    new_key, size = name + NEW, size2
                    frames[key] = frame
            rebuilt[new_key] = dict(item, data_offsets=[end, end + size])
            if new_key != key:
                rebuilt[new_key]["shape"] = [size]
            positions.append((key, new_key))
            end += size
        packed = json.dumps(rebuilt, separators=(",", ":"), ensure_ascii=False).encode()
        packed += b" " * (-len(packed) % 8)
        digests = {}
        with (src / SHARD).open("rb") as original, (model / SHARD).open("xb") as output:
            output.write(struct.pack("<Q", len(packed)))
            output.write(packed)
            for key, new_key in positions:
                a, b = meta[key]["data_offsets"]
                if key in frames:
                    frame = frames[key]
                    with frame.open("rb") as f:
                        digests[new_key] = copy_verified(
                            f, 0, frame.stat().st_size, output
                        )
                    frame.unlink()
                else:
                    digests[new_key] = copy_verified(original, start + a, b - a, output)
        with (model / SHARD).open("rb") as verify:
            actual, begin = header(verify)
            if actual != rebuilt or (model / SHARD).stat().st_size != begin + end:
                raise ValueError("inconsistent final shard header or size")
            for key, h in digests.items():
                a, b = actual[key]["data_offsets"]
                if copy_verified(verify, begin + a, b - a, None) != h:
                    raise ValueError(f"final payload differs for {key}")
        shutil.copytree(src / "tiktoken", model / "tiktoken")
        files = {}
        for p in sorted(x for x in model.rglob("*") if x.is_file()):
            with p.open("rb") as f:
                files[str(p.relative_to(model))] = dict(
                    bytes=p.stat().st_size,
                    sha256=hashlib.file_digest(f, "sha256").hexdigest(),
                )
        with (src / "tiktoken/tokenizer.model").open("rb") as f:
            if (
                hashlib.file_digest(f, "sha256").hexdigest()
                != files["tiktoken/tokenizer.model"]["sha256"]
            ):
                raise ValueError("tokenizer mismatch")
        os.rename(model, dest)
    return dict(
        files=files,
        package_bytes=sum(f["bytes"] for f in files.values()),
        planes=comparisons,
        adopted_planes=len(frames),
        verified_payloads=len(digests),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "official_shard", "encoder", "destination"):
        parser.add_argument(arg, type=Path)
    a = parser.parse_args()
    print(
        json.dumps(
            repack(a.source, a.official_shard, a.encoder.resolve(), a.destination),
            indent=2,
        )
    )
