"""Pack routed Q4 weights in independent checksummed Zstandard frames, one per expert."""
import argparse
import json
import re
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pack_engram import copy_range
from transport import exact

WEIGHT = re.compile(r"language_model\.model\.layers\.(\d+)\.mlp\.switch_mlp\.(gate|up|down)_proj\.weight")
SUFFIX = ".q38wz1"
COUNT = 512
RECORD_BYTES = 819200
MAGIC = b"Q38WZ001"


def write_plane(inp, spool, codec, decoder):
    start = spool.tell()
    prefix = struct.Struct("<8sII" + "I" * (COUNT + 1))
    spool.seek(start + prefix.size)
    offsets = [0]
    for _ in range(COUNT):
        raw = exact(inp, RECORD_BYTES)
        frame = codec.compress(raw)
        if len(frame) >= RECORD_BYTES:
            frame = raw
        elif decoder.decompress(frame) != raw:
            raise ValueError("expert weight frame did not round-trip")
        spool.write(frame)
        offsets.append(offsets[-1] + len(frame))
    spool.seek(start)
    spool.write(prefix.pack(MAGIC, COUNT, RECORD_BYTES, *offsets))
    length = prefix.size + offsets[-1]
    spool.seek(start + length)
    return length


def convert(src, dst, keys, codec, decoder):
    with src.open("rb") as inp, tempfile.TemporaryFile(dir=dst.parent) as spool:
        size, = struct.unpack("<Q", exact(inp, 8))
        if size > 32 << 20:
            raise ValueError("oversized shard header")
        header = json.loads(exact(inp, size))
        meta = header.pop("__metadata__", None)
        if set(header) != keys:
            raise ValueError(f"index and shard differ: {src.name}")
        plain, coded = [], []
        end = 0
        for key, tensor in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"]):
            lo, hi = tensor["data_offsets"]
            if lo != end or hi < lo:
                raise ValueError(f"invalid offsets: {key}")
            end = hi
            if WEIGHT.fullmatch(key):
                shape = [512, 2560, 80] if ".down_proj." in key else [512, 640, 320]
                if tensor["dtype"] != "U32" or tensor["shape"] != shape or hi-lo != COUNT*RECORD_BYTES:
                    raise ValueError(f"unsupported weight plane: {key}")
                inp.seek(8 + size + lo)
                length = write_plane(inp, spool, codec, decoder)
                coded.append((key + SUFFIX, {"dtype": "U8", "shape": [length]}, length))
            else:
                plain.append((key, tensor, hi-lo))
        if src.stat().st_size != 8 + size + end:
            raise ValueError(f"unexpected shard size: {src.name}")
        if not coded:
            shutil.copy2(src, dst)
            return end
        output = {} if meta is None else {"__metadata__": meta}
        cursor = 0
        for key, tensor, length in plain + coded:
            output[key] = {**tensor, "data_offsets": [cursor, cursor+length]}
            cursor += length
        encoded = json.dumps(output, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 64)
        with dst.open("xb") as out:
            out.write(struct.pack("<Q", len(encoded)) + encoded)
            for _, tensor, length in plain:
                copy_range(inp, out, 8 + size + tensor["data_offsets"][0], length)
            spool.seek(0)
            shutil.copyfileobj(spool, out, 8 << 20)
        return cursor


def pack(source, dest):
    source, dest = source.resolve(), dest.resolve()
    if dest.exists() or dest.is_relative_to(source):
        raise ValueError("choose a fresh destination outside the input model")
    index = "model.safetensors.index.json"
    manifest = json.loads((source / index).read_bytes())
    mapping = manifest["weight_map"]
    shards = {}
    for key, name in mapping.items():
        if key.endswith(SUFFIX) or not re.fullmatch(r"model-\d+-of-\d+\.safetensors", name):
            raise ValueError("unexpected source index")
        shards.setdefault(name, set()).add(key)
    expected = {(str(layer), proj) for layer in range(48) for proj in ("gate","up","down")}
    if {m.groups() for key in mapping if (m := WEIGHT.fullmatch(key))} != expected:
        raise ValueError("incomplete routed weight planes")
    codec = zstandard.ZstdCompressor(level=1, write_checksum=True)
    decoder = zstandard.ZstdDecompressor()
    stored = 0
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".q38-wz-") as tmp:
        model = Path(tmp) / "model"
        model.mkdir()
        for name, keys in sorted(shards.items()):
            stored += convert(source/name, model/name, keys, codec, decoder)
            for key in keys:
                if WEIGHT.fullmatch(key):
                    mapping[key + SUFFIX] = mapping.pop(key)
            print(name, (model/name).stat().st_size, flush=True)
        for p in source.iterdir():
            if p.name not in shards and p.name != index:
                if p.is_dir():
                    shutil.copytree(p, model/p.name)
                else:
                    shutil.copy2(p, model/p.name)
        manifest.setdefault("metadata", {})["total_size"] = stored
        (model/index).write_text(json.dumps(manifest, indent=2) + "\n")
        model.rename(dest)
    print(json.dumps({"tensor_bytes": stored, "file_bytes": sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "destination"):
        ap.add_argument(name, type=Path)
    args = ap.parse_args()
    pack(args.source, args.destination)
