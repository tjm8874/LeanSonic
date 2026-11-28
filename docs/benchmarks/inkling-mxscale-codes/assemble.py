"""Assemble the eight smaller MXFP8 frames and verify every payload on readback."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct

SHARD = "model-00001-of-00001.safetensors"


def read_header(source):
    size, = struct.unpack("<Q", source.read(8))
    return 8 + size, json.loads(source.read(size))


def transfer(source, size, output=None):
    digest = hashlib.sha256()
    while size:
        block = source.read(min(size, 4 << 20))
        if not block:
            raise ValueError("truncated tensor")
        digest.update(block)
        if output is not None:
            output.write(block)
        size -= len(block)
    return digest.hexdigest()


def manifest(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            result[str(path.relative_to(directory))] = {"bytes": path.stat().st_size, "sha256": digest}
    return result


def assemble(model, frames, dest):
    replacements = {}
    for layer in range(20, 24):
        for projection in ("w13", "w2"):
            prefix = f"model.llm.layers.{layer}.mlp.experts.{projection}_weight"
            replacements[prefix + ".rans_v1"] = (prefix + ".mxscale_codes_v1", frames / f"{layer}-{projection}.mx")

    # Put new files in a fresh directory. Preserve the original order and
    # byte alignment so existing unchanged tensors retain valid device inputs.
    with (model / SHARD).open("rb") as source:
        base, old_header = read_header(source)
        ordered = sorted(((k, v) for k, v in old_header.items() if k != "__metadata__"),
                         key=lambda item: item[1]["data_offsets"][0])
        header = {"__metadata__": old_header["__metadata__"]} if "__metadata__" in old_header else {}
        offset = 0
        savings = {}
        for name, tensor in ordered:
            start, end = tensor["data_offsets"]
            new_name, frame = replacements.get(name, (name, None))
            size = frame.stat().st_size if frame else end - start
            if new_name in header or offset % 8 != start % 8:
                raise ValueError(f"invalid tensor alignment or duplicate name: {new_name}")
            header[new_name] = dict(tensor, data_offsets=[offset, offset + size])
            if frame:
                if size >= end - start:
                    raise ValueError(f"replacement does not save bytes: {name}")
                header[new_name]["shape"] = [size]
                savings[name] = end - start - size
            offset += size
        if savings.keys() != replacements.keys():
            raise ValueError("missing one or more expected byte-rANS expert tensors")

        dest.mkdir()
        hashes = {}
        original_frames = {}
        encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
        encoded += b" " * (-len(encoded) % 8)
        with (dest / SHARD).open("xb") as output:
            output.write(struct.pack("<Q", len(encoded)))
            output.write(encoded)
            for name, tensor in ordered:
                start, end = tensor["data_offsets"]
                source.seek(base + start)
                if name in replacements:
                    new_name, frame = replacements[name]
                    original_frames[name] = transfer(source, end - start)
                    with frame.open("rb") as data:
                        hashes[new_name] = transfer(data, frame.stat().st_size, output)
                else:
                    hashes[name] = transfer(source, end - start, output)

    shutil.copytree(model / "tiktoken", dest / "tiktoken")
    with (dest / SHARD).open("rb") as written:
        base, actual = read_header(written)
        if actual != header:
            raise ValueError("written header differs from requested layout")
        for name, expected in hashes.items():
            start, end = actual[name]["data_offsets"]
            written.seek(base + start)
            if transfer(written, end - start) != expected:
                raise ValueError(f"written tensor differs: {name}")
    tokenizer = Path("tiktoken/tokenizer.model")
    if (model / tokenizer).read_bytes() != (dest / tokenizer).read_bytes():
        raise ValueError("written tokenizer differs")
    files = manifest(dest)
    return dict(parent_bytes=sum(f.stat().st_size for f in model.rglob("*") if f.is_file()),
                package_bytes=sum(f["bytes"] for f in files.values()), files=files,
                unchanged_payloads=len(ordered)-len(replacements), parent_frames=original_frames,
                reductions=savings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("parent", "frames", "new_model_directory"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    print(json.dumps(assemble(args.parent, args.frames, args.new_model_directory), indent=2))
