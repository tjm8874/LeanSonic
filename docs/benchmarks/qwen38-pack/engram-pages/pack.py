"""Rearrange directly loadable dictionary Engram rows into four-row tiles, losslessly."""

import argparse
import json
import os
from pathlib import Path
import struct
import tempfile

from build import indices, tensor, write_blocks


def copy_bytes(src, dst, length):
    while length:
        chunk = src.read(min(length, 8 << 20))
        if not chunk:
            raise EOFError("short shard")
        dst.write(chunk)
        length -= len(chunk)


def pack(base, dest):
    base, dest = base.resolve(), dest.resolve()
    if dest.exists() or dest.is_relative_to(base):
        raise ValueError("choose a fresh directory outside the source")
    index = "model.safetensors.index.json"
    manifest = json.loads((base / index).read_bytes())
    mapping = manifest["weight_map"]
    replacements, removed, by_file = {}, set(), {}
    rows = 2_500_012
    for i in range(128):
        prefix = f"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_{i}."
        wkey = prefix + "weight"
        skey = prefix + "scales.q38dict1"
        bkey = prefix + "biases.q38dict1"
        key = prefix + "q38tile4"
        if key in mapping or not all(k in mapping for k in (wkey, skey, bkey)):
            raise ValueError("requires the direct dictionary Engram checkpoint")
        if prefix + "scales" in mapping or prefix + "biases" in mapping:
            raise ValueError("ambiguous raw and dictionary parameters")
        w, spec = tensor(base, mapping, wkey)
        if spec["dtype"] != "U32" or spec["shape"] != [rows, 20] or len(w) != rows * 80:
            raise ValueError("invalid weight plane")
        s, sd = indices(base, mapping, prefix + "scales", rows)
        b, bd = indices(base, mapping, prefix + "biases", rows)
        length = 16 + len(sd) + len(bd) + (rows // 4) * 370
        replacements[wkey] = (key, length)
        removed.update((skey, bkey))
        for k in (wkey, skey, bkey):
            by_file.setdefault(mapping[k], set()).add(k)
    updated = dict(mapping)
    total_delta = 0
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        out = Path(tmp) / "model"
        out.mkdir()
        for path in base.iterdir():
            if path.name == index:
                continue
            if path.name not in by_file:
                os.link(path, out / path.name)
                continue
            with path.open("rb") as src:
                (size,) = struct.unpack("<Q", src.read(8))
                old = json.loads(src.read(size))
                meta = old.pop("__metadata__", None)
                if set(old) != {k for k, v in mapping.items() if v == path.name}:
                    raise ValueError("manifest and source header differ")
                entries = sorted(old.items(), key=lambda item: item[1]["data_offsets"])
                new = {} if meta is None else {"__metadata__": meta}
                cursor = 0
                for k, spec in entries:
                    if k in removed:
                        del updated[k]
                        continue
                    lo, hi = spec["data_offsets"]
                    if k in replacements:
                        newkey, length = replacements[k]
                        updated[newkey] = updated.pop(k)
                        new[newkey] = {
                            "dtype": "U8",
                            "shape": [length],
                            "data_offsets": [cursor, cursor + length],
                        }
                    else:
                        length = hi - lo
                        new[k] = {**spec, "data_offsets": [cursor, cursor + length]}
                    cursor += length
                header = json.dumps(new, separators=(",", ":")).encode()
                header += b" " * (-len(header) % 64)
                with (out / path.name).open("xb") as dst:
                    dst.write(struct.pack("<Q", len(header)))
                    dst.write(header)
                    for k, spec in entries:
                        if k in removed:
                            continue
                        if k not in replacements:
                            lo, hi = spec["data_offsets"]
                            src.seek(8 + size + lo)
                            copy_bytes(src, dst, hi - lo)
                            continue
                        prefix = k.removesuffix("weight")
                        w, _ = tensor(base, mapping, k)
                        s, sd = indices(base, mapping, prefix + "scales", rows)
                        b, bd = indices(base, mapping, prefix + "biases", rows)
                        dst.write(
                            struct.pack(
                                "<8sIHH", b"Q38TILE4", rows, len(sd) // 2, len(bd) // 2
                            )
                        )
                        dst.write(sd)
                        dst.write(bd)
                        write_blocks(dst, w, s, b, rows, 4, 370)
                    if dst.tell() != 8 + len(header) + cursor:
                        raise ValueError("incorrect output length")
                total_delta += cursor - (path.stat().st_size - 8 - size)
        manifest["weight_map"] = updated
        manifest["metadata"]["total_size"] += total_delta
        (out / index).write_text(json.dumps(manifest, indent=2) + "\n")
        out.rename(dest)
    return {
        "file_bytes": sum(p.stat().st_size for p in dest.iterdir()),
        "tensor_delta": total_delta,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    print(json.dumps(pack(a.base, a.dest)))
