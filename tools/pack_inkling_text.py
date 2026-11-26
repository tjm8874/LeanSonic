#!/usr/bin/env python3
"""Copy an Inkling checkpoint shard, omitting the vision tower for text inference."""

import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile


def pack(source: Path, destination: Path, *, prepack_head: bool = False) -> tuple[int, int]:
    """Publish a complete shard atomically without replacing an existing file."""
    if destination.exists():
        raise FileExistsError(f"destination exists: {destination}")

    with source.open("rb", buffering=0) as src:
        total = os.fstat(src.fileno()).st_size
        length_bytes = src.read(8)
        if len(length_bytes) != 8:
            raise ValueError("missing safetensors header length")
        header_len = struct.unpack("<Q", length_bytes)[0]
        if not 0 < header_len <= min(total - 8, 100_000_000):
            raise ValueError("invalid safetensors header length")
        header = json.loads(src.read(header_len))
        if not isinstance(header, dict):
            raise ValueError("expected a safetensors header object")
        if "model.llm.embed.weight" not in header or "model.llm.unembed.weight" not in header:
            raise ValueError("missing Inkling language model embeddings")

        head_name = "model.llm.unembed.weight"
        packed_prefix = "leansonic.unembed_q8_64."
        vocab, dim, padded_vocab = 200_058, 2_304, 201_024
        if prepack_head and any(key.startswith(packed_prefix) for key in header):
            raise ValueError("input already contains native output-head tensors")

        output_header = {}
        metadata = header.pop("__metadata__", None)
        if metadata is not None:
            output_header["__metadata__"] = metadata

        # Preserve the input's payload order and the bytes of every retained tensor.
        spans = []
        old_end = new_end = removed = 0
        widths = {"U8": 1, "F8_E4M3": 1, "BF16": 2, "F32": 4}
        for name, info in sorted(header.items(), key=lambda item: item[1]["data_offsets"]):
            start, end = info["data_offsets"]
            if start != old_end or end < start or end > total - 8 - header_len:
                raise ValueError(f"invalid tensor extent for {name}")
            width = widths.get(info["dtype"])
            shape = info["shape"]
            if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
                raise ValueError(f"invalid shape for {name}")
            if width is None or end - start != width * math.prod(shape):
                raise ValueError(f"unexpected dtype or byte count for {name}")
            old_end = end
            if name.startswith("model.visual."):
                removed += 1
                continue
            if prepack_head and name == head_name:
                if info["dtype"] != "BF16" or shape != [padded_vocab, dim]:
                    raise ValueError("unexpected Inkling output-head dimensions or dtype")
                for suffix, dtype, dims, size in (
                    ("codes", "I8", [vocab, dim], vocab * dim),
                    ("scales", "F32", [vocab, dim // 64], vocab * (dim // 64) * 4),
                ):
                    output_header[packed_prefix + suffix] = {
                        "dtype": dtype, "shape": dims, "data_offsets": [new_end, new_end + size]
                    }
                    new_end += size
                spans.append(None)
                continue
            output_header[name] = {**info, "data_offsets": [new_end, new_end + end - start]}
            new_end += end - start
            if spans and spans[-1] is not None and spans[-1][1] == start:
                spans[-1] = (spans[-1][0], end)
            else:
                spans.append((start, end))
        if old_end != total - 8 - header_len:
            raise ValueError("unexpected trailing payload bytes")
        if not removed and not prepack_head:
            raise ValueError("expected an Inkling shard with vision tensors")
        encoded = json.dumps(output_header, separators=(",", ":"), ensure_ascii=False).encode()
        encoded += b" " * (-len(encoded) % 8)

        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".inkling-") as dst:
            dst.write(struct.pack("<Q", len(encoded)))
            dst.write(encoded)
            for span in spans:
                if span is None:
                    dst.flush()
                    offset = dst.tell()
                    src.seek(8 + header_len + header[head_name]["data_offsets"][0])
                    manifest = Path(__file__).resolve().parents[1] / "Cargo.toml"
                    subprocess.run([
                        "cargo", "run", "--quiet", "--release", "--manifest-path", str(manifest),
                        "-p", "leansonic-core", "--example", "inkling_pack_head",
                    ], check=True, stdin=src, stdout=dst)
                    if dst.tell() - offset != vocab * dim + vocab * (dim // 64) * 4:
                        raise ValueError("unexpected native output-head payload length")
                    continue
                start, end = span
                src.seek(8 + header_len + start)
                remaining = end - start
                while remaining:
                    block = src.read(min(remaining, 8 * 1024 * 1024))
                    if not block:
                        raise ValueError("truncated tensor data")
                    dst.write(block)
                    remaining -= len(block)
            dst.flush()
            os.fsync(dst.fileno())
            os.link(dst.name, destination)
    return removed, total - (8 + len(encoded) + new_end)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="official single-file Inkling model shard")
    parser.add_argument("destination", type=Path, help="new text-only safetensors shard")
    parser.add_argument("--prepack-head", action="store_true", help="store LeanSonic native output-head weights (requires Cargo)")
    args = parser.parse_args()
    try:
        removed, saved = pack(args.source, args.destination, prepack_head=args.prepack_head)
    except (OSError, ValueError, TypeError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"pack failed: {exc}\n")
    print(f"Omitted {removed} vision tensors. Saved {saved:,} bytes.")


if __name__ == "__main__":
    main()
