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


def attention_matrices():
    dims = {
        "wq_du.weight": (3072, 2304), "wr_du.weight": (384, 2304),
        "gate_attn.weight": (3072, 2304), "attn_sink_proj.weight": (24, 2304),
        "wk_dv.weight": (1024, 2304), "wv_dv.weight": (1024, 2304),
        "wo_ud.weight": (2304, 3072),
    }
    matrices = {}
    for layer in range(24):
        prefix = f"model.llm.layers.{layer}.attn."
        parts = dict(dims)
        if layer in (5, 11, 17, 23):
            parts.update({"k_sconv.weight_proj": (32, 2304), "v_sconv.weight_proj": (32, 2304)})
        for suffix, (rows, cols) in parts.items():
            name = prefix + suffix
            matrices[name] = (rows, rows, cols, name + ".q8_64.")
    return matrices


def build_packer():
    manifest = Path(__file__).resolve().parents[1] / "Cargo.toml"
    build = subprocess.run([
        "cargo", "build", "--release", "--manifest-path", str(manifest), "--message-format=json",
        "-p", "leansonic-core", "--example", "inkling_pack_q8",
    ], check=True, stdout=subprocess.PIPE, text=True)
    for line in build.stdout.splitlines():
        record = json.loads(line)
        if record.get("target", {}).get("name") == "inkling_pack_q8" and record.get("executable"):
            return record["executable"]
    raise ValueError("Cargo did not report the Inkling packer executable")


def pack(source: Path, destination: Path, *, prepack_head: bool = False, prepack_attention: bool = False) -> tuple[int, int]:
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
        head_name = "model.llm.unembed.weight"
        head_prefix = "leansonic.unembed_q8_64."
        native_head = (header.get(head_prefix + "codes"), header.get(head_prefix + "scales"))
        has_native_head = any(plane is not None for plane in native_head)
        if has_native_head:
            if head_name in header:
                raise ValueError("ambiguous Inkling output head")
            for plane, dtype, dims in zip(native_head, ("I8", "F32"), ([200_058, 2304], [200_058, 36])):
                if not isinstance(plane, dict) or plane.get("dtype") != dtype or plane.get("shape") != dims:
                    raise ValueError("invalid native Inkling output head")
        if "model.llm.embed.weight" not in header or not (head_name in header or has_native_head):
            raise ValueError("missing Inkling language model embeddings")
        matrices = attention_matrices() if prepack_attention else {}
        if prepack_head and not has_native_head:
            matrices[head_name] = (201_024, 200_058, 2304, head_prefix)
        for name, (_, _, _, prefix) in matrices.items():
            if name not in header:
                raise ValueError(f"missing source projection {name}")
            if prefix + "codes" in header or prefix + "scales" in header:
                raise ValueError(f"input contains conflicting native tensors for {name}")

        output_header = {}
        metadata = header.pop("__metadata__", None)
        if metadata is not None:
            output_header["__metadata__"] = metadata

        # Preserve the input's payload order and the bytes of every retained tensor.
        spans = []
        old_end = new_end = removed = 0
        widths = {"I8": 1, "U8": 1, "F8_E4M3": 1, "BF16": 2, "F32": 4}
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
            if name in matrices:
                input_rows, rows, cols, prefix = matrices[name]
                if info["dtype"] != "BF16" or shape != [input_rows, cols]:
                    raise ValueError(f"unexpected dimensions or dtype for {name}")
                for suffix, dtype, dims, size in (
                    ("codes", "I8", [rows, cols], rows * cols),
                    ("scales", "F32", [rows, cols // 64], rows * (cols // 64) * 4),
                ):
                    output_header[prefix + suffix] = {
                        "dtype": dtype, "shape": dims, "data_offsets": [new_end, new_end + size]
                    }
                    new_end += size
                spans.append((start, end, (rows, cols)))
                continue
            output_header[name] = {**info, "data_offsets": [new_end, new_end + end - start]}
            new_end += end - start
            if spans and spans[-1][2] is None and spans[-1][1] == start:
                spans[-1] = (spans[-1][0], end, None)
            else:
                spans.append((start, end, None))
        if old_end != total - 8 - header_len:
            raise ValueError("unexpected trailing payload bytes")
        if not removed and not matrices:
            raise ValueError("expected an Inkling shard with vision tensors")
        encoded = json.dumps(output_header, separators=(",", ":"), ensure_ascii=False).encode()
        encoded += b" " * (-len(encoded) % 8)

        packer = build_packer() if matrices else None
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".inkling-") as dst:
            dst.write(struct.pack("<Q", len(encoded)))
            dst.write(encoded)
            for start, end, dimensions in spans:
                src.seek(8 + header_len + start)
                if dimensions is not None:
                    rows, cols = dimensions
                    dst.flush()
                    offset = dst.tell()
                    subprocess.run([packer, str(rows), str(cols)], check=True, stdin=src, stdout=dst)
                    if dst.tell() - offset != rows * cols + rows * (cols // 64) * 4:
                        raise ValueError("unexpected native projection payload length")
                    continue
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
    parser.add_argument("--prepack-attention", action="store_true", help="store native Q8_64 attention projections (requires Cargo)")
    args = parser.parse_args()
    try:
        removed, saved = pack(args.source, args.destination, prepack_head=args.prepack_head, prepack_attention=args.prepack_attention)
    except (OSError, ValueError, TypeError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"pack failed: {exc}\n")
    print(f"Omitted {removed} vision tensors. Saved {saved:,} bytes.")


if __name__ == "__main__":
    main()
