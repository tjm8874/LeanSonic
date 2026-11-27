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


def control_weight_shapes():
    shapes = {f"model.llm.layers.{layer}.mlp.gate.weight": [128, 2304] for layer in range(24)}
    for layer in range(6, 18):
        for name in ("attn", "k", "v", "mlp"):
            prefix = f"model.llm.layers.{layer}.controllers.{name}."
            shapes[prefix + "down_weight"] = [64, 4608]
            for suffix in ("gain_weight", "forget_weight"):
                shapes[prefix + suffix] = [2304, 64]
    return shapes


def expert_scale_tensors():
    tensors = {}
    for layer in range(24):
        group, dtype = (16, "F8_E4M3") if layer < 20 else (32, "U8")
        prefix = f"model.llm.layers.{layer}.mlp.experts."
        tensors[prefix + "w13_weight_scale"] = (dtype, [128, 2304, 2304 // group])
        tensors[prefix + "w2_weight_scale"] = (dtype, [128, 2304, 1152 // group])
    return tensors


def expert_weight_shapes(layers, codes_per_byte):
    tensors = {}
    for layer in layers:
        prefix = f"model.llm.layers.{layer}.mlp.experts."
        tensors[prefix + "w13_weight"] = [128, 2304, 2304 // codes_per_byte]
        tensors[prefix + "w2_weight"] = [128, 2304, 1152 // codes_per_byte]
    return tensors


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


EMBED_NAME = "model.llm.embed.weight"
EMBED_SHAPE = [201_024, 2304]


def q8_planes(header, prefix, rows, cols):
    planes = []
    for name, suffix, dtype, shape in (
        ("codes", ".rans_v1", "I8", [rows, cols]),
        ("scales", ".dict_v1", "F32", [rows, cols // 64]),
    ):
        key = prefix + name
        if key in header:
            plane = header[key]
            if (key + suffix in header or not isinstance(plane, dict)
                    or plane.get("dtype") != dtype or plane.get("shape") != shape):
                raise ValueError(f"invalid or conflicting native Q8 planes for {key}")
            planes.append(key)
        else:
            encoded = header.get(key + suffix)
            size = encoded.get("shape") if isinstance(encoded, dict) else None
            if (not isinstance(size, list) or len(size) != 1 or type(size[0]) is not int
                    or size[0] <= 0 or size[0] % 8 or encoded.get("dtype") != "U8"):
                raise ValueError(f"missing or invalid packed Q8 plane for {key}")
            planes.append(key + suffix)
    return planes


def pack(source: Path, destination: Path, *, prepack_head: bool = False, prepack_attention: bool = False,
         prepack_scales: bool = False, prepack_embedding: bool = False, prepack_mxfp8: bool = False,
         prepack_nvfp4: bool = False, contextual_nvfp4: bool = False, compact_controls: bool = False,
         entropy_scales: bool = False, neighbor_scales: bool = False, compact_q8_scales: bool = False, compact_q8_codes: bool = False) -> tuple[int, int]:
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
        native_suffixes = ("codes", "codes.rans_v1", "scales", "scales.dict_v1")
        has_native_head = any(head_prefix + key in header for key in native_suffixes)
        if has_native_head:
            if head_name in header:
                raise ValueError("ambiguous Inkling output head")
            q8_planes(header, head_prefix, 200_058, 2304)
        packed_embed = header.get(EMBED_NAME + ".prefix_v1")
        if packed_embed is not None:
            if EMBED_NAME in header:
                raise ValueError("ambiguous Inkling input embedding")
            dims = packed_embed.get("shape") if isinstance(packed_embed, dict) else None
            if (not isinstance(dims, list) or len(dims) != 1 or type(dims[0]) is not int
                    or dims[0] <= 0 or dims[0] % 8 or packed_embed.get("dtype") != "U8"):
                raise ValueError("invalid packed Inkling input embedding")
        if (EMBED_NAME not in header and packed_embed is None) or not (head_name in header or has_native_head):
            raise ValueError("missing Inkling language model embeddings")
        matrices = attention_matrices() if prepack_attention else {}
        if prepack_head and not has_native_head:
            matrices[head_name] = (201_024, 200_058, 2304, head_prefix)
        for name, (_, _, _, prefix) in matrices.items():
            if name not in header:
                raise ValueError(f"missing source projection {name}")
            if any(prefix + key in header for key in native_suffixes):
                raise ValueError(f"input contains conflicting native tensors for {name}")

        if sum((prepack_scales, entropy_scales, neighbor_scales)) > 1:
            raise ValueError("choose one expert scale format")
        scales = expert_scale_tensors() if prepack_scales or entropy_scales or neighbor_scales else {}
        for name in scales:
            if name not in header or any(name + suffix in header for suffix in (".range_v1", ".rans_v1", ".neighbor_v1")):
                raise ValueError(f"missing or conflicting source expert scales for {name}")
        scale_mode, scale_suffix = ("rans", ".rans_v1") if (entropy_scales or neighbor_scales) else ("scale", ".range_v1")
        lossless = {name: (dtype, shape, scale_mode, scale_suffix) for name, (dtype, shape) in scales.items()}
        if compact_controls:
            for name, shape in control_weight_shapes().items():
                if name not in header or name + ".bf16planes_v1" in header:
                    raise ValueError(f"missing or conflicting control weight {name}")
                lossless[name] = ("BF16", shape, "bf16planes", ".bf16planes_v1")
        if compact_q8_scales or compact_q8_codes:
            native = {**attention_matrices(), head_name: (201_024, 200_058, 2304, head_prefix)}
            converted = False
            for dense_name, (_, rows, cols, prefix) in native.items():
                if not any(prefix + key in header for key in native_suffixes):
                    continue
                if dense_name in header:
                    raise ValueError(f"conflicting native tensors for {dense_name}")
                codes, scales_key = q8_planes(header, prefix, rows, cols)
                if compact_q8_codes and codes == prefix + "codes":
                    lossless[codes] = ("I8", [rows, cols], "rans", ".rans_v1")
                    converted = True
                if compact_q8_scales and scales_key == prefix + "scales":
                    lossless[scales_key] = ("F32", [rows, cols // 64], "q8dict", ".dict_v1")
                    converted = True
            if not matrices and not converted:
                raise ValueError("no requested raw Q8 planes to compact; select --prepack-attention or --prepack-head for BF16 input")
        if prepack_nvfp4 and contextual_nvfp4:
            raise ValueError("choose one NVFP4 code format")
        for enabled, layers, per_byte, dtype in (
            (prepack_mxfp8, range(20, 24), 1, "F8_E4M3"),
            (prepack_nvfp4 or contextual_nvfp4, range(20), 2, "U8"),
        ):
            if not enabled:
                continue
            for name, shape in expert_weight_shapes(layers, per_byte).items():
                if name not in header or name + ".rans_v1" in header or name + ".nvctx_v1" in header:
                    raise ValueError(f"missing or conflicting expert weight {name}")
                mode, suffix = ("nvctx", ".nvctx_v1") if contextual_nvfp4 and per_byte == 2 else ("rans", ".rans_v1")
                lossless[name] = (dtype, shape, mode, suffix)
        if prepack_embedding and packed_embed is None:
            lossless[EMBED_NAME] = ("BF16", EMBED_SHAPE, "embedding", ".prefix_v1")
        packer = build_packer() if matrices or lossless else None
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
                arguments = [str(rows), str(cols)]
                code_bytes = rows * cols
                scale_bytes = rows * (cols // 64) * 4
                code_plane = ("codes", "I8", [rows, cols], code_bytes)
                scale_plane = ("scales", "F32", [rows, cols // 64], scale_bytes)
                if compact_q8_codes or compact_q8_scales:
                    codes = "rans" if compact_q8_codes else "raw"
                    scales_format = "dict" if compact_q8_scales else "raw"
                    src.seek(8 + header_len + start)
                    result = subprocess.run([packer, "matrix-size", codes, scales_format, *arguments], stdin=src,
                                            stdout=subprocess.PIPE, text=True, check=True)
                    csize, ssize = map(int, result.stdout.split())
                    if min(csize, ssize) <= 0 or csize % 8 or ssize % 8:
                        raise ValueError(f"invalid Q8 plane sizes for {name}")
                    if compact_q8_codes and csize < code_bytes:
                        code_plane = ("codes.rans_v1", "U8", [csize], csize)
                    else:
                        codes = "raw"
                    if compact_q8_scales and ssize < scale_bytes:
                        scale_plane = ("scales.dict_v1", "U8", [ssize], ssize)
                    else:
                        scales_format = "raw"
                    arguments = ["matrix-pack", codes, scales_format, *arguments]
                for suffix, dtype, dims, size in (code_plane, scale_plane):
                    output_header[prefix + suffix] = {
                        "dtype": dtype, "shape": dims, "data_offsets": [new_end, new_end + size]
                    }
                    new_end += size
                spans.append((start, end, (arguments, code_plane[3] + scale_plane[3])))
                continue
            if name in lossless:
                expected_dtype, expected_shape, mode, suffix = lossless[name]
                if info["dtype"] != expected_dtype or shape != expected_shape:
                    raise ValueError(f"unexpected dimensions or dtype for {name}")
                length = str(end - start)
                def encoded_size(codec, arguments):
                    src.seek(8 + header_len + start)
                    result = subprocess.run([packer, codec + "-size", *arguments], stdin=src,
                                            stdout=subprocess.PIPE, text=True, check=True)
                    size = int(result.stdout)
                    if size <= 0 or size % 8:
                        raise ValueError(f"invalid packed tensor length for {name}")
                    return size

                arguments = [length]
                size = encoded_size(mode, arguments)
                if neighbor_scales and name in scales:
                    neighbor_args = [length, str(expected_shape[-1])]
                    predicted = encoded_size("neighbor", neighbor_args)
                    if predicted < size:
                        size, mode, suffix, arguments = predicted, "neighbor", ".neighbor_v1", neighbor_args
                if size < end - start:
                    output_header[name + suffix] = {
                        "dtype": "U8", "shape": [size], "data_offsets": [new_end, new_end + size]
                    }
                    spans.append((start, end, ([mode + "-pack", *arguments], size)))
                    new_end += size
                    continue
            output_header[name] = {**info, "data_offsets": [new_end, new_end + end - start]}
            new_end += end - start
            if spans and spans[-1][2] is None and spans[-1][1] == start:
                spans[-1] = (spans[-1][0], end, None)
            else:
                spans.append((start, end, None))
        if old_end != total - 8 - header_len:
            raise ValueError("unexpected trailing payload bytes")
        if not removed and not matrices and not lossless:
            raise ValueError("expected an Inkling shard with vision tensors")
        encoded = json.dumps(output_header, separators=(",", ":"), ensure_ascii=False).encode()
        encoded += b" " * (-len(encoded) % 8)

        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".inkling-") as dst:
            dst.write(struct.pack("<Q", len(encoded)))
            dst.write(encoded)
            for start, end, transform in spans:
                src.seek(8 + header_len + start)
                if transform is not None:
                    arguments, expected_size = transform
                    dst.flush()
                    offset = dst.tell()
                    subprocess.run([packer, *arguments], check=True, stdin=src, stdout=dst)
                    if dst.tell() - offset != expected_size:
                        raise ValueError("unexpected native tensor payload length")
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
    parser.add_argument("--compact-controls", action="store_true", help="losslessly entropy-code router and controller BF16 planes when smaller")
    parser.add_argument("--compact-q8-scales", action="store_true", help="losslessly dictionary-code native Q8_64 scales where smaller")
    parser.add_argument("--compact-q8-codes", action="store_true", help="store Q8 code planes with lossless rANS where smaller")
    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument("--neighbor-scales", action="store_true", help="choose the smallest of raw, byte rANS, and row-neighbor rANS per expert scale tensor")
    scale_group.add_argument("--entropy-scales", action="store_true", help="store MoE expert scales with lossless byte rANS")
    scale_group.add_argument("--prepack-scales", action="store_true", help="store lossless packed MoE expert scales (requires Cargo)")
    parser.add_argument("--prepack-embedding", action="store_true", help="store the BF16 input embedding losslessly (requires Cargo)")
    parser.add_argument("--prepack-mxfp8", action="store_true", help="store MXFP8 expert weights losslessly (requires Cargo)")
    parser.add_argument("--prepack-nvfp4", action="store_true", help="losslessly rANS-code packed NVFP4 expert weights when smaller")
    parser.add_argument("--contextual-nvfp4", action="store_true", help="use block-context rANS for NVFP4 expert weights when smaller")
    args = parser.parse_args()
    try:
        removed, saved = pack(args.source, args.destination, prepack_head=args.prepack_head, prepack_attention=args.prepack_attention,
                              prepack_scales=args.prepack_scales, prepack_embedding=args.prepack_embedding, prepack_mxfp8=args.prepack_mxfp8, prepack_nvfp4=args.prepack_nvfp4, contextual_nvfp4=args.contextual_nvfp4, compact_controls=args.compact_controls,
                              entropy_scales=args.entropy_scales, neighbor_scales=args.neighbor_scales, compact_q8_scales=args.compact_q8_scales, compact_q8_codes=args.compact_q8_codes)
    except (OSError, ValueError, TypeError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"pack failed: {exc}\n")
    print(f"Omitted {removed} vision tensors. Saved {saved:,} bytes.")


if __name__ == "__main__":
    main()
