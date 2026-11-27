"""Store the official DSpark drafter's load-time Q8_64 planes in mtp.safetensors."""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile

from pack_inkling_text import build_packer


def projections():
    layers = (
        ("self_attn.qkv_proj", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), (3072, 1024, 1024), 2304),
        ("self_attn.kv_ctx_proj", ("self_attn.k_proj_ctx", "self_attn.v_proj_ctx"), (1024, 1024), 2304),
        ("self_attn.o_proj", ("self_attn.o_proj",), (2304,), 3072),
        ("mlp.gate_up_proj", ("mlp.gate_proj", "mlp.up_proj"), (9216, 9216), 2304),
        ("mlp.down_proj", ("mlp.down_proj",), (2304,), 9216),
    )
    for layer in range(3):
        prefix = f"layers.{layer}."
        for target, names, rows, cols in layers:
            parts = [(prefix + name + ".weight", n) for name, n in zip(names, rows, strict=True)]
            yield prefix + target, parts, cols


def read_header(file):
    size = os.fstat(file.fileno()).st_size
    n, = struct.unpack("<Q", file.read(8))
    if not 0 < n <= min(size - 8, 100_000_000):
        raise ValueError("invalid safetensors header length")
    top = json.loads(file.read(n))
    metadata = top.pop("__metadata__", {})
    pos = 0
    for name, tensor in sorted(top.items(), key=lambda item: item[1]["data_offsets"]):
        start, end = tensor["data_offsets"]
        shape = tensor["shape"]
        if (tensor["dtype"] != "BF16" or not all(type(v) is int and v > 0 for v in shape)
                or start != pos or end - start != 2 * math.prod(shape)):
            raise ValueError(f"invalid official BF16 tensor {name}")
        pos = end
    if pos + 8 + n != size:
        raise ValueError("invalid safetensors extent")
    return metadata, top, 8 + n


def validate_retained(header):
    shapes = {
        "fuse_weight": [4, 4],
        "mask_embed": [2304],
        "norm.weight": [2304],
        "markov_head.dflare_norm.weight": [2304],
        "markov_head.rnn_proj.bias": [768],
        "markov_head.markov_w1.weight": [201024, 256],
        "markov_head.hidden_out.weight": [2304, 256],
        "markov_head.hidden_out.bias": [2304],
    }
    for layer in range(3):
        for name in ("input_layernorm", "post_attention_layernorm", "dflare_norm"):
            shapes[f"layers.{layer}.{name}.weight"] = [2304]
        for name in ("q_norm", "k_norm", "k_norm_ctx"):
            shapes[f"layers.{layer}.self_attn.{name}.weight"] = [128]
    for name, shape in shapes.items():
        if header.get(name, {}).get("shape") != shape:
            raise ValueError(f"missing or invalid retained tensor {name}")


def copy_bytes(src, out, count):
    while count:
        data = src.read(min(count, 4 << 20))
        if not data:
            raise ValueError("short tensor payload")
        out.write(data)
        count -= len(data)


def pack(source, destination, packer, compact=False, compact_markov=False):
    if destination.exists():
        raise FileExistsError(destination)
    with source.open("rb") as src, tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        work = Path(temp)
        metadata, header, base = read_header(src)
        validate_retained(header)
        segments = {}
        consumed = set()

        def take(name, rows, cols):
            tensor = header.get(name)
            if not tensor or tensor["shape"] != [rows, cols] or name in consumed:
                raise ValueError(f"missing or invalid projection {name}")
            consumed.add(name)
            src.seek(base + tensor["data_offsets"][0])

        def convert(prefix, fp, rows, cols):
            fp.seek(0)
            path = work / prefix
            total = rows * cols
            code = ("codes", "I8", [rows, cols], total)
            scale = ("scales", "F32", [rows, cols // 64], total // 16)
            params = [str(rows), str(cols)]
            if compact:
                result = subprocess.run([packer, "matrix-size", "rans", "dict", *params],
                                        stdin=fp, capture_output=True, text=True, check=True)
                code_size, scale_size = map(int, result.stdout.split())
                if min(code_size, scale_size) <= 0 or code_size % 8 or scale_size % 8:
                    raise ValueError(f"invalid compact plane sizes for {prefix}")
                if code_size < code[3]:
                    code = ("codes.rans_v1", "U8", [code_size], code_size)
                if scale_size < scale[3]:
                    scale = ("scales.dict_v1", "U8", [scale_size], scale_size)
                params = ["matrix-pack", "rans" if code[1] == "U8" else "raw",
                          "dict" if scale[1] == "U8" else "raw", *params]
                fp.seek(0)
            with path.open("xb") as out:
                subprocess.run([packer, *params], stdin=fp, stdout=out, check=True)
            if path.stat().st_size != code[3] + scale[3]:
                raise ValueError(f"unexpected packed shape for {prefix}")
            for (suffix, dtype, shape, size), offset in ((code, 0), (scale, code[3])):
                segments[f"{prefix}.q8_64.{suffix}"] = (dtype, shape, path, offset, size)

        for prefix, parts, cols in projections():
            with tempfile.TemporaryFile(dir=work) as fp:
                for name, rows in parts:
                    take(name, rows, cols)
                    copy_bytes(src, fp, rows * cols * 2)
                convert(prefix, fp, sum(rows for _, rows in parts), cols)

        name = "markov_head.rnn_proj"
        take(name + ".weight", 768, 5120)
        j01 = work / "j01"
        with tempfile.TemporaryFile(dir=work) as ctx, j01.open("xb") as left:
            for _ in range(768):
                copy_bytes(src, left, 512 * 2)
                copy_bytes(src, ctx, 4608 * 2)
            convert(name + ".jctx", ctx, 768, 4608)
        segments[name + ".j01.weight"] = ("BF16", [768, 512], j01, 0, j01.stat().st_size)
        for name, tensor in header.items():
            if name in consumed:
                continue
            if name in segments:
                raise ValueError(f"conflicting output {name}")
            lo, hi = tensor["data_offsets"]
            if compact_markov and name == "markov_head.markov_w1.weight":
                src.seek(base + lo)
                frame = work / "markov.bf16planes"
                with frame.open("xb") as out:
                    subprocess.run([packer, "bf16planes-pack", str(hi - lo)],
                                   stdin=src, stdout=out, check=True)
                size = frame.stat().st_size
                if 0 < size < hi - lo:
                    segments[name + ".bf16planes_v1"] = ("U8", [size], frame, 0, size)
                    continue
            segments[name] = (tensor["dtype"], tensor["shape"], source, base + lo, hi - lo)

        final = {"__metadata__": metadata}
        cursor = 0
        for name, (dtype, shape, _, _, count) in sorted(segments.items()):
            final[name] = dict(dtype=dtype, shape=shape, data_offsets=[cursor, cursor + count])
            cursor += count
        encoded = json.dumps(final, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        output = work / "mtp.safetensors"
        with output.open("xb") as dst:
            dst.write(struct.pack("<Q", len(encoded)))
            dst.write(encoded)
            for _, _, path, start, count in (segments[k] for k in sorted(segments)):
                with path.open("rb") as data:
                    data.seek(start)
                    copy_bytes(data, dst, count)
        os.link(output, destination)
        return destination.stat().st_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compact", action="store_true", help="losslessly entropy-code each Q8 plane when smaller")
    parser.add_argument("--compact-markov", action="store_true", help="losslessly compress the BF16 Markov head")
    args = parser.parse_args()
    print(pack(args.source, args.output, build_packer(), compact=args.compact, compact_markov=args.compact_markov))
