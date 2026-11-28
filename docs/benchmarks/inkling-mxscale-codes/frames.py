"""Convert Inkling-2-Mini's eight MXFP8 expert planes using the production codec."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import tempfile


def extract(shard, base, tensor, destination):
    start, end = tensor["data_offsets"]
    shard.seek(base + start)
    with destination.open("wb") as output:
        remaining = end - start
        while remaining:
            data = shard.read(min(remaining, 4 << 20))
            if not data:
                raise ValueError("truncated tensor payload")
            output.write(data)
            remaining -= len(data)


def convert(model, packer, output):
    output.mkdir()
    with (model / "model-00001-of-00001.safetensors").open("rb") as shard:
        size, = struct.unpack("<Q", shard.read(8))
        tensors = json.loads(shard.read(size))
        with tempfile.TemporaryDirectory(dir=output) as scratch:
            code_frame = Path(scratch) / "codes.rans"
            scale_frame = Path(scratch) / "scales.rans"
            for layer in range(20, 24):
                for projection, width in (("w13", 2304), ("w2", 1152)):
                    prefix = f"model.llm.layers.{layer}.mlp.experts.{projection}_weight"
                    extract(shard, 8 + size, tensors[prefix + ".rans_v1"], code_frame)
                    extract(shard, 8 + size, tensors[prefix + "_scale.rans_v1"], scale_frame)
                    count = 128 * 2304 * width
                    destination = output / f"{layer}-{projection}.mx"
                    subprocess.run([packer, "--from-rans", str(count), code_frame,
                                    scale_frame, destination], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("packer", type=Path)
    parser.add_argument("new_frame_directory", type=Path)
    args = parser.parse_args()
    convert(args.model, args.packer.resolve(), args.new_frame_directory)
