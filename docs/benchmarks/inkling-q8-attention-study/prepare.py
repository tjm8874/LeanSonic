"""Produce a separate native weight shard for the attention-scale ablation."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

official, native, destination = map(Path, sys.argv[1:4])
shard = "model-00001-of-00001.safetensors"


def header(path):
    with path.open("rb") as stream:
        n, = struct.unpack("<Q", stream.read(8))
        return json.loads(stream.read(n)), 8 + n


source_tensors, source_base = header(official / shard)
native_tensors, native_base = header(native / shard)
destination.mkdir()
pending = destination / "model.pending"
subprocess.run(["cp", "--reflink=auto", native / shard, pending], check=True)
assert not os.path.samefile(pending, native / shard)
results = {}
for name, codes in sorted(native_tensors.items()):
    if ".attn." not in name or not name.endswith(".q8_64.codes"):
        continue
    dense_name = name.removesuffix(".q8_64.codes")
    scales = native_tensors[dense_name + ".q8_64.scales"]
    dense = source_tensors[dense_name]
    rows, cols = dense["shape"]
    assert dense["dtype"] == "BF16" and codes["shape"] == [rows, cols]
    assert scales["shape"] == [rows, cols // 64]
    result = subprocess.check_output([
        "./q8grid", str(official / shard), str(source_base + dense["data_offsets"][0]),
        str(rows * cols // 64), str(pending), str(native_base + codes["data_offsets"][0]),
        str(native_base + scales["data_offsets"][0]),
    ], text=True).split()
    results[dense_name] = [int(result[0]), int(result[1]), float(result[2]), float(result[3])]

assert len(results) == 176
pending.rename(destination / shard)
for name in ("tiktoken", "mtp.safetensors"):
    if (native / name).exists():
        (destination / name).symlink_to((native / name).resolve())
(destination / "attention-sse.json").write_text(json.dumps(results, indent=2) + "\n")
print("Blocks, changed, original SSE, candidate SSE:", list(map(sum, zip(*results.values()))))
