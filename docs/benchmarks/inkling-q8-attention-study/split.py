"""Isolate Inkling attention output projections in new native-format checkpoints."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

baseline, candidate, output = (Path(arg).resolve() for arg in sys.argv[1:4])
shard = "model-00001-of-00001.safetensors"


def header(directory):
    with (directory / shard).open("rb") as stream:
        length, = struct.unpack("<Q", stream.read(8))
        return json.loads(stream.read(length)), 8 + length


stock, offset = header(baseline)
trial, other_offset = header(candidate)
assert stock == trial and offset == other_offset
for arm in ("wo_only", "except_wo"):
    dest = output / arm
    dest.mkdir()
    pending = dest / "pending"
    subprocess.run(["cp", "--reflink=auto", baseline / shard, pending], check=True)
    assert not os.path.samefile(pending, baseline / shard)
    tensors = 0
    with pending.open("r+b") as writer, (candidate / shard).open("rb") as source:
        for name, item in stock.items():
            if ".attn." not in name or not name.endswith((".q8_64.codes", ".q8_64.scales")):
                continue
            is_output = ".attn.wo_ud.weight." in name
            if is_output != (arm == "wo_only"):
                continue
            start, end = item["data_offsets"]
            source.seek(offset + start)
            writer.seek(offset + start)
            data = source.read(end - start)
            assert len(data) == end - start
            writer.write(data)
            tensors += 1
    assert tensors == (48 if arm == "wo_only" else 304)
    pending.rename(dest / shard)
    (dest / "tiktoken").symlink_to((baseline / "tiktoken").resolve())
