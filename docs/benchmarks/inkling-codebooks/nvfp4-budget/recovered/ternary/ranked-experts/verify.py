"""Check expert membership in the shipped shards using the production decoders."""
import argparse
import json
import struct
import subprocess
from pathlib import Path

def header(path):
    with path.open("rb") as f:
        size, = struct.unpack("<Q", f.read(8))
        return 8 + size, json.loads(f.read(size))

parser = argparse.ArgumentParser(description=__doc__)
for name in ("receipt", "official_shard", "ternary_shard", "control_shard", "candidate_shard", "verifier"):
    parser.add_argument(name, type=Path)
args = parser.parse_args()
rec = json.loads(args.receipt.read_text())
key = rec["plane"]
_, official = header(args.official_shard)
width = official[key + "_scale"]["shape"][-1]
ids = rec["selected"]
assert len(ids) == len(set(ids)) and ids == rec["rank"][:len(ids)]
command = [args.verifier.resolve(), str(width), ",".join(map(str, ids))]
for path in (args.ternary_shard, args.control_shard, args.candidate_shard):
    data_start, entries = header(path)
    command.append(path)
    for name in (key + ".nvscale_direct_v1", key + "_scale.mean8_v1"):
        start, end = entries[name]["data_offsets"]
        command.extend([str(data_start + start), str(end - start)])
subprocess.run(command, check=True)
