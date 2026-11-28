"""Re-quantize all 726 official BF16 dense matrices into group-32 signed six-bit codes."""
import argparse
import json
import os
from pathlib import Path
import tempfile

from pack import INDEX, encode, projections, rewrite


def pack_q6(base, official, dest):
    manifest = json.loads((base / INDEX).read_bytes())
    if manifest["metadata"].get("leansonic_dense_format") != "symmetric-q8-group32-bf16":
        raise ValueError("expected the complete Q8 package as layout source")
    if dest.exists() or dest.resolve().is_relative_to(base.resolve()):
        raise ValueError("choose a fresh destination on the same filesystem as Q8")
    index = manifest["weight_map"]
    official_map = json.loads((official / INDEX).read_bytes())["weight_map"]
    by_shard = {}
    for prefix in sorted(projections(index)):
        filename = index[prefix + ".weight"]
        if filename != index[prefix + ".scales"]:
            raise ValueError(f"unexpected layout {prefix}")
        by_shard.setdefault(filename, []).append(prefix)
    delta = 0
    with tempfile.TemporaryDirectory(dir=dest.parent) as td:
        tmp = Path(td)
        model = tmp / "model"
        model.mkdir()
        for source in sorted(base.iterdir()):
            if source.name == INDEX:
                continue
            if source.name not in by_shard:
                os.link(source, model / source.name)
                continue
            with tempfile.TemporaryDirectory(dir=tmp) as scratch:
                changes = {}
                for prefix in by_shard[source.name]:
                    changes.update(encode(official, official_map, prefix, Path(scratch), bits=6))
                delta += rewrite(source, model / source.name, changes, set())
        manifest["metadata"]["total_size"] += delta
        manifest["metadata"]["leansonic_dense_format"] = "symmetric-q6-group32-bf16"
        (model / INDEX).write_text(json.dumps(manifest, indent=2) + "\n")
        model.rename(dest)
    return dict(matrices=sum(map(len, by_shard.values())), changed_shards=len(by_shard),
                tensor_payload_delta=delta, distribution_bytes=sum(f.stat().st_size for f in dest.iterdir()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "official", "dest"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    print(json.dumps(pack_q6(args.base, args.official, args.dest), indent=2))
