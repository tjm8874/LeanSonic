import json
import struct
from pathlib import Path

root = Path("restored").resolve()
m = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
header = {}
for i in range(48):
    for proj in ("gate_proj", "up_proj", "down_proj"):
        prefix = f"language_model.model.layers.{i}.mlp.switch_mlp.{proj}."
        fields = []
        for suffix in ("scales", "biases"):
            name = prefix + suffix
            file = root / m[name]
            if file not in header:
                with file.open("rb") as f:
                    (n,) = struct.unpack("<Q", f.read(8))
                    header[file] = (8 + n, json.loads(f.read(n)))
            offset, entries = header[file]
            t = entries[name]
            start, end = t["data_offsets"]
            assert (
                t["dtype"] == "BF16"
                and t["shape"][0] == 512
                and end - start == 512 * 51200 * 2
            )
            fields.extend([file, offset + start])
        print(*fields)
