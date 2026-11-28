"""Score native layers supplied with the official BF16 residual at each boundary."""

import json
from pathlib import Path

import numpy as np

reference = np.fromfile("layers-bf16.f32", dtype="<f4").reshape(9, 49, 10240)
native = np.fromfile("local-layers.f32", dtype="<f4").reshape(reference.shape)
natural = np.fromfile("layers-native.f32", dtype="<f4").reshape(reference.shape)
if not np.array_equal(native[:, 0], natural[:, 0]):
    raise ValueError("instrumentation changed embeddings")
fields = ["relative_l2", "cosine", "relative_to_bf16_block_update"]
scores = []
for pos, (r, n) in enumerate(zip(reference, native)):
    r, n = r.astype(np.float64), n.astype(np.float64)
    layer_scores = []
    for i in range(1, 49):
        error = np.linalg.norm(r[i] - n[i])
        nr, nn = np.linalg.norm(r[i]), np.linalg.norm(n[i])
        update = np.linalg.norm(r[i] - r[i - 1])
        layer_scores.append(
            [
                float(error / nr),
                float(np.dot(r[i], n[i]) / (nr * nn)),
                float(error / update),
            ]
        )
    scores.append(layer_scores)
print(
    json.dumps(
        {
            "input_ids": json.loads(Path("layer-cross.json").read_text())["ids"],
            "layers": list(range(48)),
            "fields": fields,
            "scores": scores,
        },
        indent=2,
    )
)
