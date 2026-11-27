"""Compare served prompt IDs with the independently tokenized official chat template."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

driver, model, ids = map(Path, sys.argv[1:])
fixtures = json.loads(Path(__file__).with_name("fixtures.json").read_text())

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    for case in fixtures:
        folder = root / case["id"]
        folder.mkdir()
        (folder / "prompt.txt").write_text(case["prompt"])
        # The existing teacher-forcing driver also requires a nonempty answer.
        (folder / "answer.txt").write_text("X")
    subprocess.run(
        [driver.resolve(), "tokenize", model, root, *[f["id"] for f in fixtures]],
        check=True,
    )
    for case in fixtures:
        name = case["id"]
        actual = list(map(int, (root / name / "prefix.ids").read_text().split()))
        expected = list(map(int, (ids / f"{name}.ids").read_text().split()))
        # The teacher-forcing driver appends the model/text channel markers after the chat prompt.
        if actual != expected + [200001, 200004]:
            raise ValueError(f"Prompt token IDs differ for {name}")
print(f"Matched every prompt token across {len(fixtures)} cases.")
