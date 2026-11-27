"""Freeze the exact prompt IDs from the original model's tokenizer and chat template."""

import json
import sys
from pathlib import Path

from jinja2 import Environment, StrictUndefined
from tokenizers import Tokenizer

weights, directory = map(Path, sys.argv[1:])
directory.mkdir()
tok = Tokenizer.from_file(str(weights / "tokenizer.json"))
source = (
    (weights / "chat_template.jinja")
    .read_text()
    .replace("{%- generation -%}", "")
    .replace("{%- endgeneration -%}", "")
)
render = Environment(undefined=StrictUndefined).from_string(source).render
for r in json.loads(Path(__file__).with_name("fixtures.json").read_text()):
    text = render(
        messages=[{"role": "user", "content": r["prompt"]}],
        tools=[],
        reasoning_effort="none",
    )
    ids = tok.encode(text, add_special_tokens=False).ids
    assert ids and ids[-1] == 200010 and len(ids) + 384 <= 512
    (directory / (r["id"] + ".ids")).write_text(" ".join(map(str, ids)) + "\n")
