"""Compare frozen inputs with the official tokenizer.json and chat_template.jinja."""
import json
import sys
from pathlib import Path

from jinja2 import Environment, StrictUndefined
from tokenizers import Tokenizer

model, root = map(Path, sys.argv[1:])
tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
# The generation markers delimit labels during training and generate no text.
template = (model / "chat_template.jinja").read_text().replace("{%- generation -%}", "").replace("{%- endgeneration -%}", "")
render = Environment(undefined=StrictUndefined).from_string(template).render
for case in (root / "order.txt").read_text().splitlines():
    d = root / case
    prompt, answer = [(d / (field + ".txt")).read_text() for field in ("prompt", "answer")]
    ids = [list(map(int, (d / (f + ".ids")).read_text().split())) for f in ("prefix", "answer")]
    text = render(messages=[dict(role="user", content=prompt), dict(role="assistant", content=answer, tool_calls=[])],
                  tools=[], reasoning_effort="none")
    encoded = tokenizer.encode(text, add_special_tokens=False).ids
    if encoded != ids[0] + ids[1] + [200010, 200006]:
        raise ValueError("official tokenization differs: " + case)
print(json.dumps(dict(checked=len((root / "order.txt").read_text().splitlines()),
                      tokenizer="official tokenizer.json", effort="none")))
