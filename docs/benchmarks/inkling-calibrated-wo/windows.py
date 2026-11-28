import hashlib
import itertools
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

train, tokenizer, out = map(Path, sys.argv[1:])
raw = train.read_bytes()
ids = (
    Tokenizer.from_file(str(tokenizer))
    .encode(raw.decode("utf-8"), add_special_tokens=False)
    .ids
)
assert len(ids) > 32 * 512
offsets = [i * (len(ids) - 512) // 31 for i in range(32)]
assert all(b - a >= 512 for a, b in itertools.pairwise(offsets))
out.write_text("".join(",".join(map(str, ids[i : i + 512])) + "\n" for i in offsets))


def sha(b):
    return hashlib.sha256(b).hexdigest()


print(
    json.dumps(
        {
            "dataset": "WikiText-2 raw train",
            "train_sha256": sha(raw),
            "tokenizer_sha256": sha(tokenizer.read_bytes()),
            "window_sha256": sha(out.read_bytes()),
            "training_token_count": len(ids),
            "offsets": offsets,
            "window_count": 32,
            "window_size": 512,
        }
    )
)
