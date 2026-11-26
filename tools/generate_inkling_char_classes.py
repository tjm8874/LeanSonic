"""Regenerate the Rust Inkling pre-tokenizer's Unicode table with tokenizers==0.23.2."""
from pathlib import Path

from tokenizers import Regex, __version__
from tokenizers.pre_tokenizers import Split

PATTERNS = [r"[\p{Lu}\p{Lt}]", r"\p{Ll}", r"[\p{Lo}\p{Lm}]", r"\p{M}", r"\p{N}", r"\s"]
OUTPUT = Path(__file__).resolve().parents[1] / "crates/leansonic-core/src/tiktoken/classes.bin"


def main():
    if __version__ != "0.23.2":
        raise RuntimeError("regenerate with tokenizers==0.23.2")
    classes = bytearray(0x110000)
    for kind, pattern in enumerate(PATTERNS, start=1):
        keep_matches = Split(Regex(pattern + "+"), behavior="removed", invert=True)
        for start in range(0, len(classes), 0x800):
            if 0xD800 <= start < 0xE000:
                continue
            text = "".join(map(chr, range(start, start + 0x800)))
            for _, (left, right) in keep_matches.pre_tokenize_str(text):
                classes[start + left:start + right] = bytes([kind]) * (right - left)
    # 256 code points per page, two 4-bit classes per byte. Share identical pages.
    pages = {}
    indices = bytearray()
    for start in range(0, len(classes), 256):
        page = bytes(classes[i] | classes[i + 1] << 4 for i in range(start, start + 256, 2))
        index = pages.setdefault(page, len(pages))
        indices.append(index)
    OUTPUT.write_bytes(indices + b"".join(pages))


if __name__ == "__main__":
    main()
