"""Recode legal pairs for one PTX PRMT. Verify each byte against original E2M1 weights."""
import hashlib
import itertools
import json
import sys
from pathlib import Path

codes = {0: 0, 1: 7, 3: 15}
table = (0x00, 0x07, 0x0f, 0x70, 0x77, 0x7f, 0xf0, 0xf7)
selector = {byte: i for i, byte in enumerate(table)}
selector[0xff] = 15

def perm(n):
    byte = table[n & 7]
    return (255 if byte & 128 else 0) if n & 8 else byte

def pair(n):
    return codes[n & 3] | codes[n >> 2] << 4

for ns in itertools.product(selector, repeat=4):
    control = sum(selector[b] << (4*i) for i, b in enumerate(ns))
    assert tuple(perm((control >> (4*i)) & 15) for i in range(4)) == ns

valid = [i for i in range(256) if all(((i >> shift) & 3) in codes for shift in (0,2,4,6))]
mapping = bytes(selector[pair(i & 15)] | selector[pair(i >> 4)] << 4 if i in valid else 0 for i in range(256))
low = bytes(perm(i & 15) for i in range(256))
high = bytes(perm(i >> 4) for i in range(256))
src, raw, dst = map(Path, sys.argv[1:])
count = 0
digest = hashlib.sha256()
with src.open('rb') as old, raw.open('rb') as original, dst.open('xb') as out:
    while block := old.read(4 << 20):
        assert not block.translate(None, bytes(valid)), 'unsupported weight code'
        expected = original.read(2 * len(block))
        encoded = block.translate(mapping)
        assert encoded.translate(low) == expected[::2] and encoded.translate(high) == expected[1::2]
        out.write(encoded)
        digest.update(encoded)
        count += len(encoded)
    assert not original.read(1)
print(json.dumps(dict(selector_bytes=count,raw_bytes=2*count,selector_sha256=digest.hexdigest(),exhaustive_words=9**4)))
