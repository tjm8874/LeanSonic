"""Safetensors file transfer and physical distribution checksums."""

import hashlib
import json
import struct

SHARD = "model-00001-of-00001.safetensors"


def read_header(source):
    (size,) = struct.unpack("<Q", source.read(8))
    return 8 + size, json.loads(source.read(size))


def transfer(source, size, output=None):
    digest = hashlib.sha256()
    while size:
        block = source.read(min(size, 4 << 20))
        if not block:
            raise ValueError("truncated tensor")
        digest.update(block)
        if output is not None:
            output.write(block)
        size -= len(block)
    return digest.hexdigest()


def manifest(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            result[str(path.relative_to(directory))] = {
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
    return result
