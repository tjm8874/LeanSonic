"""Compare reversible Q4 block transforms on stratified samples of all 144 routed planes."""
import argparse
import hashlib
import json
import re
import struct
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import zstandard as zstd

EXPERT = re.compile(r"language_model\.model\.layers\.(\d+)\.mlp\.switch_mlp\.(gate|up|down)_proj\.weight")
BLOCK = 256 * 1024
METHODS = ("raw", "center", "center_unpacked", "gray", "gray_planes", "center_zigzag_planes", "center_zigzag_pairs")


class Shards:
    def __init__(self, directory):
        mapping = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.tensors = {}
        for name in set(mapping.values()):
            path = directory / name
            with path.open("rb") as f:
                size, = struct.unpack("<Q", f.read(8))
                if size > 32 << 20:
                    raise ValueError("invalid shard header")
                header = json.loads(f.read(size))
            for key, tensor in header.items():
                if key == "__metadata__" or mapping.get(key) != name:
                    continue
                lo, hi = tensor["data_offsets"]
                self.tensors[key] = path, 8 + size + lo, hi-lo, tensor["dtype"]

    def read(self, key, offset, length, dtype):
        path, start, size, actual = self.tensors[key]
        if dtype != actual or not 0 <= offset <= size-length:
            raise ValueError(f"invalid range for {key}")
        with path.open("rb") as f:
            f.seek(start + offset)
            data = f.read(length)
        if len(data) != length:
            raise EOFError(key)
        return data


def unpack(data):
    b = np.frombuffer(data, dtype=np.uint8)
    out = np.empty(2 * len(b), dtype=np.uint8)
    out[::2] = b & 15
    out[1::2] = b >> 4
    return out


def pack(nibbles):
    return (nibbles[::2] | (nibbles[1::2] << 4)).tobytes()


def bf16(data):
    return (np.frombuffer(data, dtype="<u2").astype("<u4") << 16).view("<f4")


def encode(raw, center, method):
    if method == "raw":
        return raw
    x = unpack(raw)
    if method.startswith("center"):
        x = ((x.reshape(-1,32) - center[:,None]) & 15).ravel()
    if "zigzag" in method:
        x = ((x << 1) ^ (0 - (x >> 3))) & 15
    elif method.startswith("gray"):
        x ^= x >> 1
    if method.endswith("unpacked"):
        return x.tobytes()
    if method.endswith("planes"):
        return b"".join(np.packbits((x >> bit) & 1, bitorder="little").tobytes() for bit in range(4))
    if method.endswith("pairs"):
        return b"".join(
            np.bitwise_or.reduce(((x >> shift) & 3).reshape(-1,4) << np.array([0,2,4,6],dtype=np.uint8),axis=1).tobytes()
            for shift in (0,2)
        )
    return pack(x)


def decode(data, center, method):
    if method == "raw":
        return data
    b = np.frombuffer(data, dtype=np.uint8)
    if method.endswith("planes"):
        x = np.zeros(len(b)*2,dtype=np.uint8)
        for bit, plane in enumerate(np.split(b,4)):
            x |= np.unpackbits(plane,bitorder="little") << bit
    elif method.endswith("pairs"):
        x = np.zeros(len(b)*2,dtype=np.uint8)
        for shift, plane in zip((0,2),np.split(b,2),strict=True):
            for i in range(4):
                x[i::4] |= ((plane >> (i*2)) & 3) << shift
    elif method.endswith("unpacked"):
        x = b
    else:
        x = unpack(data)
    if "zigzag" in method:
        x = ((x >> 1) ^ (0 - (x & 1))) & 15
    elif method.startswith("gray"):
        x ^= x >> 1
        x ^= x >> 2
    if method.startswith("center"):
        x = ((x.reshape(-1,32) + center[:,None]) & 15).ravel()
    return pack(x)


def screen(directory):
    shards = Shards(directory)
    keys = sorted(k for k in shards.tensors if EXPERT.fullmatch(k))
    if len(keys)!=144:
        raise ValueError("expected 144 routed expert Q4 weight planes")
    compressor, decoder = zstd.ZstdCompressor(level=1), zstd.ZstdDecompressor()
    result = defaultdict(lambda: defaultdict(int))
    digest = hashlib.sha256()
    for key in keys:
        prefix = key.removesuffix("weight")
        blocks = shards.tensors[key][2] // BLOCK
        for sample, idx in enumerate([blocks // 7, blocks // 2, 5 * blocks // 6]):
            offset = idx * BLOCK
            raw = shards.read(key, offset, BLOCK, "U32")
            digest.update(raw)
            sc = bf16(shards.read(prefix+"scales",offset//8,BLOCK//8,"BF16"))
            bi = bf16(shards.read(prefix+"biases",offset//8,BLOCK//8,"BF16"))
            center = np.rint(np.divide(-bi, sc, out=np.zeros_like(sc), where=sc!=0)).clip(0,15).astype(np.uint8)
            for method in METHODS:
                t = time.perf_counter_ns()
                coded = compressor.compress(encode(raw,center,method))
                enc_ns = time.perf_counter_ns()-t
                t = time.perf_counter_ns()
                restored = decode(decoder.decompress(coded),center,method)
                dec_ns = time.perf_counter_ns()-t
                if restored!=raw:
                    raise ValueError((key,idx,method))
                r=result[method]
                r["source_bytes"] += BLOCK
                r["stored_bytes"] += 8+min(BLOCK,len(coded))
                r["block_count"] += 1
                r["encode_ns"] += enc_ns
                r["decode_ns"] += dec_ns
                r["offset_class_"+str(sample)] += 8+min(BLOCK,len(coded))
    return dict(block=BLOCK,zstd=zstd.__version__,sha256=digest.hexdigest(),methods=result)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original",type=Path)
    print(json.dumps(screen(parser.parse_args().original),indent=2))
