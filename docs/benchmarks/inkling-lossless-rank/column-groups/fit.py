"""Measure and round-trip a lossless, context-coded BF16 embedding frame."""
import hashlib
import json
import math
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

base, packer, out = map(Path, sys.argv[1:])
rows, cols = 201024, 2304
parent = base / "fit/embedding.frame"
original = np.memmap(base / "head.decoded", dtype="<u2", mode="r", shape=(rows, cols))
upper = np.memmap(base / "fit/upper", dtype="u1", mode="r", shape=(rows, cols))
with parent.open("rb") as src:
    magic, entries, bits, upper_bytes = struct.unpack("<4sIII", src.read(16))
    table_bytes = src.read(2 * entries)
assert magic == b"BRK1" and bits == 5
table = np.frombuffer(table_bytes, dtype="<u2")

# Model scales vary across vocabulary rows and hidden dimensions. Use mean log
# magnitude, ignoring sign; zero takes the finite lower code bound.
row_means = np.empty(rows)
col_totals = np.zeros(cols)
for i in range(0, rows, 512):
    mags = (original[i:i+512] & 32767).astype(np.float64)
    row_means[i:i+len(mags)] = mags.mean(axis=1)
    col_totals += mags.sum(axis=0)

def groups(means, count):
    order = np.argsort(means, kind="stable")
    labels = np.empty(len(order), dtype=np.uint8)
    labels[order] = np.arange(len(order)) * count // len(order)
    return labels

# Accumulate one set of fine histograms, then merge them for each candidate.
fine = 16
rr, cc = groups(row_means, fine), groups(col_totals, fine)
hist = np.zeros((fine, fine, 256), dtype=np.int64)
for i in range(0, rows, 256):
    hi = upper[i:i+256]
    ri = rr[i:i+len(hi)].astype(np.int32)
    indexes = (ri[:,None] * fine + cc[None,:]) * 256 + hi
    hist += np.bincount(indexes.ravel(), minlength=hist.size).reshape(hist.shape)
assert hist.sum() == rows * cols
def entropy_bytes(bins):
    totals = bins.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.where(bins>0, bins*np.log2(totals/bins),0).sum()/8)
candidates = []
for r in (1,2,4,8,16):
    for c in (1,2,4,8,16):
        bins = hist.reshape(r,fine//r,c,fine//c,256).sum(axis=(1,3))
        overhead = (rows if r>1 else 0) + (cols if c>1 else 0) + r*c*(512+8)
        candidates.append(dict(row_groups=r, column_groups=c, estimated_bytes=entropy_bytes(bins)+overhead))
best = min(candidates,key=lambda x:x['estimated_bytes'])
r,c = best['row_groups'],best['column_groups']
row_ids, col_ids = rr // (fine//r), cc // (fine//c)
row_sets = [np.flatnonzero(row_ids==g) for g in range(r)]
col_sets = [np.flatnonzero(col_ids==g) for g in range(c)]
streams = []
for i, row_set in enumerate(row_sets):
    for j, col_set in enumerate(col_sets):
        raw = out / "slice.raw"
        encoded = out / f"group-{i}-{j}.rans"
        with raw.open("wb") as f:
            for offset in range(0,len(row_set),512):
                f.write(upper[np.ix_(row_set[offset:offset+512],col_set)].tobytes())
        # This packer decodes its own output and compares every byte.
        subprocess.run([packer,"pack",raw,encoded],check=True)
        streams.append(encoded)
frame = out / "context.frame"
with frame.open("xb") as f:
    f.write(struct.pack("<4sIIIII", b"BRC1",rows,cols,entries,bits,r*c))
    f.write(bytes([r,c]))
    f.write(table_bytes)
    if r>1: f.write(row_ids.tobytes())
    if c>1: f.write(col_ids.tobytes())
    for stream in streams: f.write(struct.pack("<Q",stream.stat().st_size))
    with (base/"fit/lower").open("rb") as src: shutil.copyfileobj(src,f)
    for stream in streams:
        with stream.open("rb") as src: shutil.copyfileobj(src,f)
def sha(path):
    with path.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()
print(json.dumps(dict(shape=[rows,cols],row_groups=r,column_groups=c,estimates=candidates,
    frame_bytes=frame.stat().st_size,parent_bytes=parent.stat().st_size,frame_sha256=sha(frame),
    source_sha256=sha(base/"head.decoded"),upper_stream_bytes=sum(s.stat().st_size for s in streams)),indent=2))
