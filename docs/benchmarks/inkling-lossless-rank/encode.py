"""Encode original BF16 words with the smallest physically packed rank split."""
import argparse
import hashlib
import json
import math
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def encode(shard, packer, dest, low_bits=None):
    dest.mkdir()
    with shard.open('rb') as src:
        size, = struct.unpack('<Q', src.read(8))
        tensor = json.loads(src.read(size))['model.llm.embed.weight']
    start, end = tensor['data_offsets']
    assert tensor['dtype'] == 'BF16' and end-start == 2*math.prod(tensor['shape'])
    words = np.memmap(shard, dtype='<u2', mode='r', offset=8+size+start, shape=tuple(tensor['shape']))
    assert words.shape[1] % 8 == 0
    hist = np.zeros(65536, dtype=np.int64)
    for i in range(0, len(words), 256):
        hist += np.bincount(words[i:i+256].ravel(), minlength=65536)
    symbols = np.flatnonzero(hist)
    table = symbols[np.argsort(-hist[symbols], kind='stable')].astype('<u2')
    ranks = np.zeros(65536, dtype=np.uint16)
    ranks[table] = np.arange(len(table), dtype=np.uint16)
    widths = [k for k in range(1, 9) if (len(table)-1) >> k < 256]
    if low_bits is not None:
        if low_bits not in widths:
            raise ValueError('low-bit width cannot represent this rank table')
        widths = [low_bits]
    header_bytes = 16 + table.nbytes
    measured = []
    with tempfile.TemporaryDirectory(dir=dest) as workspace:
        scratch = Path(workspace)
        for k in widths:
            # Only the high-symbol entropy stream has an unknown length. The
            # low-symbol file is fixed width, so write it only for the winner.
            with (scratch/'upper').open('wb') as f:
                for i in range(0, len(words), 256):
                    f.write((ranks[words[i:i+256]] >> k).astype(np.uint8).tobytes())
            coded = scratch/f'upper-{k}.rans'
            subprocess.run([str(packer), 'pack', str(scratch/'upper'), str(coded)], check=True)
            measured.append((header_bytes + words.size*k//8 + coded.stat().st_size, k))
        size, k = min(measured)
        shutil.move(scratch/f'upper-{k}.rans', dest/'upper.rans')
    digest = hashlib.sha256()
    with (dest/'lower').open('xb') as low, (dest/'upper').open('xb') as high:
        for i in range(0, len(words), 256):
            chunk = words[i:i+256].ravel()
            ids = ranks[chunk]
            high.write((ids >> k).astype(np.uint8).tobytes())
            lo = (ids & ((1 << k)-1)).reshape(-1, 8)
            if k == 8:
                low.write(lo.astype(np.uint8).tobytes())
            else:
                packed = np.zeros(len(lo), dtype='<u8')
                for lane in range(8):
                    packed |= lo[:, lane].astype(np.uint64) << (k*lane)
                low.write(packed.view(np.uint8).reshape(-1, 8)[:, :k].tobytes())
            digest.update(chunk)
    frame = dest/'embedding.frame'
    with frame.open('xb') as f:
        f.write(struct.pack('<4sIII', b'BRK1', len(table), k, (dest/'upper.rans').stat().st_size))
        f.write(table)
        for name in ('lower', 'upper.rans'):
            with (dest/name).open('rb') as part:
                shutil.copyfileobj(part, f)
    assert frame.stat().st_size == size
    with frame.open('rb') as f:
        frame_digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(values=int(words.size), shape=tensor['shape'], distinct=len(table), low_bits=k,
                measured_frame_bytes={bits: total for total, bits in measured},
                frame_bytes=size, frame_sha256=frame_digest, decoded_sha256=digest.hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('shard', type=Path)
    parser.add_argument('packer', type=Path)
    parser.add_argument('dest', type=Path)
    parser.add_argument('--low-bits', type=int, help='reproduce a specific feasible split')
    args = parser.parse_args()
    print(json.dumps(encode(args.shard, args.packer.resolve(), args.dest, args.low_bits), indent=2))
