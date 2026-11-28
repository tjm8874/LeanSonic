from pathlib import Path
import hashlib
import sys
import numpy as np

path = Path(sys.argv[1])
raw = np.memmap(path, mode='r', dtype='u1')
freq = np.fromfile(path.with_suffix('.rans'), dtype='<u2', count=256)
symbols = np.flatnonzero(freq).astype('u1')
assert 0 < len(symbols) <= 16
index = np.full(256, 255, dtype='u1')
index[symbols] = np.arange(len(symbols),dtype='u1')
digest = hashlib.sha256(symbols.tobytes() + bytes(16-len(symbols)))
for start in range(0, len(raw), 1 << 20):
    ids = index[raw[start:start + (1 << 20)]]
    assert np.all(ids <= 15) and len(ids) % 2 == 0
    digest.update((ids[::2] | (ids[1::2] << 4)).tobytes())
print(digest.hexdigest())
