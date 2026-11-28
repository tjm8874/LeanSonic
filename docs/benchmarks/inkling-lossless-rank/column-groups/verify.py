"""Decode just the emitted frame, scatter upper symbols and check all BF16 words."""
import hashlib
import json
import struct
import sys
import subprocess
from pathlib import Path
import numpy as np

frame, decoder, official = map(Path,sys.argv[1:])
root=frame.parent
with frame.open("rb") as f:
    magic,rows,cols,entries,bits,k = struct.unpack("<4sIIIII",f.read(24))
    r,c=f.read(2)
    assert magic==b"BRC1" and r*c==k and cols%8==0
    table=np.frombuffer(f.read(2*entries),dtype="<u2")
    def labels(count,groups):
        values=np.frombuffer(f.read(count),dtype="u1") if groups>1 else np.zeros(count,dtype="u1")
        assert values.max()<groups
        return [np.flatnonzero(values==i) for i in range(groups)]
    rs,cs=labels(rows,r),labels(cols,c)
    lengths=struct.unpack(f"<{k}Q",f.read(8*k))
    lower=np.memmap(frame,dtype="u1",offset=f.tell(),mode="r",shape=(rows,cols//8,bits))
    f.seek(lower.size,1)
    upper=np.memmap(root/"verify.upper",dtype="u1",mode="w+",shape=(rows,cols))
    for a,ri in enumerate(rs):
        for b,ci in enumerate(cs):
            coded=f.read(lengths[a*c+b]); assert len(coded)==lengths[a*c+b]
            (root/"verify.rans").write_bytes(coded)
            subprocess.run([decoder,root/"verify.rans",str(len(ri)*len(ci)),root/"verify.raw"],check=True)
            data=np.memmap(root/"verify.raw",dtype="u1",mode="r",shape=(len(ri),len(ci)))
            for i in range(0,len(ri),512): upper[np.ix_(ri[i:i+512],ci)]=data[i:i+512]
            del data
    assert f.read(1)==b""
original=np.memmap(official,dtype="<u2",mode="r",shape=(rows,cols))
digest=hashlib.sha256()
for i in range(0,rows,256):
    lo=lower[i:i+256]
    lanes=np.zeros(lo.shape[:2],dtype="<u8")
    for byte in range(bits): lanes |= lo[:,:,byte].astype("<u8") << (8*byte)
    indexes=upper[i:i+len(lo)].astype("<u2")<<bits
    for lane in range(8): indexes[:,lane::8] |= ((lanes>>(bits*lane))&((1<<bits)-1)).astype("<u2")
    words=table[indexes]
    assert np.array_equal(words,original[i:i+len(lo)])
    digest.update(words.tobytes())
assert digest.hexdigest()=="86afe04b91893883cf823392f1c954bbf0881988e0b6c5f34ed7b2f938fe0a30"
print(json.dumps(dict(verified_words=rows*cols,decoded_sha256=digest.hexdigest())))
