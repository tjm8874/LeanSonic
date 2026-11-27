"""Encode every original BF16 bit pattern with a frequency-ranked, split index."""
import hashlib
import json
import math
import struct
import subprocess
import sys
from pathlib import Path
import numpy as np

shard, packer, dest = map(Path, sys.argv[1:])
dest.mkdir()
with shard.open('rb') as src:
    size, = struct.unpack('<Q', src.read(8))
    tensor = json.loads(src.read(size))['model.llm.embed.weight']
start, end = tensor['data_offsets']
assert tensor['dtype']=='BF16' and end-start==2*math.prod(tensor['shape'])
words=np.memmap(shard,dtype='<u2',mode='r',offset=8+size+start,shape=tuple(tensor['shape']))
hist=np.zeros(65536,dtype=np.int64)
for i in range(0,len(words),256):
    hist+=np.bincount(words[i:i+256].ravel(),minlength=65536)
symbols=np.flatnonzero(hist)
table=symbols[np.argsort(-hist[symbols],kind='stable')].astype('<u2')
n=len(table)
rank=np.zeros(65536,dtype=np.uint16)
rank[table]=np.arange(n,dtype=np.uint16)
candidates=[]
for k in range(1,9):
    if (n-1)>>k>=256:continue
    group=np.add.reduceat(hist[table],np.arange(0,n,1<<k))
    counts=group[group>0].astype(np.float64)
    estimate=2*n+words.size*k/8+np.sum(counts*np.log2(words.size/counts))/8
    candidates.append((float(estimate),k))
_,k=min(candidates)
assert words.shape[1]%8==0
mask=(1<<k)-1
decoded=hashlib.sha256()
with (dest/'lower').open('xb') as low_file,(dest/'upper').open('xb') as high_file:
    for i in range(0,len(words),256):
        chunk=words[i:i+256].ravel()
        ids=rank[chunk];hi=(ids>>k).astype(np.uint8)
        lo=(ids&mask).reshape(-1,8)
        packed=np.zeros(len(lo),dtype='<u8')
        for col in range(8):
            packed|=lo[:,col].astype(np.uint64)<<(k*col)
        raw=packed.view(np.uint8).reshape(-1,8)[:,:k].copy()
        low_file.write(raw);high_file.write(hi);decoded.update(chunk)
subprocess.run([str(packer),'pack',str(dest/'upper'),str(dest/'upper.rans')],check=True)
frame=dest/'embedding.frame'
with frame.open('xb') as f:
    f.write(struct.pack('<4sIII',b'BRK1',n,k,(dest/'upper.rans').stat().st_size))
    f.write(table)
    for name in ('lower','upper.rans'):
        with (dest/name).open('rb') as part:
            while buf:=part.read(4<<20):f.write(buf)
with frame.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(dict(values=int(words.size),shape=tensor['shape'],distinct=n,low_bits=k,
    candidates=candidates,frame_bytes=frame.stat().st_size,frame_sha256=digest,
    decoded_sha256=decoded.hexdigest(),baseline_prefix_bytes=618590352),indent=2))
