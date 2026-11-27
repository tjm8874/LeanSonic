"""Restore large absolute errors with literal BF16 values under a fixed embedding budget."""
from pathlib import Path
import argparse,hashlib,json,struct
import numpy as np
from encode import bf16
ap=argparse.ArgumentParser()
ap.add_argument('source',type=Path);ap.add_argument('centers',type=Path);ap.add_argument('output',type=Path)
args=ap.parse_args()
with args.source.open('rb') as f:
 n,=struct.unpack('<Q',f.read(8));t=json.loads(f.read(n))['model.llm.embed.weight']
assert t['dtype']=='BF16' and t['shape']==[201024,2304]
raw=np.memmap(args.source,'<u2','r',offset=n+8+t['data_offsets'][0],shape=tuple(t['shape']))
w=np.zeros(32768,np.int64)
for i in range(0,len(raw),512):
 w+=np.bincount((raw[i:i+512]&32767).ravel(),minlength=32768)
assert not w[0x7f80:].any()
w=w[:0x7f80]
values=bf16(np.arange(len(w))).astype(np.float64)
centers=np.array(json.loads(args.centers.read_text())['centers'][:127],np.uint16)
x=bf16(centers).astype(np.float64)
codes=np.searchsorted((x[:-1]+x[1:])*0.5,values).astype(np.uint8)
delta=np.abs(x[codes]-values)
budget=480*(1<<20)
literals=(budget-512-raw.size)//2
# Choose groups by absolute error, preserving identical treatment of equal errors.
errors,inv=np.unique(delta[w>0],return_inverse=True)
counts=np.bincount(inv,weights=w[w>0]).astype(np.int64)
keep=np.cumsum(counts[::-1])<=literals
cut=errors[::-1][keep][-1]
literal=delta>=cut
cnt=int(w[literal].sum())
lut=np.r_[centers,0,centers|0x8000,0].astype('<u2')
lookup=np.zeros(65536,np.uint8)
lookup[:len(w)]=np.where(literal,255,codes)
lookup[32768:32768+len(w)]=np.where(literal,255,codes|128)
with args.output.open('xb') as out:
 out.write(lut.tobytes())
 for i in range(0,len(raw),512):
  out.write(lookup[raw[i:i+512]].tobytes())
 for i in range(0,len(raw),512):
  r=raw[i:i+512]
  out.write(r[lookup[r]==255].tobytes())
delta[literal]=0
with args.output.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
assert args.output.stat().st_size==512+raw.size+2*cnt<=budget
print(json.dumps(dict(frame_bytes=args.output.stat().st_size,sha256=digest,budget_bytes=budget,literals=cnt,restored_min_abs_error=cut,exact_values=int(w[delta==0].sum()),values=int(w.sum()),relative_rmse=float(np.sqrt(np.dot(delta**2,w)/np.dot(values**2,w))),max_abs_error=float(delta[w>0].max())),indent=2))
