"""Fit deterministic row subsets of the tied embedding errors using actual coded byte counts."""
import hashlib,json,struct,subprocess,sys
from pathlib import Path
import numpy as np
from encode import bf16
official,centersfile,previous,converter,out=map(Path,sys.argv[1:])
fit=json.loads(previous.read_text())
budget=fit['frame_budget_bytes']
cut=max(t['threshold'] for t in fit['trials'] if t['bytes']>budget)
assert cut<fit['threshold']
with official.open('rb') as f:
 n,=struct.unpack('<Q',f.read(8));t=json.loads(f.read(n))['model.llm.embed.weight']
raw=np.memmap(official,'<u2','r',offset=n+8+t['data_offsets'][0],shape=tuple(t['shape']))
assert t['dtype']=='BF16' and raw.shape==(201024,2304)
centers=np.array(json.loads(centersfile.read_text())['centers'][:127],dtype=np.uint16)
table=np.r_[centers,0,centers|32768,0].astype('<u2')
x=bf16(centers).astype(np.float64);bits=np.arange(0x7f80)
v=bf16(bits).astype(np.float64)
code=np.searchsorted((x[1:]+x[:-1])*.5,v,side='left').astype(np.uint8)
delta=np.abs(x[code]-v)
large=delta>cut;tie=delta==cut
maps=[]
for condition in [large,large|tie]:
 lut=np.zeros(65536,dtype=np.uint8);lut[bits]=np.where(condition,255,code);lut[bits+32768]=np.where(condition,255,code|128);maps.append(lut)
selectors=(np.arange(len(raw))*73+19)%128
tied=np.zeros(128,dtype=np.int64);mag_hist=np.zeros(32768,np.int64)
for i in range(0,len(raw),256):
 block=raw[i:i+256]&32767
 assert block.max()<0x7f80
 mag_hist+=np.bincount(block.ravel(),minlength=32768)
 tied+=np.bincount(selectors[i:i+len(block)],weights=tie[block].sum(axis=1),minlength=128).astype(np.int64)
w=mag_hist[:len(v)]
energy=float(np.dot(v*v,w));rest=delta.copy();rest[large]=0
sse=float(np.dot(rest*rest,w));base_exact=int(w[(delta==0)|large].sum());base_literals=int(w[large].sum())
scratch=out/'staging';scratch.mkdir()
trials={}
best=-1
def measure(fraction):
 global best
 if fraction in trials:return trials[fraction]['bytes']
 plain=scratch/'trial.raw';coded=scratch/'trial.coded'
 def walk(f, literals):
  for i in range(0,len(raw),512):
   b=raw[i:i+512];c=maps[0][b];chosen=selectors[i:i+len(b)]<fraction
   c[chosen]=maps[1][b[chosen]]
   f.write(b[c==255].tobytes() if literals else c.tobytes())
 with plain.open('wb') as f:
  f.write(table.tobytes());walk(f,False);walk(f,True)
 subprocess.run([converter,'encode',plain,coded],check=True)
 sz=coded.stat().st_size
 kept=int(tied[:fraction].sum());literals=base_literals+kept
 assert plain.stat().st_size==512+raw.size+2*literals
 z=dict(row_classes=fraction,bytes=sz,literals=literals,exact_values=base_exact+kept,raw_bytes=plain.stat().st_size,max_abs_error=float(rest[~tie].max() if fraction==128 else cut),relative_rmse=float(np.sqrt((sse-kept*cut**2)/energy)))
 trials[fraction]=z
 print(fraction,sz,z['relative_rmse'],file=sys.stderr,flush=True)
 if sz<=budget and fraction>best:
  best=fraction;coded.replace(out/'selected.lut')
 return sz
assert measure(0)==fit['bytes']
lo,hi=0,128
while hi-lo>1:
 mid=(hi+lo)//2
 if measure(mid)<=budget:lo=mid
 else:hi=mid
assert 0<lo<128 and best==lo
result=dict(trials=list(trials.values()),values=raw.size,frame_budget_bytes=budget,row_permutation='(73*row+19)%128',tied_error=cut,**trials[lo])
with (out/'selected.lut').open('rb') as f:result['sha256']=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(result,indent=2))
