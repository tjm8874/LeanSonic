"""Measure actual archive lengths at increasing fidelity with the 217-expert package fixed."""
from pathlib import Path
import hashlib,json,struct,subprocess,sys
import numpy as np
from encode import bf16
official,centers_file,original_lut,prior_receipt,converter,targetdir=map(Path,sys.argv[1:])
budget=11663309504-json.loads(prior_receipt.read_text())['package_bytes']+original_lut.stat().st_size-4096
with official.open('rb') as f:
 n,=struct.unpack('<Q',f.read(8));t=json.loads(f.read(n))['model.llm.embed.weight']
assert t['dtype']=='BF16' and t['shape']==[201024,2304]
raw=np.memmap(official,'<u2','r',offset=n+8+t['data_offsets'][0],shape=tuple(t['shape']))
counts=np.zeros(65536,np.int64)
for i in range(0,len(raw),512):
 counts+=np.bincount(raw[i:i+512].ravel(),minlength=65536)
bits=np.flatnonzero(counts);weights=counts[bits];vals=bf16(bits).astype(np.float64)
assert np.isfinite(vals).all()
centers=np.array(json.loads(centers_file.read_text())['centers'][:127],np.uint16)
cv=bf16(centers).astype(np.float64)
magcodes=np.searchsorted((cv[:-1]+cv[1:])*0.5,np.abs(vals),side='left')
delta=np.abs(cv[magcodes]-np.abs(vals))
thresholds=np.unique(delta[delta>0])
table=np.r_[centers,0,centers|0x8000,0].astype('<u2')
signed=(magcodes|((bits>>15)<<7)).astype(np.uint8)
def entropy(h):
 h=h[h>0]
 return -float(np.dot(h,np.log2(h/h.sum())))/8 if len(h) else 0.
def estimate(cut):
 lit=delta>=cut
 return sum(entropy(np.bincount(v,weights=w,minlength=256)) for v,w in ((np.where(lit,255,signed),weights),(bits[lit]&255,weights[lit]),(bits[lit]>>8,weights[lit])))
est=[estimate(c) for c in thresholds]
guess=next((i for i,c in enumerate(est) if c<budget),len(est)-1)
scratch=targetdir/'staging';scratch.mkdir()
trials={}
def measure(i):
 if i in trials:return trials[i]['bytes']
 cut=float(thresholds[i]);lit=delta>=cut
 lookup=np.zeros(65536,np.uint8);lookup[bits]=np.where(lit,255,signed)
 plain=scratch/'trial.raw';coded=scratch/'trial.coded'
 with plain.open('wb') as f:
  f.write(table.tobytes())
  for start in range(0,len(raw),512):f.write(lookup[raw[start:start+512]].tobytes())
  for start in range(0,len(raw),512):
   block=raw[start:start+512];f.write(block[lookup[block]==255].tobytes())
 subprocess.run([converter,'encode',plain,coded],check=True)
 size=coded.stat().st_size;relative=float(np.sqrt(np.dot(delta[~lit]**2,weights[~lit])/np.dot(vals**2,weights)))
 trials[i]=dict(threshold=cut,bytes=size,raw_bytes=plain.stat().st_size,literals=int(weights[lit].sum()),exact_values=int(weights[lit|(delta==0)].sum()),max_abs_error=float(delta[~lit].max()),relative_rmse=relative,entropy_estimate=est[i])
 print(i,cut,size,relative,file=sys.stderr,flush=True)
 if size<=budget and (not (targetdir/'selected.json').exists() or i<json.loads((targetdir/'selected.json').read_text())['index']):
  coded.replace(targetdir/'selected.lut')
  (targetdir/'selected.json').write_text(json.dumps(dict(index=i,**trials[i])))
 return size
# Search adjacent discrete cutoffs. Only a physically measured file can qualify.
i=guess
if measure(i)>budget:
 lo=i;step=1
 while True:
  hi=min(len(thresholds)-1,lo+step)
  if measure(hi)<=budget:break
  assert hi>lo;lo=hi;step*=2
else:
 hi=i;step=1
 while hi>0:
  lo=max(0,hi-step)
  if measure(lo)>budget:break
  hi=lo;step*=2
 else:lo=-1
while hi-lo>1:
 mid=(lo+hi)//2
 if measure(mid)<=budget:hi=mid
 else:lo=mid
result=json.loads((targetdir/'selected.json').read_text())
assert result['index']==hi and result['bytes']<=budget
with (targetdir/'selected.lut').open('rb') as f:result['sha256']=hashlib.file_digest(f,'sha256').hexdigest()
result.update(values=raw.size,frame_budget_bytes=budget,trials=list(trials.values()))
print(json.dumps(result,indent=2))
