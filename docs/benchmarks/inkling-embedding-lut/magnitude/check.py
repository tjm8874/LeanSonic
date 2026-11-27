from pathlib import Path
import json,sys,struct,hashlib
import numpy as np
src,out=map(Path,sys.argv[1:])
with src.open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];t=json.loads(f.read(n))['model.llm.embed.weight']
a=np.memmap(src,'<u2','r',offset=8+n+t['data_offsets'][0],shape=tuple(t['shape']))
b=np.memmap(out,'<u2','r',shape=a.shape)
assert out.stat().st_size==2*a.size
err=0.;power=0.;maximum=0.;exact=0
for i in range(0,len(a),256):
 x,y=a[i:i+256],b[i:i+256]
 assert np.array_equal(x>>15,y>>15)
 exact+=int((x==y).sum())
 x,y=[(w.astype(np.uint32)<<16).view(np.float32).astype(np.float64) for w in (x,y)]
 z=x-y;err+=float(np.sum(z*z));power+=float(np.sum(x*x));maximum=max(maximum,float(np.max(np.abs(z))))
with out.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(dict(values=a.size,exact=exact,relative_rmse=(err/power)**.5,max_abs=maximum,decoded_sha256=h),indent=2))
