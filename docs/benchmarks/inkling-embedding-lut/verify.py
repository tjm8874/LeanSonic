"""Read all values back from the Rust decoder and compare to the official BF16 input tensor."""
import hashlib,json,struct,sys
from pathlib import Path
import numpy as np
from encode import bf16
src,decoded,metrics=map(Path,sys.argv[1:])
with src.open('rb') as f:
 n,=struct.unpack('<Q',f.read(8));t=json.loads(f.read(n))['model.llm.embed.weight']
raw=np.memmap(src,'<u2','r',offset=8+n+t['data_offsets'][0],shape=tuple(t['shape']))
out=np.memmap(decoded,'<u2','r',shape=raw.shape)
assert decoded.stat().st_size==raw.size*2
# The 0.00836181640625 escape threshold is frozen in the encoder record.
m=json.loads(metrics.read_text());matched=0;max_error=0.;energy=0.;sse=0.
for i in range(0,len(raw),128):
 a,b=raw[i:i+128],out[i:i+128]
 assert np.array_equal(a>>15,b>>15)
 x,y=bf16(a).astype(np.float64),bf16(b).astype(np.float64)
 d=(x-y).ravel();energy+=float(np.dot(x.ravel(),x.ravel()));sse+=float(np.dot(d,d))
 matched+=int(np.count_nonzero(a==b));max_error=max(max_error,float(np.max(np.abs(d))))
rel=float(np.sqrt(sse/energy))
assert matched==m['exact_values'] and max_error==m['max_abs_error'] and abs(rel-m['relative_rmse'])<1e-12
with decoded.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(dict(compared=raw.size,exact=matched,max_abs_error=max_error,relative_rmse=rel,decoded_sha256=digest),indent=2))
