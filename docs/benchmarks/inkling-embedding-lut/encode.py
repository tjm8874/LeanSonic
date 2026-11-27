"""Fit an empirical scalar codebook to the absolute BF16 weights, then encode both signs."""
from pathlib import Path
import argparse, hashlib, json, struct
import numpy as np

def bf16(bits):
    return (np.asarray(bits,dtype=np.uint32)<<16).view(np.float32)
def round_bf16(values):
    b=np.asarray(values,dtype=np.float32).view(np.uint32)
    return ((b+0x7fff+((b>>16)&1))>>16).astype(np.uint16)
def fit(counts,levels):
    bits=np.flatnonzero(counts)
    x=bf16(bits).astype(np.float64)
    w=counts[bits]
    def prefix(v):return np.r_[0.,np.cumsum(v,dtype=np.float64)]
    sw,sx,ss=prefix(w),prefix(w*x),prefix(w*x*x)
    n=len(x)
    assert levels<=n<=8192
    costs=np.full((n,n),np.inf)
    for end in range(n):
        total=sw[end+1]-sw[:end+1];signal=sx[end+1]-sx[:end+1]
        quant=bf16(round_bf16(signal/total)).astype(np.float64)
        costs[end,:end+1]=np.maximum(0,ss[end+1]-ss[:end+1]-2*signal*quant+total*quant*quant)
    best=np.r_[0.,np.full(n,np.inf)]
    paths=np.zeros((levels,n),np.int32)
    for k in range(levels):
        combined=costs+best[:-1]
        paths[k]=np.argmin(combined,axis=1)
        best=np.r_[np.inf,combined[np.arange(n),paths[k]]]
    centers=[]
    end=n
    for k in reversed(range(levels)):
        start=paths[k,end-1]
        centers.append(round_bf16((sx[end]-sx[start])/(sw[end]-sw[start])).item())
        end=start
    assert end==0
    centers=np.array(centers[::-1],dtype='<u2')
    assert np.all(np.diff(centers.astype(int))>0)
    return centers
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('shard',type=Path);ap.add_argument('frame',type=Path)
    a=ap.parse_args()
    with a.shard.open('rb') as f:
        length,=struct.unpack('<Q',f.read(8));meta=json.loads(f.read(length))
    t=meta['model.llm.embed.weight'];off,limit=t['data_offsets']
    assert t['dtype']=='BF16' and t['shape']==[201024,2304] and limit-off==2*np.prod(t['shape'])
    raw=np.memmap(a.shard,'<u2','r',offset=8+length+off,shape=tuple(t['shape']))
    counts=np.zeros(32768,dtype=np.int64)
    for start in range(0,len(raw),512):
        counts+=np.bincount((raw[start:start+512]&0x7fff).ravel(),minlength=32768)
    assert not counts[0x7f80:].any()
    centers=fit(counts,128)
    # Preserve signed zero exactly. Remaining centers use BF16-representable weighted means.
    centers[0]=0
    table=np.r_[centers,centers|0x8000].astype('<u2')
    vals=bf16(np.arange(0x7f80))
    cv=bf16(centers).astype(np.float64)
    cuts=(cv[:-1]+cv[1:])*0.5
    codes=np.searchsorted(cuts,vals,side='left').astype(np.uint8)
    lookup=np.zeros(65536,np.uint8)
    lookup[:len(codes)]=codes;lookup[32768:32768+len(codes)]=codes|128
    error=cv[codes]-vals
    w=counts[:len(vals)]
    with a.frame.open('xb') as f:
        f.write(table.tobytes())
        for start in range(0,len(raw),512):
            f.write(lookup[raw[start:start+512]].tobytes())
    with a.frame.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    print(json.dumps(dict(source_shape=t['shape'],observed_magnitudes=int(np.count_nonzero(w)),centers=table.tolist(),frame_bytes=a.frame.stat().st_size,frame_sha256=digest,exact_values=int(w[error==0].sum()),values=int(w.sum()),relative_rmse=float(np.sqrt(np.dot(error**2,w)/np.dot(vals.astype(np.float64)**2,w))),max_abs_error=float(np.max(np.abs(error[w>0])))),indent=2))
if __name__=='__main__':main()
