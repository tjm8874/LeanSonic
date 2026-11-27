"""Merge adjacent BF16 magnitudes by empirical distortion per bit saved."""
import heapq,json,math,struct,sys
from pathlib import Path
import numpy as np
src=Path(sys.argv[1])
with src.open('rb') as f:
    n=struct.unpack('<Q',f.read(8))[0]
    tensor=json.loads(f.read(n))['model.llm.embed.weight']
raw=np.memmap(src,'<u2','r',offset=8+n+tensor['data_offsets'][0],shape=tuple(tensor['shape']))
hist=np.zeros(32768,np.int64)
for i in range(0,len(raw),1024):hist+=np.bincount((raw[i:i+1024]&32767).ravel(),minlength=32768)
assert hist[0x7f80:].sum()==0
bf=(np.arange(0x7f80,dtype=np.uint32)<<16).view(np.float32).astype(np.float64)
present=np.flatnonzero(hist)
v=bf[present];cnt=hist[present].copy()
sums=cnt*v
energy=float(np.dot(cnt,v*v));total=int(cnt.sum());active=len(v)
# Ordered bins always round the merged centroid to an actual BF16 before measuring error.
lo=present.copy();hi=present.copy();centroid=v.copy()
left=np.arange(-1,active-1);right=np.arange(1,active+1);right[-1]=-1
sse=np.zeros(active);generation=np.zeros(active,int);heap=[]
def mass(x):return x*math.log2(x) if x else 0.
def cost(i,j):
    c=int(cnt[i]+cnt[j]);mean=(sums[i]+sums[j])/c
    mid=np.float32(mean).view(np.uint32).item()
    value=bf[(mid+32767+((mid>>16)&1))>>16]
    def extra(k):return (value-centroid[k])*(cnt[k]*(value+centroid[k])-2*sums[k])
    ds=extra(i)+extra(j)
    db=mass(c)-mass(int(cnt[i]))-mass(int(cnt[j]))
    return max(0.,ds),db,value
def offer(i):
    if i<0 or right[i]<0 or lo[i]==0:return
    j=right[i];ds,db,_=cost(i,j);heapq.heappush(heap,(ds/db,i,j,generation[i],generation[j]))
for i in range(active):offer(i)
bits=mass(total)-sum(mass(int(c)) for c in cnt)+total
err=0.;receipts=[]
for target in (502000000,498000000,490000000,465000000,450000000,400000000,350000000,260000000):
    while bits>target*8:
        _,i,j,vi,vj=heapq.heappop(heap)
        if generation[i]!=vi or generation[j]!=vj or right[i]!=j:continue
        ds,db,value=cost(i,j)
        cnt[i]+=cnt[j];sums[i]+=sums[j];hi[i]=hi[j];centroid[i]=value
        right[i]=right[j]
        if right[i]>=0:left[right[i]]=i
        generation[i]+=1;generation[j]+=1
        bits-=db;err+=ds;active-=1
        offer(left[i]);offer(i)
    mapping=np.arange(32768,dtype='<u2')
    i=0
    while i>=0:
        code=np.float32(centroid[i]).view(np.uint32).item()>>16
        mapping[lo[i]:hi[i]+1]=code;i=right[i]
    physical_err=np.sum(hist[present]*(v-bf[mapping[present]])**2)
    assert abs(physical_err-err)<1e-8*max(physical_err,1.)
    mapping.tofile(f'map-{target}.u16')
    receipts.append(dict(budget=target,entropy_bytes=bits/8,representatives=active,exact_magnitudes=int(hist[present[mapping[present]==present]].sum()),relative_rmse=math.sqrt(physical_err/energy),max_abs=float(np.max(abs(v-bf[mapping[present]])))))
print(json.dumps(dict(values=total,sign_bits=total,estimates=receipts),indent=2))
