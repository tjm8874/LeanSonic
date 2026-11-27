from concurrent.futures import ThreadPoolExecutor
import json,struct,subprocess,sys
from pathlib import Path
import numpy as np
root=Path.cwd();src=Path(sys.argv[1]).resolve()/'model-00001-of-00001.safetensors'
with src.open('rb') as f:
    size,=struct.unpack('<Q',f.read(8));h=json.loads(f.read(size));base=f.tell()
keys=sorted(k for k in h if k.endswith('_scale.mean8_v1') and '.mlp.experts.' in k)
assert len(keys)==40
def optimize(x,y,v,edges,other):
    n=len(other)+1; cols=64
    groups=np.searchsorted(other,y,side='right')
    hist=np.bincount((x*n+groups)*cols+v,minlength=(int(x.max())+1)*n*cols).reshape(-1,n,cols)
    prefix=np.pad(hist.cumsum(axis=0),((1,0),(0,0),(0,0)))
    def hcost(freq):
        total=freq.sum(axis=1)
        return float(np.sum(total*np.log2(np.maximum(total,1)))-np.sum(freq*np.log2(np.maximum(freq,1))))
    for i in range(len(edges)):
        low=edges[i-1] if i else 0; high=edges[i+1] if i+1<len(edges) else len(prefix)-1
        current=edges[i]
        def score(cut):return hcost(prefix[cut]-prefix[low])+hcost(prefix[high]-prefix[cut])
        options={current+d for d in [-32,-16,-8,-4,-2,-1,0,1,2,4,8,16,32] if low<current+d<high}
        options.add(current);edges[i]=min(options,key=lambda cut:(score(cut),abs(cut-current),cut))
def encode(key):
    directory=root/key;directory.mkdir();lo,hi=h[key]['data_offsets'];width=144 if '.w13_' in key else 72
    cmd=[str(root/'probe'),str(src),str(base+lo),str(hi-lo),str(width)]
    subprocess.run(cmd+['sample'],cwd=directory,check=True)
    head=(directory/'header').read_bytes()
    original=[list(struct.unpack_from('<'+'H'*head[o],head,o+1)) for o in [0,24]]
    left,above=(a.copy() for a in original)
    samples=np.fromfile(directory/'features',dtype=[('x','<u2'),('y','<u2'),('v','u1')])
    x=samples['x'].astype(int);y=samples['y'].astype(int);v=samples['v'].astype(int)-64
    assert v.min()>=0 and v.max()<64
    for iteration in range(3):
        optimize(x,y,v,left,above);optimize(y,x,v,above,left)
    old,new,n=map(int,subprocess.check_output(cmd+['encode',','.join(map(str,left)),','.join(map(str,above))],cwd=directory,text=True).split())
    assert old==hi-lo and n==128*2304*width and (directory/'chosen.frame').exists()==(new<old)
    row=dict(original=old,trial=new,codes=n,original_edges=original,fitted_edges=[left,above],sample_count=len(v))
    (directory/'receipt.json').write_text(json.dumps(row)+'\n');print(key,old,new,flush=True);return key,row
if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:r=dict(pool.map(encode,keys))
    Path('receipt.json').write_text(json.dumps(dict(planes=r,original_bytes=sum(t['original'] for t in r.values()),chosen_bytes=sum(min(t['trial'],t['original']) for t in r.values()),selected=sum(t['trial']<t['original'] for t in r.values()),checked_codes=sum(t['codes'] for t in r.values())),indent=2)+'\n')
