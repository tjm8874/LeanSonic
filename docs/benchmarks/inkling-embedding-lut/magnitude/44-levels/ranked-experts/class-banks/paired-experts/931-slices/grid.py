"""Price each 512-byte rANS table against the estimated symbol savings of keeping its cut."""
import hashlib,json,struct,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
parent,training,probe=map(lambda v:Path(v).resolve(),sys.argv[1:])
shard=parent/'model-00001-of-00001.safetensors'
with shard.open('rb') as f:
    n,=struct.unpack('<Q',f.read(8));header=json.loads(f.read(n));base=f.tell()
seeds=json.loads((training/'receipt.json').read_text())['planes']
def entropy(count):
    total=count.sum(axis=-1)
    return np.sum(total*np.log2(np.maximum(total,1)))-np.sum(count*np.log2(np.maximum(count,1)))
def distribution(x,y,v,left,above):
    cols=len(above)+1
    context=np.searchsorted(left,x,'right')*cols+np.searchsorted(above,y,'right')
    return np.bincount(context*64+v,minlength=(len(left)+1)*cols*64).reshape(len(left)+1,cols,64)
def cut_chain(x,y,v,initial,factor):
    left,above=[list(e) for e in initial];matrix=distribution(x,y,v,left,above)
    chain=[]
    while True:
        bits=entropy(matrix);width=len(left)+1;height=len(above)+1
        chain.append((bits*factor/8+512*width*height,[left.copy(),above.copy()]))
        options=[]
        for axis,edges in enumerate((left,above)):
            if len(edges)<=2:continue
            grid=np.swapaxes(matrix,0,axis)
            for i in range(len(edges)):
                pair=grid[i:i+2]
                delta=(entropy(pair.sum(axis=0))-entropy(pair))*factor/8-512*grid.shape[1]
                options.append((delta,axis,i))
        if not options:break
        _,axis,i=min(options);(left,above)[axis].pop(i)
        grid=np.swapaxes(matrix,0,axis).copy()
        grid[i]+=grid[i+1];matrix=np.swapaxes(np.delete(grid,i+1,axis=0),0,axis)
    return chain
def refit_axis(x,y,v,edges,other):
    groups=np.searchsorted(other,y,'right');cols=len(other)+1
    hist=np.bincount((x*cols+groups)*64+v,minlength=(int(x.max())+1)*cols*64).reshape(-1,cols,64)
    p=np.pad(hist.cumsum(axis=0),((1,0),(0,0),(0,0)))
    for i,old in enumerate(edges):
        lo=edges[i-1] if i else 0;hi=edges[i+1] if i+1<len(edges) else len(p)-1
        choices=[k for k in range(max(lo+1,old-20),min(hi,old+21))]
        choices.append(old)
        edges[i]=min(choices,key=lambda k:(entropy(p[k]-p[lo])+entropy(p[hi]-p[k]),abs(k-old),k))
def fit(key):
    directory=Path(key);directory.mkdir()
    start,end=header[key]['data_offsets'];length=end-start;width=144 if '.w13_' in key else 72
    data=np.fromfile(training/key/'features',dtype=[('x','<u2'),('y','<u2'),('v','u1')])
    x,y,v=[data[k].astype(np.int32) for k in ('x','y','v')];v-=64
    total=128*2304*width;factor=total/len(v);assert factor==32 and v.min()>=0 and v.max()<64
    with shard.open('rb') as f:
        f.seek(base+start);h=f.read(48)
    edges=[list(struct.unpack_from('<'+'H'*h[i],h,i+1)) for i in (0,24)]
    seed=seeds[key];assert edges==(seed['fitted_edges'] if seed['trial']<seed['original'] else seed['original_edges'])
    chain=cut_chain(x,y,v,edges,factor)
    best=min(chain,key=lambda pair:pair[0])
    reduced=min(chain[1:],key=lambda pair:pair[0])
    candidates=[best[1],edges]
    if best[1]==edges:candidates[0]=reduced[1]
    seen=set();records=[];chosen=length
    for i,seed_edges in enumerate(candidates):
        left,above=[list(e) for e in seed_edges]
        for _ in range(2):
            refit_axis(x,y,v,left,above);refit_axis(y,x,v,above,left)
        sig=(tuple(left),tuple(above))
        if sig in seen or [left,above]==edges:continue
        seen.add(sig);work=directory/str(i);work.mkdir()
        cmd=[str(probe),str(shard),str(base+start),str(length),str(width),'encode',','.join(map(str,left)),','.join(map(str,above))]
        old,new,codes=map(int,subprocess.check_output(cmd,cwd=work,text=True).split())
        assert old==length and codes==total
        records.append(dict(bytes=new,bins=[len(left)+1,len(above)+1],edges=[left,above]))
        if new<chosen:
            (work/'chosen.frame').replace(directory/'chosen.frame');chosen=new
    print(key,length,chosen,flush=True)
    return key,dict(original=length,trial=chosen,codes=total,trials=records)
with ThreadPoolExecutor(max_workers=2) as pool: planes=dict(pool.map(fit,sorted(seeds)))
r=dict(planes=planes,original_bytes=sum(v['original'] for v in planes.values()),chosen_bytes=sum(v['trial'] for v in planes.values()),selected=sum(v['trial']<v['original'] for v in planes.values()),checked_codes=sum(v['codes'] for v in planes.values() if v['trials']))
Path('receipt.json').write_text(json.dumps(r,indent=2)+'\n')
