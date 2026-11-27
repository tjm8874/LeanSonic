"""Measure full-frame costs for prefixes of the frozen global expert ranking."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path
import shutil

SHARD='model-00001-of-00001.safetensors'
def header(path):
    with path.open('rb') as f:
        n,=struct.unpack('<Q',f.read(8))
        return 8+n,json.loads(f.read(n))

def transfer(src,dst,n):
    h=hashlib.sha256()
    while n:
        b=src.read(min(n,4<<20));assert b
        n-=len(b);h.update(b)
        if dst is not None:dst.write(b)
    return h.hexdigest()

def pair(k):return k+'.nvscale_four_v1',k+'_scale.mean8_v1'
def refs(folder,base,meta,k):
    output=[folder/SHARD]
    for name in pair(k):
        a,b=meta[name]['data_offsets']
        output.extend((base+a,b-a))
    return output

rankfile,low,high,encoder,output=map(Path,sys.argv[1:])
r=json.loads(rankfile.read_text())
rank=r['rank']; cap=r['cap'];total=r['parent_bytes']
assert sum(p.stat().st_size for p in low.rglob('*') if p.is_file())==total
lo,lm=header(low/SHARD);ho,hm=header(high/SHARD)
assert set(lm)==set(hm)
prefixes={}
for x in rank:prefixes.setdefault(x['plane'],[]).append(x['expert'])
assert all(len(v)==len(set(v)) for v in prefixes.values())
lengths={};trials=[]
def schedule(n):
    counts={}
    for row in rank[:n]:counts[row['plane']]=counts.get(row['plane'],0)+1
    return counts
def length(task):
    (k,count),dest=task
    p=refs(low,lo,lm,k)
    q=refs(high,ho,hm,k)
    command=[encoder,144 if k.endswith('.w13_weight') else 72,
             ','.join(map(str,prefixes[k][:count])),dest,*p,*q]
    new=tuple(map(int,subprocess.check_output(list(map(str,command)),text=True).split()))
    return (k,count),(new,sum(new)-p[2]-p[4])
def size_for(n,pool):
    groups=schedule(n)
    missing=[((k,c),'-') for k,c in groups.items() if (k,c) not in lengths]
    lengths.update(pool.map(length,missing))
    cost=total+sum(lengths[k,c][1] for k,c in groups.items())
    trials.append(dict(prefix=n,estimated_archive_bytes_before_header=cost))
    print(f"trial {n} {len(groups)} {cost}",file=sys.stderr,flush=True)
    return cost

projected=0;guess=0
for x in rank:
    projected+=x['estimated_bytes']
    if projected>cap-total-4096:break
    guess+=1
assert guess>0
with ThreadPoolExecutor(max_workers=2) as pool:
    lower=0;upper=None;n=guess
    while upper is None or upper-lower>1:
        cost=size_for(n,pool)
        if cost<=cap-4096:lower=n
        else:upper=n
        if upper is None:
            if n==len(rank):break
            n=min(len(rank), max(n+1,(n*5)//4))
        else:n=(lower+upper)//2
    assert lower>0
    groups=schedule(lower)
    gain=sum(x['gain'] for x in rank[:lower])
    if gain<=r['whole_plane_gain']:raise SystemExit('measured prefix insufficient to improve the whole-plane training objective')
    Path('frames').mkdir()
    swaps={}
    jobs=[]
    for k,c in sorted(groups.items()):
        prefix=Path('frames')/k.replace('.','_')
        jobs.append(((k,c),str(prefix)))
        for z,ext in zip(pair(k),('weights','scales')):swaps[z]=prefix.with_suffix('.'+ext)
    for key,value in pool.map(length,jobs):assert lengths[key]==value

end=0;new={}
for key,entry in lm.items():
    if key=='__metadata__':new[key]=entry;continue
    a,b=entry['data_offsets']
    size=swaps[key].stat().st_size if key in swaps else b-a
    new[key]=dict(entry,data_offsets=[end,end+size])
    if key in swaps:new[key]['shape']=[size]
    end+=size
blob=json.dumps(new,ensure_ascii=False,separators=(',',':')).encode()
blob+=b' '*(-len(blob)%8)
output.mkdir()
hashes={}
with (output/SHARD).open('xb') as dst,(low/SHARD).open('rb') as src:
    dst.write(struct.pack('<Q',len(blob)));dst.write(blob)
    for k,v in lm.items():
        if k=='__metadata__':continue
        if k in swaps:
            with swaps[k].open('rb') as f:hashes[k]=transfer(f,dst,swaps[k].stat().st_size)
        else:
            a,b=v['data_offsets'];src.seek(lo+a);hashes[k]=transfer(src,dst,b-a)
shutil.copytree(low/'tiktoken',output/'tiktoken')
bo,bm=header(output/SHARD)
assert bm==new and len(hashes)==999
with (output/SHARD).open('rb') as f:
    for k,h in hashes.items():
        a,b=new[k]['data_offsets'];f.seek(bo+a);assert transfer(f,None,b-a)==h
files={}
for path in sorted(output.rglob('*')):
    if path.is_file():
        with path.open('rb') as f:
            files[str(path.relative_to(output))]=dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
size=sum(v['bytes'] for v in files.values())
assert total<size<=cap
assert (low/'tiktoken/tokenizer.model').read_bytes()==(output/'tiktoken/tokenizer.model').read_bytes()
print(json.dumps(dict(prefix=lower,cap=cap,parent_bytes=total,package_bytes=size,files=files,rank_sha256=hashlib.sha256(rankfile.read_bytes()).hexdigest(),
                      training_gain=gain,training_objective=r['parent_objective']-gain,whole_plane_gain=r['whole_plane_gain'],
                      selected={k:prefixes[k][:c] for k,c in groups.items()},trials=trials,changed_tensors=len(swaps),verified_tensors=len(hashes)),indent=2))
