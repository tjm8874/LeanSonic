"""Fit one expert-plane restoration using training-only statistics and measured archive bytes."""
import hashlib,json,os,shutil,struct,subprocess,sys
from pathlib import Path

def head(path):
    with path.open('rb') as f:
        n,=struct.unpack('<Q',f.read(8));return 8+n,json.loads(f.read(n))

def extent(meta,key):
    a,b=meta[key]['data_offsets'];return b-a

def transfer(source,dest,size):
    h=hashlib.sha256()
    while size:
        block=source.read(min(size,8<<20));assert block
        size-=len(block);h.update(block)
        if dest:dest.write(block)
    return h.hexdigest()

def run(ternary,control,official,sampling,selection,stats,routing,fitter,output):
    shard='model-00001-of-00001.safetensors'; base,hm=head(ternary/shard); cb,hc=head(control/shard); ob,orig=head(official)
    assert hm.keys()==hc.keys()
    measured= sum(p.stat().st_size for p in ternary.rglob('*') if p.is_file()); assert measured==7718950382
    cap=7764291776
    def names(k):return k+'.nvscale_direct_v1',k+'_scale.mean8_v1'
    def pairbytes(meta,k):return sum(extent(meta,n) for n in names(k))
    obs=json.loads(stats.read_text());scores={}
    for name,entry in obs.items():
        gained=entry['all']['weighted_cost']-entry['control']['weighted_cost']
        price=pairbytes(hc,name)-pairbytes(hm,name)
        assert price>0
        scores[name]=gained/price
    key=max(sorted(scores),key=scores.get)
    weight,scale=names(key);layer=int(key.split('.')[3]);width=orig[key+'_scale']['shape'][-1]
    choices=json.loads(selection.read_text())['cutoffs']
    cuts=bytes(x if x>=0 else 255 for x in choices[key]);Path('cuts').write_bytes(cuts)
    factors=json.loads(routing.read_text())['factor'][layer]
    rankseed=933+sorted(choices).index(key)
    wo,_=orig[key]['data_offsets'];so,_=orig[key+'_scale']['data_offsets']
    aa,az=orig[key+'_amax']['data_offsets'];assert az-aa==512
    with official.open('rb') as f:f.seek(ob+aa);amax=struct.unpack('<128f',f.read(512))
    # Recreate the same stratified rows and palette used by the published sample.
    rows=subprocess.check_output([sampling,official,str(ob+wo),str(ob+so),str(width),'cuts',str(rankseed)],text=True,env={**os.environ,'INKLING_OFFLINE_PALETTE':'06'}).splitlines()
    risk={}
    for e,line in enumerate(rows[:128]):
        label,idx,signal,errcontrol,errlow,errall=line.split()
        assert label=='expert' and int(idx)==e
        risk[e]=(int(errall)-int(errcontrol))*amax[e]**2*factors[e]
    assert len(risk)==128
    rank=sorted(risk,key=lambda e:(-risk[e],e));Path('rank.bin').write_bytes(bytes(rank))
    for name,folder,h,off in [('control',control,hc,cb),('ternary',ternary,hm,base)]:
        with (folder/shard).open('rb') as f:
            for suffix,k in [('weights',weight),('scales',scale)]:
                lo,hi=h[k]['data_offsets'];f.seek(off+lo)
                with Path(name+'.'+suffix).open('xb') as g:transfer(f,g,hi-lo)
    # Reserve 4 KiB for safetensors header digit/padding changes, then stat final files.
    budget=pairbytes(hm,key)+(cap-measured)-4096
    trials=subprocess.check_output([fitter,Path.cwd(),str(width),str(budget)],text=True).splitlines()
    selected=int(trials[-1].split()[1]); assert 0<selected<128
    new={};end=0
    swaps={weight:Path('fit.weights'),scale:Path('fit.scales')}
    for k,v in hm.items():
        if k=='__metadata__':new[k]=v;continue
        n=swaps[k].stat().st_size if k in swaps else extent(hm,k)
        new[k]={**v,'data_offsets':[end,end+n]};end+=n
        if k in swaps:new[k]['shape']=[n]
    hdr=json.dumps(new,separators=(',',':'),ensure_ascii=False).encode();hdr+=b' '*(-len(hdr)%8)
    output.mkdir();checks={}
    with (output/shard).open('xb') as dst,(ternary/shard).open('rb') as src:
        dst.write(struct.pack('<Q',len(hdr)));dst.write(hdr)
        for k,v in hm.items():
            if k=='__metadata__':continue
            if k in swaps:
                with swaps[k].open('rb') as f:checks[k]=transfer(f,dst,swaps[k].stat().st_size)
            else:
                a,b=v['data_offsets'];src.seek(base+a);checks[k]=transfer(src,dst,b-a)
    shutil.copytree(ternary/'tiktoken',output/'tiktoken')
    offset,final=head(output/shard);assert final==new and len(checks)==999
    with (output/shard).open('rb') as f:
        for k,h in checks.items():
            a,b=final[k]['data_offsets'];f.seek(offset+a);assert transfer(f,None,b-a)==h
    files={}
    for p in sorted(output.rglob('*')):
        if p.is_file():
            with p.open('rb') as f:files[str(p.relative_to(output))]=dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
    size=sum(v['bytes'] for v in files.values());assert measured<size<=cap
    return dict(plane=key,rank=rank,selected=rank[:selected],trials=trials,package_bytes=size,files=files,plane_score=scores,expert_score=risk,verified_tensors=len(checks),unchanged_tensors=len(checks)-2,training_sha256=hashlib.sha256(routing.read_bytes()).hexdigest())

if __name__=='__main__': print(json.dumps(run(*map(Path,sys.argv[1:])),indent=2))
