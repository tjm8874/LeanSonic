import hashlib,json,struct,subprocess,sys
from pathlib import Path
def load(folder):
    shard=folder/'model-00001-of-00001.safetensors'
    with shard.open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0]
        return shard,json.loads(f.read(length)),8+length
def digest(file,offset,size):
    file.seek(offset)
    h=hashlib.sha256()
    while size:
        data=file.read(min(size,8<<20))
        assert data
        h.update(data)
        size-=len(data)
    return h.hexdigest()
p,q,exe,receipt=map(Path,sys.argv[1:])
a,ah,base=load(p); b,bh,cand=load(q)
r=json.loads(receipt.read_text())
assert list(ah)==list(bh) and ah.get('__metadata__')==bh.get('__metadata__')
planes={}; same=0; saving=0
with a.open('rb') as fa,b.open('rb') as fb:
    for name,meta in ah.items():
        if name=='__metadata__':continue
        m=bh[name]; ao,az=meta['data_offsets'];bo,bz=m['data_offsets']
        assert m['dtype']==meta['dtype']
        if not name.endswith('.nvscale_four_v1'):
            assert meta['shape']==m['shape'] and az-ao==bz-bo
            assert digest(fa,base+ao,az-ao)==digest(fb,cand+bo,bz-bo)
            same+=1
            continue
        stem=name.removesuffix('.nvscale_four_v1')
        so,sz=ah[stem+'_scale.mean8_v1']['data_offsets']
        w=144 if stem.endswith('.w13_weight') else 72
        assert meta['shape']==[az-ao] and m['shape']==[bz-bo] and bz-bo<=az-ao
        n,changed=map(int,subprocess.check_output(list(map(str,[exe,a,b,base+ao,az-ao,cand+bo,bz-bo,base+so,sz-so,w])),text=True).split())
        assert az-ao==r['planes'][stem]['old_bytes'] and bz-bo==min(az-ao,r['planes'][stem]['trial_bytes'])
        saving+=az-ao-(bz-bo)
        planes[name]=dict(bytes_reconstructed=n,boundaries_changed=changed,saved_bytes=(az-ao)-(bz-bo))
assert same==959 and len(planes)==40 and sum(x['bytes_reconstructed'] for x in planes.values())==10192158720
assert a.stat().st_size-b.stat().st_size==saving
assert (p/'tiktoken/tokenizer.model').read_bytes()==(q/'tiktoken/tokenizer.model').read_bytes()
print(json.dumps(dict(unchanged_tensor_payloads=same,planes=planes,bytes_saved=saving,
                      package_bytes=r['package_bytes'],files=r['files']),indent=2))
