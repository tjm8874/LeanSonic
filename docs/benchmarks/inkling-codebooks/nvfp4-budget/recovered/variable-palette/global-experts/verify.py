"""Prove each shipped restored expert matches its direct-frame source, then hash unchanged tensors."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,re,struct,subprocess,sys
from pathlib import Path

recipient,low_four,low_direct,high_direct,rankfile,receipt,exe=map(Path,sys.argv[1:])
def open_header(folder):
    path=folder/'model-00001-of-00001.safetensors'
    with path.open('rb') as f:
        n,=struct.unpack('<Q',f.read(8))
        return path,8+n,json.loads(f.read(n))
def digest(f,position,size):
    f.seek(position)
    h=hashlib.sha256()
    while size:
        b=f.read(min(size,4<<20));assert b
        h.update(b);size-=len(b)
    return h.digest()

candidate,older,lo,hi=map(open_header,(recipient,low_four,low_direct,high_direct))
r=json.loads(receipt.read_text());rank=json.loads(rankfile.read_text())['rank']
assert r['rank_sha256']==hashlib.sha256(rankfile.read_bytes()).hexdigest()
selection=defaultdict(list)
for row in rank[:r['prefix']]:selection[row['plane']].append(row['expert'])
assert dict(selection)==r['selected']
assert set(candidate[2])==set(older[2])
def check_plane(item):
    key,ids=item
    cmd=[exe,144 if key.endswith('.w13_weight') else 72,','.join(map(str,ids))]
    for (path,base,meta),suffix in ((lo,'.nvscale_direct_v1'),(hi,'.nvscale_direct_v1'),(candidate,'.nvscale_four_v1')):
        cmd.append(path)
        for name in (key+suffix,key+'_scale.mean8_v1'):
            a,b=meta[name]['data_offsets'];cmd.extend((base+a,b-a))
    out=subprocess.check_output(list(map(str,cmd)),text=True).strip()
    fields=dict(kv.split('=') for kv in out.split())
    values={k:int(v) for k,v in fields.items()}
    assert int(values['experts'])==128 and values['restored']==len(ids) and values['retained']==128-len(ids)
    return key,values
with ThreadPoolExecutor(max_workers=2) as pool: planes=dict(pool.map(check_plane,selection.items()))
changed={s for key in selection for s in (key+'.nvscale_four_v1',key+'_scale.mean8_v1')}
same=0
with candidate[0].open('rb') as new,older[0].open('rb') as old:
    for name,a in candidate[2].items():
        b=older[2][name]
        if name=='__metadata__':assert a==b;continue
        if name in changed:continue
        x,y=a['data_offsets'];m,n=b['data_offsets']
        assert a['dtype']==b['dtype'] and a['shape']==b['shape'] and y-x==n-m
        assert digest(new,candidate[1]+x,y-x)==digest(old,older[1]+m,n-m)
        same+=1
assert same+len(changed)==999
assert (recipient/'tiktoken/tokenizer.model').read_bytes()==(low_four/'tiktoken/tokenizer.model').read_bytes()
assert sum(f.stat().st_size for f in recipient.rglob('*') if f.is_file())==r['package_bytes']
print(json.dumps(dict(planes=planes,unchanged_tensors=same,changed_tensors=len(changed),
                      restored=sum(p['restored'] for p in planes.values()),weight_bytes=sum(p['weight_bytes'] for p in planes.values()),
                      scale_codes=sum(p['scale_codes'] for p in planes.values()),package_bytes=r['package_bytes']),indent=2))
