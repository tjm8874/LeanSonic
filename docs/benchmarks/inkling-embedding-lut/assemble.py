"""Replace just the input embedding; retain the byte alignment of every existing tensor."""
import argparse, hashlib, json, shutil, struct
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument('parent',type=Path);ap.add_argument('embedding',type=Path);ap.add_argument('output',type=Path)
a=ap.parse_args()
name='model-00001-of-00001.safetensors'
old='model.llm.embed.weight.prefix_v1'
key='model.llm.embed.weight.lut8_v1'
def header(path):
    with path.open('rb') as f:
        n,=struct.unpack('<Q',f.read(8));return 8+n,json.loads(f.read(n))
def transfer(src,n,dst=None):
    h=hashlib.sha256()
    while n:
        b=src.read(min(n,4<<20));assert b
        if dst:dst.write(b)
        h.update(b);n-=len(b)
    return h.hexdigest()
base,original=header(a.parent/name)
offset=0;entries={};order=sorted((k for k in original if k not in ('__metadata__',old)),key=lambda k:original[k]['data_offsets'][0])
if '__metadata__' in original:entries['__metadata__']=original['__metadata__']
assert original[old]['dtype']=='U8' and not {'model.llm.embed.weight',key}.intersection(original)
for k in order:
    t=original[k];start,end=t['data_offsets'];length=end-start
    assert offset%8==start%8
    entries[k]=dict(t,data_offsets=[offset,offset+length]);offset+=length
length=a.embedding.stat().st_size
entries[key]=dict(dtype='U8',shape=[length],data_offsets=[offset,offset+length])
blob=json.dumps(entries,ensure_ascii=False,separators=(',',':')).encode();blob+=b' '*(-len(blob)%8)
a.output.mkdir()
digests={}
with (a.output/name).open('xb') as dst,(a.parent/name).open('rb') as src:
    dst.write(struct.pack('<Q',len(blob)));dst.write(blob)
    for k in order:
        i,j=original[k]['data_offsets'];src.seek(base+i);digests[k]=transfer(src,j-i,dst)
    with a.embedding.open('rb') as src:digests[key]=transfer(src,length,dst)
shutil.copytree(a.parent/'tiktoken',a.output/'tiktoken')
base2,h=header(a.output/name);assert h==entries
with (a.output/name).open('rb') as src:
    for k in [*order,key]:
        i,j=h[k]['data_offsets'];src.seek(base2+i);assert transfer(src,j-i)==digests[k]
def files(folder):
    result={}
    for p in sorted(folder.rglob('*')):
        if p.is_file():
            with p.open('rb') as f:result[str(p.relative_to(folder))]=dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
    return result
fs=files(a.output)
assert (a.output/'tiktoken/tokenizer.model').read_bytes()==(a.parent/'tiktoken/tokenizer.model').read_bytes()
before=sum(p.stat().st_size for p in a.parent.rglob('*') if p.is_file())
print(json.dumps(dict(parent_bytes=before,package_bytes=sum(x['bytes'] for x in fs.values()),unchanged_payloads=len(order),frame_sha256=digests[key],files=fs),indent=2))
