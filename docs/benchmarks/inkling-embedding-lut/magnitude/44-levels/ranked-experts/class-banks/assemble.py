from pathlib import Path
import hashlib,json,shutil,struct,sys
parent,frames,dest=map(Path,sys.argv[1:]);dest.mkdir()
shard='model-00001-of-00001.safetensors'
def header(p):
 with p.open('rb') as f:
  n,=struct.unpack('<Q',f.read(8));return 8+n,json.loads(f.read(n))
def copy(f,n,out=None):
 h=hashlib.sha256()
 while n:
  b=f.read(min(n,4<<20));assert b;n-=len(b);h.update(b)
  if out:out.write(b)
 return h.hexdigest()
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
r=json.loads(Path('fit.json').read_text());base,old=header(parent/shard);new={};hashes={};offset=0
if '__metadata__' in old:new['__metadata__']=old['__metadata__']
order=sorted(set(old)-{'__metadata__'},key=lambda k:old[k]['data_offsets'][0])
changed={k:v for k,v in r.items() if v['choice']!='original'}
for k in order:
 t=old[k];a,b=t['data_offsets'];n=changed[k]['bytes'] if k in changed else b-a
 assert (a-offset)%8==0
 new[k]=dict(t,data_offsets=[offset,offset+n]);offset+=n
 if k in changed:
  assert t['dtype']=='U8' and t['shape']==[b-a];new[k]['shape']=[n]
b=json.dumps(new,separators=(',',':')).encode();b+=b' '*(-len(b)%8)
with (parent/shard).open('rb') as src,(dest/shard).open('xb') as dst:
 dst.write(struct.pack('<Q',len(b))+b)
 for k in order:
  a,z=old[k]['data_offsets'];src.seek(a+base)
  if k in changed:
   with (frames/(k+'.frame')).open('rb') as f:hashes[k]=copy(f,changed[k]['bytes'],dst);assert not f.read(1)
  else:hashes[k]=copy(src,z-a,dst)
shutil.copytree(parent/'tiktoken',dest/'tiktoken')
base2,h=header(dest/shard);assert h==new and sum(t['data_offsets'][1]-t['data_offsets'][0] for k,t in new.items() if k!='__metadata__')==offset
with (dest/shard).open('rb') as f:
 for k in order:
  a,b=h[k]['data_offsets'];f.seek(base2+a);assert copy(f,b-a)==hashes[k]
files={str(p.relative_to(dest)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in dest.rglob('*') if p.is_file()}
before=sum(p.stat().st_size for p in parent.rglob('*') if p.is_file());after=sum(x['bytes'] for x in files.values())
assert len(order)==999 and before==11662590054 and after<=before
print(json.dumps(dict(files=files,package_bytes=after,parent_bytes=before,changed=len(changed),unchanged=len(order)-len(changed),checked_decoded_changed=sum(v['decoded_bytes'] for v in changed.values())),indent=2))
