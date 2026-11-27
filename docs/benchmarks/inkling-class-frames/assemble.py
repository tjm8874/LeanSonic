from pathlib import Path
import hashlib,json,shutil,struct,sys
parent,frames,dest=map(Path,sys.argv[1:]);dest.mkdir()
name='model-00001-of-00001.safetensors'
def header(f):
 n,=struct.unpack('<Q',f.read(8));return 8+n,json.loads(f.read(n))
def through(src,n,out=None):
 h=hashlib.sha256()
 while n:
  b=src.read(min(4<<20,n));assert b;n-=len(b);h.update(b)
  if out:out.write(b)
 return h.hexdigest()
fit=json.loads(Path('fit.json').read_text());hs={}
with (parent/name).open('rb') as f:
 base,old=header(f);keys=sorted(set(old)-{'__metadata__'},key=lambda k:old[k]['data_offsets'][0])
 new={'__metadata__':old['__metadata__']} if '__metadata__' in old else {};off=0;source={}
 for k in keys:
  a,b=old[k]['data_offsets'];n=b-a;e=old[k];nk=k
  if k in fit and fit[k]['choice']=='separate':
   assert e['dtype']=='U8' and e['shape']==[n] and k.endswith('.nvscale_four_v1')
   nk=k.removesuffix('.nvscale_four_v1')+'.nvscale_class_v1';n=fit[k]['bytes'];e=dict(e,shape=[n]);source[nk]=frames/(k+'.frame')
  assert a%8==off%8
  new[nk]=dict(e,data_offsets=[off,off+n]);off+=n
  if nk not in source:source[nk]=(a,n)
 raw=json.dumps(new,separators=(',',':')).encode();raw+=b' '*(-len(raw)%8)
 with (dest/name).open('xb') as out:
  out.write(struct.pack('<Q',len(raw))+raw)
  for k,z in source.items():
   if isinstance(z,tuple):a,n=z;f.seek(base+a);hs[k]=through(f,n,out)
   else:
    with z.open('rb') as src:hs[k]=through(src,z.stat().st_size,out)
shutil.copytree(parent/'tiktoken',dest/'tiktoken')
with (dest/name).open('rb') as f:
 base,h=header(f);assert h==new
 for k in source:
  a,b=h[k]['data_offsets'];f.seek(base+a);assert through(f,b-a)==hs[k]
def files(folder):
 r={}
 for p in sorted(folder.rglob('*')):
  if p.is_file():
   with p.open('rb') as f:r[str(p.relative_to(folder))]=dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
 return r
fs=files(dest);before=sum(p.stat().st_size for p in parent.rglob('*') if p.is_file());after=sum(x['bytes'] for x in fs.values())
assert before==11487467790 and after<=before and len(source)==999
print(json.dumps(dict(parent_bytes=before,package_bytes=after,changed=sum(x['choice']=='separate' for x in fit.values()),files=fs,verified_decoded_bytes=sum(v['decoded_bytes'] for v in fit.values())),indent=2))
