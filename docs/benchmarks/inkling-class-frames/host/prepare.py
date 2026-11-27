from pathlib import Path
import json,struct,hashlib,sys
def shard(name):
 f=(Path(name)/'model-00001-of-00001.safetensors').open('rb');n,=struct.unpack('<Q',f.read(8));h=json.loads(f.read(n));return f,h,f.tell()
def data(src,k):
 f,h,off=src;a,b=h[k]['data_offsets'];f.seek(off+a);v=f.read(b-a);assert len(v)==b-a;return v
src=shard(sys.argv[1]);dst=shard(sys.argv[2]);fit=json.loads(Path(sys.argv[3]).read_text())
r={}
for arm,w in [('w13',144),('w2',72)]:
 a=[(v['trials']['original']-v['bytes'],k) for k,v in fit.items() if '.'+arm+'_weight.' in k and v['choice']=='separate'];_,key=max(a);base=key.removesuffix('.nvscale_four_v1')
 scale=data(src,base+'_scale.mean8_v1');assert scale==data(dst,base+'_scale.mean8_v1')
 Path(arm+'-scale').write_bytes(scale)
 dig={}
 for kind,blob in [('old',data(src,key)),('new',data(dst,base+'.nvscale_class_v1'))]:
  Path(arm+'-'+kind).write_bytes(blob);dig[kind]=dict(bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest())
 r[arm]=dict(key=key,width=w,scales_sha256=hashlib.sha256(scale).hexdigest(),frames=dig)
Path('inputs.json').write_text(json.dumps(r,indent=2)+'\n')
