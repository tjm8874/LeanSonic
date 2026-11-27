import array, json,struct,hashlib,sys,subprocess as sp
from pathlib import Path
def header(f):
 n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));return n+8,h
def take(f,b,info):
 a,z=info['data_offsets'];f.seek(b+a);v=f.read(z-a);assert len(v)==z-a;return v
raw,compact,decoder=map(Path,sys.argv[1:]);records={}
with raw.open('rb') as a,compact.open('rb') as b:
 ao,ah=header(a);bo,bh=header(b);assert ah.pop('__metadata__')==bh.pop('__metadata__')
 assert len(ah)==len(bh)==59
 for k,m in ah.items():
  opts=[x for x in [k,k+'.rans_v1',k+'.dict_v1',k+'.bf16planes_v1',k+'.row_exp_v1'] if x in bh];assert len(opts)==1
  v=opts[0];original=take(a,ao,m);encoded=take(b,bo,bh[v]);decoded=encoded
  if v!=k:
   assert bh[v]['dtype']=='U8' and bh[v]['shape']==[len(encoded)] and len(encoded)<len(original)
   if v.endswith('.row_exp_v1'):
    assert m['shape']==[201024,256]
    a_len,b_len=struct.unpack_from('<QQ',encoded)
    splits=[encoded[16:16+a_len],encoded[16+a_len:16+a_len+b_len],encoded[16+a_len+b_len:]]
    rows,words=m['shape'][0],len(original)//2
    means,sign_frac,diffs=[sp.check_output([decoder,'rans',str(n)],input=f) for f,n in zip(splits,[rows,words,words],strict=True)]
    out=array.array('H',(((f&128)<<8)|(((diff+means[i//256])&255)<<7)|(f&127)
                          for i,(f,diff) in enumerate(zip(sign_frac,diffs,strict=True))))
    if sys.byteorder!='little':out.byteswap()
    decoded=out.tobytes()
   else:
    decoded=sp.check_output([decoder,'rans' if v.endswith('rans_v1') else 'dict' if v.endswith('dict_v1') else 'bf16planes',str(len(original))],input=encoded)
  else:assert (bh[v]['dtype'],bh[v]['shape'])==(m['dtype'],m['shape'])
  assert decoded==original,k
  records[v]=dict(raw=len(original),bytes=len(encoded),decoded_sha256=hashlib.sha256(decoded).hexdigest())
def file(p):
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
print(json.dumps(dict(raw=file(raw),compact=file(compact),tensors=records),indent=2))
