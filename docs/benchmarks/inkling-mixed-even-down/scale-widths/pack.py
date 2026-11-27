import json,struct,subprocess,sys
from pathlib import Path
shard=Path(sys.argv[1])/'model-00001-of-00001.safetensors'
with shard.open('rb') as f:
 n,=struct.unpack('<Q',f.read(8));h=json.loads(f.read(n));base=f.tell()
 name='model.llm.layers.0.mlp.experts.w13_weight'
 lo,hi=h[name+'_amax']['data_offsets'];assert hi-lo==512
 f.seek(base+lo);Path('amax.bin').write_bytes(f.read(hi-lo))
suffix='.nvscale_four_v1'
report={}
for k in sorted(k for k in h if k.endswith(suffix)):
 key=k[:-len(suffix)]
 args=[]
 for stem in (k,key+'_scale.mean8_v1'):
  lo,hi=h[stem]['data_offsets'];args += [str(base+lo),str(hi-lo)]
 width='144' if 'w13_weight' in key else '72'
 line=subprocess.check_output(['./pack',str(shard),*args,width,'save' if key==name else 'verify'],text=True)
 elig,raw,encoded=map(int,line.split());assert raw==128*2304*int(width)*8 and encoded==512+raw-(raw//128//4)*elig
 report[key]=dict(eligible=elig,raw=raw,encoded=encoded)
 if key==name:Path('first.json').write_text(json.dumps(report[key]))
assert len(report)==40
Path('report.json').write_text(json.dumps(report,indent=2)+'\n')
